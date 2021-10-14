#![deny(missing_docs)]

//! Safe wrapper of the memb variant of the [userspace RCU library](https://github.com/urcu/userspace-rcu).

use std::{
    cell::Cell,
    ffi::c_void,
    marker::PhantomData,
    ptr::null_mut,
    sync::atomic::{AtomicBool, Ordering},
};

#[cfg(atomic_ptr)]
use std::{ptr::read_volatile, sync::atomic::AtomicPtr};

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Zero-sized token representing whether the userspace RCU library was initialized.
#[derive(Clone, Copy)]
pub struct Rcu;

impl Rcu {
    /// Initialize the userspace RCU library.
    pub fn init() -> Self {
        assert!(!INITIALIZED.swap(true, Ordering::Relaxed));

        unsafe {
            rcu_init_memb();
        }

        Self
    }

    /// Wait for all deferred reclamation initiated prior to calling this by any thread on the system to have completed before it returns.
    pub fn barrier(&self) {
        unsafe {
            rcu_barrier_memb();
        }
    }
}

thread_local! {
    static REGISTERED: Cell<bool> = Cell::new(false);
}

/// Zero-sized token representing whether the current thread was registered with the userspace RCU library.
pub struct RcuThread {
    _marker: PhantomData<*mut ()>,
}

impl RcuThread {
    /// Register the current thread with the userspace RCU library.
    pub fn register(_rcu: &Rcu) -> Self {
        assert!(!REGISTERED.with(|registered| registered.replace(true)));

        unsafe {
            rcu_register_thread_memb();
        }

        Self {
            _marker: PhantomData,
        }
    }

    /// Establish a read side critical section.
    pub fn rscs<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&RcuRSCS) -> T,
    {
        f(&RcuRSCS::lock())
    }
}

impl Drop for RcuThread {
    fn drop(&mut self) {
        unsafe {
            rcu_unregister_thread_memb();

            REGISTERED.with(|registered| registered.set(false));
        }
    }
}

/// Zero-sized token repsenting a read side critical section.
pub struct RcuRSCS {
    _marker: PhantomData<*mut ()>,
}

impl RcuRSCS {
    fn lock() -> Self {
        unsafe {
            rcu_read_lock_memb();
        }

        Self {
            _marker: PhantomData,
        }
    }
}

impl Drop for RcuRSCS {
    fn drop(&mut self) {
        unsafe {
            rcu_read_unlock_memb();
        }
    }
}

/// A variant of `Box<T>` protected by the userspace RCU library.
pub struct RcuBox<T> {
    #[cfg(atomic_ptr)]
    ptr: AtomicPtr<RcuHead<T>>,
    #[cfg(not(atomic_ptr))]
    ptr: *mut RcuHead<T>,
}

unsafe impl<T> Send for RcuBox<T> where T: Send {}

unsafe impl<T> Sync for RcuBox<T> where T: Sync {}

impl<T> RcuBox<T> {
    /// Initialize an RCU-protected box with the given value.
    pub fn new(val: T) -> Self {
        let ptr = Box::into_raw(Box::new(RcuHead::new(val)));

        #[cfg(atomic_ptr)]
        let ptr = AtomicPtr::new(ptr);

        Self { ptr }
    }

    /// Get exclusive access to the inner value given exclusive access to the RCU-protected box.
    pub fn get_mut(&mut self) -> &mut T {
        #[cfg(atomic_ptr)]
        unsafe {
            &mut (**self.ptr.get_mut()).val
        }

        #[cfg(not(atomic_ptr))]
        unsafe {
            &mut (*self.ptr).val
        }
    }

    /// Get shared access to the inner value with a lifetime tied to a read side critical section.
    pub fn read<'a>(&'a self, _rscs: &'a RcuRSCS) -> &'a T {
        unsafe {
            #[cfg(atomic_ptr)]
            let ptr = read_volatile(
                &self.ptr as *const AtomicPtr<RcuHead<T>> as *const *const RcuHead<T>,
            );

            #[cfg(not(atomic_ptr))]
            let ptr = rcu_dereference_sym(self.ptr as *mut c_void) as *mut RcuHead<T>;

            &(*ptr).val
        }
    }
}

impl<T> RcuBox<T>
where
    T: Send,
{
    /// Assign a new value to the RCU-protected box which will eventually become visible to readers.
    pub fn update(&self, val: T) {
        let new_ptr = Box::into_raw(Box::new(RcuHead::new(val)));

        unsafe {
            #[cfg(atomic_ptr)]
            let old_ptr = self.ptr.swap(new_ptr, Ordering::AcqRel);

            #[cfg(not(atomic_ptr))]
            let old_ptr = rcu_xchg_pointer_sym(
                &self.ptr as *const *mut RcuHead<T> as *mut *mut RcuHead<T> as *mut *mut c_void,
                new_ptr as *mut c_void,
            );

            unsafe extern "C" fn drop_rcu<T>(head: *mut c_void) {
                let _ = Box::from_raw(head as *mut RcuHead<T>);
            }

            call_rcu_memb(old_ptr as *mut c_void, drop_rcu::<T>);
        }
    }
}

impl<T> Drop for RcuBox<T> {
    fn drop(&mut self) {
        unsafe {
            #[cfg(atomic_ptr)]
            let _ = Box::from_raw(*self.ptr.get_mut());

            #[cfg(not(atomic_ptr))]
            let _ = Box::from_raw(self.ptr);
        }
    }
}

#[repr(C)]
struct RcuHead<T> {
    // struct rcu_head -> next -> struct cds_wfcq_node -> next
    next: *mut c_void,
    // struct rcu_head -> func
    func: Option<fn(head: *mut c_void)>,
    val: T,
}

impl<T> RcuHead<T> {
    fn new(val: T) -> Self {
        Self {
            next: null_mut(),
            func: None,
            val,
        }
    }
}

extern "C" {
    fn rcu_init_memb();

    fn rcu_register_thread_memb();
    fn rcu_unregister_thread_memb();

    fn rcu_read_lock_memb();
    fn rcu_read_unlock_memb();

    fn rcu_barrier_memb();

    fn call_rcu_memb(head: *mut c_void, func: unsafe extern "C" fn(*mut c_void));
}

#[cfg(not(atomic_ptr))]
extern "C" {
    fn rcu_dereference_sym(ptr: *mut c_void) -> *mut c_void;
    fn rcu_xchg_pointer_sym(ptr: *mut *mut c_void, new_ptr: *mut c_void) -> *mut c_void;
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::AtomicUsize;

    use crossbeam_utils::thread::scope;

    macro_rules! assert_impl {
        ($ty:ty: $tr:path) => {{
            struct Wrapper<T>(T);

            trait AssertImpl {
                fn assert();
            }

            impl<T> AssertImpl for Wrapper<T>
            where
                T: $tr,
            {
                fn assert() {}
            }

            Wrapper::<$ty>::assert();
        }};
    }

    macro_rules! assert_not_impl {
        ($ty:ty: $tr:path) => {{
            struct Wrapper<T>(T);

            trait AssertImpl {
                fn assert();
            }

            impl<T> AssertImpl for Wrapper<T>
            where
                T: $tr,
            {
                fn assert() {}
            }

            trait AssertNotImpl {
                fn assert();
            }

            impl<T> AssertNotImpl for Wrapper<T> {
                fn assert() {}
            }

            Wrapper::<$ty>::assert();
        }};
    }

    #[test]
    fn rcu_send_and_sync() {
        assert_impl!(Rcu: Send);
        assert_impl!(Rcu: Sync);
    }

    #[test]
    fn rcu_thread_is_neither_send_nor_sync() {
        assert_not_impl!(RcuThread: Send);
        assert_not_impl!(RcuThread: Sync);
    }

    #[test]
    fn rcu_rscs_is_neither_send_nor_sync() {
        assert_not_impl!(RcuRSCS: Send);
        assert_not_impl!(RcuRSCS: Sync);
    }

    #[test]
    fn rcu_box_is_send_and_sync() {
        assert_impl!(RcuBox<()>: Send);
        assert_impl!(RcuBox<()>: Sync);
    }

    #[test]
    fn it_works() {
        let rcu = Rcu::init();

        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct CountDrops {
            reads: AtomicUsize,
            generation: usize,
        }

        impl Drop for CountDrops {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }

        let mut count_drops = RcuBox::new(CountDrops {
            reads: AtomicUsize::new(0),
            generation: 0,
        });

        const NUM_THREADS: usize = 1 << 10;
        const NUM_ITERS: usize = 1 << 10;

        scope(|scope| {
            for _ in 0..NUM_THREADS {
                scope.spawn(|_scope| {
                    let rcu = RcuThread::register(&rcu);

                    for _ in 0..NUM_ITERS / 2 {
                        rcu.rscs(|rscs| {
                            count_drops.read(rscs).reads.fetch_add(1, Ordering::Relaxed)
                        });
                    }

                    let generation = rcu.rscs(|rscs| count_drops.read(rscs).generation);

                    count_drops.update(CountDrops {
                        reads: AtomicUsize::new(0),
                        generation: generation + 1,
                    });

                    for _ in 0..NUM_ITERS / 2 {
                        rcu.rscs(|rscs| {
                            count_drops.read(rscs).reads.fetch_add(1, Ordering::Relaxed)
                        });
                    }
                });
            }
        })
        .unwrap();

        assert_ne!(count_drops.get_mut().generation, 0);

        drop(count_drops);

        rcu.barrier();

        assert_eq!(DROPS.load(Ordering::Relaxed), NUM_THREADS + 1);
    }
}
