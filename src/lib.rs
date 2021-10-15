#![deny(missing_docs)]

//! Safe wrapper of the memb variant of the [userspace RCU library](https://github.com/urcu/userspace-rcu).

use std::{
    cell::Cell,
    ffi::c_void,
    fmt,
    marker::PhantomData,
    mem::forget,
    ops::{ControlFlow, Deref},
    ptr::{null_mut, read},
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
pub struct RcuBox<T>
where
    T: Send,
{
    #[cfg(atomic_ptr)]
    ptr: AtomicPtr<RcuHead<T>>,
    #[cfg(not(atomic_ptr))]
    ptr: *mut RcuHead<T>,
}

unsafe impl<T> Send for RcuBox<T> where T: Send {}

unsafe impl<T> Sync for RcuBox<T> where T: Send + Sync {}

impl<T> RcuBox<T>
where
    T: Send,
{
    /// Initialize an RCU-protected box with the given value.
    pub fn new(val: T) -> Self {
        let ptr = Box::into_raw(Box::new(RcuHead::new(val)));

        #[cfg(atomic_ptr)]
        let ptr = AtomicPtr::new(ptr);

        Self { ptr }
    }

    /// Wait for the next grace period to take full ownership of the inner value.
    pub fn into_inner(mut self) -> T {
        #![cfg_attr(not(atomic_ptr), allow(unused_mut))]

        unsafe {
            synchronize_rcu_memb();

            #[cfg(atomic_ptr)]
            let head = read(*self.ptr.get_mut());

            #[cfg(not(atomic_ptr))]
            let head = read(self.ptr);

            forget(self);

            head.val
        }
    }

    /// Get shared access to the inner value with a lifetime tied to a read side critical section.
    pub fn read<'a>(&'a self, _rscs: &'a RcuRSCS) -> RcuRef<'a, T> {
        unsafe {
            #[cfg(atomic_ptr)]
            let ptr = read_volatile(
                &self.ptr as *const AtomicPtr<RcuHead<T>> as *const *const RcuHead<T>,
            );

            #[cfg(not(atomic_ptr))]
            let ptr = rcu_dereference_sym(self.ptr as *mut c_void) as *mut RcuHead<T>;

            RcuRef::new(ptr)
        }
    }

    /// Assign a new value to the RCU-protected box which will eventually become visible to readers.
    pub fn update(&self, val: T) -> RcuBox<T> {
        let new_ptr = Box::into_raw(Box::new(RcuHead::new(val)));

        #[cfg(atomic_ptr)]
        let old_ptr = self.ptr.swap(new_ptr, Ordering::AcqRel);

        #[cfg(not(atomic_ptr))]
        let old_ptr = unsafe {
            rcu_xchg_pointer_sym(
                &self.ptr as *const *mut RcuHead<T> as *mut *mut RcuHead<T> as *mut *mut c_void,
                new_ptr as *mut c_void,
            ) as *mut RcuHead<T>
        };

        #[cfg(atomic_ptr)]
        let old_ptr = AtomicPtr::new(old_ptr);

        Self { ptr: old_ptr }
    }

    /// Assign a new value to the RCU-protected box if it matches the given reference to the current value.
    pub fn compare_and_update<'a, R>(
        &'a self,
        mut curr: RcuRef<'a, T>,
        val: T,
        mut retry: R,
    ) -> Result<(RcuRef<'a, T>, RcuBox<T>), RcuRef<'a, T>>
    where
        R: FnMut(RcuRef<'a, T>, &mut T) -> ControlFlow<()>,
    {
        let new_ptr = Box::into_raw(Box::new(RcuHead::new(val)));

        unsafe {
            loop {
                #[cfg(atomic_ptr)]
                let result = self.ptr.compare_exchange(
                    curr.ptr as *mut RcuHead<T>,
                    new_ptr,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );

                #[cfg(not(atomic_ptr))]
                let result = {
                    let curr_ptr = rcu_cmpxchg_pointer_sym(
                        &self.ptr as *const *mut RcuHead<T> as *mut *mut RcuHead<T>
                            as *mut *mut c_void,
                        curr.ptr as *mut RcuHead<T> as *mut c_void,
                        new_ptr as *mut c_void,
                    ) as *mut RcuHead<T>;

                    if curr_ptr == curr.ptr as *mut RcuHead<T> {
                        Ok(curr_ptr)
                    } else {
                        Err(curr_ptr)
                    }
                };

                match result {
                    Ok(curr_ptr) => {
                        curr.ptr = new_ptr;

                        #[cfg(atomic_ptr)]
                        let curr_ptr = AtomicPtr::new(curr_ptr);

                        let prev = Self { ptr: curr_ptr };

                        return Ok((curr, prev));
                    }
                    Err(curr_ptr) => {
                        curr.ptr = curr_ptr;

                        if let ControlFlow::Break(()) = retry(curr, &mut (*new_ptr).val) {
                            let _ = Box::from_raw(new_ptr);

                            return Err(curr);
                        }
                    }
                }
            }
        }
    }
}

impl<T> Drop for RcuBox<T>
where
    T: Send,
{
    fn drop(&mut self) {
        unsafe extern "C" fn drop_later<T>(head: *mut c_void) {
            let _ = Box::from_raw(head as *mut RcuHead<T>);
        }

        #[cfg(atomic_ptr)]
        let ptr = *self.ptr.get_mut();

        #[cfg(not(atomic_ptr))]
        let ptr = self.ptr;

        unsafe {
            call_rcu_memb(ptr as *mut c_void, drop_later::<T>);
        }
    }
}

/// Reference to the current value of a RCU-protected box.
pub struct RcuRef<'a, T> {
    ptr: *const RcuHead<T>,
    _marker: PhantomData<&'a T>,
}

impl<T> Clone for RcuRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for RcuRef<'_, T> {}

impl<T> fmt::Debug for RcuRef<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:#x}", self.ptr as usize)
    }
}

impl<'a, T> RcuRef<'a, T> {
    unsafe fn new(ptr: *const RcuHead<T>) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Decay into reference will full lifetime associated with RSCS.
    pub fn into_ref(self) -> &'a T {
        unsafe { &(*self.ptr).val }
    }
}

impl<T> Deref for RcuRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &(*self.ptr).val }
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

    fn synchronize_rcu_memb();
    fn rcu_barrier_memb();

    fn call_rcu_memb(head: *mut c_void, func: unsafe extern "C" fn(*mut c_void));
}

#[cfg(not(atomic_ptr))]
extern "C" {
    fn rcu_dereference_sym(ptr: *mut c_void) -> *mut c_void;
    fn rcu_xchg_pointer_sym(ptr: *mut *mut c_void, new_ptr: *mut c_void) -> *mut c_void;
    fn rcu_cmpxchg_pointer_sym(
        ptr: *mut *mut c_void,
        old_ptr: *mut c_void,
        new_ptr: *mut c_void,
    ) -> *mut c_void;
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
            updates: usize,
        }

        impl Drop for CountDrops {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }

        let count_drops = RcuBox::new(CountDrops {
            reads: AtomicUsize::new(0),
            updates: 0,
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

                    rcu.rscs(|rscs| {
                        let curr = count_drops.read(rscs);

                        let val = CountDrops {
                            reads: AtomicUsize::new(0),
                            updates: curr.updates + 1,
                        };

                        let _ = count_drops
                            .compare_and_update(curr, val, |curr, val| {
                                val.updates = curr.updates + 1;

                                ControlFlow::Continue(())
                            })
                            .unwrap();
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

        assert_eq!(count_drops.into_inner().updates, NUM_THREADS);

        rcu.barrier();

        assert_eq!(DROPS.load(Ordering::Relaxed), NUM_THREADS + 1);
    }
}
