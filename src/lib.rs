#![deny(missing_docs)]

//! Safe wrapper of the memb variant of the [userspace RCU library](https://github.com/urcu/userspace-rcu).

pub mod boxed;
pub mod stack;

use std::{
    cell::Cell,
    ffi::c_void,
    fmt,
    marker::PhantomData,
    ops::Deref,
    ptr::null_mut,
    sync::atomic::{AtomicBool, Ordering},
};

#[cfg(atomic_ptr)]
use std::{ptr::read_volatile, sync::atomic::AtomicPtr};

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Zero-sized token representing that the userspace RCU library was initialized.
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

/// Zero-sized token representing that the current thread was registered with the userspace RCU library.
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

struct RcuPtr<T>
where
    T: Send,
{
    #[cfg(atomic_ptr)]
    ptr: AtomicPtr<RcuHead<T>>,
    #[cfg(not(atomic_ptr))]
    ptr: *mut RcuHead<T>,
}

impl<T> RcuPtr<T>
where
    T: Send,
{
    unsafe fn new(ptr: *mut RcuHead<T>) -> Self {
        #[cfg(atomic_ptr)]
        let ptr = AtomicPtr::new(ptr);

        Self { ptr }
    }

    fn as_ptr(&mut self) -> *mut RcuHead<T> {
        #[cfg(atomic_ptr)]
        let ptr = *self.ptr.get_mut();

        #[cfg(not(atomic_ptr))]
        let ptr = self.ptr;

        ptr
    }

    fn read<'a>(&'a self, _rscs: &'a RcuRSCS) -> RcuRef<'a, T> {
        unsafe {
            #[cfg(atomic_ptr)]
            let ptr =
                read_volatile(&self.ptr as *const AtomicPtr<RcuHead<T>> as *const *mut RcuHead<T>);

            #[cfg(not(atomic_ptr))]
            let ptr = rcu_dereference_sym(self.ptr as *mut c_void) as *mut RcuHead<T>;

            RcuRef::new(ptr)
        }
    }

    unsafe fn update(&self, new_ptr: *mut RcuHead<T>) -> *mut RcuHead<T> {
        #[cfg(atomic_ptr)]
        let old_ptr = self.ptr.swap(new_ptr, Ordering::AcqRel);

        #[cfg(not(atomic_ptr))]
        let old_ptr = unsafe {
            rcu_xchg_pointer_sym(
                &self.ptr as *const *mut RcuHead<T> as *mut *mut RcuHead<T> as *mut *mut c_void,
                new_ptr as *mut c_void,
            ) as *mut RcuHead<T>
        };

        old_ptr
    }

    unsafe fn compare_and_update<'a>(
        &'a self,
        mut curr: RcuRef<'a, T>,
        new_ptr: *mut RcuHead<T>,
    ) -> Result<(RcuRef<'a, T>, *mut RcuHead<T>), RcuRef<'a, T>> {
        #[cfg(atomic_ptr)]
        let result = self.ptr.compare_exchange(
            curr.ptr as *mut RcuHead<T>,
            new_ptr,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        #[cfg(not(atomic_ptr))]
        let result = unsafe {
            let curr_ptr = rcu_cmpxchg_pointer_sym(
                &self.ptr as *const *mut RcuHead<T> as *mut *mut RcuHead<T> as *mut *mut c_void,
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

                Ok((curr, curr_ptr))
            }
            Err(curr_ptr) => {
                curr.ptr = curr_ptr;

                Err(curr)
            }
        }
    }
}

/// The current value of an RCU-protected pointer.
pub struct RcuRef<'a, T> {
    ptr: *mut RcuHead<T>,
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
    unsafe fn new(ptr: *mut RcuHead<T>) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Access reference if the RCU-protected was not empty.
    pub fn as_ref(&self) -> Option<&T> {
        if !self.ptr.is_null() {
            unsafe { Some(&(*self.ptr).val) }
        } else {
            None
        }
    }

    /// Decay into reference will full lifetime associated with RSCS.
    pub fn into_ref(self) -> Option<&'a T> {
        if !self.ptr.is_null() {
            unsafe { Some(&(*self.ptr).val) }
        } else {
            None
        }
    }
}

impl<T> RcuRef<'_, T>
where
    T: Send,
{
    fn as_ptr(&self) -> *mut RcuHead<T> {
        self.ptr
    }
}

impl<T> Deref for RcuRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref().unwrap()
    }
}

#[repr(C)]
struct RcuHead<T> {
    // struct rcu_head -> next -> struct cds_wfcq_node -> next
    _next: *mut c_void,
    // struct rcu_head -> func
    _func: Option<fn(head: *mut c_void)>,
    val: T,
}

impl<T> RcuHead<T> {
    fn new(val: T) -> Self {
        Self {
            _next: null_mut(),
            _func: None,
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

    use lazy_static::lazy_static;
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    lazy_static! {
        pub static ref RCU: Rcu = Rcu::init();
    }

    #[test]
    fn rcu_send_and_sync() {
        assert_impl_all!(Rcu: Send, Sync);
    }

    #[test]
    fn rcu_thread_is_neither_send_nor_sync() {
        assert_not_impl_any!(RcuThread: Send, Sync);
    }

    #[test]
    fn rcu_rscs_is_neither_send_nor_sync() {
        assert_not_impl_any!(RcuRSCS: Send, Sync);
    }

    #[test]
    fn rcu_ref_is_neither_send_nor_sync() {
        assert_not_impl_any!(RcuRef<()>: Send, Sync);
    }
}
