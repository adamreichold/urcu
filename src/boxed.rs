//! A variant of `Box<T>` protected by the userspace RCU library.

use std::{ffi::c_void, mem::forget, ops::ControlFlow, ptr::null_mut};

use crate::{call_rcu_memb, synchronize_rcu_memb, RcuHead, RcuPtr, RcuRSCS, RcuRef};

/// A variant of `Box<T>` protected by the userspace RCU library.
pub struct RcuBox<T>(RcuPtr<T>)
where
    T: Send;

unsafe impl<T> Send for RcuBox<T> where T: Send {}

unsafe impl<T> Sync for RcuBox<T> where T: Send + Sync {}

impl<T> RcuBox<T>
where
    T: Send,
{
    /// Initialize an empty RCU-protected box.
    pub fn empty() -> Self {
        unsafe { Self::from_raw(null_mut()) }
    }

    /// Initialize an RCU-protected box with the given value.
    pub fn new(val: T) -> Self {
        let ptr = Box::into_raw(Box::new(RcuHead::new(val)));

        unsafe { Self::from_raw(ptr) }
    }

    pub(crate) unsafe fn from_raw(ptr: *mut RcuHead<T>) -> Self {
        Self(RcuPtr::new(ptr))
    }

    /// Wait for the next grace period to take full ownership of the inner value.
    pub fn into_inner(mut self) -> Option<T> {
        unsafe {
            synchronize_rcu_memb();

            let ptr = self.0.as_ptr();

            forget(self);

            if !ptr.is_null() {
                let head = Box::from_raw(ptr);

                Some(head.val)
            } else {
                None
            }
        }
    }

    /// Get shared access to the inner value with a lifetime tied to the given read side critical section.
    pub fn read<'a>(&'a self, rscs: &'a RcuRSCS) -> RcuRef<'a, T> {
        self.0.read(rscs)
    }

    /// Assign a new value to the RCU-protected box which will eventually become visible to readers.
    pub fn update(&self, val: T) -> RcuBox<T> {
        let mut val = Box::new(RcuHead::new(val));

        let old_ptr = unsafe { self.0.update(&mut *val) };

        forget(val);

        unsafe { Self::from_raw(old_ptr) }
    }

    /// Assign a new value to the RCU-protected box if it still matches the given reference to the current value.
    pub fn compare_and_update<'a, R>(
        &'a self,
        mut curr: RcuRef<'a, T>,
        val: T,
        mut retry: R,
    ) -> Result<(RcuRef<'a, T>, RcuBox<T>), RcuRef<'a, T>>
    where
        R: FnMut(RcuRef<'a, T>, &mut T) -> ControlFlow<()>,
    {
        let mut val = Box::new(RcuHead::new(val));

        loop {
            match unsafe { self.0.compare_and_update(curr, &mut *val) } {
                Ok((curr, old_ptr)) => {
                    forget(val);

                    return Ok((curr, unsafe { Self::from_raw(old_ptr) }));
                }
                Err(new_curr) => {
                    curr = new_curr;

                    if let ControlFlow::Break(()) = retry(curr, &mut val.val) {
                        return Err(curr);
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
        let ptr = self.0.as_ptr();
        if ptr.is_null() {
            return;
        }

        unsafe extern "C" fn drop_later<T>(head: *mut c_void) {
            let _ = Box::from_raw(head as *mut RcuHead<T>);
        }

        unsafe {
            call_rcu_memb(ptr as *mut c_void, drop_later::<T>);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};

    use crossbeam_utils::thread::scope;
    use static_assertions::assert_impl_all;

    use crate::{tests::RCU, RcuThread};

    #[test]
    fn rcu_box_is_send_and_sync() {
        assert_impl_all!(RcuBox<()>: Send, Sync);
    }

    #[test]
    fn it_works() {
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
                    let rcu = RcuThread::register(&RCU);

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

        assert_eq!(count_drops.into_inner().unwrap().updates, NUM_THREADS);

        RCU.barrier();

        assert_eq!(DROPS.load(Ordering::Relaxed), NUM_THREADS + 1);
    }
}
