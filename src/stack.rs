//! A concurrent stack make out of nodes which are protected by RCU.

use std::{
    ffi::c_void,
    iter::from_fn,
    mem::forget,
    ops::{Deref, DerefMut},
    ptr::null_mut,
};

use crate::{boxed::RcuBox, call_rcu_memb, Rcu, RcuHead, RcuPtr, RcuRSCS};

/// The nodes out of which the RCU-protected concurrent stack is assembled.
pub struct RcuNode<T>
where
    T: Send,
{
    val: T,
    next: RcuPtr<RcuNode<T>>,
}

unsafe impl<T> Send for RcuNode<T> where T: Send {}

unsafe impl<T> Sync for RcuNode<T> where T: Send + Sync {}

impl<T> RcuNode<T>
where
    T: Send,
{
    /// Decay a fully owned node into its inner value.
    pub fn into_inner(self) -> T {
        self.val
    }
}

impl<T> Deref for RcuNode<T>
where
    T: Send,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.val
    }
}

impl<T> DerefMut for RcuNode<T>
where
    T: Send,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.val
    }
}

/// A concurrent stack make out of linked nodes which are protected by RCU.
pub struct RcuStack<T>
where
    T: Send,
{
    top: RcuPtr<RcuNode<T>>,
}

unsafe impl<T> Send for RcuStack<T> where T: Send {}

unsafe impl<T> Sync for RcuStack<T> where T: Send + Sync {}

impl<T> RcuStack<T>
where
    T: Send,
{
    /// Initialize an empty stack.
    pub fn new(_rcu: &Rcu) -> Self {
        Self {
            top: unsafe { RcuPtr::new(null_mut()) },
        }
    }

    /// Push a new value onto the stack.
    pub fn push(&self, rscs: &RcuRSCS, val: T) {
        let mut old_top = self.top.read(rscs);

        let mut new_top = Box::new(RcuHead::new(RcuNode {
            val,
            next: unsafe { RcuPtr::new(old_top.as_ptr()) },
        }));

        while let Err(new_old_top) = unsafe { self.top.compare_and_update(old_top, &mut *new_top) }
        {
            old_top = new_old_top;
            new_top.val.next = unsafe { RcuPtr::new(new_old_top.as_ptr()) };
        }

        forget(new_top);
    }
}

impl<T> RcuStack<T>
where
    T: Send + Sync,
{
    /// Pop the top node off the stack.
    ///
    /// Full ownership of the inner value can be recovered after a grace period
    /// by calling `node.into_inner().into_inner()` to unwrap both the `RcuBox`
    /// and the `RcuNode` layers.
    pub fn pop(&self, rscs: &RcuRSCS) -> RcuBox<RcuNode<T>> {
        let mut old_top = self.top.read(rscs);

        loop {
            match old_top.as_ref() {
                None => return unsafe { RcuBox::from_raw(null_mut()) },
                Some(node) => {
                    let new_top = node.next.read(rscs).as_ptr();

                    match unsafe { self.top.compare_and_update(old_top, new_top) } {
                        Ok((_new_old_top, old_top)) => return unsafe { RcuBox::from_raw(old_top) },
                        Err(new_old_top) => old_top = new_old_top,
                    }
                }
            }
        }
    }

    /// Iterate through all values held in nodes reachable from the top node.
    pub fn iter<'a>(&'a self, rscs: &'a RcuRSCS) -> impl Iterator<Item = &'a T> + 'a {
        let mut curr = self.top.read(rscs).as_ref();

        from_fn(move || {
            curr.map(|node| {
                curr = node.next.read(rscs).as_ref();

                &node.val
            })
        })
    }
}

impl<T> Drop for RcuStack<T>
where
    T: Send,
{
    fn drop(&mut self) {
        let ptr = self.top.as_ptr();
        if ptr.is_null() {
            return;
        }

        unsafe extern "C" fn drop_later<T>(head: *mut c_void)
        where
            T: Send,
        {
            let mut curr = head as *mut RcuHead<RcuNode<T>>;

            while !curr.is_null() {
                let mut head = Box::from_raw(curr);

                curr = head.val.next.as_ptr();
            }
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

    use crate::{Rcu, RcuThread};

    #[test]
    fn rcu_node_is_send_and_sync() {
        assert_impl_all!(RcuNode<()>: Send, Sync);
    }

    #[test]
    fn rcu_stack_is_send_and_sync() {
        assert_impl_all!(RcuStack<()>: Send, Sync);
    }

    #[test]
    fn it_works() {
        let rcu = Rcu::init();

        let stack = RcuStack::<AtomicUsize>::new(&rcu);

        const NUM_WRITERS: usize = 1 << 4;
        const NUM_READERS: usize = 1 << 10;
        const NUM_ITERS: usize = 1 << 7;

        scope(|scope| {
            scope.spawn(|scope| {
                for _ in 0..NUM_WRITERS {
                    scope.spawn(|_scope| {
                        let rcu = RcuThread::register(&rcu);

                        for _ in 0..NUM_ITERS / 2 {
                            rcu.rscs(|rscs| {
                                stack.push(rscs, AtomicUsize::new(0));
                            });
                        }

                        for _ in 0..NUM_ITERS / 2 {
                            rcu.rscs(|rscs| {
                                let _ = stack.pop(rscs);
                            });
                        }
                    });
                }
            });

            scope.spawn(|scope| {
                for _ in 0..NUM_READERS {
                    scope.spawn(|_scope| {
                        let rcu = RcuThread::register(&rcu);

                        for _ in 0..NUM_ITERS {
                            rcu.rscs(|rscs| {
                                for val in stack.iter(rscs) {
                                    val.fetch_add(1, Ordering::Relaxed);
                                }
                            });
                        }
                    });
                }
            });
        })
        .unwrap();
    }
}
