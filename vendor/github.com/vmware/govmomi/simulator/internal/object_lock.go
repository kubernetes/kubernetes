package internal

import (
	"fmt"
	"sync"
)

// SharedLockingContext is used to identify when locks can be shared. In
// practice, simulator code uses the simulator.Context for a request, but in
// principle this could be anything.
type SharedLockingContext interface{}

// ObjectLock implements a basic "reference-counted" mutex, where a single
// SharedLockingContext can "share" the lock across code paths or child tasks.
type ObjectLock struct {
	lock sync.Locker

	stateLock sync.Mutex
	heldBy    SharedLockingContext
	count     int64
}

// NewObjectLock creates a new ObjectLock. Pass new(sync.Mutex) if you don't
// have a custom sync.Locker.
func NewObjectLock(lock sync.Locker) *ObjectLock {
	return &ObjectLock{
		lock: lock,
	}
}

// try returns true if the lock has been acquired; false otherwise
func (l *ObjectLock) try(onBehalfOf SharedLockingContext) bool {
	l.stateLock.Lock()
	defer l.stateLock.Unlock()

	if l.heldBy == onBehalfOf {
		l.count = l.count + 1
		return true
	}

	if l.heldBy == nil {
		// we expect no contention for this lock (unless the object has a custom Locker)
		l.lock.Lock()
		l.count = 1
		l.heldBy = onBehalfOf
		return true
	}

	return false
}

// wait returns when there's a chance that try() might succeed.
// It is intended to be better than busy-waiting or sleeping.
func (l *ObjectLock) wait() {
	l.lock.Lock()
	l.lock.Unlock()
}

// Release decrements the reference count. The caller should pass their
// context, which is used to sanity check that the Unlock() call is valid. If
// this is the last reference to the lock for this SharedLockingContext, the lock
// is Unlocked and can be acquired by another SharedLockingContext.
func (l *ObjectLock) Release(onBehalfOf SharedLockingContext) {
	l.stateLock.Lock()
	defer l.stateLock.Unlock()
	if l.heldBy != onBehalfOf {
		panic(fmt.Sprintf("Attempt to unlock on behalf of %#v, but is held by %#v", onBehalfOf, l.heldBy))
	}
	l.count = l.count - 1
	if l.count == 0 {
		l.heldBy = nil
		l.lock.Unlock()
	}
}

// Acquire blocks until it can acquire the lock for onBehalfOf
func (l *ObjectLock) Acquire(onBehalfOf SharedLockingContext) {
	acquired := false
	for !acquired {
		if l.try(onBehalfOf) {
			return
		} else {
			l.wait()
		}
	}
}
