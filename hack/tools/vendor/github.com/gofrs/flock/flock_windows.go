// Copyright 2015 Tim Heckman. All rights reserved.
// Use of this source code is governed by the BSD 3-Clause
// license that can be found in the LICENSE file.

package flock

import (
	"syscall"
)

// ErrorLockViolation is the error code returned from the Windows syscall when a
// lock would block and you ask to fail immediately.
const ErrorLockViolation syscall.Errno = 0x21 // 33

// Lock is a blocking call to try and take an exclusive file lock. It will wait
// until it is able to obtain the exclusive file lock. It's recommended that
// TryLock() be used over this function. This function may block the ability to
// query the current Locked() or RLocked() status due to a RW-mutex lock.
//
// If we are already locked, this function short-circuits and returns
// immediately assuming it can take the mutex lock.
func (f *Flock) Lock() error {
	return f.lock(&f.l, winLockfileExclusiveLock)
}

// RLock is a blocking call to try and take a shared file lock. It will wait
// until it is able to obtain the shared file lock. It's recommended that
// TryRLock() be used over this function. This function may block the ability to
// query the current Locked() or RLocked() status due to a RW-mutex lock.
//
// If we are already locked, this function short-circuits and returns
// immediately assuming it can take the mutex lock.
func (f *Flock) RLock() error {
	return f.lock(&f.r, winLockfileSharedLock)
}

func (f *Flock) lock(locked *bool, flag uint32) error {
	f.m.Lock()
	defer f.m.Unlock()

	if *locked {
		return nil
	}

	if f.fh == nil {
		if err := f.setFh(); err != nil {
			return err
		}
		defer f.ensureFhState()
	}

	if _, errNo := lockFileEx(syscall.Handle(f.fh.Fd()), flag, 0, 1, 0, &syscall.Overlapped{}); errNo > 0 {
		return errNo
	}

	*locked = true
	return nil
}

// Unlock is a function to unlock the file. This file takes a RW-mutex lock, so
// while it is running the Locked() and RLocked() functions will be blocked.
//
// This function short-circuits if we are unlocked already. If not, it calls
// UnlockFileEx() on the file and closes the file descriptor. It does not remove
// the file from disk. It's up to your application to do.
func (f *Flock) Unlock() error {
	f.m.Lock()
	defer f.m.Unlock()

	// if we aren't locked or if the lockfile instance is nil
	// just return a nil error because we are unlocked
	if (!f.l && !f.r) || f.fh == nil {
		return nil
	}

	// mark the file as unlocked
	if _, errNo := unlockFileEx(syscall.Handle(f.fh.Fd()), 0, 1, 0, &syscall.Overlapped{}); errNo > 0 {
		return errNo
	}

	f.fh.Close()

	f.l = false
	f.r = false
	f.fh = nil

	return nil
}

// TryLock is the preferred function for taking an exclusive file lock. This
// function does take a RW-mutex lock before it tries to lock the file, so there
// is the possibility that this function may block for a short time if another
// goroutine is trying to take any action.
//
// The actual file lock is non-blocking. If we are unable to get the exclusive
// file lock, the function will return false instead of waiting for the lock. If
// we get the lock, we also set the *Flock instance as being exclusive-locked.
func (f *Flock) TryLock() (bool, error) {
	return f.try(&f.l, winLockfileExclusiveLock)
}

// TryRLock is the preferred function for taking a shared file lock. This
// function does take a RW-mutex lock before it tries to lock the file, so there
// is the possibility that this function may block for a short time if another
// goroutine is trying to take any action.
//
// The actual file lock is non-blocking. If we are unable to get the shared file
// lock, the function will return false instead of waiting for the lock. If we
// get the lock, we also set the *Flock instance as being shared-locked.
func (f *Flock) TryRLock() (bool, error) {
	return f.try(&f.r, winLockfileSharedLock)
}

func (f *Flock) try(locked *bool, flag uint32) (bool, error) {
	f.m.Lock()
	defer f.m.Unlock()

	if *locked {
		return true, nil
	}

	if f.fh == nil {
		if err := f.setFh(); err != nil {
			return false, err
		}
		defer f.ensureFhState()
	}

	_, errNo := lockFileEx(syscall.Handle(f.fh.Fd()), flag|winLockfileFailImmediately, 0, 1, 0, &syscall.Overlapped{})

	if errNo > 0 {
		if errNo == ErrorLockViolation || errNo == syscall.ERROR_IO_PENDING {
			return false, nil
		}

		return false, errNo
	}

	*locked = true

	return true, nil
}
