// Copyright 2015 Tim Heckman. All rights reserved.
// Use of this source code is governed by the BSD 3-Clause
// license that can be found in the LICENSE file.

// Package flock implements a thread-safe interface for file locking.
// It also includes a non-blocking TryLock() function to allow locking
// without blocking execution.
//
// Package flock is released under the BSD 3-Clause License. See the LICENSE file
// for more details.
//
// While using this library, remember that the locking behaviors are not
// guaranteed to be the same on each platform. For example, some UNIX-like
// operating systems will transparently convert a shared lock to an exclusive
// lock. If you Unlock() the flock from a location where you believe that you
// have the shared lock, you may accidentally drop the exclusive lock.
package flock

import (
	"context"
	"os"
	"runtime"
	"sync"
	"time"
)

// Flock is the struct type to handle file locking. All fields are unexported,
// with access to some of the fields provided by getter methods (Path() and Locked()).
type Flock struct {
	path string
	m    sync.RWMutex
	fh   *os.File
	l    bool
	r    bool
}

// New returns a new instance of *Flock. The only parameter
// it takes is the path to the desired lockfile.
func New(path string) *Flock {
	return &Flock{path: path}
}

// NewFlock returns a new instance of *Flock. The only parameter
// it takes is the path to the desired lockfile.
//
// Deprecated: Use New instead.
func NewFlock(path string) *Flock {
	return New(path)
}

// Close is equivalent to calling Unlock.
//
// This will release the lock and close the underlying file descriptor.
// It will not remove the file from disk, that's up to your application.
func (f *Flock) Close() error {
	return f.Unlock()
}

// Path returns the path as provided in NewFlock().
func (f *Flock) Path() string {
	return f.path
}

// Locked returns the lock state (locked: true, unlocked: false).
//
// Warning: by the time you use the returned value, the state may have changed.
func (f *Flock) Locked() bool {
	f.m.RLock()
	defer f.m.RUnlock()
	return f.l
}

// RLocked returns the read lock state (locked: true, unlocked: false).
//
// Warning: by the time you use the returned value, the state may have changed.
func (f *Flock) RLocked() bool {
	f.m.RLock()
	defer f.m.RUnlock()
	return f.r
}

func (f *Flock) String() string {
	return f.path
}

// TryLockContext repeatedly tries to take an exclusive lock until one of the
// conditions is met: TryLock succeeds, TryLock fails with error, or Context
// Done channel is closed.
func (f *Flock) TryLockContext(ctx context.Context, retryDelay time.Duration) (bool, error) {
	return tryCtx(ctx, f.TryLock, retryDelay)
}

// TryRLockContext repeatedly tries to take a shared lock until one of the
// conditions is met: TryRLock succeeds, TryRLock fails with error, or Context
// Done channel is closed.
func (f *Flock) TryRLockContext(ctx context.Context, retryDelay time.Duration) (bool, error) {
	return tryCtx(ctx, f.TryRLock, retryDelay)
}

func tryCtx(ctx context.Context, fn func() (bool, error), retryDelay time.Duration) (bool, error) {
	if ctx.Err() != nil {
		return false, ctx.Err()
	}
	for {
		if ok, err := fn(); ok || err != nil {
			return ok, err
		}
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case <-time.After(retryDelay):
			// try again
		}
	}
}

func (f *Flock) setFh() error {
	// open a new os.File instance
	// create it if it doesn't exist, and open the file read-only.
	flags := os.O_CREATE
	if runtime.GOOS == "aix" {
		// AIX cannot preform write-lock (ie exclusive) on a
		// read-only file.
		flags |= os.O_RDWR
	} else {
		flags |= os.O_RDONLY
	}
	fh, err := os.OpenFile(f.path, flags, os.FileMode(0600))
	if err != nil {
		return err
	}

	// set the filehandle on the struct
	f.fh = fh
	return nil
}

// ensure the file handle is closed if no lock is held
func (f *Flock) ensureFhState() {
	if !f.l && !f.r && f.fh != nil {
		f.fh.Close()
		f.fh = nil
	}
}
