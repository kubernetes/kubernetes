/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package promise

// This file defines interfaces for promises and futures and related
// things.  These are about coordination among multiple goroutines and
// so are safe for concurrent calls --- although moderated in some
// cases by a requirement that the caller hold a certain lock.

// WriteOnce represents a variable that is initially not set and can
// be set once and is readable.  This is the common meaning for
// "promise".
type WriteOnce interface {
	// Get reads the current value of this variable.  If this variable
	// is not set yet then this call blocks until this variable gets a
	// value.
	Get() interface{}

	// IsSet returns immediately with an indication of whether this
	// variable has been set.
	IsSet() bool

	// Set normally writes a value into this variable, unblocks every
	// goroutine waiting for this variable to have a value, and
	// returns true.  In the unhappy case that this variable is
	// already set, this method returns false without modifying the
	// variable's value.
	Set(interface{}) bool
}

// LockingWriteOnce is a WriteOnce whose implementation is protected
// by a lock.
type LockingWriteOnce interface {
	WriteOnce

	// GetLocked is like Get but the caller must already hold the
	// lock.  GetLocked may release, and later re-acquire, the lock
	// any number of times.  Get may acquire, and later release, the
	// lock any number of times.
	GetLocked() interface{}

	// IsSetLocked is like IsSet but the caller must already hold the
	// lock.  IsSetLocked may release, and later re-acquire, the lock
	// any number of times.  IsSet may acquire, and later release, the
	// lock any number of times.
	IsSetLocked() bool

	// SetLocked is like Set but the caller must already hold the
	// lock.  SetLocked may release, and later re-acquire, the lock
	// any number of times.  Set may acquire, and later release, the
	// lock any number of times
	SetLocked(interface{}) bool
}
