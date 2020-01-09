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

// This file defines interfaces for promsies and futures and related
// things.

// Readable represents a variable that is initially not set and later
// becomes set.  Some instances may be set to multiple values in
// series.  A Readable for a variable that can only get one value is
// commonly known as a "future".
type Readable interface {
	// Get reads the current value of this variable.  If this variable
	// is not set yet then this call blocks until this variable gets a
	// value.
	Get() interface{}

	// IsSet returns immediately with an indication of whether this
	// variable has been set.
	IsSet() bool
}

// LockingReadable is a Readable whose implementation is protected by
// a lock
type LockingReadable interface {
	Readable

	// GetLocked is like Get but the caller must already hold the lock
	GetLocked() interface{}

	// IsSetLocked is like IsSet but the caller must already hold the lock
	IsSetLocked() bool
}

// WriteOnceOnly represents a variable that is initially not set and
// can be set once.
type WriteOnceOnly interface {
	// Set normally writes a value into this variable, unblocks every
	// goroutine waiting for this variable to have a value, and
	// returns true.  In the unhappy case that this variable is
	// already set, this method returns false.
	Set(interface{}) bool
}

// WriteOnce represents a variable that is initially not set and can
// be set once and is readable.  This is the common meaning for
// "promise".
type WriteOnce interface {
	Readable
	WriteOnceOnly
}

// LockingWriteOnce is a WriteOnce whose implementation is protected
// by a lock.
type LockingWriteOnce interface {
	LockingReadable
	WriteOnceOnly

	// SetLocked is like Set but the caller must already hold the lock
	SetLocked(interface{}) bool
}

// WriteMultipleOnly represents a variable that is initially not set
// and can be set one or more times (unlike a traditional "promise",
// which can be written only once).
type WriteMultipleOnly interface {
	// Set writes a value into this variable and unblocks every
	// goroutine waiting for this variable to have a value
	Set(interface{})
}

// WriteMultiple represents a variable that is initially not set and
// can be set one or more times (unlike a traditional "promise", which
// can be written only once) and is readable.
type WriteMultiple interface {
	Readable
	WriteMultipleOnly
}

// LockingWriteMultiple is a WriteMultiple whose implementation is
// protected by a lock.
type LockingWriteMultiple interface {
	LockingReadable
	WriteMultipleOnly

	// SetLocked is like Set but the caller must already hold the lock
	SetLocked(interface{})
}
