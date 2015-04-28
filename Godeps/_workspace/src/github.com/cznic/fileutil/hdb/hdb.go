// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

/*
WIP: Package hdb provides a "handle"/value DB like store, but actually it's
closer to the model of a process's virtual memory and its alloc, free and move
methods.

The hdb package is a thin layer around falloc.File providing stable-only
handles and the basic synchronizing primitives.  The central functionality of
hdb are the New, Set, Get and Delete methods of Store.

Conceptual analogy:
	New	    alloc(sizeof(content)), return new "memory" pointer (a handle).

	Get	    memmove() from "memory" "pointed to" by handle to the result content.
		    Note: Handle "knows" the size of its content.

	Set	    memmove() from content to "memory" pointed to by handle.
		    In contrast to real memory, the new content may have different
		    size than the previously stored one w/o additional handling
		    and the "pointer" handle remains the same.

	Delete	    free() the "memory" "pointed to" by handle.
*/
package hdb

import (
	"github.com/cznic/fileutil/falloc"
	"github.com/cznic/fileutil/storage"
)

type Store struct {
	f *falloc.File
}

// New returns a newly created Store backed by accessor, discarding its conents if any.
// If successful, methods on the returned Store can be used for I/O.
// It returns the Store and an error, if any.
func New(accessor storage.Accessor) (store *Store, err error) {
	s := &Store{}
	if s.f, err = falloc.New(accessor); err == nil {
		store = s
	}
	return
}

// Open opens the Store from accessor.
// If successful, methods on the returned Store can be used for data exchange.
// It returns the Store and an error, if any.
func Open(accessor storage.Accessor) (store *Store, err error) {
	s := &Store{}
	if s.f, err = falloc.Open(accessor); err == nil {
		store = s
	}
	return
}

// Close closes the store. Further access to the store has undefined behavior and may panic.
// It returns an error, if any.
func (s *Store) Close() (err error) {
	defer func() {
		s.f = nil
	}()

	return s.f.Close()
}

// Delete deletes the data associated with handle.
// It returns an error if any.
func (s *Store) Delete(handle falloc.Handle) (err error) {
	return s.f.Free(handle)
}

// Get gets the data associated with handle.
// It returns the data and an error, if any.
func (s *Store) Get(handle falloc.Handle) (b []byte, err error) {
	return s.f.Read(handle)
}

// New associates data with a new handle.
// It returns the handle and an error, if any.
func (s *Store) New(b []byte) (handle falloc.Handle, err error) {
	return s.f.Alloc(b)
}

// Set associates data with an existing handle.
// It returns an error, if any.
func (s *Store) Set(handle falloc.Handle, b []byte) (err error) {
	_, err = s.f.Realloc(handle, b, true)
	return
}

// Root returns the handle of the DB root (top level directory, ...).
func (s *Store) Root() falloc.Handle {
	return s.f.Root()
}

// File returns the underlying falloc.File of 's'.
func (s *Store) File() *falloc.File {
	return s.f
}

// Lock locks 's' for writing. If the lock is already locked for reading or writing,
// Lock blocks until the lock is available. To ensure that the lock eventually becomes available,
// a blocked Lock call excludes new readers from acquiring the lock.
func (s *Store) Lock() {
	s.f.Lock()
}

// RLock locks 's' for reading. If the lock is already locked for writing or there is a writer
// already waiting to release the lock, RLock blocks until the writer has released the lock.
func (s *Store) RLock() {
	s.f.RLock()
}

// Unlock unlocks 's' for writing. It's a run-time error if 's' is not locked for writing on entry to Unlock.
//
// As with Mutexes, a locked RWMutex is not associated with a particular goroutine.
// One goroutine may RLock (Lock) 's' and then arrange for another goroutine to RUnlock (Unlock) it.
func (s *Store) Unlock() {
	s.f.Unlock()
}

// RUnlock undoes a single RLock call; it does not affect other simultaneous readers.
// It's a run-time error if 's' is not locked for reading on entry to RUnlock.
func (s *Store) RUnlock() {
	s.f.RUnlock()
}

// LockedNew wraps New in a Lock/Unlock pair.
func (s *Store) LockedNew(b []byte) (handle falloc.Handle, err error) {
	return s.f.LockedAlloc(b)
}

// LockedDelete wraps Delete in a Lock/Unlock pair.
func (s *Store) LockedDelete(handle falloc.Handle) (err error) {
	return s.f.LockedFree(handle)
}

// LockedGet wraps Get in a RLock/RUnlock pair.
func (s *Store) LockedGet(handle falloc.Handle) (b []byte, err error) {
	return s.f.LockedRead(handle)
}

// LockedSet wraps Set in a Lock/Unlock pair.
func (s *Store) LockedSet(handle falloc.Handle, b []byte) (err error) {
	_, err = s.f.Realloc(handle, b, true)
	return
}
