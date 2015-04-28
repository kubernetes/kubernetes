// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

// WIP: Package storage defines and implements storage providers and store accessors.
package storage

import (
	"os"
	"sync"
	"time"
)

// FileInfo is a type implementing os.FileInfo which has setable fields, like
// the older os.FileInfo used to have. It is used wehere e.g. the Size is
// needed to be faked (encapsulated/memory only file, file cache, etc.).
type FileInfo struct {
	FName    string      // base name of the file
	FSize    int64       // length in bytes
	FMode    os.FileMode // file mode bits
	FModTime time.Time   // modification time
	FIsDir   bool        // abbreviation for Mode().IsDir()
	sys      interface{} // underlying data source (can be nil)
}

// NewFileInfo creates FileInfo from os.FileInfo fi.
func NewFileInfo(fi os.FileInfo, sys interface{}) *FileInfo {
	return &FileInfo{fi.Name(), fi.Size(), fi.Mode(), fi.ModTime(), fi.IsDir(), sys}
}

// Implementation of os.FileInfo
func (fi *FileInfo) Name() string {
	return fi.FName
}

// Implementation of os.FileInfo
func (fi *FileInfo) Size() int64 {
	return fi.FSize
}

// Implementation of os.FileInfo
func (fi *FileInfo) Mode() os.FileMode {
	return fi.FMode
}

// Implementation of os.FileInfo
func (fi *FileInfo) ModTime() time.Time {
	return fi.FModTime
}

// Implementation of os.FileInfo
func (fi *FileInfo) IsDir() bool {
	return fi.FIsDir
}

func (fi *FileInfo) Sys() interface{} {
	return fi.sys
}

// Accessor provides I/O methods to access a store.
type Accessor interface {

	// Close closes the store, rendering it unusable for I/O. It returns an
	// error, if any.
	Close() error

	// Name returns the name of the file as presented to Open.
	Name() string

	// ReadAt reads len(b) bytes from the store starting at byte offset off.
	// It returns the number of bytes read and the error, if any.
	// EOF is signaled by a zero count with err set to os.EOF.
	// ReadAt always returns a non-nil Error when n != len(b).
	ReadAt(b []byte, off int64) (n int, err error)

	// Stat returns the FileInfo structure describing the store. It returns
	// the os.FileInfo and an error, if any.
	Stat() (fi os.FileInfo, err error)

	// Sync commits the current contents of the store to stable storage.
	// Typically, this means flushing the file system's in-memory copy of
	// recently written data to disk.
	Sync() (err error)

	// Truncate changes the size of the store. It does not change the I/O
	// offset.
	Truncate(size int64) error

	// WriteAt writes len(b) bytes to the store starting at byte offset off.
	// It returns the number of bytes written and an error, if any.
	// WriteAt returns a non-nil Error when n != len(b).
	WriteAt(b []byte, off int64) (n int, err error)

	// Before every [structural] change of a store the BeginUpdate is to be
	// called and paired with EndUpdate after the change makes the store's
	// state consistent again. Invocations of BeginUpdate may nest. On
	// invoking the last non nested EndUpdate an implicit "commit" should
	// be performed by the store/provider. The concrete mechanism is
	// unspecified. It could be for example a write-ahead log.  Stores may
	// implement BeginUpdate and EndUpdate as a (documented) no op.
	BeginUpdate() error
	EndUpdate() error
}

// Mutate is a helper/wrapper for executing f in between a.BeginUpdate and
// a.EndUpdate.  Any parameters and/or return values except an error should be
// captured by a function literal passed as f. The returned err is either nil
// or the first non nil error returned from the sequence of execution:
// BeginUpdate, [f,] EndUpdate. The pair BeginUpdate/EndUpdate *is* invoked
// always regardles of any possible errors produced.  Mutate doesn't handle
// panic, it should be used only with a function [literal] which doesn't panic.
// Otherwise the pairing of BeginUpdate/EndUpdate is not guaranteed.
//
// NOTE: If BeginUpdate, which is invoked before f, returns a non-nil error,
// then f is not invoked at all (but EndUpdate still is).
func Mutate(a Accessor, f func() error) (err error) {
	defer func() {
		if e := a.EndUpdate(); e != nil && err == nil {
			err = e
		}
	}()

	if err = a.BeginUpdate(); err != nil {
		return
	}

	return f()
}

// LockedMutate wraps Mutate in yet another layer consisting of a
// l.Lock/l.Unlock pair. All other limitations apply as in Mutate, e.g. no
// panics are allowed to happen - otherwise no guarantees can be made about
// Unlock matching the Lock.
func LockedMutate(a Accessor, l sync.Locker, f func() error) (err error) {
	l.Lock()
	defer l.Unlock()

	return Mutate(a, f)
}
