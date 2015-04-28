// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

package storage

import (
	"os"
)

// FileAccessor is the concrete type returned by NewFile and OpenFile.
type FileAccessor struct {
	*os.File
}

// Implementation of Accessor.
func (f *FileAccessor) BeginUpdate() error { return nil }

// Implementation of Accessor.
func (f *FileAccessor) EndUpdate() error { return nil }

// NewFile returns an Accessor backed by an os.File named name, It opens the
// named file with specified flag (os.O_RDWR etc.) and perm, (0666 etc.) if
// applicable.  If successful, methods on the returned Accessor can be used for
// I/O.  It returns the Accessor and an Error, if any.
//
// NOTE: The returned Accessor implements BeginUpdate and EndUpdate as a no op.
func NewFile(name string, flag int, perm os.FileMode) (store Accessor, err error) {
	var f FileAccessor
	if f.File, err = os.OpenFile(name, flag, perm); err == nil {
		store = &f
	}
	return
}

// OpenFile returns an Accessor backed by an existing os.File named name, It
// opens the named file with specified flag (os.O_RDWR etc.) and perm, (0666
// etc.) if applicable.  If successful, methods on the returned Accessor can be
// used for I/O.  It returns the Accessor and an Error, if any.
//
// NOTE: The returned Accessor implements BeginUpdate and EndUpdate as a no op.
func OpenFile(name string, flag int, perm os.FileMode) (store Accessor, err error) {
	var f FileAccessor
	if f.File, err = os.OpenFile(name, flag, perm); err == nil {
		store = &f
	}
	return
}
