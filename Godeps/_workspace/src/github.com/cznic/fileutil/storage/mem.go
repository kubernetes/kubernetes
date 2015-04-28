// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

package storage

import (
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"os"
)

//TODO -> exported type w/ exported fields
type memaccessor struct {
	f  *os.File
	fi *FileInfo
	b  []byte
}

// Implementation of Accessor.
func (m *memaccessor) BeginUpdate() error { return nil }

// Implementation of Accessor.
func (f *memaccessor) EndUpdate() error { return nil }

// NewMem returns a new Accessor backed by an os.File.  The returned Accessor
// keeps all of the store content in memory.  The memory and file images are
// synced only by Sync and Close.  Recomended for small amounts of data only
// and content which may be lost on process kill/crash.  NewMem return the
// Accessor or an error of any.
//
// NOTE: The returned Accessor implements BeginUpdate and EndUpdate as a no op.
func NewMem(f *os.File) (store Accessor, err error) {
	a := &memaccessor{f: f}
	if err = f.Truncate(0); err != nil {
		return
	}

	var fi os.FileInfo
	if fi, err = a.f.Stat(); err != nil {
		return
	}

	a.fi = NewFileInfo(fi, a)
	store = a
	return
}

// OpenMem return a new Accessor backed by an os.File.  The store content is
// loaded from f.  The returned Accessor keeps all of the store content in
// memory.  The memory and file images are synced only Sync and Close.
// Recomended for small amounts of data only and content which may be lost on
// process kill/crash.  OpenMem return the Accessor or an error of any.
//
// NOTE: The returned Accessor implements BeginUpdate and EndUpdate as a no op.
func OpenMem(f *os.File) (store Accessor, err error) {
	a := &memaccessor{f: f}
	if a.b, err = ioutil.ReadAll(a.f); err != nil {
		a.f.Close()
		return
	}

	var fi os.FileInfo
	if fi, err = a.f.Stat(); err != nil {
		a.f.Close()
		return
	}

	a.fi = NewFileInfo(fi, a)
	store = a
	return
}

// Close implements Accessor. Specifically it synchronizes the memory and file images.
func (a *memaccessor) Close() (err error) {
	defer func() {
		a.b = nil
		if a.f != nil {
			if e := a.f.Close(); e != nil && err == nil {
				err = e
			}
		}
		a.f = nil
	}()

	return a.Sync()
}

func (a *memaccessor) Name() string {
	return a.f.Name()
}

func (a *memaccessor) ReadAt(b []byte, off int64) (n int, err error) {
	if off < 0 || off > math.MaxInt32 {
		return -1, fmt.Errorf("ReadAt: illegal offset %#x", off)
	}

	rq, fp := len(b), int(off)
	if fp+rq > len(a.b) {
		return -1, fmt.Errorf("ReadAt: illegal rq %#x @ offset %#x, len %#x", rq, fp, len(a.b))
	}

	copy(b, a.b[fp:])
	return
}

func (a *memaccessor) Stat() (fi os.FileInfo, err error) {
	i := a.fi
	i.FSize = int64(len(a.b))
	fi = i
	return
}

// Sync implements Accessor. Specifically it synchronizes the memory and file images.
func (a *memaccessor) Sync() (err error) {
	var n int
	if n, err = a.f.WriteAt(a.b, 0); n != len(a.b) {
		return
	}

	return a.f.Truncate(int64(len(a.b)))
}

func (a *memaccessor) Truncate(size int64) (err error) {
	defer func() {
		if e := recover(); e != nil {
			err = e.(error)
		}
	}()

	if size > math.MaxInt32 {
		panic(errors.New("truncate: illegal size"))
	}

	a.b = a.b[:int(size)]
	return
}

func (a *memaccessor) WriteAt(b []byte, off int64) (n int, err error) {
	if off < 0 || off > math.MaxInt32 {
		return -1, errors.New("WriteAt: illegal offset")
	}

	rq, fp, size := len(b), int(off), len(a.b)
	if need := rq + fp; need > size {
		if need <= cap(a.b) {
			a.b = a.b[:need]
		} else {
			nb := make([]byte, need, 2*need)
			copy(nb, a.b)
			a.b = nb
		}
	}

	copy(a.b[int(off):], b)
	return
}
