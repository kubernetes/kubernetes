// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lldb

import (
	"io"
	"os"

	"github.com/cznic/mathutil"
)

var _ Filer = (*OSFiler)(nil)

// OSFile is an os.File like minimal set of methods allowing to construct a
// Filer.
type OSFile interface {
	Name() string
	Stat() (fi os.FileInfo, err error)
	Sync() (err error)
	Truncate(size int64) (err error)
	io.Closer
	io.Reader
	io.ReaderAt
	io.Seeker
	io.Writer
	io.WriterAt
}

// OSFiler is like a SimpleFileFiler but based on an OSFile.
type OSFiler struct {
	f    OSFile
	nest int
	size int64 // not set if < 0
}

// NewOSFiler returns a Filer from an OSFile. This Filer is like the
// SimpleFileFiler, it does not implement the transaction related methods.
func NewOSFiler(f OSFile) (r *OSFiler) {
	return &OSFiler{
		f:    f,
		size: -1,
	}
}

// BeginUpdate implements Filer.
func (f *OSFiler) BeginUpdate() (err error) {
	f.nest++
	return nil
}

// Close implements Filer.
func (f *OSFiler) Close() (err error) {
	if f.nest != 0 {
		return &ErrPERM{(f.Name() + ":Close")}
	}

	return f.f.Close()
}

// EndUpdate implements Filer.
func (f *OSFiler) EndUpdate() (err error) {
	if f.nest == 0 {
		return &ErrPERM{(f.Name() + ":EndUpdate")}
	}

	f.nest--
	return
}

// Name implements Filer.
func (f *OSFiler) Name() string {
	return f.f.Name()
}

// PunchHole implements Filer.
func (f *OSFiler) PunchHole(off, size int64) (err error) {
	return
}

// ReadAt implements Filer.
func (f *OSFiler) ReadAt(b []byte, off int64) (n int, err error) {
	return f.f.ReadAt(b, off)
}

// Rollback implements Filer.
func (f *OSFiler) Rollback() (err error) { return }

// Size implements Filer.
func (f *OSFiler) Size() (n int64, err error) {
	if f.size < 0 { // boot
		fi, err := f.f.Stat()
		if err != nil {
			return 0, err
		}

		f.size = fi.Size()
	}
	return f.size, nil
}

// Sync implements Filer.
func (f *OSFiler) Sync() (err error) {
	return f.f.Sync()
}

// Truncate implements Filer.
func (f *OSFiler) Truncate(size int64) (err error) {
	if size < 0 {
		return &ErrINVAL{"Truncate size", size}
	}

	f.size = size
	return f.f.Truncate(size)
}

// WriteAt implements Filer.
func (f *OSFiler) WriteAt(b []byte, off int64) (n int, err error) {
	if f.size < 0 { // boot
		fi, err := os.Stat(f.f.Name())
		if err != nil {
			return 0, err
		}

		f.size = fi.Size()
	}
	f.size = mathutil.MaxInt64(f.size, int64(len(b))+off)
	return f.f.WriteAt(b, off)
}
