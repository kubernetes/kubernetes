// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// An abstraction of file like (persistent) storage with optional (abstracted)
// support for structural integrity.

package lldb

import (
	"fmt"

	"github.com/cznic/mathutil"
)

func doubleTrouble(first, second error) error {
	return fmt.Errorf("%q. Additionally, while attempting to recover (rollback): %q", first, second)
}

// A Filer is a []byte-like model of a file or similar entity. It may
// optionally implement support for structural transaction safety. In contrast
// to a file stream, a Filer is not sequentially accessible. ReadAt and WriteAt
// are always "addressed" by an offset and are assumed to perform atomically.
// A Filer is not safe for concurrent access, it's designed for consumption by
// the other objects in package, which should use a Filer from one goroutine
// only or via a mutex. BeginUpdate, EndUpdate and Rollback must be either all
// implemented by a Filer for structural integrity - or they should be all
// no-ops; where/if that requirement is relaxed.
//
// If a Filer wraps another Filer implementation, it usually invokes the same
// methods on the "inner" one, after some possible argument translations etc.
// If a Filer implements the structural transactions handling methods
// (BeginUpdate, EndUpdate and Rollback) as no-ops _and_ wraps another Filer:
// it then still MUST invoke those methods on the inner Filer. This is
// important for the case where a RollbackFiler exists somewhere down the
// chain.  It's also important for an Allocator - to know when it must
// invalidate its FLT cache.
type Filer interface {
	// BeginUpdate increments the "nesting" counter (initially zero). Every
	// call to BeginUpdate must be eventually "balanced" by exactly one of
	// EndUpdate or Rollback. Calls to BeginUpdate may nest.
	BeginUpdate() error

	// Analogous to os.File.Close().
	Close() error

	// EndUpdate decrements the "nesting" counter. If it's zero after that
	// then assume the "storage" has reached structural integrity (after a
	// batch of partial updates). If a Filer implements some support for
	// that (write ahead log, journal, etc.) then the appropriate actions
	// are to be taken for nesting == 0. Invocation of an unbalanced
	// EndUpdate is an error.
	EndUpdate() error

	// Analogous to os.File.Name().
	Name() string

	// PunchHole deallocates space inside a "file" in the byte range
	// starting at off and continuing for size bytes. The actual hole
	// created by PunchHole may be smaller than requested. The Filer size
	// (as reported by `Size()` does not change when hole punching, even
	// when punching the end of a file off.  In contrast to the Linux
	// implementation of FALLOC_FL_PUNCH_HOLE in `fallocate`(2); a Filer is
	// free not only to ignore `PunchHole()` (implement it as a nop), but
	// additionally no guarantees about the content of the hole, when
	// eventually read back, are required, i.e.  any data, not only zeros,
	// can be read from the "hole", including just anything what was left
	// there - with all of the possible security problems.
	PunchHole(off, size int64) error

	// As os.File.ReadAt. Note: `off` is an absolute "file pointer"
	// address and cannot be negative even when a Filer is a InnerFiler.
	ReadAt(b []byte, off int64) (n int, err error)

	// Rollback cancels and undoes the innermost pending update level.
	// Rollback decrements the "nesting" counter.  If a Filer implements
	// some support for keeping structural integrity (write ahead log,
	// journal, etc.) then the appropriate actions are to be taken.
	// Invocation of an unbalanced Rollback is an error.
	Rollback() error

	// Analogous to os.File.FileInfo().Size().
	Size() (int64, error)

	// Analogous to os.Sync().
	Sync() (err error)

	// Analogous to os.File.Truncate().
	Truncate(size int64) error

	// Analogous to os.File.WriteAt(). Note: `off` is an absolute "file
	// pointer" address and cannot be negative even when a Filer is a
	// InnerFiler.
	WriteAt(b []byte, off int64) (n int, err error)
}

var _ Filer = &InnerFiler{} // Ensure InnerFiler is a Filer.

// A InnerFiler is a Filer with added addressing/size translation.
type InnerFiler struct {
	outer Filer
	off   int64
}

// NewInnerFiler returns a new InnerFiler wrapped by `outer` in a way which
// adds `off` to every access.
//
// For example, considering:
//
// 	inner := NewInnerFiler(outer, 10)
//
// then
//
// 	inner.WriteAt([]byte{42}, 4)
//
// translates to
//
// 	outer.WriteAt([]byte{42}, 14)
//
// But an attempt to emulate
//
// 	outer.WriteAt([]byte{17}, 9)
//
// by
//
// 	inner.WriteAt([]byte{17}, -1)
//
// will fail as the `off` parameter can never be < 0. Also note that
//
// 	inner.Size() == outer.Size() - off,
//
// i.e. `inner` pretends no `outer` exists. Finally, after e.g.
//
// 	inner.Truncate(7)
// 	outer.Size() == 17
//
// will be true.
func NewInnerFiler(outer Filer, off int64) *InnerFiler { return &InnerFiler{outer, off} }

// BeginUpdate implements Filer.
func (f *InnerFiler) BeginUpdate() error { return f.outer.BeginUpdate() }

// Close implements Filer.
func (f *InnerFiler) Close() (err error) { return f.outer.Close() }

// EndUpdate implements Filer.
func (f *InnerFiler) EndUpdate() error { return f.outer.EndUpdate() }

// Name implements Filer.
func (f *InnerFiler) Name() string { return f.outer.Name() }

// PunchHole implements Filer. `off`, `size` must be >= 0.
func (f *InnerFiler) PunchHole(off, size int64) error { return f.outer.PunchHole(f.off+off, size) }

// ReadAt implements Filer. `off` must be >= 0.
func (f *InnerFiler) ReadAt(b []byte, off int64) (n int, err error) {
	if off < 0 {
		return 0, &ErrINVAL{f.outer.Name() + ":ReadAt invalid off", off}
	}

	return f.outer.ReadAt(b, f.off+off)
}

// Rollback implements Filer.
func (f *InnerFiler) Rollback() error { return f.outer.Rollback() }

// Size implements Filer.
func (f *InnerFiler) Size() (int64, error) {
	sz, err := f.outer.Size()
	if err != nil {
		return 0, err
	}

	return mathutil.MaxInt64(sz-f.off, 0), nil
}

// Sync() implements Filer.
func (f *InnerFiler) Sync() (err error) {
	return f.outer.Sync()
}

// Truncate implements Filer.
func (f *InnerFiler) Truncate(size int64) error { return f.outer.Truncate(size + f.off) }

// WriteAt implements Filer. `off` must be >= 0.
func (f *InnerFiler) WriteAt(b []byte, off int64) (n int, err error) {
	if off < 0 {
		return 0, &ErrINVAL{f.outer.Name() + ":WriteAt invalid off", off}
	}

	return f.outer.WriteAt(b, f.off+off)
}
