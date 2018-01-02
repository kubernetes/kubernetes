// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mmap provides a way to memory-map a file.
package mmap

import (
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"syscall"
	"unsafe"
)

// debug is whether to print debugging messages for manual testing.
//
// The runtime.SetFinalizer documentation says that, "The finalizer for x is
// scheduled to run at some arbitrary time after x becomes unreachable. There
// is no guarantee that finalizers will run before a program exits", so we
// cannot automatically test that the finalizer runs. Instead, set this to true
// when running the manual test.
const debug = false

// ReaderAt reads a memory-mapped file.
//
// Like any io.ReaderAt, clients can execute parallel ReadAt calls, but it is
// not safe to call Close and reading methods concurrently.
type ReaderAt struct {
	data []byte
}

// Close closes the reader.
func (r *ReaderAt) Close() error {
	if r.data == nil {
		return nil
	}
	data := r.data
	r.data = nil
	if debug {
		var p *byte
		if len(data) != 0 {
			p = &data[0]
		}
		println("munmap", r, p)
	}
	runtime.SetFinalizer(r, nil)
	return syscall.UnmapViewOfFile(uintptr(unsafe.Pointer(&data[0])))
}

// Len returns the length of the underlying memory-mapped file.
func (r *ReaderAt) Len() int {
	return len(r.data)
}

// At returns the byte at index i.
func (r *ReaderAt) At(i int) byte {
	return r.data[i]
}

// ReadAt implements the io.ReaderAt interface.
func (r *ReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if r.data == nil {
		return 0, errors.New("mmap: closed")
	}
	if off < 0 || int64(len(r.data)) < off {
		return 0, fmt.Errorf("mmap: invalid ReadAt offset %d", off)
	}
	n := copy(p, r.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

// Open memory-maps the named file for reading.
func Open(filename string) (*ReaderAt, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}

	size := fi.Size()
	if size == 0 {
		return &ReaderAt{}, nil
	}
	if size < 0 {
		return nil, fmt.Errorf("mmap: file %q has negative size", filename)
	}
	if size != int64(int(size)) {
		return nil, fmt.Errorf("mmap: file %q is too large", filename)
	}

	low, high := uint32(size), uint32(size>>32)
	fmap, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READONLY, high, low, nil)
	if err != nil {
		return nil, err
	}
	defer syscall.CloseHandle(fmap)
	ptr, err := syscall.MapViewOfFile(fmap, syscall.FILE_MAP_READ, 0, 0, uintptr(size))
	if err != nil {
		return nil, err
	}
	data := (*[1 << 30]byte)(unsafe.Pointer(ptr))[:size]

	r := &ReaderAt{data: data}
	if debug {
		var p *byte
		if len(data) != 0 {
			p = &data[0]
		}
		println("mmap", r, p)
	}
	runtime.SetFinalizer(r, (*ReaderAt).Close)
	return r, nil
}
