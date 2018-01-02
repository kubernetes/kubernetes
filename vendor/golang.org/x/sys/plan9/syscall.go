// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build plan9

// Package plan9 contains an interface to the low-level operating system
// primitives. OS details vary depending on the underlying system, and
// by default, godoc will display the OS-specific documentation for the current
// system. If you want godoc to display documentation for another
// system, set $GOOS and $GOARCH to the desired system. For example, if
// you want to view documentation for freebsd/arm on linux/amd64, set $GOOS
// to freebsd and $GOARCH to arm.
// The primary use of this package is inside other packages that provide a more
// portable interface to the system, such as "os", "time" and "net".  Use
// those packages rather than this one if you can.
// For details of the functions and data types in this package consult
// the manuals for the appropriate operating system.
// These calls return err == nil to indicate success; otherwise
// err represents an operating system error describing the failure and
// holds a value of type syscall.ErrorString.
package plan9 // import "golang.org/x/sys/plan9"

import "unsafe"

// ByteSliceFromString returns a NUL-terminated slice of bytes
// containing the text of s. If s contains a NUL byte at any
// location, it returns (nil, EINVAL).
func ByteSliceFromString(s string) ([]byte, error) {
	for i := 0; i < len(s); i++ {
		if s[i] == 0 {
			return nil, EINVAL
		}
	}
	a := make([]byte, len(s)+1)
	copy(a, s)
	return a, nil
}

// BytePtrFromString returns a pointer to a NUL-terminated array of
// bytes containing the text of s. If s contains a NUL byte at any
// location, it returns (nil, EINVAL).
func BytePtrFromString(s string) (*byte, error) {
	a, err := ByteSliceFromString(s)
	if err != nil {
		return nil, err
	}
	return &a[0], nil
}

// Single-word zero for use when we need a valid pointer to 0 bytes.
// See mksyscall.pl.
var _zero uintptr

func (ts *Timespec) Unix() (sec int64, nsec int64) {
	return int64(ts.Sec), int64(ts.Nsec)
}

func (tv *Timeval) Unix() (sec int64, nsec int64) {
	return int64(tv.Sec), int64(tv.Usec) * 1000
}

func (ts *Timespec) Nano() int64 {
	return int64(ts.Sec)*1e9 + int64(ts.Nsec)
}

func (tv *Timeval) Nano() int64 {
	return int64(tv.Sec)*1e9 + int64(tv.Usec)*1000
}

// use is a no-op, but the compiler cannot see that it is.
// Calling use(p) ensures that p is kept live until that point.
//go:noescape
func use(p unsafe.Pointer)
