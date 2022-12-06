// Copyright 2020 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux
// +build linux

package gensupport

import "syscall"

func init() {
	// Initialize syscallRetryable to return true on transient socket-level
	// errors. These errors are specific to Linux.
	syscallRetryable = func(err error) bool { return err == syscall.ECONNRESET || err == syscall.ECONNREFUSED }
}
