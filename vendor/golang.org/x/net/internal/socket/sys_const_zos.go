// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos
// +build zos

package socket

import "syscall"

const (
	sysAF_UNSPEC = syscall.AF_UNSPEC
	sysAF_INET   = syscall.AF_INET
	sysAF_INET6  = syscall.AF_INET6

	sysSOCK_RAW = syscall.SOCK_RAW
)
