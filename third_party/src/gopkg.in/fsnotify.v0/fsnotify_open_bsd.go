// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd openbsd netbsd

package fsnotify

import "syscall"

const open_FLAGS = syscall.O_NONBLOCK | syscall.O_RDONLY
