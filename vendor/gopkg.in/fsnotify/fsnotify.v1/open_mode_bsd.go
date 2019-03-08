// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd openbsd netbsd dragonfly

package fsnotify

import "syscall"

const openMode = syscall.O_NONBLOCK | syscall.O_RDONLY
