// Based on ssh/terminal:
// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine

package logrus

import "syscall"

const ioctlReadTermios = syscall.TCGETS

type Termios syscall.Termios
