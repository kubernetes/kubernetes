// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd netbsd openbsd

package socket

import "unsafe"

func probeProtocolStack() int {
	var p uintptr
	return int(unsafe.Sizeof(p))
}
