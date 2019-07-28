// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd netbsd openbsd

package socket

import (
	"runtime"
	"unsafe"
)

func probeProtocolStack() int {
	if runtime.GOOS == "openbsd" && runtime.GOARCH == "arm" {
		return 8
	}
	var p uintptr
	return int(unsafe.Sizeof(p))
}
