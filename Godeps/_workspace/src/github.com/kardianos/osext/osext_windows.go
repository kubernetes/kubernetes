// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package osext

import (
	"syscall"
	"unicode/utf16"
	"unsafe"
)

var (
	kernel                = syscall.MustLoadDLL("kernel32.dll")
	getModuleFileNameProc = kernel.MustFindProc("GetModuleFileNameW")
)

// GetModuleFileName() with hModule = NULL
func executable() (exePath string, err error) {
	return getModuleFileName()
}

func getModuleFileName() (string, error) {
	var n uint32
	b := make([]uint16, syscall.MAX_PATH)
	size := uint32(len(b))

	r0, _, e1 := getModuleFileNameProc.Call(0, uintptr(unsafe.Pointer(&b[0])), uintptr(size))
	n = uint32(r0)
	if n == 0 {
		return "", e1
	}
	return string(utf16.Decode(b[0:n])), nil
}
