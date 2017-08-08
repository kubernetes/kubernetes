// Copyright 2012 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

import (
	"syscall"
	"unsafe"
)

func (obj *ITaskbarList3) SetProgressValue(hwnd HWND, current uint32, length uint32) HRESULT {
	ret, _, _ := syscall.Syscall6(obj.LpVtbl.SetProgressValue, 4,
		uintptr(unsafe.Pointer(obj)),
		uintptr(hwnd),
		uintptr(current),
		uintptr(length),
		0,
		0)

	return HRESULT(ret)
}
