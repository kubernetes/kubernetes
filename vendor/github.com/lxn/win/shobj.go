// Copyright 2012 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

import (
	"syscall"
	"unsafe"
)

var (
	CLSID_TaskbarList = CLSID{0x56FDF344, 0xFD6D, 0x11d0, [8]byte{0x95, 0x8A, 0x00, 0x60, 0x97, 0xC9, 0xA0, 0x90}}
	IID_ITaskbarList3 = IID{0xea1afb91, 0x9e28, 0x4b86, [8]byte{0x90, 0xe9, 0x9e, 0x9f, 0x8a, 0x5e, 0xef, 0xaf}}
)

//TBPFLAG
const (
	TBPF_NOPROGRESS    = 0
	TBPF_INDETERMINATE = 0x1
	TBPF_NORMAL        = 0x2
	TBPF_ERROR         = 0x4
	TBPF_PAUSED        = 0x8
)

type ITaskbarList3Vtbl struct {
	QueryInterface        uintptr
	AddRef                uintptr
	Release               uintptr
	HrInit                uintptr
	AddTab                uintptr
	DeleteTab             uintptr
	ActivateTab           uintptr
	SetActiveAlt          uintptr
	MarkFullscreenWindow  uintptr
	SetProgressValue      uintptr
	SetProgressState      uintptr
	RegisterTab           uintptr
	UnregisterTab         uintptr
	SetTabOrder           uintptr
	SetTabActive          uintptr
	ThumbBarAddButtons    uintptr
	ThumbBarUpdateButtons uintptr
	ThumbBarSetImageList  uintptr
	SetOverlayIcon        uintptr
	SetThumbnailTooltip   uintptr
	SetThumbnailClip      uintptr
}

type ITaskbarList3 struct {
	LpVtbl *ITaskbarList3Vtbl
}

func (obj *ITaskbarList3) SetProgressState(hwnd HWND, state int) HRESULT {
	ret, _, _ := syscall.Syscall(obj.LpVtbl.SetProgressState, 3,
		uintptr(unsafe.Pointer(obj)),
		uintptr(hwnd),
		uintptr(state))
	return HRESULT(ret)
}
