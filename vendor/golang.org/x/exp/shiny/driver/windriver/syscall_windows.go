// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run $GOROOT/src/syscall/mksyscall_windows.go -output zsyscall_windows.go syscall_windows.go

package windriver

import (
	"unsafe"
)

type _COLORREF uint32

func _RGB(r, g, b byte) _COLORREF {
	return _COLORREF(r) | _COLORREF(g)<<8 | _COLORREF(b)<<16
}

type _POINT struct {
	X int32
	Y int32
}

type _RECT struct {
	Left   int32
	Top    int32
	Right  int32
	Bottom int32
}

type _BITMAPINFOHEADER struct {
	Size          uint32
	Width         int32
	Height        int32
	Planes        uint16
	BitCount      uint16
	Compression   uint32
	SizeImage     uint32
	XPelsPerMeter int32
	YPelsPerMeter int32
	ClrUsed       uint32
	ClrImportant  uint32
}

type _RGBQUAD struct {
	Blue     byte
	Green    byte
	Red      byte
	Reserved byte
}

type _BITMAPINFO struct {
	Header _BITMAPINFOHEADER
	Colors [1]_RGBQUAD
}

type _BLENDFUNCTION struct {
	BlendOp             byte
	BlendFlags          byte
	SourceConstantAlpha byte
	AlphaFormat         byte
}

// ToUintptr helps to pass bf to syscall.Syscall.
func (bf _BLENDFUNCTION) ToUintptr() uintptr {
	return *((*uintptr)(unsafe.Pointer(&bf)))
}

type _XFORM struct {
	eM11 float32
	eM12 float32
	eM21 float32
	eM22 float32
	eDx  float32
	eDy  float32
}

const (
	_WM_PAINT            = 15
	_WM_WINDOWPOSCHANGED = 71
	_WM_KEYDOWN          = 256
	_WM_KEYUP            = 257
	_WM_SYSKEYDOWN       = 260
	_WM_SYSKEYUP         = 261
	_WM_MOUSEMOVE        = 512
	_WM_LBUTTONDOWN      = 513
	_WM_LBUTTONUP        = 514
	_WM_RBUTTONDOWN      = 516
	_WM_RBUTTONUP        = 517
	_WM_MBUTTONDOWN      = 519
	_WM_MBUTTONUP        = 520
	_WM_USER             = 0x0400
)

const (
	_WS_OVERLAPPED       = 0x00000000
	_WS_CAPTION          = 0x00C00000
	_WS_SYSMENU          = 0x00080000
	_WS_THICKFRAME       = 0x00040000
	_WS_MINIMIZEBOX      = 0x00020000
	_WS_MAXIMIZEBOX      = 0x00010000
	_WS_OVERLAPPEDWINDOW = _WS_OVERLAPPED | _WS_CAPTION | _WS_SYSMENU | _WS_THICKFRAME | _WS_MINIMIZEBOX | _WS_MAXIMIZEBOX
)

const (
	_COLOR_BTNFACE = 15
)

const (
	_IDI_APPLICATION = 32512
	_IDC_ARROW       = 32512
)

const (
	_BI_RGB         = 0
	_DIB_RGB_COLORS = 0

	_AC_SRC_OVER  = 0x00
	_AC_SRC_ALPHA = 0x01

	_SRCCOPY = 0x00cc0020
)

const (
	_GM_COMPATIBLE = 1
	_GM_ADVANCED   = 2

	_MWT_IDENTITY = 1
)

func _GET_X_LPARAM(lp uintptr) int32 {
	return int32(_LOWORD(lp))
}

func _GET_Y_LPARAM(lp uintptr) int32 {
	return int32(_HIWORD(lp))
}

func _LOWORD(l uintptr) uint16 {
	return uint16(uint32(l))
}

func _HIWORD(l uintptr) uint16 {
	return uint16(uint32(l >> 16))
}

// notes to self
// UINT = uint32
// callbacks = uintptr
// strings = *uint16

//sys	_AlphaBlend(dcdest syscall.Handle, xoriginDest int32, yoriginDest int32, wDest int32, hDest int32, dcsrc syscall.Handle, xoriginSrc int32, yoriginSrc int32, wsrc int32, hsrc int32, ftn uintptr) (err error) = msimg32.AlphaBlend
//sys	_BitBlt(dcdest syscall.Handle, xdest int32, ydest int32, width int32, height int32, dcsrc syscall.Handle, xsrc int32, ysrc int32, rop uint32) (err error) = gdi32.BitBlt
//sys	_CreateCompatibleBitmap(dc syscall.Handle, width int32, height int32) (bitmap syscall.Handle, err error) = gdi32.CreateCompatibleBitmap
//sys	_CreateCompatibleDC(dc syscall.Handle) (newdc syscall.Handle, err error) = gdi32.CreateCompatibleDC
//sys	_CreateDIBSection(dc syscall.Handle, bmi *_BITMAPINFO, usage uint32, bits **byte, section syscall.Handle, offset uint32) (bitmap syscall.Handle, err error) = gdi32.CreateDIBSection
//sys	_CreateSolidBrush(color _COLORREF) (brush syscall.Handle, err error) = gdi32.CreateSolidBrush
//sys	_DeleteDC(dc syscall.Handle) (err error) = gdi32.DeleteDC
//sys	_DeleteObject(object syscall.Handle) (err error) = gdi32.DeleteObject
//sys	_FillRect(dc syscall.Handle, rc *_RECT, brush syscall.Handle) (err error) = user32.FillRect
//sys	_ModifyWorldTransform(dc syscall.Handle, x *_XFORM, mode uint32) (err error) = gdi32.ModifyWorldTransform
//sys	_SelectObject(dc syscall.Handle, gdiobj syscall.Handle) (newobj syscall.Handle, err error) = gdi32.SelectObject
//sys	_SetGraphicsMode(dc syscall.Handle, mode int32) (oldmode int32, err error) = gdi32.SetGraphicsMode
//sys	_SetWorldTransform(dc syscall.Handle, x *_XFORM) (err error) = gdi32.SetWorldTransform
//sys	_StretchBlt(dcdest syscall.Handle, xdest int32, ydest int32, wdest int32, hdest int32, dcsrc syscall.Handle, xsrc int32, ysrc int32, wsrc int32, hsrc int32, rop uint32) (err error) = gdi32.StretchBlt
