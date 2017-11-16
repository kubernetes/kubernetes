// Copyright 2011 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

import (
	"syscall"
	"unsafe"
)

// for second parameter of WglSwapLayerBuffers
const (
	WGL_SWAP_MAIN_PLANE = (1 << 0)
	WGL_SWAP_OVERLAY1   = (1 << 1)
	WGL_SWAP_OVERLAY2   = (1 << 2)
	WGL_SWAP_OVERLAY3   = (1 << 3)
	WGL_SWAP_OVERLAY4   = (1 << 4)
	WGL_SWAP_OVERLAY5   = (1 << 5)
	WGL_SWAP_OVERLAY6   = (1 << 6)
	WGL_SWAP_OVERLAY7   = (1 << 7)
	WGL_SWAP_OVERLAY8   = (1 << 8)
	WGL_SWAP_OVERLAY9   = (1 << 9)
	WGL_SWAP_OVERLAY10  = (1 << 10)
	WGL_SWAP_OVERLAY11  = (1 << 11)
	WGL_SWAP_OVERLAY12  = (1 << 12)
	WGL_SWAP_OVERLAY13  = (1 << 13)
	WGL_SWAP_OVERLAY14  = (1 << 14)
	WGL_SWAP_OVERLAY15  = (1 << 15)
	WGL_SWAP_UNDERLAY1  = (1 << 16)
	WGL_SWAP_UNDERLAY2  = (1 << 17)
	WGL_SWAP_UNDERLAY3  = (1 << 18)
	WGL_SWAP_UNDERLAY4  = (1 << 19)
	WGL_SWAP_UNDERLAY5  = (1 << 20)
	WGL_SWAP_UNDERLAY6  = (1 << 21)
	WGL_SWAP_UNDERLAY7  = (1 << 22)
	WGL_SWAP_UNDERLAY8  = (1 << 23)
	WGL_SWAP_UNDERLAY9  = (1 << 24)
	WGL_SWAP_UNDERLAY10 = (1 << 25)
	WGL_SWAP_UNDERLAY11 = (1 << 26)
	WGL_SWAP_UNDERLAY12 = (1 << 27)
	WGL_SWAP_UNDERLAY13 = (1 << 28)
	WGL_SWAP_UNDERLAY14 = (1 << 29)
	WGL_SWAP_UNDERLAY15 = (1 << 30)
)

type (
	HGLRC HANDLE
)

type LAYERPLANEDESCRIPTOR struct {
	NSize           uint16
	NVersion        uint16
	DwFlags         uint32
	IPixelType      uint8
	CColorBits      uint8
	CRedBits        uint8
	CRedShift       uint8
	CGreenBits      uint8
	CGreenShift     uint8
	CBlueBits       uint8
	CBlueShift      uint8
	CAlphaBits      uint8
	CAlphaShift     uint8
	CAccumBits      uint8
	CAccumRedBits   uint8
	CAccumGreenBits uint8
	CAccumBlueBits  uint8
	CAccumAlphaBits uint8
	CDepthBits      uint8
	CStencilBits    uint8
	CAuxBuffers     uint8
	ILayerType      uint8
	BReserved       uint8
	CrTransparent   COLORREF
}

type POINTFLOAT struct {
	X, Y float32
}

type GLYPHMETRICSFLOAT struct {
	GmfBlackBoxX     float32
	GmfBlackBoxY     float32
	GmfptGlyphOrigin POINTFLOAT
	GmfCellIncX      float32
	GmfCellIncY      float32
}

var (
	// Library
	lib uintptr

	// Functions
	wglCopyContext            uintptr
	wglCreateContext          uintptr
	wglCreateLayerContext     uintptr
	wglDeleteContext          uintptr
	wglDescribeLayerPlane     uintptr
	wglGetCurrentContext      uintptr
	wglGetCurrentDC           uintptr
	wglGetLayerPaletteEntries uintptr
	wglGetProcAddress         uintptr
	wglMakeCurrent            uintptr
	wglRealizeLayerPalette    uintptr
	wglSetLayerPaletteEntries uintptr
	wglShareLists             uintptr
	wglSwapLayerBuffers       uintptr
	wglUseFontBitmaps         uintptr
	wglUseFontOutlines        uintptr
)

func init() {
	// Library
	lib = MustLoadLibrary("opengl32.dll")

	// Functions
	wglCopyContext = MustGetProcAddress(lib, "wglCopyContext")
	wglCreateContext = MustGetProcAddress(lib, "wglCreateContext")
	wglCreateLayerContext = MustGetProcAddress(lib, "wglCreateLayerContext")
	wglDeleteContext = MustGetProcAddress(lib, "wglDeleteContext")
	wglDescribeLayerPlane = MustGetProcAddress(lib, "wglDescribeLayerPlane")
	wglGetCurrentContext = MustGetProcAddress(lib, "wglGetCurrentContext")
	wglGetCurrentDC = MustGetProcAddress(lib, "wglGetCurrentDC")
	wglGetLayerPaletteEntries = MustGetProcAddress(lib, "wglGetLayerPaletteEntries")
	wglGetProcAddress = MustGetProcAddress(lib, "wglGetProcAddress")
	wglMakeCurrent = MustGetProcAddress(lib, "wglMakeCurrent")
	wglRealizeLayerPalette = MustGetProcAddress(lib, "wglRealizeLayerPalette")
	wglSetLayerPaletteEntries = MustGetProcAddress(lib, "wglSetLayerPaletteEntries")
	wglShareLists = MustGetProcAddress(lib, "wglShareLists")
	wglSwapLayerBuffers = MustGetProcAddress(lib, "wglSwapLayerBuffers")
	wglUseFontBitmaps = MustGetProcAddress(lib, "wglUseFontBitmapsW")
	wglUseFontOutlines = MustGetProcAddress(lib, "wglUseFontOutlinesW")
}

func WglCopyContext(hglrcSrc, hglrcDst HGLRC, mask uint) bool {
	ret, _, _ := syscall.Syscall(wglCopyContext, 3,
		uintptr(hglrcSrc),
		uintptr(hglrcDst),
		uintptr(mask))

	return ret != 0
}

func WglCreateContext(hdc HDC) HGLRC {
	ret, _, _ := syscall.Syscall(wglCreateContext, 1,
		uintptr(hdc),
		0,
		0)

	return HGLRC(ret)
}

func WglCreateLayerContext(hdc HDC, iLayerPlane int) HGLRC {
	ret, _, _ := syscall.Syscall(wglCreateLayerContext, 2,
		uintptr(hdc),
		uintptr(iLayerPlane),
		0)

	return HGLRC(ret)
}

func WglDeleteContext(hglrc HGLRC) bool {
	ret, _, _ := syscall.Syscall(wglDeleteContext, 1,
		uintptr(hglrc),
		0,
		0)

	return ret != 0
}

func WglDescribeLayerPlane(hdc HDC, iPixelFormat, iLayerPlane int, nBytes uint8, plpd *LAYERPLANEDESCRIPTOR) bool {
	ret, _, _ := syscall.Syscall6(wglDescribeLayerPlane, 5,
		uintptr(hdc),
		uintptr(iPixelFormat),
		uintptr(iLayerPlane),
		uintptr(nBytes),
		uintptr(unsafe.Pointer(plpd)),
		0)

	return ret != 0
}

func WglGetCurrentContext() HGLRC {
	ret, _, _ := syscall.Syscall(wglGetCurrentContext, 0,
		0,
		0,
		0)

	return HGLRC(ret)
}

func WglGetCurrentDC() HDC {
	ret, _, _ := syscall.Syscall(wglGetCurrentDC, 0,
		0,
		0,
		0)

	return HDC(ret)
}

func WglGetLayerPaletteEntries(hdc HDC, iLayerPlane, iStart, cEntries int, pcr *COLORREF) int {
	ret, _, _ := syscall.Syscall6(wglGetLayerPaletteEntries, 5,
		uintptr(hdc),
		uintptr(iLayerPlane),
		uintptr(iStart),
		uintptr(cEntries),
		uintptr(unsafe.Pointer(pcr)),
		0)

	return int(ret)
}

func WglGetProcAddress(lpszProc *byte) uintptr {
	ret, _, _ := syscall.Syscall(wglGetProcAddress, 1,
		uintptr(unsafe.Pointer(lpszProc)),
		0,
		0)

	return uintptr(ret)
}

func WglMakeCurrent(hdc HDC, hglrc HGLRC) bool {
	ret, _, _ := syscall.Syscall(wglMakeCurrent, 2,
		uintptr(hdc),
		uintptr(hglrc),
		0)

	return ret != 0
}

func WglRealizeLayerPalette(hdc HDC, iLayerPlane int, bRealize bool) bool {
	ret, _, _ := syscall.Syscall(wglRealizeLayerPalette, 3,
		uintptr(hdc),
		uintptr(iLayerPlane),
		uintptr(BoolToBOOL(bRealize)))

	return ret != 0
}

func WglSetLayerPaletteEntries(hdc HDC, iLayerPlane, iStart, cEntries int, pcr *COLORREF) int {
	ret, _, _ := syscall.Syscall6(wglSetLayerPaletteEntries, 5,
		uintptr(hdc),
		uintptr(iLayerPlane),
		uintptr(iStart),
		uintptr(cEntries),
		uintptr(unsafe.Pointer(pcr)),
		0)

	return int(ret)
}

func WglShareLists(hglrc1, hglrc2 HGLRC) bool {
	ret, _, _ := syscall.Syscall(wglShareLists, 2,
		uintptr(hglrc1),
		uintptr(hglrc2),
		0)

	return ret != 0
}

func WglSwapLayerBuffers(hdc HDC, fuPlanes uint) bool {
	ret, _, _ := syscall.Syscall(wglSwapLayerBuffers, 2,
		uintptr(hdc),
		uintptr(fuPlanes),
		0)

	return ret != 0
}

func WglUseFontBitmaps(hdc HDC, first, count, listbase uint32) bool {
	ret, _, _ := syscall.Syscall6(wglUseFontBitmaps, 4,
		uintptr(hdc),
		uintptr(first),
		uintptr(count),
		uintptr(listbase),
		0,
		0)

	return ret != 0
}

func WglUseFontOutlines(hdc HDC, first, count, listbase uint32, deviation, extrusion float32, format int, pgmf *GLYPHMETRICSFLOAT) bool {
	ret, _, _ := syscall.Syscall12(wglUseFontBitmaps, 8,
		uintptr(hdc),
		uintptr(first),
		uintptr(count),
		uintptr(listbase),
		uintptr(deviation),
		uintptr(extrusion),
		uintptr(format),
		uintptr(unsafe.Pointer(pgmf)),
		0,
		0,
		0,
		0)

	return ret != 0
}
