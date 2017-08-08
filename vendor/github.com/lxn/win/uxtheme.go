// Copyright 2010 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

import (
	"syscall"
	"unsafe"
)

// CheckBox parts
const (
	BP_CHECKBOX = 3
)

// CheckBox states
const (
	CBS_UNCHECKEDNORMAL = 1
)

// LISTVIEW parts
const (
	LVP_LISTITEM         = 1
	LVP_LISTGROUP        = 2
	LVP_LISTDETAIL       = 3
	LVP_LISTSORTEDDETAIL = 4
	LVP_EMPTYTEXT        = 5
	LVP_GROUPHEADER      = 6
	LVP_GROUPHEADERLINE  = 7
	LVP_EXPANDBUTTON     = 8
	LVP_COLLAPSEBUTTON   = 9
	LVP_COLUMNDETAIL     = 10
)

// LVP_LISTITEM states
const (
	LISS_NORMAL           = 1
	LISS_HOT              = 2
	LISS_SELECTED         = 3
	LISS_DISABLED         = 4
	LISS_SELECTEDNOTFOCUS = 5
	LISS_HOTSELECTED      = 6
)

// TAB parts
const (
	TABP_TABITEM = 1
)

// TABP_TABITEM states
const (
	TIS_NORMAL   = 1
	TIS_HOT      = 2
	TIS_SELECTED = 3
	TIS_DISABLED = 4
	TIS_FOCUSED  = 5
)

// TREEVIEW parts
const (
	TVP_TREEITEM = 1
	TVP_GLYPH    = 2
	TVP_BRANCH   = 3
	TVP_HOTGLYPH = 4
)

// TVP_TREEITEM states
const (
	TREIS_NORMAL           = 1
	TREIS_HOT              = 2
	TREIS_SELECTED         = 3
	TREIS_DISABLED         = 4
	TREIS_SELECTEDNOTFOCUS = 5
	TREIS_HOTSELECTED      = 6
)

// DTTOPTS flags
const (
	DTT_TEXTCOLOR    = 1 << 0
	DTT_BORDERCOLOR  = 1 << 1
	DTT_SHADOWCOLOR  = 1 << 2
	DTT_SHADOWTYPE   = 1 << 3
	DTT_SHADOWOFFSET = 1 << 4
	DTT_BORDERSIZE   = 1 << 5
	DTT_FONTPROP     = 1 << 6
	DTT_COLORPROP    = 1 << 7
	DTT_STATEID      = 1 << 8
	DTT_CALCRECT     = 1 << 9
	DTT_APPLYOVERLAY = 1 << 10
	DTT_GLOWSIZE     = 1 << 11
	DTT_CALLBACK     = 1 << 12
	DTT_COMPOSITED   = 1 << 13
	DTT_VALIDBITS    = DTT_TEXTCOLOR |
		DTT_BORDERCOLOR |
		DTT_SHADOWCOLOR |
		DTT_SHADOWTYPE |
		DTT_SHADOWOFFSET |
		DTT_BORDERSIZE |
		DTT_FONTPROP |
		DTT_COLORPROP |
		DTT_STATEID |
		DTT_CALCRECT |
		DTT_APPLYOVERLAY |
		DTT_GLOWSIZE |
		DTT_COMPOSITED
)

type HTHEME HANDLE

type THEMESIZE int

const (
	TS_MIN THEMESIZE = iota
	TS_TRUE
	TS_DRAW
)

type DTTOPTS struct {
	DwSize              uint32
	DwFlags             uint32
	CrText              COLORREF
	CrBorder            COLORREF
	CrShadow            COLORREF
	ITextShadowType     int
	PtShadowOffset      POINT
	IBorderSize         int
	IFontPropId         int
	IColorPropId        int
	IStateId            int
	FApplyOverlay       BOOL
	IGlowSize           int
	PfnDrawTextCallback uintptr
	LParam              uintptr
}

var (
	// Library
	libuxtheme uintptr

	// Functions
	closeThemeData      uintptr
	drawThemeBackground uintptr
	drawThemeTextEx     uintptr
	getThemePartSize    uintptr
	getThemeTextExtent  uintptr
	isAppThemed         uintptr
	openThemeData       uintptr
	setWindowTheme      uintptr
)

func init() {
	// Library
	libuxtheme = MustLoadLibrary("uxtheme.dll")

	// Functions
	closeThemeData = MustGetProcAddress(libuxtheme, "CloseThemeData")
	drawThemeBackground = MustGetProcAddress(libuxtheme, "DrawThemeBackground")
	drawThemeTextEx = MustGetProcAddress(libuxtheme, "DrawThemeTextEx")
	getThemePartSize = MustGetProcAddress(libuxtheme, "GetThemePartSize")
	getThemeTextExtent = MustGetProcAddress(libuxtheme, "GetThemeTextExtent")
	isAppThemed = MustGetProcAddress(libuxtheme, "IsAppThemed")
	openThemeData = MustGetProcAddress(libuxtheme, "OpenThemeData")
	setWindowTheme = MustGetProcAddress(libuxtheme, "SetWindowTheme")
}

func CloseThemeData(hTheme HTHEME) HRESULT {
	ret, _, _ := syscall.Syscall(closeThemeData, 1,
		uintptr(hTheme),
		0,
		0)

	return HRESULT(ret)
}

func DrawThemeBackground(hTheme HTHEME, hdc HDC, iPartId, iStateId int32, pRect, pClipRect *RECT) HRESULT {
	ret, _, _ := syscall.Syscall6(drawThemeBackground, 6,
		uintptr(hTheme),
		uintptr(hdc),
		uintptr(iPartId),
		uintptr(iStateId),
		uintptr(unsafe.Pointer(pRect)),
		uintptr(unsafe.Pointer(pClipRect)))

	return HRESULT(ret)
}

func DrawThemeTextEx(hTheme HTHEME, hdc HDC, iPartId, iStateId int32, pszText *uint16, iCharCount int32, dwFlags uint32, pRect *RECT, pOptions *DTTOPTS) HRESULT {
	ret, _, _ := syscall.Syscall9(drawThemeTextEx, 9,
		uintptr(hTheme),
		uintptr(hdc),
		uintptr(iPartId),
		uintptr(iStateId),
		uintptr(unsafe.Pointer(pszText)),
		uintptr(iCharCount),
		uintptr(dwFlags),
		uintptr(unsafe.Pointer(pRect)),
		uintptr(unsafe.Pointer(pOptions)))

	return HRESULT(ret)
}

func GetThemePartSize(hTheme HTHEME, hdc HDC, iPartId, iStateId int32, prc *RECT, eSize THEMESIZE, psz *SIZE) HRESULT {
	ret, _, _ := syscall.Syscall9(getThemePartSize, 7,
		uintptr(hTheme),
		uintptr(hdc),
		uintptr(iPartId),
		uintptr(iStateId),
		uintptr(unsafe.Pointer(prc)),
		uintptr(eSize),
		uintptr(unsafe.Pointer(psz)),
		0,
		0)

	return HRESULT(ret)
}

func GetThemeTextExtent(hTheme HTHEME, hdc HDC, iPartId, iStateId int32, pszText *uint16, iCharCount int32, dwTextFlags uint32, pBoundingRect, pExtentRect *RECT) HRESULT {
	ret, _, _ := syscall.Syscall9(getThemeTextExtent, 9,
		uintptr(hTheme),
		uintptr(hdc),
		uintptr(iPartId),
		uintptr(iStateId),
		uintptr(unsafe.Pointer(pszText)),
		uintptr(iCharCount),
		uintptr(dwTextFlags),
		uintptr(unsafe.Pointer(pBoundingRect)),
		uintptr(unsafe.Pointer(pExtentRect)))

	return HRESULT(ret)
}

func IsAppThemed() bool {
	ret, _, _ := syscall.Syscall(isAppThemed, 0,
		0,
		0,
		0)

	return ret != 0
}

func OpenThemeData(hwnd HWND, pszClassList *uint16) HTHEME {
	ret, _, _ := syscall.Syscall(openThemeData, 2,
		uintptr(hwnd),
		uintptr(unsafe.Pointer(pszClassList)),
		0)

	return HTHEME(ret)
}

func SetWindowTheme(hwnd HWND, pszSubAppName, pszSubIdList *uint16) HRESULT {
	ret, _, _ := syscall.Syscall(setWindowTheme, 3,
		uintptr(hwnd),
		uintptr(unsafe.Pointer(pszSubAppName)),
		uintptr(unsafe.Pointer(pszSubIdList)))

	return HRESULT(ret)
}
