// Copyright 2010 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

import (
	"unsafe"
)

// ToolTip styles
const (
	TTS_ALWAYSTIP = 0x01
	TTS_NOPREFIX  = 0x02
	TTS_NOANIMATE = 0x10
	TTS_NOFADE    = 0x20
	TTS_BALLOON   = 0x40
	TTS_CLOSE     = 0x80
)

// ToolTip messages
const (
	TTM_ACTIVATE        = WM_USER + 1
	TTM_SETDELAYTIME    = WM_USER + 3
	TTM_ADDTOOL         = WM_USER + 50
	TTM_DELTOOL         = WM_USER + 51
	TTM_NEWTOOLRECT     = WM_USER + 52
	TTM_RELAYEVENT      = WM_USER + 7
	TTM_GETTOOLINFO     = WM_USER + 53
	TTM_SETTOOLINFO     = WM_USER + 54
	TTM_HITTEST         = WM_USER + 55
	TTM_GETTEXT         = WM_USER + 56
	TTM_UPDATETIPTEXT   = WM_USER + 57
	TTM_GETTOOLCOUNT    = WM_USER + 13
	TTM_ENUMTOOLS       = WM_USER + 58
	TTM_GETCURRENTTOOL  = WM_USER + 59
	TTM_WINDOWFROMPOINT = WM_USER + 16
	TTM_TRACKACTIVATE   = WM_USER + 17
	TTM_TRACKPOSITION   = WM_USER + 18
	TTM_SETTIPBKCOLOR   = WM_USER + 19
	TTM_SETTIPTEXTCOLOR = WM_USER + 20
	TTM_GETDELAYTIME    = WM_USER + 21
	TTM_GETTIPBKCOLOR   = WM_USER + 22
	TTM_GETTIPTEXTCOLOR = WM_USER + 23
	TTM_SETMAXTIPWIDTH  = WM_USER + 24
	TTM_GETMAXTIPWIDTH  = WM_USER + 25
	TTM_SETMARGIN       = WM_USER + 26
	TTM_GETMARGIN       = WM_USER + 27
	TTM_POP             = WM_USER + 28
	TTM_UPDATE          = WM_USER + 29
	TTM_GETBUBBLESIZE   = WM_USER + 30
	TTM_ADJUSTRECT      = WM_USER + 31
	TTM_SETTITLE        = WM_USER + 33
	TTM_POPUP           = WM_USER + 34
	TTM_GETTITLE        = WM_USER + 35
)

// ToolTip flags
const (
	TTF_IDISHWND    = 0x0001
	TTF_CENTERTIP   = 0x0002
	TTF_RTLREADING  = 0x0004
	TTF_SUBCLASS    = 0x0010
	TTF_TRACK       = 0x0020
	TTF_ABSOLUTE    = 0x0080
	TTF_TRANSPARENT = 0x0100
	TTF_DI_SETITEM  = 0x8000
)

// ToolTip icons
const (
	TTI_NONE    = 0
	TTI_INFO    = 1
	TTI_WARNING = 2
	TTI_ERROR   = 3
)

type TOOLINFO struct {
	CbSize     uint32
	UFlags     uint32
	Hwnd       HWND
	UId        uintptr
	Rect       RECT
	Hinst      HINSTANCE
	LpszText   *uint16
	LParam     uintptr
	LpReserved unsafe.Pointer
}

type TTGETTITLE struct {
	DwSize       uint32
	UTitleBitmap uint32
	Cch          uint32
	PszTitle     *uint16
}
