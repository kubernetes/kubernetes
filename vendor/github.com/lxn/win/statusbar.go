// Copyright 2013 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

// Styles
const (
	SBARS_SIZEGRIP = 0x100
	SBARS_TOOLTIPS = 0x800
)

// Messages
const (
	SB_SETPARTS         = WM_USER + 4
	SB_GETPARTS         = WM_USER + 6
	SB_GETBORDERS       = WM_USER + 7
	SB_SETMINHEIGHT     = WM_USER + 8
	SB_SIMPLE           = WM_USER + 9
	SB_GETRECT          = WM_USER + 10
	SB_SETTEXT          = WM_USER + 11
	SB_GETTEXTLENGTH    = WM_USER + 12
	SB_GETTEXT          = WM_USER + 13
	SB_ISSIMPLE         = WM_USER + 14
	SB_SETICON          = WM_USER + 15
	SB_SETTIPTEXT       = WM_USER + 17
	SB_GETTIPTEXT       = WM_USER + 19
	SB_GETICON          = WM_USER + 20
	SB_SETUNICODEFORMAT = CCM_SETUNICODEFORMAT
	SB_GETUNICODEFORMAT = CCM_GETUNICODEFORMAT
	SB_SETBKCOLOR       = CCM_SETBKCOLOR
)

// SB_SETTEXT options
const (
	SBT_NOBORDERS    = 0x100
	SBT_POPOUT       = 0x200
	SBT_RTLREADING   = 0x400
	SBT_NOTABPARSING = 0x800
	SBT_OWNERDRAW    = 0x1000
)

const (
	SBN_FIRST            = -880
	SBN_SIMPLEMODECHANGE = SBN_FIRST - 0
)

const SB_SIMPLEID = 0xff
