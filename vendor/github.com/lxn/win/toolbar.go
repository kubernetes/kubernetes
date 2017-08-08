// Copyright 2010 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

// ToolBar messages
const (
	TB_THUMBPOSITION         = 4
	TB_THUMBTRACK            = 5
	TB_ENDTRACK              = 8
	TB_ENABLEBUTTON          = WM_USER + 1
	TB_CHECKBUTTON           = WM_USER + 2
	TB_PRESSBUTTON           = WM_USER + 3
	TB_HIDEBUTTON            = WM_USER + 4
	TB_INDETERMINATE         = WM_USER + 5
	TB_MARKBUTTON            = WM_USER + 6
	TB_ISBUTTONENABLED       = WM_USER + 9
	TB_ISBUTTONCHECKED       = WM_USER + 10
	TB_ISBUTTONPRESSED       = WM_USER + 11
	TB_ISBUTTONHIDDEN        = WM_USER + 12
	TB_ISBUTTONINDETERMINATE = WM_USER + 13
	TB_ISBUTTONHIGHLIGHTED   = WM_USER + 14
	TB_SETSTATE              = WM_USER + 17
	TB_GETSTATE              = WM_USER + 18
	TB_ADDBITMAP             = WM_USER + 19
	TB_DELETEBUTTON          = WM_USER + 22
	TB_GETBUTTON             = WM_USER + 23
	TB_BUTTONCOUNT           = WM_USER + 24
	TB_COMMANDTOINDEX        = WM_USER + 25
	TB_SAVERESTORE           = WM_USER + 76
	TB_CUSTOMIZE             = WM_USER + 27
	TB_ADDSTRING             = WM_USER + 77
	TB_GETITEMRECT           = WM_USER + 29
	TB_BUTTONSTRUCTSIZE      = WM_USER + 30
	TB_SETBUTTONSIZE         = WM_USER + 31
	TB_SETBITMAPSIZE         = WM_USER + 32
	TB_AUTOSIZE              = WM_USER + 33
	TB_GETTOOLTIPS           = WM_USER + 35
	TB_SETTOOLTIPS           = WM_USER + 36
	TB_SETPARENT             = WM_USER + 37
	TB_SETROWS               = WM_USER + 39
	TB_GETROWS               = WM_USER + 40
	TB_GETBITMAPFLAGS        = WM_USER + 41
	TB_SETCMDID              = WM_USER + 42
	TB_CHANGEBITMAP          = WM_USER + 43
	TB_GETBITMAP             = WM_USER + 44
	TB_GETBUTTONTEXT         = WM_USER + 75
	TB_REPLACEBITMAP         = WM_USER + 46
	TB_GETBUTTONSIZE         = WM_USER + 58
	TB_SETBUTTONWIDTH        = WM_USER + 59
	TB_SETINDENT             = WM_USER + 47
	TB_SETIMAGELIST          = WM_USER + 48
	TB_GETIMAGELIST          = WM_USER + 49
	TB_LOADIMAGES            = WM_USER + 50
	TB_GETRECT               = WM_USER + 51
	TB_SETHOTIMAGELIST       = WM_USER + 52
	TB_GETHOTIMAGELIST       = WM_USER + 53
	TB_SETDISABLEDIMAGELIST  = WM_USER + 54
	TB_GETDISABLEDIMAGELIST  = WM_USER + 55
	TB_SETSTYLE              = WM_USER + 56
	TB_GETSTYLE              = WM_USER + 57
	TB_SETMAXTEXTROWS        = WM_USER + 60
	TB_GETTEXTROWS           = WM_USER + 61
	TB_GETOBJECT             = WM_USER + 62
	TB_GETBUTTONINFO         = WM_USER + 63
	TB_SETBUTTONINFO         = WM_USER + 64
	TB_INSERTBUTTON          = WM_USER + 67
	TB_ADDBUTTONS            = WM_USER + 68
	TB_HITTEST               = WM_USER + 69
	TB_SETDRAWTEXTFLAGS      = WM_USER + 70
	TB_GETHOTITEM            = WM_USER + 71
	TB_SETHOTITEM            = WM_USER + 72
	TB_SETANCHORHIGHLIGHT    = WM_USER + 73
	TB_GETANCHORHIGHLIGHT    = WM_USER + 74
	TB_GETINSERTMARK         = WM_USER + 79
	TB_SETINSERTMARK         = WM_USER + 80
	TB_INSERTMARKHITTEST     = WM_USER + 81
	TB_MOVEBUTTON            = WM_USER + 82
	TB_GETMAXSIZE            = WM_USER + 83
	TB_SETEXTENDEDSTYLE      = WM_USER + 84
	TB_GETEXTENDEDSTYLE      = WM_USER + 85
	TB_GETPADDING            = WM_USER + 86
	TB_SETPADDING            = WM_USER + 87
	TB_SETINSERTMARKCOLOR    = WM_USER + 88
	TB_GETINSERTMARKCOLOR    = WM_USER + 89
	TB_MAPACCELERATOR        = WM_USER + 90
	TB_GETSTRING             = WM_USER + 91
	TB_GETIDEALSIZE          = WM_USER + 99
	TB_SETCOLORSCHEME        = CCM_SETCOLORSCHEME
	TB_GETCOLORSCHEME        = CCM_GETCOLORSCHEME
	TB_SETUNICODEFORMAT      = CCM_SETUNICODEFORMAT
	TB_GETUNICODEFORMAT      = CCM_GETUNICODEFORMAT
)

// ToolBar notifications
const (
	TBN_FIRST    = -700
	TBN_DROPDOWN = TBN_FIRST - 10
)

// TBN_DROPDOWN return codes
const (
	TBDDRET_DEFAULT      = 0
	TBDDRET_NODEFAULT    = 1
	TBDDRET_TREATPRESSED = 2
)

// ToolBar state constants
const (
	TBSTATE_CHECKED       = 1
	TBSTATE_PRESSED       = 2
	TBSTATE_ENABLED       = 4
	TBSTATE_HIDDEN        = 8
	TBSTATE_INDETERMINATE = 16
	TBSTATE_WRAP          = 32
	TBSTATE_ELLIPSES      = 0x40
	TBSTATE_MARKED        = 0x0080
)

// ToolBar style constants
const (
	TBSTYLE_BUTTON       = 0
	TBSTYLE_SEP          = 1
	TBSTYLE_CHECK        = 2
	TBSTYLE_GROUP        = 4
	TBSTYLE_CHECKGROUP   = TBSTYLE_GROUP | TBSTYLE_CHECK
	TBSTYLE_DROPDOWN     = 8
	TBSTYLE_AUTOSIZE     = 16
	TBSTYLE_NOPREFIX     = 32
	TBSTYLE_TOOLTIPS     = 256
	TBSTYLE_WRAPABLE     = 512
	TBSTYLE_ALTDRAG      = 1024
	TBSTYLE_FLAT         = 2048
	TBSTYLE_LIST         = 4096
	TBSTYLE_CUSTOMERASE  = 8192
	TBSTYLE_REGISTERDROP = 0x4000
	TBSTYLE_TRANSPARENT  = 0x8000
)

// ToolBar extended style constants
const (
	TBSTYLE_EX_DRAWDDARROWS       = 0x00000001
	TBSTYLE_EX_MIXEDBUTTONS       = 8
	TBSTYLE_EX_HIDECLIPPEDBUTTONS = 16
	TBSTYLE_EX_DOUBLEBUFFER       = 0x80
)

// ToolBar button style constants
const (
	BTNS_BUTTON        = TBSTYLE_BUTTON
	BTNS_SEP           = TBSTYLE_SEP
	BTNS_CHECK         = TBSTYLE_CHECK
	BTNS_GROUP         = TBSTYLE_GROUP
	BTNS_CHECKGROUP    = TBSTYLE_CHECKGROUP
	BTNS_DROPDOWN      = TBSTYLE_DROPDOWN
	BTNS_AUTOSIZE      = TBSTYLE_AUTOSIZE
	BTNS_NOPREFIX      = TBSTYLE_NOPREFIX
	BTNS_WHOLEDROPDOWN = 0x0080
	BTNS_SHOWTEXT      = 0x0040
)

// TBBUTTONINFO mask flags
const (
	TBIF_IMAGE   = 0x00000001
	TBIF_TEXT    = 0x00000002
	TBIF_STATE   = 0x00000004
	TBIF_STYLE   = 0x00000008
	TBIF_LPARAM  = 0x00000010
	TBIF_COMMAND = 0x00000020
	TBIF_SIZE    = 0x00000040
	TBIF_BYINDEX = 0x80000000
)

type NMMOUSE struct {
	Hdr        NMHDR
	DwItemSpec uintptr
	DwItemData uintptr
	Pt         POINT
	DwHitInfo  uintptr
}

type NMTOOLBAR struct {
	Hdr      NMHDR
	IItem    int32
	TbButton TBBUTTON
	CchText  int32
	PszText  *uint16
	RcButton RECT
}

type TBBUTTON struct {
	IBitmap   int32
	IdCommand int32
	FsState   byte
	FsStyle   byte
	//#ifdef _WIN64
	//    BYTE bReserved[6]          // padding for alignment
	//#elif defined(_WIN32)
	BReserved [2]byte // padding for alignment
	//#endif
	DwData  uintptr
	IString uintptr
}

type TBBUTTONINFO struct {
	CbSize    uint32
	DwMask    uint32
	IdCommand int32
	IImage    int32
	FsState   byte
	FsStyle   byte
	Cx        uint16
	LParam    uintptr
	PszText   uintptr
	CchText   int32
}
