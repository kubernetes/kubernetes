// Copyright 2016 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

import (
	"syscall"
	"unsafe"
)

// Button control messages
const (
	BCM_FIRST            = 0x1600
	BCM_GETIDEALSIZE     = BCM_FIRST + 0x0001
	BCM_SETIMAGELIST     = BCM_FIRST + 0x0002
	BCM_GETIMAGELIST     = BCM_FIRST + 0x0003
	BCM_SETTEXTMARGIN    = BCM_FIRST + 0x0004
	BCM_GETTEXTMARGIN    = BCM_FIRST + 0x0005
	BCM_SETDROPDOWNSTATE = BCM_FIRST + 0x0006
	BCM_SETSPLITINFO     = BCM_FIRST + 0x0007
	BCM_GETSPLITINFO     = BCM_FIRST + 0x0008
	BCM_SETNOTE          = BCM_FIRST + 0x0009
	BCM_GETNOTE          = BCM_FIRST + 0x000A
	BCM_GETNOTELENGTH    = BCM_FIRST + 0x000B
)

const (
	CCM_FIRST            = 0x2000
	CCM_LAST             = CCM_FIRST + 0x200
	CCM_SETBKCOLOR       = 8193
	CCM_SETCOLORSCHEME   = 8194
	CCM_GETCOLORSCHEME   = 8195
	CCM_GETDROPTARGET    = 8196
	CCM_SETUNICODEFORMAT = 8197
	CCM_GETUNICODEFORMAT = 8198
	CCM_SETVERSION       = 0x2007
	CCM_GETVERSION       = 0x2008
	CCM_SETNOTIFYWINDOW  = 0x2009
	CCM_SETWINDOWTHEME   = 0x200b
	CCM_DPISCALE         = 0x200c
)

// Common controls styles
const (
	CCS_TOP           = 1
	CCS_NOMOVEY       = 2
	CCS_BOTTOM        = 3
	CCS_NORESIZE      = 4
	CCS_NOPARENTALIGN = 8
	CCS_ADJUSTABLE    = 32
	CCS_NODIVIDER     = 64
	CCS_VERT          = 128
	CCS_LEFT          = 129
	CCS_NOMOVEX       = 130
	CCS_RIGHT         = 131
)

// InitCommonControlsEx flags
const (
	ICC_LISTVIEW_CLASSES   = 1
	ICC_TREEVIEW_CLASSES   = 2
	ICC_BAR_CLASSES        = 4
	ICC_TAB_CLASSES        = 8
	ICC_UPDOWN_CLASS       = 16
	ICC_PROGRESS_CLASS     = 32
	ICC_HOTKEY_CLASS       = 64
	ICC_ANIMATE_CLASS      = 128
	ICC_WIN95_CLASSES      = 255
	ICC_DATE_CLASSES       = 256
	ICC_USEREX_CLASSES     = 512
	ICC_COOL_CLASSES       = 1024
	ICC_INTERNET_CLASSES   = 2048
	ICC_PAGESCROLLER_CLASS = 4096
	ICC_NATIVEFNTCTL_CLASS = 8192
	INFOTIPSIZE            = 1024
	ICC_STANDARD_CLASSES   = 0x00004000
	ICC_LINK_CLASS         = 0x00008000
)

// WM_NOTITY messages
const (
	NM_FIRST              = 0
	NM_OUTOFMEMORY        = ^uint32(0)  // NM_FIRST - 1
	NM_CLICK              = ^uint32(1)  // NM_FIRST - 2
	NM_DBLCLK             = ^uint32(2)  // NM_FIRST - 3
	NM_RETURN             = ^uint32(3)  // NM_FIRST - 4
	NM_RCLICK             = ^uint32(4)  // NM_FIRST - 5
	NM_RDBLCLK            = ^uint32(5)  // NM_FIRST - 6
	NM_SETFOCUS           = ^uint32(6)  // NM_FIRST - 7
	NM_KILLFOCUS          = ^uint32(7)  // NM_FIRST - 8
	NM_CUSTOMDRAW         = ^uint32(11) // NM_FIRST - 12
	NM_HOVER              = ^uint32(12) // NM_FIRST - 13
	NM_NCHITTEST          = ^uint32(13) // NM_FIRST - 14
	NM_KEYDOWN            = ^uint32(14) // NM_FIRST - 15
	NM_RELEASEDCAPTURE    = ^uint32(15) // NM_FIRST - 16
	NM_SETCURSOR          = ^uint32(16) // NM_FIRST - 17
	NM_CHAR               = ^uint32(17) // NM_FIRST - 18
	NM_TOOLTIPSCREATED    = ^uint32(18) // NM_FIRST - 19
	NM_LAST               = ^uint32(98) // NM_FIRST - 99
	TRBN_THUMBPOSCHANGING = 0xfffffa22  // TRBN_FIRST - 1
)

// ProgressBar messages
const (
	PBM_SETPOS      = WM_USER + 2
	PBM_DELTAPOS    = WM_USER + 3
	PBM_SETSTEP     = WM_USER + 4
	PBM_STEPIT      = WM_USER + 5
	PBM_SETMARQUEE  = WM_USER + 10
	PBM_SETRANGE32  = 1030
	PBM_GETRANGE    = 1031
	PBM_GETPOS      = 1032
	PBM_SETBARCOLOR = 1033
	PBM_SETBKCOLOR  = CCM_SETBKCOLOR
)

// ProgressBar styles
const (
	PBS_SMOOTH   = 0x01
	PBS_VERTICAL = 0x04
	PBS_MARQUEE  = 0x08
)

// TrackBar (Slider) messages
const (
	TBM_GETPOS      = WM_USER
	TBM_GETRANGEMIN = WM_USER + 1
	TBM_GETRANGEMAX = WM_USER + 2
	TBM_SETPOS      = WM_USER + 5
	TBM_SETRANGEMIN = WM_USER + 7
	TBM_SETRANGEMAX = WM_USER + 8
)

// TrackBar (Slider) styles
const (
	TBS_VERT     = 0x002
	TBS_TOOLTIPS = 0x100
)

// ImageList creation flags
const (
	ILC_MASK          = 0x00000001
	ILC_COLOR         = 0x00000000
	ILC_COLORDDB      = 0x000000FE
	ILC_COLOR4        = 0x00000004
	ILC_COLOR8        = 0x00000008
	ILC_COLOR16       = 0x00000010
	ILC_COLOR24       = 0x00000018
	ILC_COLOR32       = 0x00000020
	ILC_PALETTE       = 0x00000800
	ILC_MIRROR        = 0x00002000
	ILC_PERITEMMIRROR = 0x00008000
)

const (
	CDDS_PREPAINT      = 0x00000001
	CDDS_POSTPAINT     = 0x00000002
	CDDS_PREERASE      = 0x00000003
	CDDS_POSTERASE     = 0x00000004
	CDDS_ITEM          = 0x00010000
	CDDS_ITEMPREPAINT  = CDDS_ITEM | CDDS_PREPAINT
	CDDS_ITEMPOSTPAINT = CDDS_ITEM | CDDS_POSTPAINT
	CDDS_ITEMPREERASE  = CDDS_ITEM | CDDS_PREERASE
	CDDS_ITEMPOSTERASE = CDDS_ITEM | CDDS_POSTERASE
	CDDS_SUBITEM       = 0x00020000
)

const (
	CDIS_SELECTED         = 0x0001
	CDIS_GRAYED           = 0x0002
	CDIS_DISABLED         = 0x0004
	CDIS_CHECKED          = 0x0008
	CDIS_FOCUS            = 0x0010
	CDIS_DEFAULT          = 0x0020
	CDIS_HOT              = 0x0040
	CDIS_MARKED           = 0x0080
	CDIS_INDETERMINATE    = 0x0100
	CDIS_SHOWKEYBOARDCUES = 0x0200
	CDIS_NEARHOT          = 0x0400
	CDIS_OTHERSIDEHOT     = 0x0800
	CDIS_DROPHILITED      = 0x1000
)

const (
	CDRF_DODEFAULT         = 0x00000000
	CDRF_NEWFONT           = 0x00000002
	CDRF_SKIPDEFAULT       = 0x00000004
	CDRF_DOERASE           = 0x00000008
	CDRF_NOTIFYPOSTPAINT   = 0x00000010
	CDRF_NOTIFYITEMDRAW    = 0x00000020
	CDRF_NOTIFYSUBITEMDRAW = 0x00000020
	CDRF_NOTIFYPOSTERASE   = 0x00000040
	CDRF_SKIPPOSTPAINT     = 0x00000100
)

const (
	LPSTR_TEXTCALLBACK = ^uintptr(0)
	I_CHILDRENCALLBACK = -1
)

type HIMAGELIST HANDLE

type INITCOMMONCONTROLSEX struct {
	DwSize, DwICC uint32
}

type NMCUSTOMDRAW struct {
	Hdr         NMHDR
	DwDrawStage uint32
	Hdc         HDC
	Rc          RECT
	DwItemSpec  uintptr
	UItemState  uint32
	LItemlParam uintptr
}

var (
	// Library
	libcomctl32 uintptr

	// Functions
	imageList_Add         uintptr
	imageList_AddMasked   uintptr
	imageList_Create      uintptr
	imageList_Destroy     uintptr
	imageList_ReplaceIcon uintptr
	initCommonControlsEx  uintptr
)

func init() {
	// Library
	libcomctl32 = MustLoadLibrary("comctl32.dll")

	// Functions
	imageList_Add = MustGetProcAddress(libcomctl32, "ImageList_Add")
	imageList_AddMasked = MustGetProcAddress(libcomctl32, "ImageList_AddMasked")
	imageList_Create = MustGetProcAddress(libcomctl32, "ImageList_Create")
	imageList_Destroy = MustGetProcAddress(libcomctl32, "ImageList_Destroy")
	imageList_ReplaceIcon = MustGetProcAddress(libcomctl32, "ImageList_ReplaceIcon")
	initCommonControlsEx = MustGetProcAddress(libcomctl32, "InitCommonControlsEx")

	// Initialize the common controls we support
	var initCtrls INITCOMMONCONTROLSEX
	initCtrls.DwSize = uint32(unsafe.Sizeof(initCtrls))
	initCtrls.DwICC = ICC_LINK_CLASS | ICC_LISTVIEW_CLASSES | ICC_PROGRESS_CLASS | ICC_TAB_CLASSES | ICC_TREEVIEW_CLASSES

	InitCommonControlsEx(&initCtrls)
}

func ImageList_Add(himl HIMAGELIST, hbmImage, hbmMask HBITMAP) int32 {
	ret, _, _ := syscall.Syscall(imageList_Add, 3,
		uintptr(himl),
		uintptr(hbmImage),
		uintptr(hbmMask))

	return int32(ret)
}

func ImageList_AddMasked(himl HIMAGELIST, hbmImage HBITMAP, crMask COLORREF) int32 {
	ret, _, _ := syscall.Syscall(imageList_AddMasked, 3,
		uintptr(himl),
		uintptr(hbmImage),
		uintptr(crMask))

	return int32(ret)
}

func ImageList_Create(cx, cy int32, flags uint32, cInitial, cGrow int32) HIMAGELIST {
	ret, _, _ := syscall.Syscall6(imageList_Create, 5,
		uintptr(cx),
		uintptr(cy),
		uintptr(flags),
		uintptr(cInitial),
		uintptr(cGrow),
		0)

	return HIMAGELIST(ret)
}

func ImageList_Destroy(hIml HIMAGELIST) bool {
	ret, _, _ := syscall.Syscall(imageList_Destroy, 1,
		uintptr(hIml),
		0,
		0)

	return ret != 0
}

func ImageList_ReplaceIcon(himl HIMAGELIST, i int32, hicon HICON) int32 {
	ret, _, _ := syscall.Syscall(imageList_ReplaceIcon, 3,
		uintptr(himl),
		uintptr(i),
		uintptr(hicon))

	return int32(ret)
}

func InitCommonControlsEx(lpInitCtrls *INITCOMMONCONTROLSEX) bool {
	ret, _, _ := syscall.Syscall(initCommonControlsEx, 1,
		uintptr(unsafe.Pointer(lpInitCtrls)),
		0,
		0)

	return ret != 0
}
