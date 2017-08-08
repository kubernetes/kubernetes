// Copyright 2010 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

import (
	"syscall"
	"unsafe"
)

// Common error codes
const (
	CDERR_DIALOGFAILURE   = 0xFFFF
	CDERR_FINDRESFAILURE  = 0x0006
	CDERR_INITIALIZATION  = 0x0002
	CDERR_LOADRESFAILURE  = 0x0007
	CDERR_LOADSTRFAILURE  = 0x0005
	CDERR_LOCKRESFAILURE  = 0x0008
	CDERR_MEMALLOCFAILURE = 0x0009
	CDERR_MEMLOCKFAILURE  = 0x000A
	CDERR_NOHINSTANCE     = 0x0004
	CDERR_NOHOOK          = 0x000B
	CDERR_NOTEMPLATE      = 0x0003
	CDERR_REGISTERMSGFAIL = 0x000C
	CDERR_STRUCTSIZE      = 0x0001
)

// CHOOSECOLOR flags
const (
	CC_ANYCOLOR             = 0x00000100
	CC_ENABLEHOOK           = 0x00000010
	CC_ENABLETEMPLATE       = 0x00000020
	CC_ENABLETEMPLATEHANDLE = 0x00000040
	CC_FULLOPEN             = 0x00000002
	CC_PREVENTFULLOPEN      = 0x00000004
	CC_RGBINIT              = 0x00000001
	CC_SHOWHELP             = 0x00000008
	CC_SOLIDCOLOR           = 0x00000080
)

type CHOOSECOLOR struct {
	LStructSize    uint32
	HwndOwner      HWND
	HInstance      HWND
	RgbResult      COLORREF
	LpCustColors   *COLORREF
	Flags          uint32
	LCustData      uintptr
	LpfnHook       uintptr
	LpTemplateName *uint16
}

// PrintDlg specific error codes
const (
	PDERR_CREATEICFAILURE  = 0x100A
	PDERR_DEFAULTDIFFERENT = 0x100C
	PDERR_DNDMMISMATCH     = 0x1009
	PDERR_GETDEVMODEFAIL   = 0x1005
	PDERR_INITFAILURE      = 0x1006
	PDERR_LOADDRVFAILURE   = 0x1004
	PDERR_NODEFAULTPRN     = 0x1008
	PDERR_NODEVICES        = 0x1007
	PDERR_PARSEFAILURE     = 0x1002
	PDERR_PRINTERNOTFOUND  = 0x100B
	PDERR_RETDEFFAILURE    = 0x1003
	PDERR_SETUPFAILURE     = 0x1001
)

// ChooseFont specific error codes
const (
	CFERR_MAXLESSTHANMIN = 0x2002
	CFERR_NOFONTS        = 0x2001
)

// GetOpenFileName and GetSaveFileName specific error codes
const (
	FNERR_BUFFERTOOSMALL  = 0x3003
	FNERR_INVALIDFILENAME = 0x3002
	FNERR_SUBCLASSFAILURE = 0x3001
)

// FindText and ReplaceText specific error codes
const (
	FRERR_BUFFERLENGTHZERO = 0x4001
)

// GetOpenFileName and GetSaveFileName flags
const (
	OFN_ALLOWMULTISELECT     = 0x00000200
	OFN_CREATEPROMPT         = 0x00002000
	OFN_DONTADDTORECENT      = 0x02000000
	OFN_ENABLEHOOK           = 0x00000020
	OFN_ENABLEINCLUDENOTIFY  = 0x00400000
	OFN_ENABLESIZING         = 0x00800000
	OFN_ENABLETEMPLATE       = 0x00000040
	OFN_ENABLETEMPLATEHANDLE = 0x00000080
	OFN_EXPLORER             = 0x00080000
	OFN_EXTENSIONDIFFERENT   = 0x00000400
	OFN_FILEMUSTEXIST        = 0x00001000
	OFN_FORCESHOWHIDDEN      = 0x10000000
	OFN_HIDEREADONLY         = 0x00000004
	OFN_LONGNAMES            = 0x00200000
	OFN_NOCHANGEDIR          = 0x00000008
	OFN_NODEREFERENCELINKS   = 0x00100000
	OFN_NOLONGNAMES          = 0x00040000
	OFN_NONETWORKBUTTON      = 0x00020000
	OFN_NOREADONLYRETURN     = 0x00008000
	OFN_NOTESTFILECREATE     = 0x00010000
	OFN_NOVALIDATE           = 0x00000100
	OFN_OVERWRITEPROMPT      = 0x00000002
	OFN_PATHMUSTEXIST        = 0x00000800
	OFN_READONLY             = 0x00000001
	OFN_SHAREAWARE           = 0x00004000
	OFN_SHOWHELP             = 0x00000010
)

// GetOpenFileName and GetSaveFileName extended flags
const (
	OFN_EX_NOPLACESBAR = 0x00000001
)

// PrintDlg[Ex] result actions
const (
	PD_RESULT_APPLY  = 2
	PD_RESULT_CANCEL = 0
	PD_RESULT_PRINT  = 1
)

// PrintDlg[Ex] flags
const (
	PD_ALLPAGES                   = 0x00000000
	PD_COLLATE                    = 0x00000010
	PD_CURRENTPAGE                = 0x00400000
	PD_DISABLEPRINTTOFILE         = 0x00080000
	PD_ENABLEPRINTTEMPLATE        = 0x00004000
	PD_ENABLEPRINTTEMPLATEHANDLE  = 0x00010000
	PD_EXCLUSIONFLAGS             = 0x01000000
	PD_HIDEPRINTTOFILE            = 0x00100000
	PD_NOCURRENTPAGE              = 0x00800000
	PD_NOPAGENUMS                 = 0x00000008
	PD_NOSELECTION                = 0x00000004
	PD_NOWARNING                  = 0x00000080
	PD_PAGENUMS                   = 0x00000002
	PD_PRINTTOFILE                = 0x00000020
	PD_RETURNDC                   = 0x00000100
	PD_RETURNDEFAULT              = 0x00000400
	PD_RETURNIC                   = 0x00000200
	PD_SELECTION                  = 0x00000001
	PD_USEDEVMODECOPIES           = 0x00040000
	PD_USEDEVMODECOPIESANDCOLLATE = 0x00040000
	PD_USELARGETEMPLATE           = 0x10000000
)

// PrintDlgEx exclusion flags
const (
	PD_EXCL_COPIESANDCOLLATE = DM_COPIES | DM_COLLATE
)

const START_PAGE_GENERAL = 0xffffffff

type (
	LPOFNHOOKPROC  uintptr
	HPROPSHEETPAGE HANDLE
	LPUNKNOWN      uintptr
)

type OPENFILENAME struct {
	LStructSize       uint32
	HwndOwner         HWND
	HInstance         HINSTANCE
	LpstrFilter       *uint16
	LpstrCustomFilter *uint16
	NMaxCustFilter    uint32
	NFilterIndex      uint32
	LpstrFile         *uint16
	NMaxFile          uint32
	LpstrFileTitle    *uint16
	NMaxFileTitle     uint32
	LpstrInitialDir   *uint16
	LpstrTitle        *uint16
	Flags             uint32
	NFileOffset       uint16
	NFileExtension    uint16
	LpstrDefExt       *uint16
	LCustData         uintptr
	LpfnHook          LPOFNHOOKPROC
	LpTemplateName    *uint16
	PvReserved        unsafe.Pointer
	DwReserved        uint32
	FlagsEx           uint32
}

type PRINTPAGERANGE struct {
	NFromPage uint32
	NToPage   uint32
}

type DEVNAMES struct {
	WDriverOffset uint16
	WDeviceOffset uint16
	WOutputOffset uint16
	WDefault      uint16
}

type PRINTDLGEX struct {
	LStructSize         uint32
	HwndOwner           HWND
	HDevMode            HGLOBAL
	HDevNames           HGLOBAL
	HDC                 HDC
	Flags               uint32
	Flags2              uint32
	ExclusionFlags      uint32
	NPageRanges         uint32
	NMaxPageRanges      uint32
	LpPageRanges        *PRINTPAGERANGE
	NMinPage            uint32
	NMaxPage            uint32
	NCopies             uint32
	HInstance           HINSTANCE
	LpPrintTemplateName *uint16
	LpCallback          LPUNKNOWN
	NPropertyPages      uint32
	LphPropertyPages    *HPROPSHEETPAGE
	NStartPage          uint32
	DwResultAction      uint32
}

var (
	// Library
	libcomdlg32 uintptr

	// Functions
	chooseColor          uintptr
	commDlgExtendedError uintptr
	getOpenFileName      uintptr
	getSaveFileName      uintptr
	printDlgEx           uintptr
)

func init() {
	// Library
	libcomdlg32 = MustLoadLibrary("comdlg32.dll")

	// Functions
	chooseColor = MustGetProcAddress(libcomdlg32, "ChooseColorW")
	commDlgExtendedError = MustGetProcAddress(libcomdlg32, "CommDlgExtendedError")
	getOpenFileName = MustGetProcAddress(libcomdlg32, "GetOpenFileNameW")
	getSaveFileName = MustGetProcAddress(libcomdlg32, "GetSaveFileNameW")
	printDlgEx = MustGetProcAddress(libcomdlg32, "PrintDlgExW")
}

func ChooseColor(lpcc *CHOOSECOLOR) bool {
	ret, _, _ := syscall.Syscall(chooseColor, 1,
		uintptr(unsafe.Pointer(lpcc)),
		0,
		0)

	return ret != 0
}

func CommDlgExtendedError() uint32 {
	ret, _, _ := syscall.Syscall(commDlgExtendedError, 0,
		0,
		0,
		0)

	return uint32(ret)
}

func GetOpenFileName(lpofn *OPENFILENAME) bool {
	ret, _, _ := syscall.Syscall(getOpenFileName, 1,
		uintptr(unsafe.Pointer(lpofn)),
		0,
		0)

	return ret != 0
}

func GetSaveFileName(lpofn *OPENFILENAME) bool {
	ret, _, _ := syscall.Syscall(getSaveFileName, 1,
		uintptr(unsafe.Pointer(lpofn)),
		0,
		0)

	return ret != 0
}

func PrintDlgEx(lppd *PRINTDLGEX) HRESULT {
	ret, _, _ := syscall.Syscall(printDlgEx, 1,
		uintptr(unsafe.Pointer(lppd)),
		0,
		0)

	return HRESULT(ret)
}
