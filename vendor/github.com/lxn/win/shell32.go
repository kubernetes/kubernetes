// Copyright 2010 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

import (
	"syscall"
	"unsafe"
)

type CSIDL uint32
type HDROP HANDLE

const (
	CSIDL_DESKTOP                 = 0x00
	CSIDL_INTERNET                = 0x01
	CSIDL_PROGRAMS                = 0x02
	CSIDL_CONTROLS                = 0x03
	CSIDL_PRINTERS                = 0x04
	CSIDL_PERSONAL                = 0x05
	CSIDL_FAVORITES               = 0x06
	CSIDL_STARTUP                 = 0x07
	CSIDL_RECENT                  = 0x08
	CSIDL_SENDTO                  = 0x09
	CSIDL_BITBUCKET               = 0x0A
	CSIDL_STARTMENU               = 0x0B
	CSIDL_MYDOCUMENTS             = 0x0C
	CSIDL_MYMUSIC                 = 0x0D
	CSIDL_MYVIDEO                 = 0x0E
	CSIDL_DESKTOPDIRECTORY        = 0x10
	CSIDL_DRIVES                  = 0x11
	CSIDL_NETWORK                 = 0x12
	CSIDL_NETHOOD                 = 0x13
	CSIDL_FONTS                   = 0x14
	CSIDL_TEMPLATES               = 0x15
	CSIDL_COMMON_STARTMENU        = 0x16
	CSIDL_COMMON_PROGRAMS         = 0x17
	CSIDL_COMMON_STARTUP          = 0x18
	CSIDL_COMMON_DESKTOPDIRECTORY = 0x19
	CSIDL_APPDATA                 = 0x1A
	CSIDL_PRINTHOOD               = 0x1B
	CSIDL_LOCAL_APPDATA           = 0x1C
	CSIDL_ALTSTARTUP              = 0x1D
	CSIDL_COMMON_ALTSTARTUP       = 0x1E
	CSIDL_COMMON_FAVORITES        = 0x1F
	CSIDL_INTERNET_CACHE          = 0x20
	CSIDL_COOKIES                 = 0x21
	CSIDL_HISTORY                 = 0x22
	CSIDL_COMMON_APPDATA          = 0x23
	CSIDL_WINDOWS                 = 0x24
	CSIDL_SYSTEM                  = 0x25
	CSIDL_PROGRAM_FILES           = 0x26
	CSIDL_MYPICTURES              = 0x27
	CSIDL_PROFILE                 = 0x28
	CSIDL_SYSTEMX86               = 0x29
	CSIDL_PROGRAM_FILESX86        = 0x2A
	CSIDL_PROGRAM_FILES_COMMON    = 0x2B
	CSIDL_PROGRAM_FILES_COMMONX86 = 0x2C
	CSIDL_COMMON_TEMPLATES        = 0x2D
	CSIDL_COMMON_DOCUMENTS        = 0x2E
	CSIDL_COMMON_ADMINTOOLS       = 0x2F
	CSIDL_ADMINTOOLS              = 0x30
	CSIDL_CONNECTIONS             = 0x31
	CSIDL_COMMON_MUSIC            = 0x35
	CSIDL_COMMON_PICTURES         = 0x36
	CSIDL_COMMON_VIDEO            = 0x37
	CSIDL_RESOURCES               = 0x38
	CSIDL_RESOURCES_LOCALIZED     = 0x39
	CSIDL_COMMON_OEM_LINKS        = 0x3A
	CSIDL_CDBURN_AREA             = 0x3B
	CSIDL_COMPUTERSNEARME         = 0x3D
	CSIDL_FLAG_CREATE             = 0x8000
	CSIDL_FLAG_DONT_VERIFY        = 0x4000
	CSIDL_FLAG_NO_ALIAS           = 0x1000
	CSIDL_FLAG_PER_USER_INIT      = 0x8000
	CSIDL_FLAG_MASK               = 0xFF00
)

// NotifyIcon flags
const (
	NIF_MESSAGE = 0x00000001
	NIF_ICON    = 0x00000002
	NIF_TIP     = 0x00000004
	NIF_STATE   = 0x00000008
	NIF_INFO    = 0x00000010
)

// NotifyIcon messages
const (
	NIM_ADD        = 0x00000000
	NIM_MODIFY     = 0x00000001
	NIM_DELETE     = 0x00000002
	NIM_SETFOCUS   = 0x00000003
	NIM_SETVERSION = 0x00000004
)

// NotifyIcon states
const (
	NIS_HIDDEN     = 0x00000001
	NIS_SHAREDICON = 0x00000002
)

// NotifyIcon info flags
const (
	NIIF_NONE    = 0x00000000
	NIIF_INFO    = 0x00000001
	NIIF_WARNING = 0x00000002
	NIIF_ERROR   = 0x00000003
	NIIF_USER    = 0x00000004
	NIIF_NOSOUND = 0x00000010
)

const NOTIFYICON_VERSION = 3

// SHGetFileInfo flags
const (
	SHGFI_LARGEICON         = 0x000000000
	SHGFI_SMALLICON         = 0x000000001
	SHGFI_OPENICON          = 0x000000002
	SHGFI_SHELLICONSIZE     = 0x000000004
	SHGFI_PIDL              = 0x000000008
	SHGFI_USEFILEATTRIBUTES = 0x000000010
	SHGFI_ADDOVERLAYS       = 0x000000020
	SHGFI_OVERLAYINDEX      = 0x000000040
	SHGFI_ICON              = 0x000000100
	SHGFI_DISPLAYNAME       = 0x000000200
	SHGFI_TYPENAME          = 0x000000400
	SHGFI_ATTRIBUTES        = 0x000000800
	SHGFI_ICONLOCATION      = 0x000001000
	SHGFI_EXETYPE           = 0x000002000
	SHGFI_SYSICONINDEX      = 0x000004000
	SHGFI_LINKOVERLAY       = 0x000008000
	SHGFI_SELECTED          = 0x000010000
	SHGFI_ATTR_SPECIFIED    = 0x000020000
)

type NOTIFYICONDATA struct {
	CbSize           uint32
	HWnd             HWND
	UID              uint32
	UFlags           uint32
	UCallbackMessage uint32
	HIcon            HICON
	SzTip            [128]uint16
	DwState          uint32
	DwStateMask      uint32
	SzInfo           [256]uint16
	UVersion         uint32
	SzInfoTitle      [64]uint16
	DwInfoFlags      uint32
	GuidItem         syscall.GUID
}

type SHFILEINFO struct {
	HIcon         HICON
	IIcon         int32
	DwAttributes  uint32
	SzDisplayName [MAX_PATH]uint16
	SzTypeName    [80]uint16
}

type BROWSEINFO struct {
	HwndOwner      HWND
	PidlRoot       uintptr
	PszDisplayName *uint16
	LpszTitle      *uint16
	UlFlags        uint32
	Lpfn           uintptr
	LParam         uintptr
	IImage         int32
}

var (
	// Library
	libshell32 uintptr

	// Functions
	dragAcceptFiles        uintptr
	dragFinish             uintptr
	dragQueryFile          uintptr
	shBrowseForFolder      uintptr
	shGetFileInfo          uintptr
	shGetPathFromIDList    uintptr
	shGetSpecialFolderPath uintptr
	shParseDisplayName     uintptr
	shell_NotifyIcon       uintptr
)

func init() {
	// Library
	libshell32 = MustLoadLibrary("shell32.dll")

	// Functions
	dragAcceptFiles = MustGetProcAddress(libshell32, "DragAcceptFiles")
	dragFinish = MustGetProcAddress(libshell32, "DragFinish")
	dragQueryFile = MustGetProcAddress(libshell32, "DragQueryFileW")
	shBrowseForFolder = MustGetProcAddress(libshell32, "SHBrowseForFolderW")
	shGetFileInfo = MustGetProcAddress(libshell32, "SHGetFileInfoW")
	shGetPathFromIDList = MustGetProcAddress(libshell32, "SHGetPathFromIDListW")
	shGetSpecialFolderPath = MustGetProcAddress(libshell32, "SHGetSpecialFolderPathW")
	shParseDisplayName = MustGetProcAddress(libshell32, "SHParseDisplayName")
	shell_NotifyIcon = MustGetProcAddress(libshell32, "Shell_NotifyIconW")
}

func DragAcceptFiles(hWnd HWND, fAccept bool) bool {
	ret, _, _ := syscall.Syscall(dragAcceptFiles, 2,
		uintptr(hWnd),
		uintptr(BoolToBOOL(fAccept)),
		0)

	return ret != 0
}

func DragQueryFile(hDrop HDROP, iFile uint, lpszFile *uint16, cch uint) uint {
	ret, _, _ := syscall.Syscall6(dragQueryFile, 4,
		uintptr(hDrop),
		uintptr(iFile),
		uintptr(unsafe.Pointer(lpszFile)),
		uintptr(cch),
		0,
		0)

	return uint(ret)
}

func DragFinish(hDrop HDROP) {
	syscall.Syscall(dragAcceptFiles, 1,
		uintptr(hDrop),
		0,
		0)
}

func SHBrowseForFolder(lpbi *BROWSEINFO) uintptr {
	ret, _, _ := syscall.Syscall(shBrowseForFolder, 1,
		uintptr(unsafe.Pointer(lpbi)),
		0,
		0)

	return ret
}

func SHGetFileInfo(pszPath *uint16, dwFileAttributes uint32, psfi *SHFILEINFO, cbFileInfo, uFlags uint32) uintptr {
	ret, _, _ := syscall.Syscall6(shGetFileInfo, 5,
		uintptr(unsafe.Pointer(pszPath)),
		uintptr(dwFileAttributes),
		uintptr(unsafe.Pointer(psfi)),
		uintptr(cbFileInfo),
		uintptr(uFlags),
		0)

	return ret
}

func SHGetPathFromIDList(pidl uintptr, pszPath *uint16) bool {
	ret, _, _ := syscall.Syscall(shGetPathFromIDList, 2,
		pidl,
		uintptr(unsafe.Pointer(pszPath)),
		0)

	return ret != 0
}

func SHGetSpecialFolderPath(hwndOwner HWND, lpszPath *uint16, csidl CSIDL, fCreate bool) bool {
	ret, _, _ := syscall.Syscall6(shGetSpecialFolderPath, 4,
		uintptr(hwndOwner),
		uintptr(unsafe.Pointer(lpszPath)),
		uintptr(csidl),
		uintptr(BoolToBOOL(fCreate)),
		0,
		0)

	return ret != 0
}

func SHParseDisplayName(pszName *uint16, pbc uintptr, ppidl *uintptr, sfgaoIn uint32, psfgaoOut *uint32) HRESULT {
	ret, _, _ := syscall.Syscall6(shParseDisplayName, 5,
		uintptr(unsafe.Pointer(pszName)),
		pbc,
		uintptr(unsafe.Pointer(ppidl)),
		0,
		uintptr(unsafe.Pointer(psfgaoOut)),
		0)

	return HRESULT(ret)
}

func Shell_NotifyIcon(dwMessage uint32, lpdata *NOTIFYICONDATA) bool {
	ret, _, _ := syscall.Syscall(shell_NotifyIcon, 2,
		uintptr(dwMessage),
		uintptr(unsafe.Pointer(lpdata)),
		0)

	return ret != 0
}
