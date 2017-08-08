// Copyright 2010 The win Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win

import (
	"syscall"
	"unsafe"
)

const (
	DOCHOSTUIDBLCLK_DEFAULT        = 0
	DOCHOSTUIDBLCLK_SHOWPROPERTIES = 1
	DOCHOSTUIDBLCLK_SHOWCODE       = 2
)

const (
	DOCHOSTUIFLAG_DIALOG                     = 0x1
	DOCHOSTUIFLAG_DISABLE_HELP_MENU          = 0x2
	DOCHOSTUIFLAG_NO3DBORDER                 = 0x4
	DOCHOSTUIFLAG_SCROLL_NO                  = 0x8
	DOCHOSTUIFLAG_DISABLE_SCRIPT_INACTIVE    = 0x10
	DOCHOSTUIFLAG_OPENNEWWIN                 = 0x20
	DOCHOSTUIFLAG_DISABLE_OFFSCREEN          = 0x40
	DOCHOSTUIFLAG_FLAT_SCROLLBAR             = 0x80
	DOCHOSTUIFLAG_DIV_BLOCKDEFAULT           = 0x100
	DOCHOSTUIFLAG_ACTIVATE_CLIENTHIT_ONLY    = 0x200
	DOCHOSTUIFLAG_OVERRIDEBEHAVIORFACTORY    = 0x400
	DOCHOSTUIFLAG_CODEPAGELINKEDFONTS        = 0x800
	DOCHOSTUIFLAG_URL_ENCODING_DISABLE_UTF8  = 0x1000
	DOCHOSTUIFLAG_URL_ENCODING_ENABLE_UTF8   = 0x2000
	DOCHOSTUIFLAG_ENABLE_FORMS_AUTOCOMPLETE  = 0x4000
	DOCHOSTUIFLAG_ENABLE_INPLACE_NAVIGATION  = 0x10000
	DOCHOSTUIFLAG_IME_ENABLE_RECONVERSION    = 0x20000
	DOCHOSTUIFLAG_THEME                      = 0x40000
	DOCHOSTUIFLAG_NOTHEME                    = 0x80000
	DOCHOSTUIFLAG_NOPICS                     = 0x100000
	DOCHOSTUIFLAG_NO3DOUTERBORDER            = 0x200000
	DOCHOSTUIFLAG_DISABLE_EDIT_NS_FIXUP      = 0x400000
	DOCHOSTUIFLAG_LOCAL_MACHINE_ACCESS_CHECK = 0x800000
	DOCHOSTUIFLAG_DISABLE_UNTRUSTEDPROTOCOL  = 0x1000000
)

// BrowserNavConstants
const (
	NavOpenInNewWindow       = 0x1
	NavNoHistory             = 0x2
	NavNoReadFromCache       = 0x4
	NavNoWriteToCache        = 0x8
	NavAllowAutosearch       = 0x10
	NavBrowserBar            = 0x20
	NavHyperlink             = 0x40
	NavEnforceRestricted     = 0x80
	NavNewWindowsManaged     = 0x0100
	NavUntrustedForDownload  = 0x0200
	NavTrustedForActiveX     = 0x0400
	NavOpenInNewTab          = 0x0800
	NavOpenInBackgroundTab   = 0x1000
	NavKeepWordWheelText     = 0x2000
	NavVirtualTab            = 0x4000
	NavBlockRedirectsXDomain = 0x8000
	NavOpenNewForegroundTab  = 0x10000
)

var (
	CLSID_WebBrowser        = CLSID{0x8856F961, 0x340A, 0x11D0, [8]byte{0xA9, 0x6B, 0x00, 0xC0, 0x4F, 0xD7, 0x05, 0xA2}}
	DIID_DWebBrowserEvents2 = IID{0x34A715A0, 0x6587, 0x11D0, [8]byte{0x92, 0x4A, 0x00, 0x20, 0xAF, 0xC7, 0xAC, 0x4D}}
	IID_IWebBrowser2        = IID{0xD30C1661, 0xCDAF, 0x11D0, [8]byte{0x8A, 0x3E, 0x00, 0xC0, 0x4F, 0xC9, 0xE2, 0x6E}}
	IID_IDocHostUIHandler   = IID{0xBD3F23C0, 0xD43E, 0x11CF, [8]byte{0x89, 0x3B, 0x00, 0xAA, 0x00, 0xBD, 0xCE, 0x1A}}
)

type DWebBrowserEvents2Vtbl struct {
	QueryInterface   uintptr
	AddRef           uintptr
	Release          uintptr
	GetTypeInfoCount uintptr
	GetTypeInfo      uintptr
	GetIDsOfNames    uintptr
	Invoke           uintptr
}

type DWebBrowserEvents2 struct {
	LpVtbl *DWebBrowserEvents2Vtbl
}

type IWebBrowser2Vtbl struct {
	QueryInterface           uintptr
	AddRef                   uintptr
	Release                  uintptr
	GetTypeInfoCount         uintptr
	GetTypeInfo              uintptr
	GetIDsOfNames            uintptr
	Invoke                   uintptr
	GoBack                   uintptr
	GoForward                uintptr
	GoHome                   uintptr
	GoSearch                 uintptr
	Navigate                 uintptr
	Refresh                  uintptr
	Refresh2                 uintptr
	Stop                     uintptr
	Get_Application          uintptr
	Get_Parent               uintptr
	Get_Container            uintptr
	Get_Document             uintptr
	Get_TopLevelContainer    uintptr
	Get_Type                 uintptr
	Get_Left                 uintptr
	Put_Left                 uintptr
	Get_Top                  uintptr
	Put_Top                  uintptr
	Get_Width                uintptr
	Put_Width                uintptr
	Get_Height               uintptr
	Put_Height               uintptr
	Get_LocationName         uintptr
	Get_LocationURL          uintptr
	Get_Busy                 uintptr
	Quit                     uintptr
	ClientToWindow           uintptr
	PutProperty              uintptr
	GetProperty              uintptr
	Get_Name                 uintptr
	Get_HWND                 uintptr
	Get_FullName             uintptr
	Get_Path                 uintptr
	Get_Visible              uintptr
	Put_Visible              uintptr
	Get_StatusBar            uintptr
	Put_StatusBar            uintptr
	Get_StatusText           uintptr
	Put_StatusText           uintptr
	Get_ToolBar              uintptr
	Put_ToolBar              uintptr
	Get_MenuBar              uintptr
	Put_MenuBar              uintptr
	Get_FullScreen           uintptr
	Put_FullScreen           uintptr
	Navigate2                uintptr
	QueryStatusWB            uintptr
	ExecWB                   uintptr
	ShowBrowserBar           uintptr
	Get_ReadyState           uintptr
	Get_Offline              uintptr
	Put_Offline              uintptr
	Get_Silent               uintptr
	Put_Silent               uintptr
	Get_RegisterAsBrowser    uintptr
	Put_RegisterAsBrowser    uintptr
	Get_RegisterAsDropTarget uintptr
	Put_RegisterAsDropTarget uintptr
	Get_TheaterMode          uintptr
	Put_TheaterMode          uintptr
	Get_AddressBar           uintptr
	Put_AddressBar           uintptr
	Get_Resizable            uintptr
	Put_Resizable            uintptr
}

type IWebBrowser2 struct {
	LpVtbl *IWebBrowser2Vtbl
}

func (wb2 *IWebBrowser2) Release() HRESULT {
	ret, _, _ := syscall.Syscall(wb2.LpVtbl.Release, 1,
		uintptr(unsafe.Pointer(wb2)),
		0,
		0)

	return HRESULT(ret)
}

func (wb2 *IWebBrowser2) Refresh() HRESULT {
	ret, _, _ := syscall.Syscall(wb2.LpVtbl.Refresh, 1,
		uintptr(unsafe.Pointer(wb2)),
		0,
		0)

	return HRESULT(ret)
}

func (wb2 *IWebBrowser2) Put_Left(Left int32) HRESULT {
	ret, _, _ := syscall.Syscall(wb2.LpVtbl.Put_Left, 2,
		uintptr(unsafe.Pointer(wb2)),
		uintptr(Left),
		0)

	return HRESULT(ret)
}

func (wb2 *IWebBrowser2) Put_Top(Top int32) HRESULT {
	ret, _, _ := syscall.Syscall(wb2.LpVtbl.Put_Top, 2,
		uintptr(unsafe.Pointer(wb2)),
		uintptr(Top),
		0)

	return HRESULT(ret)
}

func (wb2 *IWebBrowser2) Put_Width(Width int32) HRESULT {
	ret, _, _ := syscall.Syscall(wb2.LpVtbl.Put_Width, 2,
		uintptr(unsafe.Pointer(wb2)),
		uintptr(Width),
		0)

	return HRESULT(ret)
}

func (wb2 *IWebBrowser2) Put_Height(Height int32) HRESULT {
	ret, _, _ := syscall.Syscall(wb2.LpVtbl.Put_Height, 2,
		uintptr(unsafe.Pointer(wb2)),
		uintptr(Height),
		0)

	return HRESULT(ret)
}

func (wb2 *IWebBrowser2) Get_LocationURL(pbstrLocationURL **uint16 /*BSTR*/) HRESULT {
	ret, _, _ := syscall.Syscall(wb2.LpVtbl.Get_LocationURL, 2,
		uintptr(unsafe.Pointer(wb2)),
		uintptr(unsafe.Pointer(pbstrLocationURL)),
		0)

	return HRESULT(ret)
}

func (wb2 *IWebBrowser2) Navigate2(URL *VAR_BSTR, Flags *VAR_I4, TargetFrameName *VAR_BSTR, PostData unsafe.Pointer, Headers *VAR_BSTR) HRESULT {
	ret, _, _ := syscall.Syscall6(wb2.LpVtbl.Navigate2, 6,
		uintptr(unsafe.Pointer(wb2)),
		uintptr(unsafe.Pointer(URL)),
		uintptr(unsafe.Pointer(Flags)),
		uintptr(unsafe.Pointer(TargetFrameName)),
		uintptr(PostData),
		uintptr(unsafe.Pointer(Headers)))

	return HRESULT(ret)
}

type IDocHostUIHandlerVtbl struct {
	QueryInterface        uintptr
	AddRef                uintptr
	Release               uintptr
	ShowContextMenu       uintptr
	GetHostInfo           uintptr
	ShowUI                uintptr
	HideUI                uintptr
	UpdateUI              uintptr
	EnableModeless        uintptr
	OnDocWindowActivate   uintptr
	OnFrameWindowActivate uintptr
	ResizeBorder          uintptr
	TranslateAccelerator  uintptr
	GetOptionKeyPath      uintptr
	GetDropTarget         uintptr
	GetExternal           uintptr
	TranslateUrl          uintptr
	FilterDataObject      uintptr
}

type IDocHostUIHandler struct {
	LpVtbl *IDocHostUIHandlerVtbl
}

type DOCHOSTUIINFO struct {
	CbSize        uint32
	DwFlags       uint32
	DwDoubleClick uint32
	PchHostCss    *uint16
	PchHostNS     *uint16
}
