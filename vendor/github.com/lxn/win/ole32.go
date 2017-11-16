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
	CLSCTX_INPROC_SERVER          = 0x1
	CLSCTX_INPROC_HANDLER         = 0x2
	CLSCTX_LOCAL_SERVER           = 0x4
	CLSCTX_INPROC_SERVER16        = 0x8
	CLSCTX_REMOTE_SERVER          = 0x10
	CLSCTX_INPROC_HANDLER16       = 0x20
	CLSCTX_RESERVED1              = 0x40
	CLSCTX_RESERVED2              = 0x80
	CLSCTX_RESERVED3              = 0x100
	CLSCTX_RESERVED4              = 0x200
	CLSCTX_NO_CODE_DOWNLOAD       = 0x400
	CLSCTX_RESERVED5              = 0x800
	CLSCTX_NO_CUSTOM_MARSHAL      = 0x1000
	CLSCTX_ENABLE_CODE_DOWNLOAD   = 0x2000
	CLSCTX_NO_FAILURE_LOG         = 0x4000
	CLSCTX_DISABLE_AAA            = 0x8000
	CLSCTX_ENABLE_AAA             = 0x10000
	CLSCTX_FROM_DEFAULT_CONTEXT   = 0x20000
	CLSCTX_ACTIVATE_32_BIT_SERVER = 0x40000
	CLSCTX_ACTIVATE_64_BIT_SERVER = 0x80000
	CLSCTX_ENABLE_CLOAKING        = 0x100000
	CLSCTX_PS_DLL                 = 0x80000000
	CLSCTX_ALL                    = CLSCTX_INPROC_SERVER | CLSCTX_INPROC_HANDLER | CLSCTX_LOCAL_SERVER | CLSCTX_REMOTE_SERVER
)

// Verbs for IOleObject.DoVerb
const (
	OLEIVERB_PRIMARY          = 0
	OLEIVERB_SHOW             = -1
	OLEIVERB_OPEN             = -2
	OLEIVERB_HIDE             = -3
	OLEIVERB_UIACTIVATE       = -4
	OLEIVERB_INPLACEACTIVATE  = -5
	OLEIVERB_DISCARDUNDOSTATE = -6
)

// OLECLOSE constants
const (
	OLECLOSE_SAVEIFDIRTY = 0
	OLECLOSE_NOSAVE      = 1
	OLECLOSE_PROMPTSAVE  = 2
)

type IID syscall.GUID
type CLSID syscall.GUID
type REFIID *IID
type REFCLSID *CLSID

var (
	IID_IClassFactory             = IID{0x00000001, 0x0000, 0x0000, [8]byte{0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}}
	IID_IConnectionPointContainer = IID{0xB196B284, 0xBAB4, 0x101A, [8]byte{0xB6, 0x9C, 0x00, 0xAA, 0x00, 0x34, 0x1D, 0x07}}
	IID_IOleClientSite            = IID{0x00000118, 0x0000, 0x0000, [8]byte{0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}}
	IID_IOleInPlaceObject         = IID{0x00000113, 0x0000, 0x0000, [8]byte{0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}}
	IID_IOleInPlaceSite           = IID{0x00000119, 0x0000, 0x0000, [8]byte{0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}}
	IID_IOleObject                = IID{0x00000112, 0x0000, 0x0000, [8]byte{0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}}
	IID_IUnknown                  = IID{0x00000000, 0x0000, 0x0000, [8]byte{0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}}
)

func EqualREFIID(a, b REFIID) bool {
	if a == b {
		return true
	}
	if a == nil || b == nil {
		return false
	}

	if a.Data1 != b.Data1 || a.Data2 != b.Data2 || a.Data3 != b.Data3 {
		return false
	}

	for i := 0; i < 8; i++ {
		if a.Data4[i] != b.Data4[i] {
			return false
		}
	}

	return true
}

type IClassFactoryVtbl struct {
	QueryInterface uintptr
	AddRef         uintptr
	Release        uintptr
	CreateInstance uintptr
	LockServer     uintptr
}

type IClassFactory struct {
	LpVtbl *IClassFactoryVtbl
}

func (cf *IClassFactory) Release() uint32 {
	ret, _, _ := syscall.Syscall(cf.LpVtbl.Release, 1,
		uintptr(unsafe.Pointer(cf)),
		0,
		0)

	return uint32(ret)
}

func (cf *IClassFactory) CreateInstance(pUnkOuter *IUnknown, riid REFIID, ppvObject *unsafe.Pointer) HRESULT {
	ret, _, _ := syscall.Syscall6(cf.LpVtbl.CreateInstance, 4,
		uintptr(unsafe.Pointer(cf)),
		uintptr(unsafe.Pointer(pUnkOuter)),
		uintptr(unsafe.Pointer(riid)),
		uintptr(unsafe.Pointer(ppvObject)),
		0,
		0)

	return HRESULT(ret)
}

type IConnectionPointVtbl struct {
	QueryInterface              uintptr
	AddRef                      uintptr
	Release                     uintptr
	GetConnectionInterface      uintptr
	GetConnectionPointContainer uintptr
	Advise                      uintptr
	Unadvise                    uintptr
	EnumConnections             uintptr
}

type IConnectionPoint struct {
	LpVtbl *IConnectionPointVtbl
}

func (cp *IConnectionPoint) Release() uint32 {
	ret, _, _ := syscall.Syscall(cp.LpVtbl.Release, 1,
		uintptr(unsafe.Pointer(cp)),
		0,
		0)

	return uint32(ret)
}

func (cp *IConnectionPoint) Advise(pUnkSink unsafe.Pointer, pdwCookie *uint32) HRESULT {
	ret, _, _ := syscall.Syscall(cp.LpVtbl.Advise, 3,
		uintptr(unsafe.Pointer(cp)),
		uintptr(pUnkSink),
		uintptr(unsafe.Pointer(pdwCookie)))

	return HRESULT(ret)
}

type IConnectionPointContainerVtbl struct {
	QueryInterface       uintptr
	AddRef               uintptr
	Release              uintptr
	EnumConnectionPoints uintptr
	FindConnectionPoint  uintptr
}

type IConnectionPointContainer struct {
	LpVtbl *IConnectionPointContainerVtbl
}

func (cpc *IConnectionPointContainer) Release() uint32 {
	ret, _, _ := syscall.Syscall(cpc.LpVtbl.Release, 1,
		uintptr(unsafe.Pointer(cpc)),
		0,
		0)

	return uint32(ret)
}

func (cpc *IConnectionPointContainer) FindConnectionPoint(riid REFIID, ppCP **IConnectionPoint) HRESULT {
	ret, _, _ := syscall.Syscall(cpc.LpVtbl.FindConnectionPoint, 3,
		uintptr(unsafe.Pointer(cpc)),
		uintptr(unsafe.Pointer(riid)),
		uintptr(unsafe.Pointer(ppCP)))

	return HRESULT(ret)
}

type IOleClientSiteVtbl struct {
	QueryInterface         uintptr
	AddRef                 uintptr
	Release                uintptr
	SaveObject             uintptr
	GetMoniker             uintptr
	GetContainer           uintptr
	ShowObject             uintptr
	OnShowWindow           uintptr
	RequestNewObjectLayout uintptr
}

type IOleClientSite struct {
	LpVtbl *IOleClientSiteVtbl
}

type IOleInPlaceFrameVtbl struct {
	QueryInterface       uintptr
	AddRef               uintptr
	Release              uintptr
	GetWindow            uintptr
	ContextSensitiveHelp uintptr
	GetBorder            uintptr
	RequestBorderSpace   uintptr
	SetBorderSpace       uintptr
	SetActiveObject      uintptr
	InsertMenus          uintptr
	SetMenu              uintptr
	RemoveMenus          uintptr
	SetStatusText        uintptr
	EnableModeless       uintptr
	TranslateAccelerator uintptr
}

type IOleInPlaceFrame struct {
	LpVtbl *IOleInPlaceFrameVtbl
}

type IOleInPlaceObjectVtbl struct {
	QueryInterface       uintptr
	AddRef               uintptr
	Release              uintptr
	GetWindow            uintptr
	ContextSensitiveHelp uintptr
	InPlaceDeactivate    uintptr
	UIDeactivate         uintptr
	SetObjectRects       uintptr
	ReactivateAndUndo    uintptr
}

type IOleInPlaceObject struct {
	LpVtbl *IOleInPlaceObjectVtbl
}

func (obj *IOleInPlaceObject) Release() uint32 {
	ret, _, _ := syscall.Syscall(obj.LpVtbl.Release, 1,
		uintptr(unsafe.Pointer(obj)),
		0,
		0)

	return uint32(ret)
}

func (obj *IOleInPlaceObject) SetObjectRects(lprcPosRect, lprcClipRect *RECT) HRESULT {
	ret, _, _ := syscall.Syscall(obj.LpVtbl.SetObjectRects, 3,
		uintptr(unsafe.Pointer(obj)),
		uintptr(unsafe.Pointer(lprcPosRect)),
		uintptr(unsafe.Pointer(lprcClipRect)))

	return HRESULT(ret)
}

type IOleInPlaceSiteVtbl struct {
	QueryInterface       uintptr
	AddRef               uintptr
	Release              uintptr
	GetWindow            uintptr
	ContextSensitiveHelp uintptr
	CanInPlaceActivate   uintptr
	OnInPlaceActivate    uintptr
	OnUIActivate         uintptr
	GetWindowContext     uintptr
	Scroll               uintptr
	OnUIDeactivate       uintptr
	OnInPlaceDeactivate  uintptr
	DiscardUndoState     uintptr
	DeactivateAndUndo    uintptr
	OnPosRectChange      uintptr
}

type IOleInPlaceSite struct {
	LpVtbl *IOleInPlaceSiteVtbl
}

type IOleObjectVtbl struct {
	QueryInterface   uintptr
	AddRef           uintptr
	Release          uintptr
	SetClientSite    uintptr
	GetClientSite    uintptr
	SetHostNames     uintptr
	Close            uintptr
	SetMoniker       uintptr
	GetMoniker       uintptr
	InitFromData     uintptr
	GetClipboardData uintptr
	DoVerb           uintptr
	EnumVerbs        uintptr
	Update           uintptr
	IsUpToDate       uintptr
	GetUserClassID   uintptr
	GetUserType      uintptr
	SetExtent        uintptr
	GetExtent        uintptr
	Advise           uintptr
	Unadvise         uintptr
	EnumAdvise       uintptr
	GetMiscStatus    uintptr
	SetColorScheme   uintptr
}

type IOleObject struct {
	LpVtbl *IOleObjectVtbl
}

func (obj *IOleObject) QueryInterface(riid REFIID, ppvObject *unsafe.Pointer) HRESULT {
	ret, _, _ := syscall.Syscall(obj.LpVtbl.QueryInterface, 3,
		uintptr(unsafe.Pointer(obj)),
		uintptr(unsafe.Pointer(riid)),
		uintptr(unsafe.Pointer(ppvObject)))

	return HRESULT(ret)
}

func (obj *IOleObject) Release() uint32 {
	ret, _, _ := syscall.Syscall(obj.LpVtbl.Release, 1,
		uintptr(unsafe.Pointer(obj)),
		0,
		0)

	return uint32(ret)
}

func (obj *IOleObject) SetClientSite(pClientSite *IOleClientSite) HRESULT {
	ret, _, _ := syscall.Syscall(obj.LpVtbl.SetClientSite, 2,
		uintptr(unsafe.Pointer(obj)),
		uintptr(unsafe.Pointer(pClientSite)),
		0)

	return HRESULT(ret)
}

func (obj *IOleObject) SetHostNames(szContainerApp, szContainerObj *uint16) HRESULT {
	ret, _, _ := syscall.Syscall(obj.LpVtbl.SetHostNames, 3,
		uintptr(unsafe.Pointer(obj)),
		uintptr(unsafe.Pointer(szContainerApp)),
		uintptr(unsafe.Pointer(szContainerObj)))

	return HRESULT(ret)
}

func (obj *IOleObject) Close(dwSaveOption uint32) HRESULT {
	ret, _, _ := syscall.Syscall(obj.LpVtbl.Close, 2,
		uintptr(unsafe.Pointer(obj)),
		uintptr(dwSaveOption),
		0)

	return HRESULT(ret)
}

func (obj *IOleObject) DoVerb(iVerb int32, lpmsg *MSG, pActiveSite *IOleClientSite, lindex int32, hwndParent HWND, lprcPosRect *RECT) HRESULT {
	ret, _, _ := syscall.Syscall9(obj.LpVtbl.DoVerb, 7,
		uintptr(unsafe.Pointer(obj)),
		uintptr(iVerb),
		uintptr(unsafe.Pointer(lpmsg)),
		uintptr(unsafe.Pointer(pActiveSite)),
		uintptr(lindex),
		uintptr(hwndParent),
		uintptr(unsafe.Pointer(lprcPosRect)),
		0,
		0)

	return HRESULT(ret)
}

type IUnknownVtbl struct {
	QueryInterface uintptr
	AddRef         uintptr
	Release        uintptr
}

type IUnknown struct {
	LpVtbl *IUnknownVtbl
}

type OLEINPLACEFRAMEINFO struct {
	Cb            uint32
	FMDIApp       BOOL
	HwndFrame     HWND
	Haccel        HACCEL
	CAccelEntries uint32
}

type COAUTHIDENTITY struct {
	User           *uint16
	UserLength     uint32
	Domain         *uint16
	DomainLength   uint32
	Password       *uint16
	PasswordLength uint32
	Flags          uint32
}

type COAUTHINFO struct {
	dwAuthnSvc           uint32
	dwAuthzSvc           uint32
	pwszServerPrincName  *uint16
	dwAuthnLevel         uint32
	dwImpersonationLevel uint32
	pAuthIdentityData    *COAUTHIDENTITY
	dwCapabilities       uint32
}

type COSERVERINFO struct {
	dwReserved1 uint32
	pwszName    *uint16
	pAuthInfo   *COAUTHINFO
	dwReserved2 uint32
}

var (
	// Library
	libole32 uintptr

	// Functions
	coCreateInstance      uintptr
	coGetClassObject      uintptr
	coTaskMemFree         uintptr
	oleInitialize         uintptr
	oleSetContainedObject uintptr
	oleUninitialize       uintptr
)

func init() {
	// Library
	libole32 = MustLoadLibrary("ole32.dll")

	// Functions
	coCreateInstance = MustGetProcAddress(libole32, "CoCreateInstance")
	coGetClassObject = MustGetProcAddress(libole32, "CoGetClassObject")
	coTaskMemFree = MustGetProcAddress(libole32, "CoTaskMemFree")
	oleInitialize = MustGetProcAddress(libole32, "OleInitialize")
	oleSetContainedObject = MustGetProcAddress(libole32, "OleSetContainedObject")
	oleUninitialize = MustGetProcAddress(libole32, "OleUninitialize")
}

func CoCreateInstance(rclsid REFCLSID, pUnkOuter *IUnknown, dwClsContext uint32, riid REFIID, ppv *unsafe.Pointer) HRESULT {
	ret, _, _ := syscall.Syscall6(coCreateInstance, 5,
		uintptr(unsafe.Pointer(rclsid)),
		uintptr(unsafe.Pointer(pUnkOuter)),
		uintptr(dwClsContext),
		uintptr(unsafe.Pointer(riid)),
		uintptr(unsafe.Pointer(ppv)),
		0)

	return HRESULT(ret)
}

func CoGetClassObject(rclsid REFCLSID, dwClsContext uint32, pServerInfo *COSERVERINFO, riid REFIID, ppv *unsafe.Pointer) HRESULT {
	ret, _, _ := syscall.Syscall6(coGetClassObject, 5,
		uintptr(unsafe.Pointer(rclsid)),
		uintptr(dwClsContext),
		uintptr(unsafe.Pointer(pServerInfo)),
		uintptr(unsafe.Pointer(riid)),
		uintptr(unsafe.Pointer(ppv)),
		0)

	return HRESULT(ret)
}

func CoTaskMemFree(pv uintptr) {
	syscall.Syscall(coTaskMemFree, 1,
		pv,
		0,
		0)
}

func OleInitialize() HRESULT {
	ret, _, _ := syscall.Syscall(oleInitialize, 1, // WTF, why does 0 not work here?
		0,
		0,
		0)

	return HRESULT(ret)
}

func OleSetContainedObject(pUnknown *IUnknown, fContained bool) HRESULT {
	ret, _, _ := syscall.Syscall(oleSetContainedObject, 2,
		uintptr(unsafe.Pointer(pUnknown)),
		uintptr(BoolToBOOL(fContained)),
		0)

	return HRESULT(ret)
}

func OleUninitialize() {
	syscall.Syscall(oleUninitialize, 0,
		0,
		0,
		0)
}
