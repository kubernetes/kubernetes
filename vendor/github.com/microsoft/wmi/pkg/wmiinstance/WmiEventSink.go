// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// This class wraps a typicaly SWbemSink object. Its implementation is based on the
// SWbemSink documentation: https://docs.microsoft.com/en-us/windows/win32/wmisdk/receiving-asynchronous-event-notifications
// as well as the winsock example: https://github.com/go-ole/go-ole/blob/master/_example/winsock/winsock.go

// Note: Please consider the security implications of makig asynchronous calls.
// Documentation:
// https://docs.microsoft.com/en-us/windows/win32/wmisdk/making-an-asynchronous-call-with-vbscript
// https://docs.microsoft.com/en-us/windows/win32/wmisdk/setting-security-on-an-asynchronous-call

package cim

import (
	"reflect"
	"syscall"
	"unsafe"

	ole "github.com/go-ole/go-ole"
	"github.com/go-ole/go-ole/oleutil"
)

var IID_ISWbemObjectPath = &ole.GUID{0x5791BC27, 0xCE9C, 0x11d1, [8]byte{0x97, 0xBF, 0x00, 0x00, 0xF8, 0x1E, 0x84, 0x9C}}
var IID_ISWbemSinkEvents = &ole.GUID{0x75718CA0, 0xF029, 0x11d1, [8]byte{0xA1, 0xAC, 0x00, 0xC0, 0x4F, 0xB6, 0xC2, 0x23}}

const (
	eventSinkObjectName  = "WbemScripting.SWbemSink"
	iSWbemSinkEventsGuid = "{75718CA0-F029-11D1-A1AC-00C04FB6C223}"
)

type WmiEventSink struct {
	lpVtbl          *WmiEventSinkVtbl
	ref             int32
	instance        *ole.IDispatch
	unknown         *ole.IUnknown
	closed          bool
	session         *WmiSession
	onObjectReady   func(interface{}, []*WmiInstance)
	onCompleted     func(interface{}, []*WmiInstance)
	onProgress      func(interface{}, []*WmiInstance)
	onObjectPut     func(interface{}, []*WmiInstance)
	callbackContext interface{}
}

type WmiEventSinkVtbl struct {
	pQueryInterface   uintptr
	pAddRef           uintptr
	pRelease          uintptr
	pGetTypeInfoCount uintptr
	pGetTypeInfo      uintptr
	pGetIDsOfNames    uintptr
	pInvoke           uintptr
}

// DISPPARAMS are the arguments that passed to methods or property.
type DISPPARAMS struct {
	rgvarg            uintptr
	rgdispidNamedArgs uintptr
	cArgs             uint32
	cNamedArgs        uint32
}

func CreateWmiEventSink(session *WmiSession, callbackContext interface{}, onObjectReady func(interface{}, []*WmiInstance), onCompleted func(interface{}, []*WmiInstance), onProgress func(interface{}, []*WmiInstance), onObjectPut func(interface{}, []*WmiInstance)) (*WmiEventSink, error) {
	eventSinkObject, err := oleutil.CreateObject(eventSinkObjectName)
	if err != nil {
		return nil, err
	}

	eventSinkInstance, err := eventSinkObject.QueryInterface(ole.IID_IDispatch)
	if err != nil {
		return nil, err
	}

	wmiEventSink := &WmiEventSink{}
	wmiEventSink.lpVtbl = &WmiEventSinkVtbl{}
	wmiEventSink.lpVtbl.pQueryInterface = syscall.NewCallback(queryInterface)
	wmiEventSink.lpVtbl.pAddRef = syscall.NewCallback(addRef)
	wmiEventSink.lpVtbl.pRelease = syscall.NewCallback(release)
	wmiEventSink.lpVtbl.pGetTypeInfoCount = syscall.NewCallback(getTypeInfoCount)
	wmiEventSink.lpVtbl.pGetTypeInfo = syscall.NewCallback(getTypeInfo)
	wmiEventSink.lpVtbl.pGetIDsOfNames = syscall.NewCallback(getIDsOfNames)
	wmiEventSink.lpVtbl.pInvoke = syscall.NewCallback(invoke)
	wmiEventSink.onObjectReady = onObjectReady
	wmiEventSink.onCompleted = onCompleted
	wmiEventSink.onProgress = onProgress
	wmiEventSink.onObjectPut = onObjectPut
	wmiEventSink.callbackContext = callbackContext
	wmiEventSink.instance = eventSinkInstance
	wmiEventSink.unknown = eventSinkObject
	wmiEventSink.session = session

	return wmiEventSink, nil
}

func (c *WmiEventSink) Connect() (cookie uint32, err error) {
	cookie = 0
	err = nil

	connectionPointContainer, err := c.instance.QueryInterface(ole.IID_IConnectionPointContainer)
	if err != nil {
		return
	}
	defer connectionPointContainer.Release()

	container := (*ole.IConnectionPointContainer)(unsafe.Pointer(connectionPointContainer))

	var point *ole.IConnectionPoint
	err = container.FindConnectionPoint(IID_ISWbemSinkEvents, &point)
	if err != nil {
		return
	}

	return point.Advise((*ole.IUnknown)(unsafe.Pointer(c)))
}

func (c *WmiEventSink) GetAndDispatchMessages() {
	for c.ref != 0 {
		var m ole.Msg
		ole.GetMessage(&m, 0, 0, 0)
		ole.DispatchMessage(&m)
	}
}

func (c *WmiEventSink) IsReadyToClose() bool {
	return (c.ref == 0)
}

func (c *WmiEventSink) PeekAndDispatchMessages() bool {
	var m ole.Msg
	msgAvailable, err := PeekMessage(&m, 0, 0, 0, PM_REMOVE)
	if err != nil {
		return false
	}

	if msgAvailable {
		ole.DispatchMessage(&m)
	}

	return msgAvailable
}

func (c *WmiEventSink) IsClosed() bool {
	return c.closed
}
func (c *WmiEventSink) Close() {
	if c.instance != nil {
		c.instance.Release()
		c.instance = nil
	}
	if c.unknown != nil {
		c.unknown.Release()
		c.unknown = nil
	}
	c.closed = true
}

/////////////////////////////// Private methods and callbacks /////////////////////////////////////////////////////

func queryInterface(this *ole.IUnknown, iid *ole.GUID, punk **ole.IUnknown) uintptr {
	s, _ := ole.StringFromCLSID(iid)

	*punk = nil
	if ole.IsEqualGUID(iid, ole.IID_IUnknown) ||
		ole.IsEqualGUID(iid, ole.IID_IDispatch) {
		addRef(this)
		*punk = this
		return ole.S_OK
	}
	if s == iSWbemSinkEventsGuid {
		addRef(this)
		*punk = this
		return ole.S_OK
	}

	return ole.E_NOINTERFACE
}

func addRef(this *ole.IUnknown) uintptr {
	pthis := (*WmiEventSink)(unsafe.Pointer(this))
	pthis.ref++
	return uintptr(pthis.ref)
}

func release(this *ole.IUnknown) uintptr {
	pthis := (*WmiEventSink)(unsafe.Pointer(this))
	pthis.ref--
	return uintptr(pthis.ref)
}

func getIDsOfNames(this *ole.IUnknown, iid *ole.GUID, wnames **uint16, namelen int, lcid int, pdisp *int32) uintptr {
	var pdispSlice []int32
	sliceHeader := (*reflect.SliceHeader)((unsafe.Pointer(&pdispSlice)))
	sliceHeader.Cap = namelen
	sliceHeader.Len = namelen
	sliceHeader.Data = uintptr(unsafe.Pointer(pdisp))

	var pwnamesSlice []*uint16
	sliceHeader2 := (*reflect.SliceHeader)((unsafe.Pointer(&pwnamesSlice)))
	sliceHeader2.Cap = namelen
	sliceHeader2.Len = namelen
	sliceHeader2.Data = uintptr(unsafe.Pointer(wnames))

	for n := 0; n < namelen; n++ {
		pdispSlice[n] = int32(n)
	}
	return uintptr(ole.S_OK)
}

func getTypeInfoCount(pcount *int) uintptr {
	if pcount != nil {
		*pcount = 0
	}
	return uintptr(ole.S_OK)
}

func getTypeInfo(ptypeif *uintptr) uintptr {
	return uintptr(ole.E_NOTIMPL)
}

func invoke(this *ole.IDispatch, dispid int, riid *ole.GUID, lcid int, flags int16, rawdispparams *DISPPARAMS, result *ole.VARIANT, pexcepinfo *ole.EXCEPINFO, nerr *uint) uintptr {
	pthis := (*WmiEventSink)(unsafe.Pointer(this))
	if pthis.IsClosed() {
		return ole.S_OK
	}

	dispparams := GetDispParamsFromRaw(rawdispparams)
	wmiEventInstances, err := GetVariantArrayAsWmiInstances(dispparams.rgvarg, pthis.session)
	if err != nil {
		return ole.S_OK
	}
	switch dispid {
	case 1:
		pthis.onObjectReady(pthis.callbackContext, wmiEventInstances)
		return ole.S_OK
	case 2:
		pthis.onCompleted(pthis.callbackContext, wmiEventInstances)
		return ole.S_OK
	case 3:
		pthis.onProgress(pthis.callbackContext, wmiEventInstances)
		return ole.S_OK
	case 4:
		pthis.onObjectPut(pthis.callbackContext, wmiEventInstances)
		return ole.S_OK
	default:
	}
	return ole.E_NOTIMPL
}
