// +build windows

package oleutil

import (
	"reflect"
	"unsafe"

	ole "github.com/go-ole/go-ole"
)

type stdDispatch struct {
	lpVtbl  *stdDispatchVtbl
	ref     int32
	iid     *ole.GUID
	iface   interface{}
	funcMap map[string]int32
}

type stdDispatchVtbl struct {
	pQueryInterface   uintptr
	pAddRef           uintptr
	pRelease          uintptr
	pGetTypeInfoCount uintptr
	pGetTypeInfo      uintptr
	pGetIDsOfNames    uintptr
	pInvoke           uintptr
}

func dispQueryInterface(this *ole.IUnknown, iid *ole.GUID, punk **ole.IUnknown) uint32 {
	pthis := (*stdDispatch)(unsafe.Pointer(this))
	*punk = nil
	if ole.IsEqualGUID(iid, ole.IID_IUnknown) ||
		ole.IsEqualGUID(iid, ole.IID_IDispatch) {
		dispAddRef(this)
		*punk = this
		return ole.S_OK
	}
	if ole.IsEqualGUID(iid, pthis.iid) {
		dispAddRef(this)
		*punk = this
		return ole.S_OK
	}
	return ole.E_NOINTERFACE
}

func dispAddRef(this *ole.IUnknown) int32 {
	pthis := (*stdDispatch)(unsafe.Pointer(this))
	pthis.ref++
	return pthis.ref
}

func dispRelease(this *ole.IUnknown) int32 {
	pthis := (*stdDispatch)(unsafe.Pointer(this))
	pthis.ref--
	return pthis.ref
}

func dispGetIDsOfNames(this *ole.IUnknown, iid *ole.GUID, wnames []*uint16, namelen int, lcid int, pdisp []int32) uintptr {
	pthis := (*stdDispatch)(unsafe.Pointer(this))
	names := make([]string, len(wnames))
	for i := 0; i < len(names); i++ {
		names[i] = ole.LpOleStrToString(wnames[i])
	}
	for n := 0; n < namelen; n++ {
		if id, ok := pthis.funcMap[names[n]]; ok {
			pdisp[n] = id
		}
	}
	return ole.S_OK
}

func dispGetTypeInfoCount(pcount *int) uintptr {
	if pcount != nil {
		*pcount = 0
	}
	return ole.S_OK
}

func dispGetTypeInfo(ptypeif *uintptr) uintptr {
	return ole.E_NOTIMPL
}

func dispInvoke(this *ole.IDispatch, dispid int32, riid *ole.GUID, lcid int, flags int16, dispparams *ole.DISPPARAMS, result *ole.VARIANT, pexcepinfo *ole.EXCEPINFO, nerr *uint) uintptr {
	pthis := (*stdDispatch)(unsafe.Pointer(this))
	found := ""
	for name, id := range pthis.funcMap {
		if id == dispid {
			found = name
		}
	}
	if found != "" {
		rv := reflect.ValueOf(pthis.iface).Elem()
		rm := rv.MethodByName(found)
		rr := rm.Call([]reflect.Value{})
		println(len(rr))
		return ole.S_OK
	}
	return ole.E_NOTIMPL
}
