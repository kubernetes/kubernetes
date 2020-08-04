// +build windows

package oleutil

import (
	"reflect"
	"syscall"
	"unsafe"

	ole "github.com/go-ole/go-ole"
)

// ConnectObject creates a connection point between two services for communication.
func ConnectObject(disp *ole.IDispatch, iid *ole.GUID, idisp interface{}) (cookie uint32, err error) {
	unknown, err := disp.QueryInterface(ole.IID_IConnectionPointContainer)
	if err != nil {
		return
	}

	container := (*ole.IConnectionPointContainer)(unsafe.Pointer(unknown))
	var point *ole.IConnectionPoint
	err = container.FindConnectionPoint(iid, &point)
	if err != nil {
		return
	}
	if edisp, ok := idisp.(*ole.IUnknown); ok {
		cookie, err = point.Advise(edisp)
		container.Release()
		if err != nil {
			return
		}
	}
	rv := reflect.ValueOf(disp).Elem()
	if rv.Type().Kind() == reflect.Struct {
		dest := &stdDispatch{}
		dest.lpVtbl = &stdDispatchVtbl{}
		dest.lpVtbl.pQueryInterface = syscall.NewCallback(dispQueryInterface)
		dest.lpVtbl.pAddRef = syscall.NewCallback(dispAddRef)
		dest.lpVtbl.pRelease = syscall.NewCallback(dispRelease)
		dest.lpVtbl.pGetTypeInfoCount = syscall.NewCallback(dispGetTypeInfoCount)
		dest.lpVtbl.pGetTypeInfo = syscall.NewCallback(dispGetTypeInfo)
		dest.lpVtbl.pGetIDsOfNames = syscall.NewCallback(dispGetIDsOfNames)
		dest.lpVtbl.pInvoke = syscall.NewCallback(dispInvoke)
		dest.iface = disp
		dest.iid = iid
		cookie, err = point.Advise((*ole.IUnknown)(unsafe.Pointer(dest)))
		container.Release()
		if err != nil {
			point.Release()
			return
		}
		return
	}

	container.Release()

	return 0, ole.NewError(ole.E_INVALIDARG)
}
