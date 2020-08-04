package ole

import "unsafe"

type IConnectionPointContainer struct {
	IUnknown
}

type IConnectionPointContainerVtbl struct {
	IUnknownVtbl
	EnumConnectionPoints uintptr
	FindConnectionPoint  uintptr
}

func (v *IConnectionPointContainer) VTable() *IConnectionPointContainerVtbl {
	return (*IConnectionPointContainerVtbl)(unsafe.Pointer(v.RawVTable))
}
