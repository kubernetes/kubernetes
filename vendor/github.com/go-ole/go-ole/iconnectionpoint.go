package ole

import "unsafe"

type IConnectionPoint struct {
	IUnknown
}

type IConnectionPointVtbl struct {
	IUnknownVtbl
	GetConnectionInterface      uintptr
	GetConnectionPointContainer uintptr
	Advise                      uintptr
	Unadvise                    uintptr
	EnumConnections             uintptr
}

func (v *IConnectionPoint) VTable() *IConnectionPointVtbl {
	return (*IConnectionPointVtbl)(unsafe.Pointer(v.RawVTable))
}
