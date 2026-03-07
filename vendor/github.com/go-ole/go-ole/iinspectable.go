package ole

import "unsafe"

type IInspectable struct {
	IUnknown
}

type IInspectableVtbl struct {
	IUnknownVtbl
	GetIIds             uintptr
	GetRuntimeClassName uintptr
	GetTrustLevel       uintptr
}

func (v *IInspectable) VTable() *IInspectableVtbl {
	return (*IInspectableVtbl)(unsafe.Pointer(v.RawVTable))
}
