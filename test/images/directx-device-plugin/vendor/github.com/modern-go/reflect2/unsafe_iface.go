package reflect2

import (
	"reflect"
	"unsafe"
)

type iface struct {
	itab *itab
	data unsafe.Pointer
}

type itab struct {
	ignore unsafe.Pointer
	rtype  unsafe.Pointer
}

func IFaceToEFace(ptr unsafe.Pointer) interface{} {
	iface := (*iface)(ptr)
	if iface.itab == nil {
		return nil
	}
	return packEFace(iface.itab.rtype, iface.data)
}

type UnsafeIFaceType struct {
	unsafeType
}

func newUnsafeIFaceType(cfg *frozenConfig, type1 reflect.Type) *UnsafeIFaceType {
	return &UnsafeIFaceType{
		unsafeType: *newUnsafeType(cfg, type1),
	}
}

func (type2 *UnsafeIFaceType) Indirect(obj interface{}) interface{} {
	objEFace := unpackEFace(obj)
	assertType("Type.Indirect argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeIndirect(objEFace.data)
}

func (type2 *UnsafeIFaceType) UnsafeIndirect(ptr unsafe.Pointer) interface{} {
	return IFaceToEFace(ptr)
}

func (type2 *UnsafeIFaceType) IsNil(obj interface{}) bool {
	if obj == nil {
		return true
	}
	objEFace := unpackEFace(obj)
	assertType("Type.IsNil argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeIsNil(objEFace.data)
}

func (type2 *UnsafeIFaceType) UnsafeIsNil(ptr unsafe.Pointer) bool {
	if ptr == nil {
		return true
	}
	iface := (*iface)(ptr)
	if iface.itab == nil {
		return true
	}
	return false
}
