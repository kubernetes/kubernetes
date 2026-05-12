package reflect2

import (
	"reflect"
	"unsafe"
)

type UnsafePtrType struct {
	unsafeType
}

func newUnsafePtrType(cfg *frozenConfig, type1 reflect.Type) *UnsafePtrType {
	return &UnsafePtrType{
		unsafeType: *newUnsafeType(cfg, type1),
	}
}

func (type2 *UnsafePtrType) IsNil(obj interface{}) bool {
	if obj == nil {
		return true
	}
	objEFace := unpackEFace(obj)
	assertType("Type.IsNil argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeIsNil(objEFace.data)
}

func (type2 *UnsafePtrType) UnsafeIsNil(ptr unsafe.Pointer) bool {
	if ptr == nil {
		return true
	}
	return *(*unsafe.Pointer)(ptr) == nil
}

func (type2 *UnsafePtrType) LikePtr() bool {
	return true
}

func (type2 *UnsafePtrType) Indirect(obj interface{}) interface{} {
	objEFace := unpackEFace(obj)
	assertType("Type.Indirect argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeIndirect(objEFace.data)
}

func (type2 *UnsafePtrType) UnsafeIndirect(ptr unsafe.Pointer) interface{} {
	return packEFace(type2.rtype, *(*unsafe.Pointer)(ptr))
}
