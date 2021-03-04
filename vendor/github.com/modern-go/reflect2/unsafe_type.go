package reflect2

import (
	"reflect"
	"unsafe"
)

type unsafeType struct {
	safeType
	rtype    unsafe.Pointer
	ptrRType unsafe.Pointer
}

func newUnsafeType(cfg *frozenConfig, type1 reflect.Type) *unsafeType {
	return &unsafeType{
		safeType: safeType{
			Type: type1,
			cfg:  cfg,
		},
		rtype:    unpackEFace(type1).data,
		ptrRType: unpackEFace(reflect.PtrTo(type1)).data,
	}
}

func (type2 *unsafeType) Set(obj interface{}, val interface{}) {
	objEFace := unpackEFace(obj)
	assertType("Type.Set argument 1", type2.ptrRType, objEFace.rtype)
	valEFace := unpackEFace(val)
	assertType("Type.Set argument 2", type2.ptrRType, valEFace.rtype)
	type2.UnsafeSet(objEFace.data, valEFace.data)
}

func (type2 *unsafeType) UnsafeSet(ptr unsafe.Pointer, val unsafe.Pointer) {
	typedmemmove(type2.rtype, ptr, val)
}

func (type2 *unsafeType) IsNil(obj interface{}) bool {
	objEFace := unpackEFace(obj)
	assertType("Type.IsNil argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeIsNil(objEFace.data)
}

func (type2 *unsafeType) UnsafeIsNil(ptr unsafe.Pointer) bool {
	return ptr == nil
}

func (type2 *unsafeType) UnsafeNew() unsafe.Pointer {
	return unsafe_New(type2.rtype)
}

func (type2 *unsafeType) New() interface{} {
	return packEFace(type2.ptrRType, type2.UnsafeNew())
}

func (type2 *unsafeType) PackEFace(ptr unsafe.Pointer) interface{} {
	return packEFace(type2.ptrRType, ptr)
}

func (type2 *unsafeType) RType() uintptr {
	return uintptr(type2.rtype)
}

func (type2 *unsafeType) Indirect(obj interface{}) interface{} {
	objEFace := unpackEFace(obj)
	assertType("Type.Indirect argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeIndirect(objEFace.data)
}

func (type2 *unsafeType) UnsafeIndirect(obj unsafe.Pointer) interface{} {
	return packEFace(type2.rtype, obj)
}

func (type2 *unsafeType) LikePtr() bool {
	return false
}

func assertType(where string, expectRType unsafe.Pointer, actualRType unsafe.Pointer) {
	if expectRType != actualRType {
		expectType := reflect.TypeOf(0)
		(*iface)(unsafe.Pointer(&expectType)).data = expectRType
		actualType := reflect.TypeOf(0)
		(*iface)(unsafe.Pointer(&actualType)).data = actualRType
		panic(where + ": expect " + expectType.String() + ", actual " + actualType.String())
	}
}
