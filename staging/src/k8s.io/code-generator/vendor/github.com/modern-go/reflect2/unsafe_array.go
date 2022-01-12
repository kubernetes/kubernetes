package reflect2

import (
	"reflect"
	"unsafe"
)

type UnsafeArrayType struct {
	unsafeType
	elemRType  unsafe.Pointer
	pElemRType unsafe.Pointer
	elemSize   uintptr
	likePtr    bool
}

func newUnsafeArrayType(cfg *frozenConfig, type1 reflect.Type) *UnsafeArrayType {
	return &UnsafeArrayType{
		unsafeType: *newUnsafeType(cfg, type1),
		elemRType:  unpackEFace(type1.Elem()).data,
		pElemRType: unpackEFace(reflect.PtrTo(type1.Elem())).data,
		elemSize:   type1.Elem().Size(),
		likePtr:    likePtrType(type1),
	}
}

func (type2 *UnsafeArrayType) LikePtr() bool {
	return type2.likePtr
}

func (type2 *UnsafeArrayType) Indirect(obj interface{}) interface{} {
	objEFace := unpackEFace(obj)
	assertType("Type.Indirect argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeIndirect(objEFace.data)
}

func (type2 *UnsafeArrayType) UnsafeIndirect(ptr unsafe.Pointer) interface{} {
	if type2.likePtr {
		return packEFace(type2.rtype, *(*unsafe.Pointer)(ptr))
	}
	return packEFace(type2.rtype, ptr)
}

func (type2 *UnsafeArrayType) SetIndex(obj interface{}, index int, elem interface{}) {
	objEFace := unpackEFace(obj)
	assertType("ArrayType.SetIndex argument 1", type2.ptrRType, objEFace.rtype)
	elemEFace := unpackEFace(elem)
	assertType("ArrayType.SetIndex argument 3", type2.pElemRType, elemEFace.rtype)
	type2.UnsafeSetIndex(objEFace.data, index, elemEFace.data)
}

func (type2 *UnsafeArrayType) UnsafeSetIndex(obj unsafe.Pointer, index int, elem unsafe.Pointer) {
	elemPtr := arrayAt(obj, index, type2.elemSize, "i < s.Len")
	typedmemmove(type2.elemRType, elemPtr, elem)
}

func (type2 *UnsafeArrayType) GetIndex(obj interface{}, index int) interface{} {
	objEFace := unpackEFace(obj)
	assertType("ArrayType.GetIndex argument 1", type2.ptrRType, objEFace.rtype)
	elemPtr := type2.UnsafeGetIndex(objEFace.data, index)
	return packEFace(type2.pElemRType, elemPtr)
}

func (type2 *UnsafeArrayType) UnsafeGetIndex(obj unsafe.Pointer, index int) unsafe.Pointer {
	return arrayAt(obj, index, type2.elemSize, "i < s.Len")
}
