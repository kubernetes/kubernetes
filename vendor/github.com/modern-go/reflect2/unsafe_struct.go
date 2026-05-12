package reflect2

import (
	"reflect"
	"unsafe"
)

type UnsafeStructType struct {
	unsafeType
	likePtr bool
}

func newUnsafeStructType(cfg *frozenConfig, type1 reflect.Type) *UnsafeStructType {
	return &UnsafeStructType{
		unsafeType: *newUnsafeType(cfg, type1),
		likePtr:    likePtrType(type1),
	}
}

func (type2 *UnsafeStructType) LikePtr() bool {
	return type2.likePtr
}

func (type2 *UnsafeStructType) Indirect(obj interface{}) interface{} {
	objEFace := unpackEFace(obj)
	assertType("Type.Indirect argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeIndirect(objEFace.data)
}

func (type2 *UnsafeStructType) UnsafeIndirect(ptr unsafe.Pointer) interface{} {
	if type2.likePtr {
		return packEFace(type2.rtype, *(*unsafe.Pointer)(ptr))
	}
	return packEFace(type2.rtype, ptr)
}

func (type2 *UnsafeStructType) FieldByName(name string) StructField {
	structField, found := type2.Type.FieldByName(name)
	if !found {
		return nil
	}
	return newUnsafeStructField(type2, structField)
}

func (type2 *UnsafeStructType) Field(i int) StructField {
	return newUnsafeStructField(type2, type2.Type.Field(i))
}

func (type2 *UnsafeStructType) FieldByIndex(index []int) StructField {
	return newUnsafeStructField(type2, type2.Type.FieldByIndex(index))
}

func (type2 *UnsafeStructType) FieldByNameFunc(match func(string) bool) StructField {
	structField, found := type2.Type.FieldByNameFunc(match)
	if !found {
		panic("field match condition not found in " + type2.Type.String())
	}
	return newUnsafeStructField(type2, structField)
}
