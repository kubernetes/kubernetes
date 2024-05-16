package reflect2

import (
	"reflect"
	"unsafe"
)

type UnsafeStructField struct {
	reflect.StructField
	structType *UnsafeStructType
	rtype      unsafe.Pointer
	ptrRType   unsafe.Pointer
}

func newUnsafeStructField(structType *UnsafeStructType, structField reflect.StructField) *UnsafeStructField {
	return &UnsafeStructField{
		StructField: structField,
		rtype:       unpackEFace(structField.Type).data,
		ptrRType:    unpackEFace(reflect.PtrTo(structField.Type)).data,
		structType:  structType,
	}
}

func (field *UnsafeStructField) Offset() uintptr {
	return field.StructField.Offset
}

func (field *UnsafeStructField) Name() string {
	return field.StructField.Name
}

func (field *UnsafeStructField) PkgPath() string {
	return field.StructField.PkgPath
}

func (field *UnsafeStructField) Type() Type {
	return field.structType.cfg.Type2(field.StructField.Type)
}

func (field *UnsafeStructField) Tag() reflect.StructTag {
	return field.StructField.Tag
}

func (field *UnsafeStructField) Index() []int {
	return field.StructField.Index
}

func (field *UnsafeStructField) Anonymous() bool {
	return field.StructField.Anonymous
}

func (field *UnsafeStructField) Set(obj interface{}, value interface{}) {
	objEFace := unpackEFace(obj)
	assertType("StructField.SetIndex argument 1", field.structType.ptrRType, objEFace.rtype)
	valueEFace := unpackEFace(value)
	assertType("StructField.SetIndex argument 2", field.ptrRType, valueEFace.rtype)
	field.UnsafeSet(objEFace.data, valueEFace.data)
}

func (field *UnsafeStructField) UnsafeSet(obj unsafe.Pointer, value unsafe.Pointer) {
	fieldPtr := add(obj, field.StructField.Offset, "same as non-reflect &v.field")
	typedmemmove(field.rtype, fieldPtr, value)
}

func (field *UnsafeStructField) Get(obj interface{}) interface{} {
	objEFace := unpackEFace(obj)
	assertType("StructField.GetIndex argument 1", field.structType.ptrRType, objEFace.rtype)
	value := field.UnsafeGet(objEFace.data)
	return packEFace(field.ptrRType, value)
}

func (field *UnsafeStructField) UnsafeGet(obj unsafe.Pointer) unsafe.Pointer {
	return add(obj, field.StructField.Offset, "same as non-reflect &v.field")
}
