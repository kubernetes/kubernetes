package reflect2

import (
	"reflect"
	"unsafe"
)

// sliceHeader is a safe version of SliceHeader used within this package.
type sliceHeader struct {
	Data unsafe.Pointer
	Len  int
	Cap  int
}

type UnsafeSliceType struct {
	unsafeType
	elemRType  unsafe.Pointer
	pElemRType unsafe.Pointer
	elemSize   uintptr
}

func newUnsafeSliceType(cfg *frozenConfig, type1 reflect.Type) SliceType {
	elemType := type1.Elem()
	return &UnsafeSliceType{
		unsafeType: *newUnsafeType(cfg, type1),
		pElemRType: unpackEFace(reflect.PtrTo(elemType)).data,
		elemRType:  unpackEFace(elemType).data,
		elemSize:   elemType.Size(),
	}
}

func (type2 *UnsafeSliceType) Set(obj interface{}, val interface{}) {
	objEFace := unpackEFace(obj)
	assertType("Type.Set argument 1", type2.ptrRType, objEFace.rtype)
	valEFace := unpackEFace(val)
	assertType("Type.Set argument 2", type2.ptrRType, valEFace.rtype)
	type2.UnsafeSet(objEFace.data, valEFace.data)
}

func (type2 *UnsafeSliceType) UnsafeSet(ptr unsafe.Pointer, val unsafe.Pointer) {
	*(*sliceHeader)(ptr) = *(*sliceHeader)(val)
}

func (type2 *UnsafeSliceType) IsNil(obj interface{}) bool {
	if obj == nil {
		return true
	}
	objEFace := unpackEFace(obj)
	assertType("Type.IsNil argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeIsNil(objEFace.data)
}

func (type2 *UnsafeSliceType) UnsafeIsNil(ptr unsafe.Pointer) bool {
	if ptr == nil {
		return true
	}
	return (*sliceHeader)(ptr).Data == nil
}

func (type2 *UnsafeSliceType) SetNil(obj interface{}) {
	objEFace := unpackEFace(obj)
	assertType("SliceType.SetNil argument 1", type2.ptrRType, objEFace.rtype)
	type2.UnsafeSetNil(objEFace.data)
}

func (type2 *UnsafeSliceType) UnsafeSetNil(ptr unsafe.Pointer) {
	header := (*sliceHeader)(ptr)
	header.Len = 0
	header.Cap = 0
	header.Data = nil
}

func (type2 *UnsafeSliceType) MakeSlice(length int, cap int) interface{} {
	return packEFace(type2.ptrRType, type2.UnsafeMakeSlice(length, cap))
}

func (type2 *UnsafeSliceType) UnsafeMakeSlice(length int, cap int) unsafe.Pointer {
	header := &sliceHeader{unsafe_NewArray(type2.elemRType, cap), length, cap}
	return unsafe.Pointer(header)
}

func (type2 *UnsafeSliceType) LengthOf(obj interface{}) int {
	objEFace := unpackEFace(obj)
	assertType("SliceType.Len argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeLengthOf(objEFace.data)
}

func (type2 *UnsafeSliceType) UnsafeLengthOf(obj unsafe.Pointer) int {
	header := (*sliceHeader)(obj)
	return header.Len
}

func (type2 *UnsafeSliceType) SetIndex(obj interface{}, index int, elem interface{}) {
	objEFace := unpackEFace(obj)
	assertType("SliceType.SetIndex argument 1", type2.ptrRType, objEFace.rtype)
	elemEFace := unpackEFace(elem)
	assertType("SliceType.SetIndex argument 3", type2.pElemRType, elemEFace.rtype)
	type2.UnsafeSetIndex(objEFace.data, index, elemEFace.data)
}

func (type2 *UnsafeSliceType) UnsafeSetIndex(obj unsafe.Pointer, index int, elem unsafe.Pointer) {
	header := (*sliceHeader)(obj)
	elemPtr := arrayAt(header.Data, index, type2.elemSize, "i < s.Len")
	typedmemmove(type2.elemRType, elemPtr, elem)
}

func (type2 *UnsafeSliceType) GetIndex(obj interface{}, index int) interface{} {
	objEFace := unpackEFace(obj)
	assertType("SliceType.GetIndex argument 1", type2.ptrRType, objEFace.rtype)
	elemPtr := type2.UnsafeGetIndex(objEFace.data, index)
	return packEFace(type2.pElemRType, elemPtr)
}

func (type2 *UnsafeSliceType) UnsafeGetIndex(obj unsafe.Pointer, index int) unsafe.Pointer {
	header := (*sliceHeader)(obj)
	return arrayAt(header.Data, index, type2.elemSize, "i < s.Len")
}

func (type2 *UnsafeSliceType) Append(obj interface{}, elem interface{}) {
	objEFace := unpackEFace(obj)
	assertType("SliceType.Append argument 1", type2.ptrRType, objEFace.rtype)
	elemEFace := unpackEFace(elem)
	assertType("SliceType.Append argument 2", type2.pElemRType, elemEFace.rtype)
	type2.UnsafeAppend(objEFace.data, elemEFace.data)
}

func (type2 *UnsafeSliceType) UnsafeAppend(obj unsafe.Pointer, elem unsafe.Pointer) {
	header := (*sliceHeader)(obj)
	oldLen := header.Len
	type2.UnsafeGrow(obj, oldLen+1)
	type2.UnsafeSetIndex(obj, oldLen, elem)
}

func (type2 *UnsafeSliceType) Cap(obj interface{}) int {
	objEFace := unpackEFace(obj)
	assertType("SliceType.Cap argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeCap(objEFace.data)
}

func (type2 *UnsafeSliceType) UnsafeCap(ptr unsafe.Pointer) int {
	return (*sliceHeader)(ptr).Cap
}

func (type2 *UnsafeSliceType) Grow(obj interface{}, newLength int) {
	objEFace := unpackEFace(obj)
	assertType("SliceType.Grow argument 1", type2.ptrRType, objEFace.rtype)
	type2.UnsafeGrow(objEFace.data, newLength)
}

func (type2 *UnsafeSliceType) UnsafeGrow(obj unsafe.Pointer, newLength int) {
	header := (*sliceHeader)(obj)
	if newLength <= header.Cap {
		header.Len = newLength
		return
	}
	newCap := calcNewCap(header.Cap, newLength)
	newHeader := (*sliceHeader)(type2.UnsafeMakeSlice(header.Len, newCap))
	typedslicecopy(type2.elemRType, *newHeader, *header)
	header.Data = newHeader.Data
	header.Cap = newHeader.Cap
	header.Len = newLength
}

func calcNewCap(cap int, expectedCap int) int {
	if cap == 0 {
		cap = expectedCap
	} else {
		for cap < expectedCap {
			if cap < 1024 {
				cap += cap
			} else {
				cap += cap / 4
			}
		}
	}
	return cap
}
