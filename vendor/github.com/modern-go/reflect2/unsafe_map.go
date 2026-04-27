package reflect2

import (
	"reflect"
	"unsafe"
)

type UnsafeMapType struct {
	unsafeType
	pKeyRType  unsafe.Pointer
	pElemRType unsafe.Pointer
}

func newUnsafeMapType(cfg *frozenConfig, type1 reflect.Type) MapType {
	return &UnsafeMapType{
		unsafeType: *newUnsafeType(cfg, type1),
		pKeyRType:  unpackEFace(reflect.PtrTo(type1.Key())).data,
		pElemRType: unpackEFace(reflect.PtrTo(type1.Elem())).data,
	}
}

func (type2 *UnsafeMapType) IsNil(obj interface{}) bool {
	if obj == nil {
		return true
	}
	objEFace := unpackEFace(obj)
	assertType("Type.IsNil argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeIsNil(objEFace.data)
}

func (type2 *UnsafeMapType) UnsafeIsNil(ptr unsafe.Pointer) bool {
	if ptr == nil {
		return true
	}
	return *(*unsafe.Pointer)(ptr) == nil
}

func (type2 *UnsafeMapType) LikePtr() bool {
	return true
}

func (type2 *UnsafeMapType) Indirect(obj interface{}) interface{} {
	objEFace := unpackEFace(obj)
	assertType("MapType.Indirect argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeIndirect(objEFace.data)
}

func (type2 *UnsafeMapType) UnsafeIndirect(ptr unsafe.Pointer) interface{} {
	return packEFace(type2.rtype, *(*unsafe.Pointer)(ptr))
}

func (type2 *UnsafeMapType) Key() Type {
	return type2.cfg.Type2(type2.Type.Key())
}

func (type2 *UnsafeMapType) MakeMap(cap int) interface{} {
	return packEFace(type2.ptrRType, type2.UnsafeMakeMap(cap))
}

func (type2 *UnsafeMapType) UnsafeMakeMap(cap int) unsafe.Pointer {
	m := makeMapWithSize(type2.rtype, cap)
	return unsafe.Pointer(&m)
}

func (type2 *UnsafeMapType) SetIndex(obj interface{}, key interface{}, elem interface{}) {
	objEFace := unpackEFace(obj)
	assertType("MapType.SetIndex argument 1", type2.ptrRType, objEFace.rtype)
	keyEFace := unpackEFace(key)
	assertType("MapType.SetIndex argument 2", type2.pKeyRType, keyEFace.rtype)
	elemEFace := unpackEFace(elem)
	assertType("MapType.SetIndex argument 3", type2.pElemRType, elemEFace.rtype)
	type2.UnsafeSetIndex(objEFace.data, keyEFace.data, elemEFace.data)
}

func (type2 *UnsafeMapType) UnsafeSetIndex(obj unsafe.Pointer, key unsafe.Pointer, elem unsafe.Pointer) {
	mapassign(type2.rtype, *(*unsafe.Pointer)(obj), key, elem)
}

func (type2 *UnsafeMapType) TryGetIndex(obj interface{}, key interface{}) (interface{}, bool) {
	objEFace := unpackEFace(obj)
	assertType("MapType.TryGetIndex argument 1", type2.ptrRType, objEFace.rtype)
	keyEFace := unpackEFace(key)
	assertType("MapType.TryGetIndex argument 2", type2.pKeyRType, keyEFace.rtype)
	elemPtr := type2.UnsafeGetIndex(objEFace.data, keyEFace.data)
	if elemPtr == nil {
		return nil, false
	}
	return packEFace(type2.pElemRType, elemPtr), true
}

func (type2 *UnsafeMapType) GetIndex(obj interface{}, key interface{}) interface{} {
	objEFace := unpackEFace(obj)
	assertType("MapType.GetIndex argument 1", type2.ptrRType, objEFace.rtype)
	keyEFace := unpackEFace(key)
	assertType("MapType.GetIndex argument 2", type2.pKeyRType, keyEFace.rtype)
	elemPtr := type2.UnsafeGetIndex(objEFace.data, keyEFace.data)
	return packEFace(type2.pElemRType, elemPtr)
}

func (type2 *UnsafeMapType) UnsafeGetIndex(obj unsafe.Pointer, key unsafe.Pointer) unsafe.Pointer {
	return mapaccess(type2.rtype, *(*unsafe.Pointer)(obj), key)
}

func (type2 *UnsafeMapType) Iterate(obj interface{}) MapIterator {
	objEFace := unpackEFace(obj)
	assertType("MapType.Iterate argument 1", type2.ptrRType, objEFace.rtype)
	return type2.UnsafeIterate(objEFace.data)
}

type UnsafeMapIterator struct {
	*hiter
	pKeyRType  unsafe.Pointer
	pElemRType unsafe.Pointer
}

func (iter *UnsafeMapIterator) HasNext() bool {
	return iter.key != nil
}

func (iter *UnsafeMapIterator) Next() (interface{}, interface{}) {
	key, elem := iter.UnsafeNext()
	return packEFace(iter.pKeyRType, key), packEFace(iter.pElemRType, elem)
}

func (iter *UnsafeMapIterator) UnsafeNext() (unsafe.Pointer, unsafe.Pointer) {
	key := iter.key
	elem := iter.value
	mapiternext(iter.hiter)
	return key, elem
}
