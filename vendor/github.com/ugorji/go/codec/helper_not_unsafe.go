// +build !go1.7 safe appengine

// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

import (
	"reflect"
	"sync/atomic"
)

// stringView returns a view of the []byte as a string.
// In unsafe mode, it doesn't incur allocation and copying caused by conversion.
// In regular safe mode, it is an allocation and copy.
//
// Usage: Always maintain a reference to v while result of this call is in use,
//        and call keepAlive4BytesView(v) at point where done with view.
func stringView(v []byte) string {
	return string(v)
}

// bytesView returns a view of the string as a []byte.
// In unsafe mode, it doesn't incur allocation and copying caused by conversion.
// In regular safe mode, it is an allocation and copy.
//
// Usage: Always maintain a reference to v while result of this call is in use,
//        and call keepAlive4BytesView(v) at point where done with view.
func bytesView(v string) []byte {
	return []byte(v)
}

func definitelyNil(v interface{}) bool {
	return false
	// rv := reflect.ValueOf(v)
	// switch rv.Kind() {
	// case reflect.Invalid:
	// 	return true
	// case reflect.Ptr, reflect.Interface, reflect.Chan, reflect.Slice, reflect.Map, reflect.Func:
	// 	return rv.IsNil()
	// default:
	// 	return false
	// }
}

// // keepAlive4BytesView maintains a reference to the input parameter for bytesView.
// //
// // Usage: call this at point where done with the bytes view.
// func keepAlive4BytesView(v string) {}

// // keepAlive4BytesView maintains a reference to the input parameter for stringView.
// //
// // Usage: call this at point where done with the string view.
// func keepAlive4StringView(v []byte) {}

func rv2i(rv reflect.Value) interface{} {
	return rv.Interface()
}

func rt2id(rt reflect.Type) uintptr {
	return reflect.ValueOf(rt).Pointer()
}

func rv2rtid(rv reflect.Value) uintptr {
	return reflect.ValueOf(rv.Type()).Pointer()
}

// --------------------------
// type ptrToRvMap struct{}

// func (_ *ptrToRvMap) init() {}
// func (_ *ptrToRvMap) get(i interface{}) reflect.Value {
// 	return reflect.ValueOf(i).Elem()
// }

// --------------------------
type atomicTypeInfoSlice struct {
	v atomic.Value
}

func (x *atomicTypeInfoSlice) load() *[]rtid2ti {
	i := x.v.Load()
	if i == nil {
		return nil
	}
	return i.(*[]rtid2ti)
}

func (x *atomicTypeInfoSlice) store(p *[]rtid2ti) {
	x.v.Store(p)
}

// --------------------------
func (d *Decoder) raw(f *codecFnInfo, rv reflect.Value) {
	rv.SetBytes(d.rawBytes())
}

func (d *Decoder) kString(f *codecFnInfo, rv reflect.Value) {
	rv.SetString(d.d.DecodeString())
}

func (d *Decoder) kBool(f *codecFnInfo, rv reflect.Value) {
	rv.SetBool(d.d.DecodeBool())
}

func (d *Decoder) kFloat32(f *codecFnInfo, rv reflect.Value) {
	rv.SetFloat(d.d.DecodeFloat(true))
}

func (d *Decoder) kFloat64(f *codecFnInfo, rv reflect.Value) {
	rv.SetFloat(d.d.DecodeFloat(false))
}

func (d *Decoder) kInt(f *codecFnInfo, rv reflect.Value) {
	rv.SetInt(d.d.DecodeInt(intBitsize))
}

func (d *Decoder) kInt8(f *codecFnInfo, rv reflect.Value) {
	rv.SetInt(d.d.DecodeInt(8))
}

func (d *Decoder) kInt16(f *codecFnInfo, rv reflect.Value) {
	rv.SetInt(d.d.DecodeInt(16))
}

func (d *Decoder) kInt32(f *codecFnInfo, rv reflect.Value) {
	rv.SetInt(d.d.DecodeInt(32))
}

func (d *Decoder) kInt64(f *codecFnInfo, rv reflect.Value) {
	rv.SetInt(d.d.DecodeInt(64))
}

func (d *Decoder) kUint(f *codecFnInfo, rv reflect.Value) {
	rv.SetUint(d.d.DecodeUint(uintBitsize))
}

func (d *Decoder) kUintptr(f *codecFnInfo, rv reflect.Value) {
	rv.SetUint(d.d.DecodeUint(uintBitsize))
}

func (d *Decoder) kUint8(f *codecFnInfo, rv reflect.Value) {
	rv.SetUint(d.d.DecodeUint(8))
}

func (d *Decoder) kUint16(f *codecFnInfo, rv reflect.Value) {
	rv.SetUint(d.d.DecodeUint(16))
}

func (d *Decoder) kUint32(f *codecFnInfo, rv reflect.Value) {
	rv.SetUint(d.d.DecodeUint(32))
}

func (d *Decoder) kUint64(f *codecFnInfo, rv reflect.Value) {
	rv.SetUint(d.d.DecodeUint(64))
}
