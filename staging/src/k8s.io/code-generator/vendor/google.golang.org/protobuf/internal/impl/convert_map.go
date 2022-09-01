// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"fmt"
	"reflect"

	pref "google.golang.org/protobuf/reflect/protoreflect"
)

type mapConverter struct {
	goType           reflect.Type // map[K]V
	keyConv, valConv Converter
}

func newMapConverter(t reflect.Type, fd pref.FieldDescriptor) *mapConverter {
	if t.Kind() != reflect.Map {
		panic(fmt.Sprintf("invalid Go type %v for field %v", t, fd.FullName()))
	}
	return &mapConverter{
		goType:  t,
		keyConv: newSingularConverter(t.Key(), fd.MapKey()),
		valConv: newSingularConverter(t.Elem(), fd.MapValue()),
	}
}

func (c *mapConverter) PBValueOf(v reflect.Value) pref.Value {
	if v.Type() != c.goType {
		panic(fmt.Sprintf("invalid type: got %v, want %v", v.Type(), c.goType))
	}
	return pref.ValueOfMap(&mapReflect{v, c.keyConv, c.valConv})
}

func (c *mapConverter) GoValueOf(v pref.Value) reflect.Value {
	return v.Map().(*mapReflect).v
}

func (c *mapConverter) IsValidPB(v pref.Value) bool {
	mapv, ok := v.Interface().(*mapReflect)
	if !ok {
		return false
	}
	return mapv.v.Type() == c.goType
}

func (c *mapConverter) IsValidGo(v reflect.Value) bool {
	return v.IsValid() && v.Type() == c.goType
}

func (c *mapConverter) New() pref.Value {
	return c.PBValueOf(reflect.MakeMap(c.goType))
}

func (c *mapConverter) Zero() pref.Value {
	return c.PBValueOf(reflect.Zero(c.goType))
}

type mapReflect struct {
	v       reflect.Value // map[K]V
	keyConv Converter
	valConv Converter
}

func (ms *mapReflect) Len() int {
	return ms.v.Len()
}
func (ms *mapReflect) Has(k pref.MapKey) bool {
	rk := ms.keyConv.GoValueOf(k.Value())
	rv := ms.v.MapIndex(rk)
	return rv.IsValid()
}
func (ms *mapReflect) Get(k pref.MapKey) pref.Value {
	rk := ms.keyConv.GoValueOf(k.Value())
	rv := ms.v.MapIndex(rk)
	if !rv.IsValid() {
		return pref.Value{}
	}
	return ms.valConv.PBValueOf(rv)
}
func (ms *mapReflect) Set(k pref.MapKey, v pref.Value) {
	rk := ms.keyConv.GoValueOf(k.Value())
	rv := ms.valConv.GoValueOf(v)
	ms.v.SetMapIndex(rk, rv)
}
func (ms *mapReflect) Clear(k pref.MapKey) {
	rk := ms.keyConv.GoValueOf(k.Value())
	ms.v.SetMapIndex(rk, reflect.Value{})
}
func (ms *mapReflect) Mutable(k pref.MapKey) pref.Value {
	if _, ok := ms.valConv.(*messageConverter); !ok {
		panic("invalid Mutable on map with non-message value type")
	}
	v := ms.Get(k)
	if !v.IsValid() {
		v = ms.NewValue()
		ms.Set(k, v)
	}
	return v
}
func (ms *mapReflect) Range(f func(pref.MapKey, pref.Value) bool) {
	iter := mapRange(ms.v)
	for iter.Next() {
		k := ms.keyConv.PBValueOf(iter.Key()).MapKey()
		v := ms.valConv.PBValueOf(iter.Value())
		if !f(k, v) {
			return
		}
	}
}
func (ms *mapReflect) NewValue() pref.Value {
	return ms.valConv.New()
}
func (ms *mapReflect) IsValid() bool {
	return !ms.v.IsNil()
}
func (ms *mapReflect) protoUnwrap() interface{} {
	return ms.v.Interface()
}
