/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package value

import (
	"encoding/base64"
	"fmt"
	"reflect"
)

// NewValueReflect creates a Value backed by an "interface{}" type,
// typically an structured object in Kubernetes world that is uses reflection to expose.
// The provided "interface{}" value must be a pointer so that the value can be modified via reflection.
// The provided "interface{}" may contain structs and types that are converted to Values
// by the jsonMarshaler interface.
func NewValueReflect(value interface{}) (Value, error) {
	if value == nil {
		return NewValueInterface(nil), nil
	}
	v := reflect.ValueOf(value)
	if v.Kind() != reflect.Ptr {
		// The root value to reflect on must be a pointer so that map.Set() and map.Delete() operations are possible.
		return nil, fmt.Errorf("value provided to NewValueReflect must be a pointer")
	}
	return wrapValueReflect(v, nil, nil)
}

// wrapValueReflect wraps the provide reflect.Value as a value. If parent in the data tree is a map, parentMap
// and parentMapKey must be provided so that the returned value may be set and deleted.
func wrapValueReflect(value reflect.Value, parentMap, parentMapKey *reflect.Value) (Value, error) {
	val := HeapAllocator.allocValueReflect()
	return val.reuse(value, nil, parentMap, parentMapKey)
}

// wrapValueReflect wraps the provide reflect.Value as a value, and panics if there is an error. If parent in the data
// tree is a map, parentMap and parentMapKey must be provided so that the returned value may be set and deleted.
func mustWrapValueReflect(value reflect.Value, parentMap, parentMapKey *reflect.Value) Value {
	v, err := wrapValueReflect(value, parentMap, parentMapKey)
	if err != nil {
		panic(err)
	}
	return v
}

// the value interface doesn't care about the type for value.IsNull, so we can use a constant
var nilType = reflect.TypeOf(&struct{}{})

// reuse replaces the value of the valueReflect. If parent in the data tree is a map, parentMap and parentMapKey
// must be provided so that the returned value may be set and deleted.
func (r *valueReflect) reuse(value reflect.Value, cacheEntry *TypeReflectCacheEntry, parentMap, parentMapKey *reflect.Value) (Value, error) {
	if cacheEntry == nil {
		cacheEntry = TypeReflectEntryOf(value.Type())
	}
	if cacheEntry.CanConvertToUnstructured() {
		u, err := cacheEntry.ToUnstructured(value)
		if err != nil {
			return nil, err
		}
		if u == nil {
			value = reflect.Zero(nilType)
		} else {
			value = reflect.ValueOf(u)
		}
	}
	r.Value = dereference(value)
	r.ParentMap = parentMap
	r.ParentMapKey = parentMapKey
	r.kind = kind(r.Value)
	return r, nil
}

// mustReuse replaces the value of the valueReflect and panics if there is an error. If parent in the data tree is a
// map, parentMap and parentMapKey must be provided so that the returned value may be set and deleted.
func (r *valueReflect) mustReuse(value reflect.Value, cacheEntry *TypeReflectCacheEntry, parentMap, parentMapKey *reflect.Value) Value {
	v, err := r.reuse(value, cacheEntry, parentMap, parentMapKey)
	if err != nil {
		panic(err)
	}
	return v
}

func dereference(val reflect.Value) reflect.Value {
	kind := val.Kind()
	if (kind == reflect.Interface || kind == reflect.Ptr) && !safeIsNil(val) {
		return val.Elem()
	}
	return val
}

type valueReflect struct {
	ParentMap    *reflect.Value
	ParentMapKey *reflect.Value
	Value        reflect.Value
	kind         reflectType
}

func (r valueReflect) IsMap() bool {
	return r.kind == mapType || r.kind == structMapType
}

func (r valueReflect) IsList() bool {
	return r.kind == listType
}

func (r valueReflect) IsBool() bool {
	return r.kind == boolType
}

func (r valueReflect) IsInt() bool {
	return r.kind == intType || r.kind == uintType
}

func (r valueReflect) IsFloat() bool {
	return r.kind == floatType
}

func (r valueReflect) IsString() bool {
	return r.kind == stringType || r.kind == byteStringType
}

func (r valueReflect) IsNull() bool {
	return r.kind == nullType
}

type reflectType = int

const (
	mapType = iota
	structMapType
	listType
	intType
	uintType
	floatType
	stringType
	byteStringType
	boolType
	nullType
)

func kind(v reflect.Value) reflectType {
	typ := v.Type()
	rk := typ.Kind()
	switch rk {
	case reflect.Map:
		if v.IsNil() {
			return nullType
		}
		return mapType
	case reflect.Struct:
		return structMapType
	case reflect.Int, reflect.Int64, reflect.Int32, reflect.Int16, reflect.Int8:
		return intType
	case reflect.Uint, reflect.Uint32, reflect.Uint16, reflect.Uint8:
		// Uint64 deliberately excluded, see valueUnstructured.Int.
		return uintType
	case reflect.Float64, reflect.Float32:
		return floatType
	case reflect.String:
		return stringType
	case reflect.Bool:
		return boolType
	case reflect.Slice:
		if v.IsNil() {
			return nullType
		}
		elemKind := typ.Elem().Kind()
		if elemKind == reflect.Uint8 {
			return byteStringType
		}
		return listType
	case reflect.Chan, reflect.Func, reflect.Ptr, reflect.UnsafePointer, reflect.Interface:
		if v.IsNil() {
			return nullType
		}
		panic(fmt.Sprintf("unsupported type: %v", v.Type()))
	default:
		panic(fmt.Sprintf("unsupported type: %v", v.Type()))
	}
}

// TODO find a cleaner way to avoid panics from reflect.IsNil()
func safeIsNil(v reflect.Value) bool {
	k := v.Kind()
	switch k {
	case reflect.Chan, reflect.Func, reflect.Map, reflect.Ptr, reflect.UnsafePointer, reflect.Interface, reflect.Slice:
		return v.IsNil()
	}
	return false
}

func (r valueReflect) AsMap() Map {
	return r.AsMapUsing(HeapAllocator)
}

func (r valueReflect) AsMapUsing(a Allocator) Map {
	switch r.kind {
	case structMapType:
		v := a.allocStructReflect()
		v.valueReflect = r
		return v
	case mapType:
		v := a.allocMapReflect()
		v.valueReflect = r
		return v
	default:
		panic("value is not a map or struct")
	}
}

func (r valueReflect) AsList() List {
	return r.AsListUsing(HeapAllocator)
}

func (r valueReflect) AsListUsing(a Allocator) List {
	if r.IsList() {
		v := a.allocListReflect()
		v.Value = r.Value
		return v
	}
	panic("value is not a list")
}

func (r valueReflect) AsBool() bool {
	if r.IsBool() {
		return r.Value.Bool()
	}
	panic("value is not a bool")
}

func (r valueReflect) AsInt() int64 {
	if r.kind == intType {
		return r.Value.Int()
	}
	if r.kind == uintType {
		return int64(r.Value.Uint())
	}

	panic("value is not an int")
}

func (r valueReflect) AsFloat() float64 {
	if r.IsFloat() {
		return r.Value.Float()
	}
	panic("value is not a float")
}

func (r valueReflect) AsString() string {
	switch r.kind {
	case stringType:
		return r.Value.String()
	case byteStringType:
		return base64.StdEncoding.EncodeToString(r.Value.Bytes())
	}
	panic("value is not a string")
}

func (r valueReflect) Unstructured() interface{} {
	val := r.Value
	switch {
	case r.IsNull():
		return nil
	case val.Kind() == reflect.Struct:
		return structReflect{r}.Unstructured()
	case val.Kind() == reflect.Map:
		return mapReflect{valueReflect: r}.Unstructured()
	case r.IsList():
		return listReflect{r.Value}.Unstructured()
	case r.IsString():
		return r.AsString()
	case r.IsInt():
		return r.AsInt()
	case r.IsBool():
		return r.AsBool()
	case r.IsFloat():
		return r.AsFloat()
	default:
		panic(fmt.Sprintf("value of type %s is not a supported by value reflector", val.Type()))
	}
}
