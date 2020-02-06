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
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
)

var reflectPool = sync.Pool{
	New: func() interface{} {
		return &valueReflect{}
	},
}

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
	return wrapValueReflect(nil, nil, v)
}

func wrapValueReflect(parentMap, parentMapKey *reflect.Value, value reflect.Value) (Value, error) {
	// TODO: conversion of json.Marshaller interface types is expensive. This can be mostly optimized away by
	// introducing conversion functions that do not require going through JSON and using those here.
	if marshaler, ok := getMarshaler(value); ok {
		return toUnstructured(marshaler, value)
	}
	value = dereference(value)
	val := reflectPool.Get().(*valueReflect)
	val.Value = value
	val.ParentMap = parentMap
	val.ParentMapKey = parentMapKey
	return Value(val), nil
}

func mustWrapValueReflect(value reflect.Value) Value {
	v, err := wrapValueReflect(nil, nil, value)
	if err != nil {
		panic(err)
	}
	return v
}

func mustWrapValueReflectMapItem(parentMap, parentMapKey *reflect.Value, value reflect.Value) Value {
	v, err := wrapValueReflect(parentMap, parentMapKey, value)
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
}

func (r valueReflect) IsMap() bool {
	return r.isKind(reflect.Map, reflect.Struct)
}

func (r valueReflect) IsList() bool {
	typ := r.Value.Type()
	return typ.Kind() == reflect.Slice && !(typ.Elem().Kind() == reflect.Uint8)
}

func (r valueReflect) IsBool() bool {
	return r.isKind(reflect.Bool)
}

func (r valueReflect) IsInt() bool {
	// Uint64 deliberately excluded, see valueUnstructured.Int.
	return r.isKind(reflect.Int, reflect.Int64, reflect.Int32, reflect.Int16, reflect.Int8, reflect.Uint, reflect.Uint32, reflect.Uint16, reflect.Uint8)
}

func (r valueReflect) IsFloat() bool {
	return r.isKind(reflect.Float64, reflect.Float32)
}

func (r valueReflect) IsString() bool {
	kind := r.Value.Kind()
	if kind == reflect.String {
		return true
	}
	if kind == reflect.Slice && r.Value.Type().Elem().Kind() == reflect.Uint8 {
		return true
	}
	return false
}

func (r valueReflect) IsNull() bool {
	return safeIsNil(r.Value)
}

func (r valueReflect) isKind(kinds ...reflect.Kind) bool {
	kind := r.Value.Kind()
	for _, k := range kinds {
		if kind == k {
			return true
		}
	}
	return false
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
	val := r.Value
	switch val.Kind() {
	case reflect.Struct:
		return structReflect{r}
	case reflect.Map:
		return mapReflect{r}
	default:
		panic("value is not a map or struct")
	}
}

func (r *valueReflect) Recycle() {
	reflectPool.Put(r)
}

func (r valueReflect) AsList() List {
	if r.IsList() {
		return listReflect{r.Value}
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
	if r.isKind(reflect.Int, reflect.Int64, reflect.Int32, reflect.Int16, reflect.Int8) {
		return r.Value.Int()
	}
	if r.isKind(reflect.Uint, reflect.Uint32, reflect.Uint16, reflect.Uint8) {
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
	kind := r.Value.Kind()
	if kind == reflect.String {
		return r.Value.String()
	}
	if kind == reflect.Slice && r.Value.Type().Elem().Kind() == reflect.Uint8 {
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
		return mapReflect{r}.Unstructured()
	case r.IsList():
		return listReflect{Value: r.Value}.Unstructured()
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

// The below getMarshaler and toUnstructured functions are based on
// https://github.com/kubernetes/kubernetes/blob/40df9f82d0572a123f5ad13f48312978a2ff5877/staging/src/k8s.io/apimachinery/pkg/runtime/converter.go#L509
// and should somehow be consolidated with it

var marshalerType = reflect.TypeOf(new(json.Marshaler)).Elem()

func getMarshaler(v reflect.Value) (json.Marshaler, bool) {
	// Check value receivers if v is not a pointer and pointer receivers if v is a pointer
	if v.Type().Implements(marshalerType) {
		return v.Interface().(json.Marshaler), true
	}
	// Check pointer receivers if v is not a pointer
	if v.Kind() != reflect.Ptr && v.CanAddr() {
		v = v.Addr()
		if v.Type().Implements(marshalerType) {
			return v.Interface().(json.Marshaler), true
		}
	}
	return nil, false
}

var (
	nullBytes  = []byte("null")
	trueBytes  = []byte("true")
	falseBytes = []byte("false")
)

func toUnstructured(marshaler json.Marshaler, sv reflect.Value) (Value, error) {
	data, err := marshaler.MarshalJSON()
	if err != nil {
		return nil, err
	}
	switch {
	case len(data) == 0:
		return nil, fmt.Errorf("error decoding from json: empty value")

	case bytes.Equal(data, nullBytes):
		// We're done - we don't need to store anything.
		return NewValueInterface(nil), nil

	case bytes.Equal(data, trueBytes):
		return NewValueInterface(true), nil

	case bytes.Equal(data, falseBytes):
		return NewValueInterface(false), nil

	case data[0] == '"':
		var result string
		err := json.Unmarshal(data, &result)
		if err != nil {
			return nil, fmt.Errorf("error decoding string from json: %v", err)
		}
		return NewValueInterface(result), nil

	case data[0] == '{':
		result := make(map[string]interface{})
		err := json.Unmarshal(data, &result)
		if err != nil {
			return nil, fmt.Errorf("error decoding object from json: %v", err)
		}
		return NewValueInterface(result), nil

	case data[0] == '[':
		result := make([]interface{}, 0)
		err := json.Unmarshal(data, &result)
		if err != nil {
			return nil, fmt.Errorf("error decoding array from json: %v", err)
		}
		return NewValueInterface(result), nil

	default:
		var (
			resultInt   int64
			resultFloat float64
			err         error
		)
		if err = json.Unmarshal(data, &resultInt); err == nil {
			return NewValueInterface(resultInt), nil
		}
		if err = json.Unmarshal(data, &resultFloat); err == nil {
			return NewValueInterface(resultFloat), nil
		}
		return nil, fmt.Errorf("error decoding number from json: %v", err)
	}
}
