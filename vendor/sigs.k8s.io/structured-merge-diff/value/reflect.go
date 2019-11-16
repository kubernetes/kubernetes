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
	"fmt"
	"reflect"
)

type ReflectValue struct {
	Value interface{}
}

func (r ReflectValue) IsMap() bool {
	return isKind(r.Value, reflect.Map, reflect.Struct)
}

func (r ReflectValue) IsList() bool {
	return isKind(r.Value, reflect.Slice, reflect.Array)
}
func (r ReflectValue) IsBool() bool {
	return isKind(r.Value, reflect.Bool)
}
func (r ReflectValue) IsInt() bool {
	// This feels wrong. Very wrong.
	return isKind(r.Value, reflect.Int, reflect.Int64, reflect.Int32, reflect.Int16, reflect.Int8, reflect.Uint64, reflect.Uint, reflect.Uint32, reflect.Uint16, reflect.Uint8)
}
func (r ReflectValue) IsFloat() bool {
	return isKind(r.Value, reflect.Float64, reflect.Float32)
}
func (r ReflectValue) IsString() bool {
	return isKind(r.Value, reflect.String)
}
func (r ReflectValue) IsNull() bool {
	return reflect.ValueOf(r.Value).IsNil()
}
func (r ReflectValue) Map() Map {
	if r.IsMap() {
		return ReflectMap{r.Value}
	}
	panic("illegal cast")
}

func (r ReflectValue) List() List {
	if r.IsList() {
		return ReflectList{r.Value}
	}
	panic("illegal cast")
}
func (r ReflectValue) Bool() bool {
	if r.IsBool() {
		return deref(r.Value).Bool()
	}
	panic("illegal cast")
}
func (r ReflectValue) Int() int64 {
	if r.IsInt() {
		return deref(r.Value).Int()
	}
	panic("illegal cast")
}
func (r ReflectValue) Float() float64 {
	if r.IsFloat() {
		return deref(r.Value).Float()
	}
	panic("illegal cast")
}
func (r ReflectValue) String() string {
	if r.IsString() {
		return deref(r.Value).String()
	}
	panic("illegal cast")
}
func (r ReflectValue) Interface() interface{} {
	return r.Value
}

type ReflectMap struct {
	Value interface{}
}

func (r ReflectMap) Length() int {
	rval := deref(r.Value)
	switch rval.Type().Kind() {
	case reflect.Struct:
		return rval.NumField()
	case reflect.Map:
		return rval.Len()
	default:
		panic(fmt.Sprintf("unsupported kind: %v", reflect.TypeOf(r.Value).Kind()))
	}
}

func (r ReflectMap) Get(key string) (Value, bool) {
	var val reflect.Value
	rval := deref(r.Value)
	switch rval.Type().Kind() {
	case reflect.Struct:
		val = rval.FieldByName(key)
	case reflect.Map:
		val = rval.MapIndex(reflect.ValueOf(key))
	default:
		panic(fmt.Sprintf("unsupported kind: %v", reflect.TypeOf(r.Value).Kind()))
	}
	zero := reflect.Value{}
	return ReflectValue{val.Interface()}, val != zero
}

func (r ReflectMap) Set(key string, val Value) {
	rval := deref(r.Value)
	switch rval.Type().Kind() {
	case reflect.Struct:
		rval.FieldByName(key).Set(rval)
	case reflect.Map:
		rval.SetMapIndex(reflect.ValueOf(key), rval)
	default:
		panic(fmt.Sprintf("unsupported kind: %v", reflect.TypeOf(r.Value).Kind()))
	}
}

func (r ReflectMap) Delete(key string) {
	rval := deref(r.Value)
	switch rval.Type().Kind() {
	case reflect.Struct:
		rval.FieldByName(key).Set(reflect.Value{})
	case reflect.Map:
		rval.SetMapIndex(reflect.ValueOf(key), reflect.Value{})
	default:
		panic(fmt.Sprintf("unsupported kind: %v", reflect.TypeOf(r.Value).Kind()))
	}
}

func (r ReflectMap) Iterate(fn func(string, Value) bool) {
	rval := deref(r.Value)
	switch rval.Type().Kind() {
	case reflect.Struct:
		for i := 0; i < rval.NumField(); i++ {
			fn(rval.Type().Field(i).Name, ReflectValue{rval.Field(i).Interface()})
		}
	case reflect.Map:
		iter := rval.MapRange()
		for iter.Next() {
			if !fn(iter.Key().String(), ReflectValue{iter.Value().Interface()}) {
				return
			}
		}
	default:
		panic(fmt.Sprintf("unsupported kind: %v", reflect.TypeOf(r.Value).Kind()))
	}
}

type ReflectList struct {
	Value interface{}
}

func (r ReflectList) Interface() []interface{} {
	return r.Value.([]interface{}) // TODO: This function should not be part of the interface
}

func (r ReflectList) Length() int {
	rval := deref(r.Value)
	return rval.Len()
}

func (r ReflectList) Iterate(fn func(int, Value)) {
	rval := deref(r.Value)
	length := rval.Len()
	for i := 0; i < length; i++ {
		fn(i, ReflectValue{rval.Index(i).Interface()})
	}
}

func (r ReflectList) At(i int) Value {
	rval := deref(r.Value)
	return ReflectValue{rval.Index(i).Interface()}
}

func isKind(val interface{}, kinds ...reflect.Kind) bool {
	return reflectIsKind(reflect.ValueOf(val), kinds...)
}

func reflectIsKind(rval reflect.Value, kinds ...reflect.Kind) bool {
	kind := rval.Kind()
	if kind == reflect.Ptr || kind == reflect.Interface {
		return reflectIsKind(rval.Elem(), kinds...)
	}
	for _, k := range kinds {
		if kind == k {
			return true
		}
	}
	return false
}

func deref(val interface{}) reflect.Value {
	rval := reflect.ValueOf(val)
	kind := rval.Type().Kind()
	if kind == reflect.Interface || kind == reflect.Ptr {
		return rval.Elem()
	}
	return rval
}