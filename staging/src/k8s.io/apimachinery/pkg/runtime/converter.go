/*
Copyright 2017 The Kubernetes Authors.

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

package runtime

import (
	"bytes"
	encodingjson "encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/util/json"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"

	"k8s.io/klog"
)

// UnstructuredConverter is an interface for converting between interface{}
// and map[string]interface representation.
type UnstructuredConverter interface {
	ToUnstructured(obj interface{}) (map[string]interface{}, error)
	FromUnstructured(u map[string]interface{}, obj interface{}) error
}

type structField struct {
	structType reflect.Type
	field      int
}

type fieldInfo struct {
	name      string
	nameValue reflect.Value
	omitempty bool
}

type fieldsCacheMap map[structField]*fieldInfo

type fieldsCache struct {
	sync.Mutex
	value atomic.Value
}

func newFieldsCache() *fieldsCache {
	cache := &fieldsCache{}
	cache.value.Store(make(fieldsCacheMap))
	return cache
}

var (
	marshalerType          = reflect.TypeOf(new(encodingjson.Marshaler)).Elem()
	unmarshalerType        = reflect.TypeOf(new(encodingjson.Unmarshaler)).Elem()
	mapStringInterfaceType = reflect.TypeOf(map[string]interface{}{})
	stringType             = reflect.TypeOf(string(""))
	int64Type              = reflect.TypeOf(int64(0))
	float64Type            = reflect.TypeOf(float64(0))
	boolType               = reflect.TypeOf(bool(false))
	fieldCache             = newFieldsCache()

	// DefaultUnstructuredConverter performs unstructured to Go typed object conversions.
	DefaultUnstructuredConverter = &unstructuredConverter{
		mismatchDetection: parseBool(os.Getenv("KUBE_PATCH_CONVERSION_DETECTOR")),
		comparison: conversion.EqualitiesOrDie(
			func(a, b time.Time) bool {
				return a.UTC() == b.UTC()
			},
		),
	}
)

func parseBool(key string) bool {
	if len(key) == 0 {
		return false
	}
	value, err := strconv.ParseBool(key)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't parse '%s' as bool for unstructured mismatch detection", key))
	}
	return value
}

// unstructuredConverter knows how to convert between interface{} and
// Unstructured in both ways.
type unstructuredConverter struct {
	// If true, we will be additionally running conversion via json
	// to ensure that the result is true.
	// This is supposed to be set only in tests.
	mismatchDetection bool
	// comparison is the default test logic used to compare
	comparison conversion.Equalities
}

// NewTestUnstructuredConverter creates an UnstructuredConverter that accepts JSON typed maps and translates them
// to Go types via reflection. It performs mismatch detection automatically and is intended for use by external
// test tools. Use DefaultUnstructuredConverter if you do not explicitly need mismatch detection.
func NewTestUnstructuredConverter(comparison conversion.Equalities) UnstructuredConverter {
	return &unstructuredConverter{
		mismatchDetection: true,
		comparison:        comparison,
	}
}

// FromUnstructured converts an object from map[string]interface{} representation into a concrete type.
// It uses encoding/json/Unmarshaler if object implements it or reflection if not.
func (c *unstructuredConverter) FromUnstructured(u map[string]interface{}, obj interface{}) error {
	t := reflect.TypeOf(obj)
	value := reflect.ValueOf(obj)
	if t.Kind() != reflect.Ptr || value.IsNil() {
		return fmt.Errorf("FromUnstructured requires a non-nil pointer to an object, got %v", t)
	}
	err := fromUnstructured(reflect.ValueOf(u), value.Elem())
	if c.mismatchDetection {
		newObj := reflect.New(t.Elem()).Interface()
		newErr := fromUnstructuredViaJSON(u, newObj)
		if (err != nil) != (newErr != nil) {
			klog.Fatalf("FromUnstructured unexpected error for %v: error: %v", u, err)
		}
		if err == nil && !c.comparison.DeepEqual(obj, newObj) {
			klog.Fatalf("FromUnstructured mismatch\nobj1: %#v\nobj2: %#v", obj, newObj)
		}
	}
	return err
}

func fromUnstructuredViaJSON(u map[string]interface{}, obj interface{}) error {
	data, err := json.Marshal(u)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, obj)
}

func fromUnstructured(sv, dv reflect.Value) error {
	sv = unwrapInterface(sv)
	if !sv.IsValid() {
		dv.Set(reflect.Zero(dv.Type()))
		return nil
	}
	st, dt := sv.Type(), dv.Type()

	switch dt.Kind() {
	case reflect.Map, reflect.Slice, reflect.Ptr, reflect.Struct, reflect.Interface:
		// Those require non-trivial conversion.
	default:
		// This should handle all simple types.
		if st.AssignableTo(dt) {
			dv.Set(sv)
			return nil
		}
		// We cannot simply use "ConvertibleTo", as JSON doesn't support conversions
		// between those four groups: bools, integers, floats and string. We need to
		// do the same.
		if st.ConvertibleTo(dt) {
			switch st.Kind() {
			case reflect.String:
				switch dt.Kind() {
				case reflect.String:
					dv.Set(sv.Convert(dt))
					return nil
				}
			case reflect.Bool:
				switch dt.Kind() {
				case reflect.Bool:
					dv.Set(sv.Convert(dt))
					return nil
				}
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
				reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
				switch dt.Kind() {
				case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
					reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
					dv.Set(sv.Convert(dt))
					return nil
				}
			case reflect.Float32, reflect.Float64:
				switch dt.Kind() {
				case reflect.Float32, reflect.Float64:
					dv.Set(sv.Convert(dt))
					return nil
				}
				if sv.Float() == math.Trunc(sv.Float()) {
					dv.Set(sv.Convert(dt))
					return nil
				}
			}
			return fmt.Errorf("cannot convert %s to %s", st.String(), dt.String())
		}
	}

	// Check if the object has a custom JSON marshaller/unmarshaller.
	if reflect.PtrTo(dt).Implements(unmarshalerType) {
		data, err := json.Marshal(sv.Interface())
		if err != nil {
			return fmt.Errorf("error encoding %s to json: %v", st.String(), err)
		}
		unmarshaler := dv.Addr().Interface().(encodingjson.Unmarshaler)
		return unmarshaler.UnmarshalJSON(data)
	}

	switch dt.Kind() {
	case reflect.Map:
		return mapFromUnstructured(sv, dv)
	case reflect.Slice:
		return sliceFromUnstructured(sv, dv)
	case reflect.Ptr:
		return pointerFromUnstructured(sv, dv)
	case reflect.Struct:
		return structFromUnstructured(sv, dv)
	case reflect.Interface:
		return interfaceFromUnstructured(sv, dv)
	default:
		return fmt.Errorf("unrecognized type: %v", dt.Kind())
	}
}

func fieldInfoFromField(structType reflect.Type, field int) *fieldInfo {
	fieldCacheMap := fieldCache.value.Load().(fieldsCacheMap)
	if info, ok := fieldCacheMap[structField{structType, field}]; ok {
		return info
	}

	// Cache miss - we need to compute the field name.
	info := &fieldInfo{}
	typeField := structType.Field(field)
	jsonTag := typeField.Tag.Get("json")
	if len(jsonTag) == 0 {
		// Make the first character lowercase.
		if typeField.Name == "" {
			info.name = typeField.Name
		} else {
			info.name = strings.ToLower(typeField.Name[:1]) + typeField.Name[1:]
		}
	} else {
		items := strings.Split(jsonTag, ",")
		info.name = items[0]
		for i := range items {
			if items[i] == "omitempty" {
				info.omitempty = true
			}
		}
	}
	info.nameValue = reflect.ValueOf(info.name)

	fieldCache.Lock()
	defer fieldCache.Unlock()
	fieldCacheMap = fieldCache.value.Load().(fieldsCacheMap)
	newFieldCacheMap := make(fieldsCacheMap)
	for k, v := range fieldCacheMap {
		newFieldCacheMap[k] = v
	}
	newFieldCacheMap[structField{structType, field}] = info
	fieldCache.value.Store(newFieldCacheMap)
	return info
}

func unwrapInterface(v reflect.Value) reflect.Value {
	for v.Kind() == reflect.Interface {
		v = v.Elem()
	}
	return v
}

func mapFromUnstructured(sv, dv reflect.Value) error {
	st, dt := sv.Type(), dv.Type()
	if st.Kind() != reflect.Map {
		return fmt.Errorf("cannot restore map from %v", st.Kind())
	}

	if !st.Key().AssignableTo(dt.Key()) && !st.Key().ConvertibleTo(dt.Key()) {
		return fmt.Errorf("cannot copy map with non-assignable keys: %v %v", st.Key(), dt.Key())
	}

	if sv.IsNil() {
		dv.Set(reflect.Zero(dt))
		return nil
	}
	dv.Set(reflect.MakeMap(dt))
	for _, key := range sv.MapKeys() {
		value := reflect.New(dt.Elem()).Elem()
		if val := unwrapInterface(sv.MapIndex(key)); val.IsValid() {
			if err := fromUnstructured(val, value); err != nil {
				return err
			}
		} else {
			value.Set(reflect.Zero(dt.Elem()))
		}
		if st.Key().AssignableTo(dt.Key()) {
			dv.SetMapIndex(key, value)
		} else {
			dv.SetMapIndex(key.Convert(dt.Key()), value)
		}
	}
	return nil
}

func sliceFromUnstructured(sv, dv reflect.Value) error {
	st, dt := sv.Type(), dv.Type()
	if st.Kind() == reflect.String && dt.Elem().Kind() == reflect.Uint8 {
		// We store original []byte representation as string.
		// This conversion is allowed, but we need to be careful about
		// marshaling data appropriately.
		if len(sv.Interface().(string)) > 0 {
			marshalled, err := json.Marshal(sv.Interface())
			if err != nil {
				return fmt.Errorf("error encoding %s to json: %v", st, err)
			}
			// TODO: Is this Unmarshal needed?
			var data []byte
			err = json.Unmarshal(marshalled, &data)
			if err != nil {
				return fmt.Errorf("error decoding from json: %v", err)
			}
			dv.SetBytes(data)
		} else {
			dv.Set(reflect.Zero(dt))
		}
		return nil
	}
	if st.Kind() != reflect.Slice {
		return fmt.Errorf("cannot restore slice from %v", st.Kind())
	}

	if sv.IsNil() {
		dv.Set(reflect.Zero(dt))
		return nil
	}
	dv.Set(reflect.MakeSlice(dt, sv.Len(), sv.Cap()))
	for i := 0; i < sv.Len(); i++ {
		if err := fromUnstructured(sv.Index(i), dv.Index(i)); err != nil {
			return err
		}
	}
	return nil
}

func pointerFromUnstructured(sv, dv reflect.Value) error {
	st, dt := sv.Type(), dv.Type()

	if st.Kind() == reflect.Ptr && sv.IsNil() {
		dv.Set(reflect.Zero(dt))
		return nil
	}
	dv.Set(reflect.New(dt.Elem()))
	switch st.Kind() {
	case reflect.Ptr, reflect.Interface:
		return fromUnstructured(sv.Elem(), dv.Elem())
	default:
		return fromUnstructured(sv, dv.Elem())
	}
}

func structFromUnstructured(sv, dv reflect.Value) error {
	st, dt := sv.Type(), dv.Type()
	if st.Kind() != reflect.Map {
		return fmt.Errorf("cannot restore struct from: %v", st.Kind())
	}

	for i := 0; i < dt.NumField(); i++ {
		fieldInfo := fieldInfoFromField(dt, i)
		fv := dv.Field(i)

		if len(fieldInfo.name) == 0 {
			// This field is inlined.
			if err := fromUnstructured(sv, fv); err != nil {
				return err
			}
		} else {
			value := unwrapInterface(sv.MapIndex(fieldInfo.nameValue))
			if value.IsValid() {
				if err := fromUnstructured(value, fv); err != nil {
					return err
				}
			} else {
				fv.Set(reflect.Zero(fv.Type()))
			}
		}
	}
	return nil
}

func interfaceFromUnstructured(sv, dv reflect.Value) error {
	// TODO: Is this conversion safe?
	dv.Set(sv)
	return nil
}

// ToUnstructured converts an object into map[string]interface{} representation.
// It uses encoding/json/Marshaler if object implements it or reflection if not.
func (c *unstructuredConverter) ToUnstructured(obj interface{}) (map[string]interface{}, error) {
	var u map[string]interface{}
	var err error
	if unstr, ok := obj.(Unstructured); ok {
		u = unstr.UnstructuredContent()
	} else {
		t := reflect.TypeOf(obj)
		value := reflect.ValueOf(obj)
		if t.Kind() != reflect.Ptr || value.IsNil() {
			return nil, fmt.Errorf("ToUnstructured requires a non-nil pointer to an object, got %v", t)
		}
		u = map[string]interface{}{}
		err = toUnstructured(value.Elem(), reflect.ValueOf(&u).Elem())
	}
	if c.mismatchDetection {
		newUnstr := map[string]interface{}{}
		newErr := toUnstructuredViaJSON(obj, &newUnstr)
		if (err != nil) != (newErr != nil) {
			klog.Fatalf("ToUnstructured unexpected error for %v: error: %v; newErr: %v", obj, err, newErr)
		}
		if err == nil && !c.comparison.DeepEqual(u, newUnstr) {
			klog.Fatalf("ToUnstructured mismatch\nobj1: %#v\nobj2: %#v", u, newUnstr)
		}
	}
	if err != nil {
		return nil, err
	}
	return u, nil
}

// DeepCopyJSON deep copies the passed value, assuming it is a valid JSON representation i.e. only contains
// types produced by json.Unmarshal() and also int64.
// bool, int64, float64, string, []interface{}, map[string]interface{}, json.Number and nil
func DeepCopyJSON(x map[string]interface{}) map[string]interface{} {
	return DeepCopyJSONValue(x).(map[string]interface{})
}

// DeepCopyJSONValue deep copies the passed value, assuming it is a valid JSON representation i.e. only contains
// types produced by json.Unmarshal() and also int64.
// bool, int64, float64, string, []interface{}, map[string]interface{}, json.Number and nil
func DeepCopyJSONValue(x interface{}) interface{} {
	switch x := x.(type) {
	case map[string]interface{}:
		if x == nil {
			// Typed nil - an interface{} that contains a type map[string]interface{} with a value of nil
			return x
		}
		clone := make(map[string]interface{}, len(x))
		for k, v := range x {
			clone[k] = DeepCopyJSONValue(v)
		}
		return clone
	case []interface{}:
		if x == nil {
			// Typed nil - an interface{} that contains a type []interface{} with a value of nil
			return x
		}
		clone := make([]interface{}, len(x))
		for i, v := range x {
			clone[i] = DeepCopyJSONValue(v)
		}
		return clone
	case string, int64, bool, float64, nil, encodingjson.Number:
		return x
	default:
		panic(fmt.Errorf("cannot deep copy %T", x))
	}
}

func toUnstructuredViaJSON(obj interface{}, u *map[string]interface{}) error {
	data, err := json.Marshal(obj)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, u)
}

var (
	nullBytes  = []byte("null")
	trueBytes  = []byte("true")
	falseBytes = []byte("false")
)

func getMarshaler(v reflect.Value) (encodingjson.Marshaler, bool) {
	// Check value receivers if v is not a pointer and pointer receivers if v is a pointer
	if v.Type().Implements(marshalerType) {
		return v.Interface().(encodingjson.Marshaler), true
	}
	// Check pointer receivers if v is not a pointer
	if v.Kind() != reflect.Ptr && v.CanAddr() {
		v = v.Addr()
		if v.Type().Implements(marshalerType) {
			return v.Interface().(encodingjson.Marshaler), true
		}
	}
	return nil, false
}

func toUnstructured(sv, dv reflect.Value) error {
	// Check if the object has a custom JSON marshaller/unmarshaller.
	if marshaler, ok := getMarshaler(sv); ok {
		if sv.Kind() == reflect.Ptr && sv.IsNil() {
			// We're done - we don't need to store anything.
			return nil
		}

		data, err := marshaler.MarshalJSON()
		if err != nil {
			return err
		}
		switch {
		case len(data) == 0:
			return fmt.Errorf("error decoding from json: empty value")

		case bytes.Equal(data, nullBytes):
			// We're done - we don't need to store anything.

		case bytes.Equal(data, trueBytes):
			dv.Set(reflect.ValueOf(true))

		case bytes.Equal(data, falseBytes):
			dv.Set(reflect.ValueOf(false))

		case data[0] == '"':
			var result string
			err := json.Unmarshal(data, &result)
			if err != nil {
				return fmt.Errorf("error decoding string from json: %v", err)
			}
			dv.Set(reflect.ValueOf(result))

		case data[0] == '{':
			result := make(map[string]interface{})
			err := json.Unmarshal(data, &result)
			if err != nil {
				return fmt.Errorf("error decoding object from json: %v", err)
			}
			dv.Set(reflect.ValueOf(result))

		case data[0] == '[':
			result := make([]interface{}, 0)
			err := json.Unmarshal(data, &result)
			if err != nil {
				return fmt.Errorf("error decoding array from json: %v", err)
			}
			dv.Set(reflect.ValueOf(result))

		default:
			var (
				resultInt   int64
				resultFloat float64
				err         error
			)
			if err = json.Unmarshal(data, &resultInt); err == nil {
				dv.Set(reflect.ValueOf(resultInt))
			} else if err = json.Unmarshal(data, &resultFloat); err == nil {
				dv.Set(reflect.ValueOf(resultFloat))
			} else {
				return fmt.Errorf("error decoding number from json: %v", err)
			}
		}

		return nil
	}

	st, dt := sv.Type(), dv.Type()
	switch st.Kind() {
	case reflect.String:
		if dt.Kind() == reflect.Interface && dv.NumMethod() == 0 {
			dv.Set(reflect.New(stringType))
		}
		dv.Set(reflect.ValueOf(sv.String()))
		return nil
	case reflect.Bool:
		if dt.Kind() == reflect.Interface && dv.NumMethod() == 0 {
			dv.Set(reflect.New(boolType))
		}
		dv.Set(reflect.ValueOf(sv.Bool()))
		return nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if dt.Kind() == reflect.Interface && dv.NumMethod() == 0 {
			dv.Set(reflect.New(int64Type))
		}
		dv.Set(reflect.ValueOf(sv.Int()))
		return nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		uVal := sv.Uint()
		if uVal > math.MaxInt64 {
			return fmt.Errorf("unsigned value %d does not fit into int64 (overflow)", uVal)
		}
		if dt.Kind() == reflect.Interface && dv.NumMethod() == 0 {
			dv.Set(reflect.New(int64Type))
		}
		dv.Set(reflect.ValueOf(int64(uVal)))
		return nil
	case reflect.Float32, reflect.Float64:
		if dt.Kind() == reflect.Interface && dv.NumMethod() == 0 {
			dv.Set(reflect.New(float64Type))
		}
		dv.Set(reflect.ValueOf(sv.Float()))
		return nil
	case reflect.Map:
		return mapToUnstructured(sv, dv)
	case reflect.Slice:
		return sliceToUnstructured(sv, dv)
	case reflect.Ptr:
		return pointerToUnstructured(sv, dv)
	case reflect.Struct:
		return structToUnstructured(sv, dv)
	case reflect.Interface:
		return interfaceToUnstructured(sv, dv)
	default:
		return fmt.Errorf("unrecognized type: %v", st.Kind())
	}
}

func mapToUnstructured(sv, dv reflect.Value) error {
	st, dt := sv.Type(), dv.Type()
	if sv.IsNil() {
		dv.Set(reflect.Zero(dt))
		return nil
	}
	if dt.Kind() == reflect.Interface && dv.NumMethod() == 0 {
		if st.Key().Kind() == reflect.String {
			switch st.Elem().Kind() {
			// TODO It should be possible to reuse the slice for primitive types.
			// However, it is panicing in the following form.
			// case reflect.String, reflect.Bool,
			// 	reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			// 	reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			// 	sv.Set(sv)
			// 	return nil
			default:
				// We need to do a proper conversion.
			}
		}
		dv.Set(reflect.MakeMap(mapStringInterfaceType))
		dv = dv.Elem()
		dt = dv.Type()
	}
	if dt.Kind() != reflect.Map {
		return fmt.Errorf("cannot convert struct to: %v", dt.Kind())
	}

	if !st.Key().AssignableTo(dt.Key()) && !st.Key().ConvertibleTo(dt.Key()) {
		return fmt.Errorf("cannot copy map with non-assignable keys: %v %v", st.Key(), dt.Key())
	}

	for _, key := range sv.MapKeys() {
		value := reflect.New(dt.Elem()).Elem()
		if err := toUnstructured(sv.MapIndex(key), value); err != nil {
			return err
		}
		if st.Key().AssignableTo(dt.Key()) {
			dv.SetMapIndex(key, value)
		} else {
			dv.SetMapIndex(key.Convert(dt.Key()), value)
		}
	}
	return nil
}

func sliceToUnstructured(sv, dv reflect.Value) error {
	st, dt := sv.Type(), dv.Type()
	if sv.IsNil() {
		dv.Set(reflect.Zero(dt))
		return nil
	}
	if st.Elem().Kind() == reflect.Uint8 {
		dv.Set(reflect.New(stringType))
		data, err := json.Marshal(sv.Bytes())
		if err != nil {
			return err
		}
		var result string
		if err = json.Unmarshal(data, &result); err != nil {
			return err
		}
		dv.Set(reflect.ValueOf(result))
		return nil
	}
	if dt.Kind() == reflect.Interface && dv.NumMethod() == 0 {
		switch st.Elem().Kind() {
		// TODO It should be possible to reuse the slice for primitive types.
		// However, it is panicing in the following form.
		// case reflect.String, reflect.Bool,
		// 	reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		// 	reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		// 	sv.Set(sv)
		// 	return nil
		default:
			// We need to do a proper conversion.
			dv.Set(reflect.MakeSlice(reflect.SliceOf(dt), sv.Len(), sv.Cap()))
			dv = dv.Elem()
			dt = dv.Type()
		}
	}
	if dt.Kind() != reflect.Slice {
		return fmt.Errorf("cannot convert slice to: %v", dt.Kind())
	}
	for i := 0; i < sv.Len(); i++ {
		if err := toUnstructured(sv.Index(i), dv.Index(i)); err != nil {
			return err
		}
	}
	return nil
}

func pointerToUnstructured(sv, dv reflect.Value) error {
	if sv.IsNil() {
		// We're done - we don't need to store anything.
		return nil
	}
	return toUnstructured(sv.Elem(), dv)
}

func isZero(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Map, reflect.Slice:
		// TODO: It seems that 0-len maps are ignored in it.
		return v.IsNil() || v.Len() == 0
	case reflect.Ptr, reflect.Interface:
		return v.IsNil()
	}
	return false
}

func structToUnstructured(sv, dv reflect.Value) error {
	st, dt := sv.Type(), dv.Type()
	if dt.Kind() == reflect.Interface && dv.NumMethod() == 0 {
		dv.Set(reflect.MakeMap(mapStringInterfaceType))
		dv = dv.Elem()
		dt = dv.Type()
	}
	if dt.Kind() != reflect.Map {
		return fmt.Errorf("cannot convert struct to: %v", dt.Kind())
	}
	realMap := dv.Interface().(map[string]interface{})

	for i := 0; i < st.NumField(); i++ {
		fieldInfo := fieldInfoFromField(st, i)
		fv := sv.Field(i)

		if fieldInfo.name == "-" {
			// This field should be skipped.
			continue
		}
		if fieldInfo.omitempty && isZero(fv) {
			// omitempty fields should be ignored.
			continue
		}
		if len(fieldInfo.name) == 0 {
			// This field is inlined.
			if err := toUnstructured(fv, dv); err != nil {
				return err
			}
			continue
		}
		switch fv.Type().Kind() {
		case reflect.String:
			realMap[fieldInfo.name] = fv.String()
		case reflect.Bool:
			realMap[fieldInfo.name] = fv.Bool()
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			realMap[fieldInfo.name] = fv.Int()
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			realMap[fieldInfo.name] = fv.Uint()
		case reflect.Float32, reflect.Float64:
			realMap[fieldInfo.name] = fv.Float()
		default:
			subv := reflect.New(dt.Elem()).Elem()
			if err := toUnstructured(fv, subv); err != nil {
				return err
			}
			dv.SetMapIndex(fieldInfo.nameValue, subv)
		}
	}
	return nil
}

func interfaceToUnstructured(sv, dv reflect.Value) error {
	if !sv.IsValid() || sv.IsNil() {
		dv.Set(reflect.Zero(dv.Type()))
		return nil
	}
	return toUnstructured(sv.Elem(), dv)
}
