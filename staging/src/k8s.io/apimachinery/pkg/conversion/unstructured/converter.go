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

package unstructured

import (
	"bytes"
	encodingjson "encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/json"
)

type structField struct {
	structType reflect.Type
	field      int
}

type fieldInfo struct {
	name      string
	nameValue reflect.Value
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
	fieldCache             = newFieldsCache()
)

// Converter knows how to convert betweek runtime.Object and
// Unstructured in both ways.
type Converter struct {
}

func NewConverter() *Converter {
	return &Converter{}
}

func (c *Converter) FromUnstructured(u map[string]interface{}, obj runtime.Object) error {
	return fromUnstructured(reflect.ValueOf(u), reflect.ValueOf(obj).Elem())
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
			}
			return fmt.Errorf("cannot convert %s to %d", st.String(), dt.String())
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
		info.name = strings.Split(jsonTag, ",")[0]
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

func (c *Converter) ToUnstructured(obj runtime.Object, u *map[string]interface{}) error {
	return toUnstructured(reflect.ValueOf(obj).Elem(), reflect.ValueOf(u).Elem())
}

func toUnstructured(sv, dv reflect.Value) error {
	st, dt := sv.Type(), dv.Type()

	// Check if the object has a custom JSON marshaller/unmarshaller.
	if st.Implements(marshalerType) {
		if sv.Kind() == reflect.Ptr && sv.IsNil() {
			// We're done - we don't need to store anything.
			return nil
		}

		marshaler := sv.Interface().(encodingjson.Marshaler)
		data, err := marshaler.MarshalJSON()
		if err != nil {
			return err
		}
		if bytes.Equal(data, []byte("null")) {
			// We're done - we don't need to store anything.
		} else {
			switch {
			case len(data) > 0 && data[0] == '"':
				var result string
				err := json.Unmarshal(data, &result)
				if err != nil {
					return fmt.Errorf("error decoding from json: %v", err)
				}
				dv.Set(reflect.ValueOf(result))
			case len(data) > 0 && data[0] == '{':
				result := make(map[string]interface{})
				err := json.Unmarshal(data, &result)
				if err != nil {
					return fmt.Errorf("error decoding from json: %v", err)
				}
				dv.Set(reflect.ValueOf(result))
			default:
				var result int64
				err := json.Unmarshal(data, &result)
				if err != nil {
					return fmt.Errorf("error decoding from json: %v", err)
				}
				dv.Set(reflect.ValueOf(result))
			}
		}
		return nil
	}

	switch st.Kind() {
	case reflect.String, reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		if dt.Kind() == reflect.Interface && dv.NumMethod() == 0 {
			dv.Set(reflect.New(st))
		}
		dv.Set(sv)
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

		if len(fieldInfo.name) == 0 {
			// This field is inlined.
			if err := toUnstructured(fv, dv); err != nil {
				return err
			}
			continue
		}
		if !fv.IsValid() {
			// No fource field, skip.
			continue
		}
		switch fv.Type().Kind() {
		case reflect.String, reflect.Bool,
			reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			realMap[fieldInfo.name] = fv.Interface()
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
