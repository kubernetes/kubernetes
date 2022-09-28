/*
Copyright 2022 The Kubernetes Authors.

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

package roundtrip

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func defaultFillFuncs() map[reflect.Type]FillFunc {
	funcs := map[reflect.Type]FillFunc{}
	funcs[reflect.TypeOf(&runtime.RawExtension{})] = func(s string, i int, obj interface{}) {
		// generate a raw object in normalized form
		// TODO: test non-normalized round-tripping... YAMLToJSON normalizes and makes exact comparisons fail
		obj.(*runtime.RawExtension).Raw = []byte(`{"apiVersion":"example.com/v1","kind":"CustomType","spec":{"replicas":1},"status":{"available":1}}`)
	}
	funcs[reflect.TypeOf(&metav1.TypeMeta{})] = func(s string, i int, obj interface{}) {
		// APIVersion and Kind are not serialized in all formats (notably protobuf), so clear by default for cross-format checking.
		obj.(*metav1.TypeMeta).APIVersion = ""
		obj.(*metav1.TypeMeta).Kind = ""
	}
	funcs[reflect.TypeOf(&metav1.FieldsV1{})] = func(s string, i int, obj interface{}) {
		obj.(*metav1.FieldsV1).Raw = []byte(`{}`)
	}
	funcs[reflect.TypeOf(&metav1.Time{})] = func(s string, i int, obj interface{}) {
		// use the integer as an offset from the year
		obj.(*metav1.Time).Time = time.Date(2000+i, 1, 1, 1, 1, 1, 0, time.UTC)
	}
	funcs[reflect.TypeOf(&metav1.MicroTime{})] = func(s string, i int, obj interface{}) {
		// use the integer as an offset from the year, and as a microsecond
		obj.(*metav1.MicroTime).Time = time.Date(2000+i, 1, 1, 1, 1, 1, i*int(time.Microsecond), time.UTC)
	}
	funcs[reflect.TypeOf(&intstr.IntOrString{})] = func(s string, i int, obj interface{}) {
		// use the string as a string value
		obj.(*intstr.IntOrString).Type = intstr.String
		obj.(*intstr.IntOrString).StrVal = s + "Value"
	}
	return funcs
}

// CompatibilityTestObject returns a deterministically filled object for the specified GVK
func CompatibilityTestObject(scheme *runtime.Scheme, gvk schema.GroupVersionKind, fillFuncs map[reflect.Type]FillFunc) (runtime.Object, error) {
	// Construct the object
	obj, err := scheme.New(gvk)
	if err != nil {
		return nil, err
	}

	fill("", 0, reflect.TypeOf(obj), reflect.ValueOf(obj), fillFuncs, map[reflect.Type]bool{})

	// Set the kind and apiVersion
	if typeAcc, err := apimeta.TypeAccessor(obj); err != nil {
		return nil, err
	} else {
		typeAcc.SetKind(gvk.Kind)
		typeAcc.SetAPIVersion(gvk.GroupVersion().String())
	}

	return obj, nil
}

func fill(dataString string, dataInt int, t reflect.Type, v reflect.Value, fillFuncs map[reflect.Type]FillFunc, filledTypes map[reflect.Type]bool) {
	if filledTypes[t] {
		// we already filled this type, avoid recursing infinitely
		return
	}
	filledTypes[t] = true
	defer delete(filledTypes, t)

	// if nil, populate pointers with a zero-value instance of the underlying type
	if t.Kind() == reflect.Pointer && v.IsNil() {
		if v.CanSet() {
			v.Set(reflect.New(t.Elem()))
		} else if v.IsNil() {
			panic(fmt.Errorf("unsettable nil pointer of type %v in field %s", t, dataString))
		}
	}

	if f, ok := fillFuncs[t]; ok {
		// use the custom fill function for this type
		f(dataString, dataInt, v.Interface())
		return
	}

	switch t.Kind() {
	case reflect.Slice:
		// populate with a single-item slice
		v.Set(reflect.MakeSlice(t, 1, 1))
		// recurse to populate the item, preserving the data context
		fill(dataString, dataInt, t.Elem(), v.Index(0), fillFuncs, filledTypes)

	case reflect.Map:
		// construct the key, which must be a string type, possibly converted to a type alias of string
		key := reflect.ValueOf(dataString + "Key").Convert(t.Key())
		// construct a zero-value item
		item := reflect.New(t.Elem())
		// recurse to populate the item, preserving the data context
		fill(dataString, dataInt, t.Elem(), item.Elem(), fillFuncs, filledTypes)
		// store in the map
		v.Set(reflect.MakeMap(t))
		v.SetMapIndex(key, item.Elem())

	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			field := t.Field(i)

			if !field.IsExported() {
				continue
			}

			// use the json field name, which must be stable
			dataString := strings.Split(field.Tag.Get("json"), ",")[0]
			if len(dataString) == 0 {
				// fall back to the struct field name if there is no json field name
				dataString = "<no json tag> " + field.Name
			}

			// use the protobuf tag, which must be stable
			dataInt := 0
			if protobufTagParts := strings.Split(field.Tag.Get("protobuf"), ","); len(protobufTagParts) > 1 {
				if tag, err := strconv.Atoi(protobufTagParts[1]); err != nil {
					panic(err)
				} else {
					dataInt = tag
				}
			}
			if dataInt == 0 {
				// fall back to the length of dataString as a backup
				dataInt = -len(dataString)
			}

			fieldType := field.Type
			fieldValue := v.Field(i)

			fill(dataString, dataInt, reflect.PointerTo(fieldType), fieldValue.Addr(), fillFuncs, filledTypes)
		}

	case reflect.Pointer:
		fill(dataString, dataInt, t.Elem(), v.Elem(), fillFuncs, filledTypes)

	case reflect.String:
		// use Convert to set into string alias types correctly
		v.Set(reflect.ValueOf(dataString + "Value").Convert(t))

	case reflect.Bool:
		// set to true to ensure we serialize omitempty fields
		v.Set(reflect.ValueOf(true).Convert(t))

	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		// use Convert to set into int alias types and different int widths correctly
		v.Set(reflect.ValueOf(dataInt).Convert(t))

	default:
		panic(fmt.Errorf("unhandled type %v in field %s", t, dataString))
	}
}
