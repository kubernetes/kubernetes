/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"strings"

	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/json"

	"github.com/golang/glog"
)

// Converter knows how to convert betweek runtime.Object and
// Unstructured in both ways.
type Converter struct {
}

func NewConverter() *Converter {
	return &Converter{}
}

func (c *Converter) FromUnstructured(u map[string]interface{}, obj runtime.Object) error {
	return c.fromUnstructured(reflect.ValueOf(u), reflect.ValueOf(obj).Elem())
}

func (c *Converter) fromUnstructured(sv, dv reflect.Value) error {
	sv = unwrapInterface(sv)
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
		if st.ConvertibleTo(dt) {
			dv.Set(sv.Convert(dt))
			return nil
		}
	}

	// If the from value is string, check if there is custom marshaller
	// for the out type and try to use it.
	//if st.Kind() == reflect.String && dv.CanAddr() {
	if dv.CanAddr() {
		// Check if the object has a custom JSON marshaller/unmarshaller.
		unmarshal := dv.Addr().MethodByName("UnmarshalJSON")
		if unmarshal.IsValid() {
			// UnmarshalJSON takes []byte as an argument. However, it assumes
			// that this is json encoded, whereas here it isn't. We need to
			// encode it first.
			data, err := json.Marshal(sv.Interface())
			if err != nil {
				return fmt.Errorf("Error encoding to json")
			}
			ret := unmarshal.Call([]reflect.Value{reflect.ValueOf(data)})[0].Interface()
			if ret != nil {
				return ret.(error)
			}
			return nil
		}
	}

	switch dt.Kind() {
	case reflect.Map:
		return c.mapFromUnstructured(sv, dv)
	case reflect.Slice:
		return c.sliceFromUnstructured(sv, dv)
	case reflect.Ptr:
		return c.pointerFromUnstructured(sv, dv)
	case reflect.Struct:
		return c.structFromUnstructured(sv, dv)
	case reflect.Interface:
		return c.interfaceFromUnstructured(sv, dv)
	default:
		return fmt.Errorf("Unrecognized type: %v", dt.Kind())
	}
}

func fieldNameFromField(field *reflect.StructField) string {
	jsonTag := field.Tag.Get("json")
	if len(jsonTag) == 0 {
		// FIXME: This should start with small letter.
		return field.Name
	}
	return strings.Split(jsonTag, ",")[0]
}

func unwrapInterface(v reflect.Value) reflect.Value {
	for v.Kind() == reflect.Interface {
		v = v.Elem()
	}
	return v
}

func (c *Converter) mapFromUnstructured(sv, dv reflect.Value) error {
	st, dt := sv.Type(), dv.Type()
	if st.Kind() != reflect.Map {
		return fmt.Errorf("Cannot restore map from %v", st.Kind())
	}

	if !st.Key().AssignableTo(dt.Key()) && !st.Key().ConvertibleTo(dt.Key()) {
		return fmt.Errorf("Cannot copy map with non-assignable keys: %v %v", st.Key(), dt.Key())
	}

	if sv.IsNil() {
		dv.Set(reflect.Zero(dt))
		return nil
	}
	dv.Set(reflect.MakeMap(dt))
	for _, key := range sv.MapKeys() {
		value := reflect.New(dt.Elem()).Elem()
		if val := unwrapInterface(sv.MapIndex(key)); val.IsValid() {
			if err := c.fromUnstructured(val, value); err != nil {
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

func (c *Converter) sliceFromUnstructured(sv, dv reflect.Value) error {
	st, dt := sv.Type(), dv.Type()
	if st.Kind() == reflect.String && dt.Elem().Kind() == reflect.Uint8 {
		// We store original []byte representation as string.
		// This conversion is allowed, but we need to be careful about
		// marshaling data appropriately.
		if len(sv.Interface().(string)) > 0 {
			marshalled, err := json.Marshal(sv.Interface())
			if err != nil {
				return fmt.Errorf("Error encoding to json: %v", err)
			}
			var data []byte
			err = json.Unmarshal(marshalled, &data)
			if err != nil {
				return fmt.Errorf("Error decoding from json: %v", err)
			}
			dv.SetBytes(data)
		} else {
			dv.Set(reflect.Zero(dt))
		}
		return nil
	}
	if st.Kind() != reflect.Slice {
		return fmt.Errorf("Cannot restore slice from %v", st.Kind())
	}

	if sv.IsNil() {
		dv.Set(reflect.Zero(dt))
		return nil
	}
	dv.Set(reflect.MakeSlice(dt, sv.Len(), sv.Cap()))
	for i := 0; i < sv.Len(); i++ {
		if err := c.fromUnstructured(sv.Index(i), dv.Index(i)); err != nil {
			return err
		}
	}
	return nil
}

func (c *Converter) pointerFromUnstructured(sv, dv reflect.Value) error {
	st, dt := sv.Type(), dv.Type()

	if st.Kind() == reflect.Ptr && sv.IsNil() {
		dv.Set(reflect.Zero(dt))
		return nil
	}
	dv.Set(reflect.New(dt.Elem()))
	switch st.Kind() {
	case reflect.Ptr, reflect.Interface:
		return c.fromUnstructured(sv.Elem(), dv.Elem())
	default:
		return c.fromUnstructured(sv, dv.Elem())
	}
}

func (c *Converter) structFromUnstructured(sv, dv reflect.Value) error {
	st, dt := sv.Type(), dv.Type()
	if st.Kind() != reflect.Map {
		return fmt.Errorf("Cannot restore struct from: %v", st.Kind())
	}

	for i := 0; i < dt.NumField(); i++ {
		field := dt.Field(i)
		glog.Errorf("--> %s", field.Name)
		fieldName := fieldNameFromField(&field)
		fv := dv.Field(i)

		if len(fieldName) == 0 {
			// This field is inlined.
			if err := c.fromUnstructured(sv, fv); err != nil {
				return err
			}
		} else {
			value := unwrapInterface(sv.MapIndex(reflect.ValueOf(fieldName)))
			if value.IsValid() {
				if err := c.fromUnstructured(value, fv); err != nil {
					return err
				}
			} else {
				fv.Set(reflect.Zero(fv.Type()))
			}
		}
	}
	return nil
}

func (c *Converter) interfaceFromUnstructured(sv, dv reflect.Value) error {
	return fmt.Errorf("Interface conversion unsupported")
}
