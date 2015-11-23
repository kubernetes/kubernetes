/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"reflect"

	"k8s.io/kubernetes/pkg/conversion"
)

// fieldPtr puts the address of fieldName, which must be a member of v,
// into dest, which must be an address of a variable to which this field's
// address can be assigned.
func FieldPtr(v reflect.Value, fieldName string, dest interface{}) error {
	field := v.FieldByName(fieldName)
	if !field.IsValid() {
		return fmt.Errorf("couldn't find %v field in %#v", fieldName, v.Interface())
	}
	v, err := conversion.EnforcePtr(dest)
	if err != nil {
		return err
	}
	field = field.Addr()
	if field.Type().AssignableTo(v.Type()) {
		v.Set(field)
		return nil
	}
	if field.Type().ConvertibleTo(v.Type()) {
		v.Set(field.Convert(v.Type()))
		return nil
	}
	return fmt.Errorf("couldn't assign/convert %v to %v", field.Type(), v.Type())
}

// DecodeList alters the list in place, attempting to decode any objects found in
// the list that have the runtime.Unknown type. Any errors that occur are returned
// after the entire list is processed. Decoders are tried in order.
func DecodeList(objects []Object, decoders ...ObjectDecoder) []error {
	errs := []error(nil)
	for i, obj := range objects {
		switch t := obj.(type) {
		case *Unknown:
			for _, decoder := range decoders {
				if !decoder.Recognizes(t.APIVersion, t.Kind) {
					continue
				}
				obj, err := decoder.Decode(t.RawJSON)
				if err != nil {
					errs = append(errs, err)
					break
				}
				objects[i] = obj
				break
			}
		}
	}
	return errs
}

// MultiObjectTyper returns the types of objects across multiple schemes in order.
type MultiObjectTyper []ObjectTyper

var _ ObjectTyper = MultiObjectTyper{}

func (m MultiObjectTyper) DataVersionAndKind(data []byte) (version, kind string, err error) {
	for _, t := range m {
		version, kind, err = t.DataVersionAndKind(data)
		if err == nil {
			return
		}
	}
	return
}

func (m MultiObjectTyper) ObjectVersionAndKind(obj Object) (version, kind string, err error) {
	for _, t := range m {
		version, kind, err = t.ObjectVersionAndKind(obj)
		if err == nil {
			return
		}
	}
	return
}

func (m MultiObjectTyper) Recognizes(version, kind string) bool {
	for _, t := range m {
		if t.Recognizes(version, kind) {
			return true
		}
	}
	return false
}
