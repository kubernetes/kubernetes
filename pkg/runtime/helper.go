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

// IsListType returns true if the provided Object has a slice called Items
func IsListType(obj Object) bool {
	_, err := GetItemsPtr(obj)
	return err == nil
}

// GetItemsPtr returns a pointer to the list object's Items member.
// If 'list' doesn't have an Items member, it's not really a list type
// and an error will be returned.
// This function will either return a pointer to a slice, or an error, but not both.
func GetItemsPtr(list Object) (interface{}, error) {
	v, err := conversion.EnforcePtr(list)
	if err != nil {
		return nil, err
	}
	items := v.FieldByName("Items")
	if !items.IsValid() {
		return nil, fmt.Errorf("no Items field in %#v", list)
	}
	switch items.Kind() {
	case reflect.Interface, reflect.Ptr:
		target := reflect.TypeOf(items.Interface()).Elem()
		if target.Kind() != reflect.Slice {
			return nil, fmt.Errorf("items: Expected slice, got %s", target.Kind())
		}
		return items.Interface(), nil
	case reflect.Slice:
		return items.Addr().Interface(), nil
	default:
		return nil, fmt.Errorf("items: Expected slice, got %s", items.Kind())
	}
}

// ExtractList returns obj's Items element as an array of runtime.Objects.
// Returns an error if obj is not a List type (does not have an Items member).
// TODO: move me to pkg/api/meta
func ExtractList(obj Object) ([]Object, error) {
	itemsPtr, err := GetItemsPtr(obj)
	if err != nil {
		return nil, err
	}
	items, err := conversion.EnforcePtr(itemsPtr)
	if err != nil {
		return nil, err
	}
	list := make([]Object, items.Len())
	for i := range list {
		raw := items.Index(i)
		switch item := raw.Interface().(type) {
		case Object:
			list[i] = item
		case RawExtension:
			list[i] = &Unknown{
				RawJSON: item.RawJSON,
			}
		default:
			var found bool
			if list[i], found = raw.Addr().Interface().(Object); !found {
				return nil, fmt.Errorf("%v: item[%v]: Expected object, got %#v(%s)", obj, i, raw.Interface(), raw.Kind())
			}
		}
	}
	return list, nil
}

// objectSliceType is the type of a slice of Objects
var objectSliceType = reflect.TypeOf([]Object{})

// SetList sets the given list object's Items member have the elements given in
// objects.
// Returns an error if list is not a List type (does not have an Items member),
// or if any of the objects are not of the right type.
// TODO: move me to pkg/api/meta
func SetList(list Object, objects []Object) error {
	itemsPtr, err := GetItemsPtr(list)
	if err != nil {
		return err
	}
	items, err := conversion.EnforcePtr(itemsPtr)
	if err != nil {
		return err
	}
	if items.Type() == objectSliceType {
		items.Set(reflect.ValueOf(objects))
		return nil
	}
	slice := reflect.MakeSlice(items.Type(), len(objects), len(objects))
	for i := range objects {
		dest := slice.Index(i)
		src, err := conversion.EnforcePtr(objects[i])
		if err != nil {
			return err
		}
		if src.Type().AssignableTo(dest.Type()) {
			dest.Set(src)
		} else if src.Type().ConvertibleTo(dest.Type()) {
			dest.Set(src.Convert(dest.Type()))
		} else {
			return fmt.Errorf("item[%d]: Type mismatch: Expected %v, got %v", i, dest.Type(), src.Type())
		}
	}
	items.Set(slice)
	return nil
}

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
