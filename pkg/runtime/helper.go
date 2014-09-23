/*
Copyright 2014 Google Inc. All rights reserved.

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
)

// GetItemsPtr returns a pointer to the list object's Items member.
// If 'list' doesn't have an Items member, it's not really a list type
// and an error will be returned.
// This function will either return a pointer to a slice, or an error, but not both.
func GetItemsPtr(list Object) (interface{}, error) {
	v := reflect.ValueOf(list)
	if !v.IsValid() {
		return nil, fmt.Errorf("nil list object")
	}
	items := v.Elem().FieldByName("Items")
	if !items.IsValid() {
		return nil, fmt.Errorf("no Items field in %#v", list)
	}
	if items.Kind() != reflect.Slice {
		return nil, fmt.Errorf("Items field is not a slice")
	}
	return items.Addr().Interface(), nil
}

// ExtractList returns obj's Items element as an array of runtime.Objects.
// Returns an error if obj is not a List type (does not have an Items member).
func ExtractList(obj Object) ([]Object, error) {
	itemsPtr, err := GetItemsPtr(obj)
	if err != nil {
		return nil, err
	}
	items := reflect.ValueOf(itemsPtr).Elem()
	list := make([]Object, items.Len())
	for i := range list {
		raw := items.Index(i)
		item, ok := raw.Addr().Interface().(Object)
		if !ok {
			return nil, fmt.Errorf("item in index %v isn't an object: %#v", i, raw.Interface())
		}
		list[i] = item
	}
	return list, nil
}

// SetList sets the given list object's Items member have the elements given in
// objects.
// Returns an error if list is not a List type (does not have an Items member),
// or if any of the objects are not of the right type.
func SetList(list Object, objects []Object) error {
	itemsPtr, err := GetItemsPtr(list)
	if err != nil {
		return err
	}
	items := reflect.ValueOf(itemsPtr).Elem()
	slice := reflect.MakeSlice(items.Type(), len(objects), len(objects))
	for i := range objects {
		dest := slice.Index(i)
		src := reflect.ValueOf(objects[i])
		if !src.IsValid() || src.IsNil() {
			return fmt.Errorf("an object was nil")
		}
		src = src.Elem() // Object is a pointer, but the items in slice are not.
		if src.Type().AssignableTo(dest.Type()) {
			dest.Set(src)
		} else if src.Type().ConvertibleTo(dest.Type()) {
			dest.Set(src.Convert(dest.Type()))
		} else {
			return fmt.Errorf("wrong type: need %v, got %v", dest.Type(), src.Type())
		}
	}
	items.Set(slice)
	return nil
}
