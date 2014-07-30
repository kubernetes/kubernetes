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

package api

import (
	"fmt"
	"reflect"
)

// DefaultCopy copies API objects to/from their corresponding types
// in a versioned package (e.g., v1beta1). Only suitable for types
// in which no fields changed.
// dest and src must both be pointers to API objects.
// Not safe for objects with cyclic references!
// TODO: Allow overrides, using the same function mechanism that
// util.Fuzzer allows.
func DefaultCopy(src, dest interface{}) error {
	dv, sv := reflect.ValueOf(dest), reflect.ValueOf(src)
	if dv.Kind() != reflect.Ptr {
		return fmt.Errorf("Need pointer, but got %#v", dest)
	}
	if sv.Kind() != reflect.Ptr {
		return fmt.Errorf("Need pointer, but got %#v", src)
	}
	dv = dv.Elem()
	sv = sv.Elem()
	if !dv.CanAddr() {
		return fmt.Errorf("Can't write to dest")
	}

	// Ensure there's no reversed src/dest bugs by making src unwriteable.
	sv = reflect.ValueOf(sv.Interface())
	if sv.CanAddr() {
		return fmt.Errorf("Can write to src, shouldn't be able to.")
	}

	return copyValue(sv, dv)
}

// Recursively copy sv into dv
func copyValue(sv, dv reflect.Value) error {
	dt, st := dv.Type(), sv.Type()
	if dt.Name() != st.Name() {
		return fmt.Errorf("Type names don't match: %v, %v", dt.Name(), st.Name())
	}

	// This should handle all simple types.
	if st.AssignableTo(dt) {
		dv.Set(sv)
		return nil
	} else if st.ConvertibleTo(dt) {
		dv.Set(sv.Convert(dt))
		return nil
	}

	// For debugging, should you need to do that.
	if false {
		fmt.Printf("copyVal of %v.%v (%v) -> %v.%v (%v)\n",
			st.PkgPath(), st.Name(), st.Kind(),
			dt.PkgPath(), dt.Name(), dt.Kind())
	}

	switch dv.Kind() {
	case reflect.Struct:
		for i := 0; i < dt.NumField(); i++ {
			f := dv.Type().Field(i)
			df := dv.FieldByName(f.Name)
			sf := sv.FieldByName(f.Name)
			if !df.IsValid() || !sf.IsValid() {
				return fmt.Errorf("%v not present in source and dest.", f.Name)
			}
			if err := copyValue(sf, df); err != nil {
				return err
			}
		}
	case reflect.Slice:
		if sv.IsNil() {
			// Don't make a zero-length slice.
			dv.Set(reflect.Zero(dt))
			return nil
		}
		dv.Set(reflect.MakeSlice(dt, sv.Len(), sv.Cap()))
		for i := 0; i < sv.Len(); i++ {
			if err := copyValue(sv.Index(i), dv.Index(i)); err != nil {
				return err
			}
		}
	case reflect.Ptr:
		if sv.IsNil() {
			// Don't copy a nil ptr!
			dv.Set(reflect.Zero(dt))
			return nil
		}
		dv.Set(reflect.New(dt.Elem()))
		return copyValue(sv.Elem(), dv.Elem())
	case reflect.Map:
		if sv.IsNil() {
			// Don't copy a nil ptr!
			dv.Set(reflect.Zero(dt))
			return nil
		}
		dv.Set(reflect.MakeMap(dt))
		for _, sk := range sv.MapKeys() {
			dk := reflect.New(dt.Key()).Elem()
			if err := copyValue(sk, dk); err != nil {
				return err
			}
			dkv := reflect.New(dt.Elem()).Elem()
			if err := copyValue(sv.MapIndex(sk), dkv); err != nil {
				return err
			}
			dv.SetMapIndex(dk, dkv)
		}
	default:
		return fmt.Errorf("Couldn't copy %#v (%v) into %#v (%v)",
			sv.Interface(), sv.Kind(), dv.Interface(), dv.Kind())
	}
	return nil
}
