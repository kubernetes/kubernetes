package awsutil

import (
	"io"
	"reflect"
)

// Copy deeply copies a src structure to dst. Useful for copying request and
// response structures.
//
// Can copy between structs of different type, but will only copy fields which
// are assignable, and exist in both structs. Fields which are not assignable,
// or do not exist in both structs are ignored.
func Copy(dst, src interface{}) {
	dstval := reflect.ValueOf(dst)
	if !dstval.IsValid() {
		panic("Copy dst cannot be nil")
	}

	rcopy(dstval, reflect.ValueOf(src), true)
}

// CopyOf returns a copy of src while also allocating the memory for dst.
// src must be a pointer type or this operation will fail.
func CopyOf(src interface{}) (dst interface{}) {
	dsti := reflect.New(reflect.TypeOf(src).Elem())
	dst = dsti.Interface()
	rcopy(dsti, reflect.ValueOf(src), true)
	return
}

// rcopy performs a recursive copy of values from the source to destination.
//
// root is used to skip certain aspects of the copy which are not valid
// for the root node of a object.
func rcopy(dst, src reflect.Value, root bool) {
	if !src.IsValid() {
		return
	}

	switch src.Kind() {
	case reflect.Ptr:
		if _, ok := src.Interface().(io.Reader); ok {
			if dst.Kind() == reflect.Ptr && dst.Elem().CanSet() {
				dst.Elem().Set(src)
			} else if dst.CanSet() {
				dst.Set(src)
			}
		} else {
			e := src.Type().Elem()
			if dst.CanSet() && !src.IsNil() {
				dst.Set(reflect.New(e))
			}
			if src.Elem().IsValid() {
				// Keep the current root state since the depth hasn't changed
				rcopy(dst.Elem(), src.Elem(), root)
			}
		}
	case reflect.Struct:
		if !root {
			dst.Set(reflect.New(src.Type()).Elem())
		}

		t := dst.Type()
		for i := 0; i < t.NumField(); i++ {
			name := t.Field(i).Name
			srcval := src.FieldByName(name)
			if srcval.IsValid() {
				rcopy(dst.FieldByName(name), srcval, false)
			}
		}
	case reflect.Slice:
		if src.IsNil() {
			break
		}

		s := reflect.MakeSlice(src.Type(), src.Len(), src.Cap())
		dst.Set(s)
		for i := 0; i < src.Len(); i++ {
			rcopy(dst.Index(i), src.Index(i), false)
		}
	case reflect.Map:
		if src.IsNil() {
			break
		}

		s := reflect.MakeMap(src.Type())
		dst.Set(s)
		for _, k := range src.MapKeys() {
			v := src.MapIndex(k)
			v2 := reflect.New(v.Type()).Elem()
			rcopy(v2, v, false)
			dst.SetMapIndex(k, v2)
		}
	default:
		// Assign the value if possible. If its not assignable, the value would
		// need to be converted and the impact of that may be unexpected, or is
		// not compatible with the dst type.
		if src.Type().AssignableTo(dst.Type()) {
			dst.Set(src)
		}
	}
}
