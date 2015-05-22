package awsutil

import (
	"io"
	"reflect"
)

// Copy deeply copies a src structure to dst. Useful for copying request and
// response structures.
func Copy(dst, src interface{}) {
	rcopy(reflect.ValueOf(dst), reflect.ValueOf(src))
}

// CopyOf returns a copy of src while also allocating the memory for dst.
// src must be a pointer type or this operation will fail.
func CopyOf(src interface{}) (dst interface{}) {
	dsti := reflect.New(reflect.TypeOf(src).Elem())
	dst = dsti.Interface()
	rcopy(dsti, reflect.ValueOf(src))
	return
}

// rcopy performs a recursive copy of values from the source to destination.
func rcopy(dst, src reflect.Value) {
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
			if dst.CanSet() {
				dst.Set(reflect.New(e))
			}
			if src.Elem().IsValid() {
				rcopy(dst.Elem(), src.Elem())
			}
		}
	case reflect.Struct:
		dst.Set(reflect.New(src.Type()).Elem())
		for i := 0; i < src.NumField(); i++ {
			rcopy(dst.Field(i), src.Field(i))
		}
	case reflect.Slice:
		s := reflect.MakeSlice(src.Type(), src.Len(), src.Cap())
		dst.Set(s)
		for i := 0; i < src.Len(); i++ {
			rcopy(dst.Index(i), src.Index(i))
		}
	case reflect.Map:
		s := reflect.MakeMap(src.Type())
		dst.Set(s)
		for _, k := range src.MapKeys() {
			v := src.MapIndex(k)
			v2 := reflect.New(v.Type()).Elem()
			rcopy(v2, v)
			dst.SetMapIndex(k, v2)
		}
	default:
		dst.Set(src)
	}
}
