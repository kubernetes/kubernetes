/*
Copyright 2024 The Kubernetes Authors.

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

package clientcmd

import (
	"fmt"
	"reflect"
	"strings"
)

// recursively merges src into dst:
// - non-pointer struct fields with any exported fields are recursively merged
// - non-pointer struct fields with only unexported fields prefer src if the field is non-zero
// - maps are shallow merged with src keys taking priority over dst
// - non-zero src fields encountered during recursion that are not maps or structs overwrite and recursion stops
func merge[T any](dst, src *T) error {
	if dst == nil {
		return fmt.Errorf("cannot merge into nil pointer")
	}
	if src == nil {
		return nil
	}
	return mergeValues(nil, reflect.ValueOf(dst).Elem(), reflect.ValueOf(src).Elem())
}

func mergeValues(fieldNames []string, dst, src reflect.Value) error {
	dstType := dst.Type()
	// no-op if we can't read the src
	if !src.IsValid() {
		return nil
	}
	// sanity check types match
	if srcType := src.Type(); dstType != srcType {
		return fmt.Errorf("cannot merge mismatched types (%s, %s) at %s", dstType, srcType, strings.Join(fieldNames, "."))
	}

	switch dstType.Kind() {
	case reflect.Struct:
		if hasExportedField(dstType) {
			// recursively merge
			for i, n := 0, dstType.NumField(); i < n; i++ {
				if err := mergeValues(append(fieldNames, dstType.Field(i).Name), dst.Field(i), src.Field(i)); err != nil {
					return err
				}
			}
		} else if dst.CanSet() {
			// If all fields are unexported, overwrite with src.
			// Using src.IsZero() would make more sense but that's not what mergo did.
			dst.Set(src)
		}

	case reflect.Map:
		if dst.CanSet() && !src.IsZero() {
			// initialize dst if needed
			if dst.IsZero() {
				dst.Set(reflect.MakeMap(dstType))
			}
			// shallow-merge overwriting dst keys with src keys
			for _, mapKey := range src.MapKeys() {
				dst.SetMapIndex(mapKey, src.MapIndex(mapKey))
			}
		}

	case reflect.Slice:
		if dst.CanSet() && src.Len() > 0 {
			// overwrite dst with non-empty src slice
			dst.Set(src)
		}

	case reflect.Pointer:
		if dst.CanSet() && !src.IsZero() {
			// overwrite dst with non-zero values for other types
			if dstType.Elem().Kind() == reflect.Struct {
				// use struct pointer as-is
				dst.Set(src)
			} else {
				// shallow-copy non-struct pointer (interfaces, primitives, etc)
				dst.Set(reflect.New(dstType.Elem()))
				dst.Elem().Set(src.Elem())
			}
		}

	default:
		if dst.CanSet() && !src.IsZero() {
			// overwrite dst with non-zero values for other types
			dst.Set(src)
		}
	}

	return nil
}

// hasExportedField returns true if the given type has any exported fields,
// or if it has any anonymous/embedded struct fields with exported fields
func hasExportedField(dstType reflect.Type) bool {
	for i, n := 0, dstType.NumField(); i < n; i++ {
		field := dstType.Field(i)
		if field.Anonymous && field.Type.Kind() == reflect.Struct {
			if hasExportedField(dstType.Field(i).Type) {
				return true
			}
		} else if len(field.PkgPath) == 0 {
			return true
		}
	}
	return false
}
