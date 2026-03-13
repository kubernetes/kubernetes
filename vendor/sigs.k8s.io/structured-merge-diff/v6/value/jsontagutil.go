/*
Copyright 2019 The Kubernetes Authors.

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

package value

import (
	"fmt"
	"reflect"
	"strings"
)

type isZeroer interface {
	IsZero() bool
}

var isZeroerType = reflect.TypeOf((*isZeroer)(nil)).Elem()

func reflectIsZero(dv reflect.Value) bool {
	return dv.IsZero()
}

// OmitZeroFunc returns a function for a type for a given struct field
// which determines if the value for that field is a zero value, matching
// how the stdlib JSON implementation.
func OmitZeroFunc(t reflect.Type) func(reflect.Value) bool {
	// Provide a function that uses a type's IsZero method.
	// This matches the go 1.24 custom IsZero() implementation matching
	switch {
	case t.Kind() == reflect.Interface && t.Implements(isZeroerType):
		return func(v reflect.Value) bool {
			// Avoid panics calling IsZero on a nil interface or
			// non-nil interface with nil pointer.
			return safeIsNil(v) ||
				(v.Elem().Kind() == reflect.Pointer && v.Elem().IsNil()) ||
				v.Interface().(isZeroer).IsZero()
		}
	case t.Kind() == reflect.Pointer && t.Implements(isZeroerType):
		return func(v reflect.Value) bool {
			// Avoid panics calling IsZero on nil pointer.
			return safeIsNil(v) || v.Interface().(isZeroer).IsZero()
		}
	case t.Implements(isZeroerType):
		return func(v reflect.Value) bool {
			return v.Interface().(isZeroer).IsZero()
		}
	case reflect.PointerTo(t).Implements(isZeroerType):
		return func(v reflect.Value) bool {
			if !v.CanAddr() {
				// Temporarily box v so we can take the address.
				v2 := reflect.New(v.Type()).Elem()
				v2.Set(v)
				v = v2
			}
			return v.Addr().Interface().(isZeroer).IsZero()
		}
	default:
		// default to the reflect.IsZero implementation
		return reflectIsZero
	}
}

// TODO: This implements the same functionality as https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/apimachinery/pkg/runtime/converter.go#L236
// but is based on the highly efficient approach from https://golang.org/src/encoding/json/encode.go

func lookupJsonTags(f reflect.StructField) (name string, omit bool, inline bool, omitempty bool, omitzero func(reflect.Value) bool) {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "", true, false, false, nil
	}
	name, opts := parseTag(tag)
	if name == "" {
		name = f.Name
	}

	if opts.Contains("omitzero") {
		omitzero = OmitZeroFunc(f.Type)
	}

	return name, false, opts.Contains("inline"), opts.Contains("omitempty"), omitzero
}

func isEmpty(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	case reflect.Chan, reflect.Func:
		panic(fmt.Sprintf("unsupported type: %v", v.Type()))
	}
	return false
}

type tagOptions string

// parseTag splits a struct field's json tag into its name and
// comma-separated options.
func parseTag(tag string) (string, tagOptions) {
	if idx := strings.Index(tag, ","); idx != -1 {
		return tag[:idx], tagOptions(tag[idx+1:])
	}
	return tag, tagOptions("")
}

// Contains reports whether a comma-separated list of options
// contains a particular substr flag. substr must be surrounded by a
// string boundary or commas.
func (o tagOptions) Contains(optionName string) bool {
	if len(o) == 0 {
		return false
	}
	s := string(o)
	for s != "" {
		var next string
		i := strings.Index(s, ",")
		if i >= 0 {
			s, next = s[:i], s[i+1:]
		}
		if s == optionName {
			return true
		}
		s = next
	}
	return false
}
