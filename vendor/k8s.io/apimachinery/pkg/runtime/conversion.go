/*
Copyright 2014 The Kubernetes Authors.

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

// Package runtime defines conversions between generic types and structs to map query strings
// to struct objects.
package runtime

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/conversion"
)

// DefaultMetaV1FieldSelectorConversion auto-accepts metav1 values for name and namespace.
// A cluster scoped resource specifying namespace empty works fine and specifying a particular
// namespace will return no results, as expected.
func DefaultMetaV1FieldSelectorConversion(label, value string) (string, string, error) {
	switch label {
	case "metadata.name":
		return label, value, nil
	case "metadata.namespace":
		return label, value, nil
	default:
		return "", "", fmt.Errorf("%q is not a known field selector: only %q, %q", label, "metadata.name", "metadata.namespace")
	}
}

// JSONKeyMapper uses the struct tags on a conversion to determine the key value for
// the other side. Use when mapping from a map[string]* to a struct or vice versa.
func JSONKeyMapper(key string, sourceTag, destTag reflect.StructTag) (string, string) {
	if s := destTag.Get("json"); len(s) > 0 {
		return strings.SplitN(s, ",", 2)[0], key
	}
	if s := sourceTag.Get("json"); len(s) > 0 {
		return key, strings.SplitN(s, ",", 2)[0]
	}
	return key, key
}

func Convert_Slice_string_To_string(in *[]string, out *string, s conversion.Scope) error {
	if len(*in) == 0 {
		*out = ""
		return nil
	}
	*out = (*in)[0]
	return nil
}

func Convert_Slice_string_To_int(in *[]string, out *int, s conversion.Scope) error {
	if len(*in) == 0 {
		*out = 0
		return nil
	}
	str := (*in)[0]
	i, err := strconv.Atoi(str)
	if err != nil {
		return err
	}
	*out = i
	return nil
}

// Convert_Slice_string_To_bool will convert a string parameter to boolean.
// Only the absence of a value (i.e. zero-length slice), a value of "false", or a
// value of "0" resolve to false.
// Any other value (including empty string) resolves to true.
func Convert_Slice_string_To_bool(in *[]string, out *bool, s conversion.Scope) error {
	if len(*in) == 0 {
		*out = false
		return nil
	}
	switch {
	case (*in)[0] == "0", strings.EqualFold((*in)[0], "false"):
		*out = false
	default:
		*out = true
	}
	return nil
}

// Convert_Slice_string_To_bool will convert a string parameter to boolean.
// Only the absence of a value (i.e. zero-length slice), a value of "false", or a
// value of "0" resolve to false.
// Any other value (including empty string) resolves to true.
func Convert_Slice_string_To_Pointer_bool(in *[]string, out **bool, s conversion.Scope) error {
	if len(*in) == 0 {
		boolVar := false
		*out = &boolVar
		return nil
	}
	switch {
	case (*in)[0] == "0", strings.EqualFold((*in)[0], "false"):
		boolVar := false
		*out = &boolVar
	default:
		boolVar := true
		*out = &boolVar
	}
	return nil
}

func string_to_int64(in string) (int64, error) {
	return strconv.ParseInt(in, 10, 64)
}

func Convert_string_To_int64(in *string, out *int64, s conversion.Scope) error {
	if in == nil {
		*out = 0
		return nil
	}
	i, err := string_to_int64(*in)
	if err != nil {
		return err
	}
	*out = i
	return nil
}

func Convert_Slice_string_To_int64(in *[]string, out *int64, s conversion.Scope) error {
	if len(*in) == 0 {
		*out = 0
		return nil
	}
	i, err := string_to_int64((*in)[0])
	if err != nil {
		return err
	}
	*out = i
	return nil
}

func Convert_string_To_Pointer_int64(in *string, out **int64, s conversion.Scope) error {
	if in == nil {
		*out = nil
		return nil
	}
	i, err := string_to_int64(*in)
	if err != nil {
		return err
	}
	*out = &i
	return nil
}

func Convert_Slice_string_To_Pointer_int64(in *[]string, out **int64, s conversion.Scope) error {
	if len(*in) == 0 {
		*out = nil
		return nil
	}
	i, err := string_to_int64((*in)[0])
	if err != nil {
		return err
	}
	*out = &i
	return nil
}

func RegisterStringConversions(s *Scheme) error {
	if err := s.AddConversionFunc((*[]string)(nil), (*string)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return Convert_Slice_string_To_string(a.(*[]string), b.(*string), scope)
	}); err != nil {
		return err
	}
	if err := s.AddConversionFunc((*[]string)(nil), (*int)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return Convert_Slice_string_To_int(a.(*[]string), b.(*int), scope)
	}); err != nil {
		return err
	}
	if err := s.AddConversionFunc((*[]string)(nil), (*bool)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return Convert_Slice_string_To_bool(a.(*[]string), b.(*bool), scope)
	}); err != nil {
		return err
	}
	if err := s.AddConversionFunc((*[]string)(nil), (*int64)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return Convert_Slice_string_To_int64(a.(*[]string), b.(*int64), scope)
	}); err != nil {
		return err
	}
	return nil
}
