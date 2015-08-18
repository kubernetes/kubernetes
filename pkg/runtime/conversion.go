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

// Defines conversions between generic types and structs to map query strings
// to struct objects.
package runtime

import (
	"reflect"
	"strconv"
	"strings"

	"k8s.io/kubernetes/pkg/conversion"
)

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

// Converts the source value from CSV to an array if the dest field has a "csv" tag.
// Used when mapping a CSV query param to the corresponding []string field.
func CSVFieldValueMapper(sv, dv reflect.Value, sTag, dTag reflect.StructTag) (reflect.Value, reflect.Value) {
	// Check if the field has csv tag.
	jsonTag := dTag.Get("json")
	if len(jsonTag) == 0 {
		return sv, dv
	}
	tags := strings.Split(jsonTag, ",")
	hasCSVTag := false
	for _, tag := range tags {
		if tag == "csv" {
			hasCSVTag = true
		}
	}
	if !hasCSVTag {
		return sv, dv
	}
	if sv.Type().String() != "[]string" || dv.Type().String() != "[]string" {
		return sv, dv
	}

	// Calculate the number of elements in new slice.
	newSliceLen := 0
	for svIndex := 0; svIndex < sv.Len(); svIndex++ {
		arr := strings.Split(sv.Index(svIndex).String(), ",")
		newSliceLen += len(arr)
	}
	if newSliceLen == sv.Len() {
		// Splitting didnt have any effect.
		return sv, dv
	}
	newSlice := reflect.MakeSlice(sv.Type(), newSliceLen, newSliceLen)
	newSliceIndex := 0
	for svIndex := 0; svIndex < sv.Len(); svIndex++ {
		arr := strings.Split(sv.Index(svIndex).String(), ",")
		for i := 0; i < len(arr); i++ {
			newSlice.Index(newSliceIndex).SetString(arr[i])
			newSliceIndex++
		}
	}
	return newSlice, dv
}

// DefaultStringConversions are helpers for converting []string and string to real values.
var DefaultStringConversions = []interface{}{
	convertStringSliceToString,
	convertStringSliceToInt,
	convertStringSliceToBool,
	convertStringSliceToInt64,
}

func convertStringSliceToString(input *[]string, out *string, s conversion.Scope) error {
	if len(*input) == 0 {
		*out = ""
	}
	*out = (*input)[0]
	return nil
}

func convertStringSliceToInt(input *[]string, out *int, s conversion.Scope) error {
	if len(*input) == 0 {
		*out = 0
	}
	str := (*input)[0]
	i, err := strconv.Atoi(str)
	if err != nil {
		return err
	}
	*out = i
	return nil
}

// converStringSliceToBool will convert a string parameter to boolean.
// Only the absence of a value, a value of "false", or a value of "0" resolve to false.
// Any other value (including empty string) resolves to true.
func convertStringSliceToBool(input *[]string, out *bool, s conversion.Scope) error {
	if len(*input) == 0 {
		*out = false
		return nil
	}
	switch strings.ToLower((*input)[0]) {
	case "false", "0":
		*out = false
	default:
		*out = true
	}
	return nil
}

func convertStringSliceToInt64(input *[]string, out *int64, s conversion.Scope) error {
	if len(*input) == 0 {
		*out = 0
	}
	str := (*input)[0]
	i, err := strconv.ParseInt(str, 10, 64)
	if err != nil {
		return err
	}
	*out = i
	return nil
}
