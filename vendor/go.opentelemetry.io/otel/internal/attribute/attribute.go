// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
Package attribute provide several helper functions for some commonly used
logic of processing attributes.
*/
package attribute // import "go.opentelemetry.io/otel/internal/attribute"

import (
	"reflect"
)

// SliceValue convert a slice into an array with same elements as slice.
func SliceValue[T bool | int64 | float64 | string](v []T) any {
	var zero T
	cp := reflect.New(reflect.ArrayOf(len(v), reflect.TypeOf(zero)))
	copy(cp.Elem().Slice(0, len(v)).Interface().([]T), v)
	return cp.Elem().Interface()
}

// AsSlice convert an array into a slice into with same elements as array.
func AsSlice[T bool | int64 | float64 | string](v any) []T {
	rv := reflect.ValueOf(v)
	if rv.Type().Kind() != reflect.Array {
		return nil
	}
	var zero T
	correctLen := rv.Len()
	correctType := reflect.ArrayOf(correctLen, reflect.TypeOf(zero))
	cpy := reflect.New(correctType)
	_ = reflect.Copy(cpy.Elem(), rv)
	return cpy.Elem().Slice(0, correctLen).Interface().([]T)
}
