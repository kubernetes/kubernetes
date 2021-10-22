/*
Copyright 2020 The Kubernetes Authors.

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
	"reflect"
	"testing"
)

type CustomValue struct {
	data []byte
}

// MarshalJSON has a value receiver on this type.
func (c CustomValue) MarshalJSON() ([]byte, error) {
	return c.data, nil
}

type CustomPointer struct {
	data []byte
}

// MarshalJSON has a pointer receiver on this type.
func (c *CustomPointer) MarshalJSON() ([]byte, error) {
	return c.data, nil
}

func TestToUnstructured(t *testing.T) {
	testcases := []struct {
		Data     string
		Expected interface{}
	}{
		{Data: `null`, Expected: nil},
		{Data: `true`, Expected: true},
		{Data: `false`, Expected: false},
		{Data: `[]`, Expected: []interface{}{}},
		{Data: `[1]`, Expected: []interface{}{int64(1)}},
		{Data: `{}`, Expected: map[string]interface{}{}},
		{Data: `{"a":1}`, Expected: map[string]interface{}{"a": int64(1)}},
		{Data: `0`, Expected: int64(0)},
		{Data: `0.0`, Expected: float64(0)},
	}

	for _, tc := range testcases {
		tc := tc
		t.Run(tc.Data, func(t *testing.T) {
			t.Parallel()
			custom := []interface{}{
				CustomValue{data: []byte(tc.Data)},
				&CustomValue{data: []byte(tc.Data)},
				&CustomPointer{data: []byte(tc.Data)},
			}
			for _, custom := range custom {
				rv := reflect.ValueOf(custom)
				result, err := TypeReflectEntryOf(rv.Type()).ToUnstructured(rv)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(result, tc.Expected) {
					t.Errorf("expected %#v but got %#v", tc.Expected, result)
				}
			}
		})
	}
}
