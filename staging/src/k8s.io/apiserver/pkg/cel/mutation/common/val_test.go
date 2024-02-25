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

package common

import (
	"reflect"
	"testing"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

func TestOptional(t *testing.T) {
	for _, tc := range []struct {
		name     string
		fields   map[string]ref.Val
		expected map[string]any
	}{
		{
			name: "present",
			fields: map[string]ref.Val{
				"zero": types.OptionalOf(types.IntZero),
			},
			expected: map[string]any{
				"zero": int64(0),
			},
		},
		{
			name: "none",
			fields: map[string]ref.Val{
				"absent": types.OptionalNone,
			},
			expected: map[string]any{
				// right now no way to differ from a plain null.
				// we will need to filter out optional.none() before this conversion.
				"absent": nil,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v := &ObjectVal{
				typeRef: nil, // safe in this test, otherwise put a mock
				fields:  tc.fields,
			}
			converted := v.Value()
			if !reflect.DeepEqual(tc.expected, converted) {
				t.Errorf("wrong result, expected %v but got %v", tc.expected, converted)
			}
		})
	}
}
