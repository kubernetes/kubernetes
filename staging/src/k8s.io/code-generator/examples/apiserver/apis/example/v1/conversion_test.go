/*
Copyright 2025 The Kubernetes Authors.

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

package v1

import (
	"reflect"
	"testing"

	"k8s.io/code-generator/examples/apiserver/apis/example"
)

func TestConversion(t *testing.T) {
	testcases := []struct {
		name string
		in   *ConversionCustomContainer
	}{
		{
			name: "nil",
			in:   &ConversionCustomContainer{},
		},
		{
			name: "empty",
			in: &ConversionCustomContainer{
				Slice:   []ConversionCustom{},
				SliceP:  []*ConversionCustom{},
				Map:     map[string]ConversionCustom{},
				MapP:    map[string]*ConversionCustom{},
				Struct:  ConversionCustom{},
				StructP: &ConversionCustom{},
			},
		},
		{
			name: "nil_entries",
			in: &ConversionCustomContainer{
				Slice:  []ConversionCustom{{}},
				SliceP: []*ConversionCustom{nil},
				Map:    map[string]ConversionCustom{"key": {}},
				MapP:   map[string]*ConversionCustom{"key": nil},
			},
		},
		{
			name: "set_entries",
			in: &ConversionCustomContainer{
				Slice:   []ConversionCustom{{PublicField: "test1"}},
				SliceP:  []*ConversionCustom{{PublicField: "test2"}},
				Map:     map[string]ConversionCustom{"key": {PublicField: "test3"}},
				MapP:    map[string]*ConversionCustom{"key": {PublicField: "test4"}},
				Struct:  ConversionCustom{PublicField: "test5"},
				StructP: &ConversionCustom{PublicField: "test6"},
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			original := tc.in.DeepCopy()

			out := &example.ConversionCustomContainer{}
			if err := Convert_v1_ConversionCustomContainer_To_example_ConversionCustomContainer(tc.in, out, nil); err != nil {
				t.Fatal(err)
			}

			roundtrip := &ConversionCustomContainer{}
			if err := Convert_example_ConversionCustomContainer_To_v1_ConversionCustomContainer(out, roundtrip, nil); err != nil {
				t.Fatal(err)
			}

			if !reflect.DeepEqual(original, roundtrip) {
				t.Fatalf("expected:\n%#v\ngot:\n%#v", original, roundtrip)
			}
		})
	}
}
