/*
Copyright 2017 The Kubernetes Authors.

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

package v1_test

import (
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestMapToLabelSelectorRoundTrip(t *testing.T) {
	// We should be able to round-trip a map-only selector through LabelSelector.
	inputs := []map[string]string{
		nil,
		{},
		{"one": "foo"},
		{"one": "foo", "two": "bar"},
	}
	for _, in := range inputs {
		ls := &v1.LabelSelector{}
		if err := v1.Convert_map_to_unversioned_LabelSelector(&in, ls, nil); err != nil {
			t.Errorf("Convert_map_to_unversioned_LabelSelector(%#v): %v", in, err)
			continue
		}
		out := map[string]string{}
		if err := v1.Convert_unversioned_LabelSelector_to_map(ls, &out, nil); err != nil {
			t.Errorf("Convert_unversioned_LabelSelector_to_map(%#v): %v", ls, err)
			continue
		}
		if !apiequality.Semantic.DeepEqual(in, out) {
			t.Errorf("map-selector conversion round-trip failed: got %v; want %v", out, in)
		}
	}
}

func TestConvertSliceStringToDeletionPropagation(t *testing.T) {
	tcs := []struct {
		Input  []string
		Output v1.DeletionPropagation
	}{
		{
			Input:  nil,
			Output: "",
		},
		{
			Input:  []string{},
			Output: "",
		},
		{
			Input:  []string{"foo"},
			Output: "foo",
		},
		{
			Input:  []string{"bar", "foo"},
			Output: "bar",
		},
	}

	for _, tc := range tcs {
		var dp v1.DeletionPropagation
		if err := v1.Convert_Slice_string_To_v1_DeletionPropagation(&tc.Input, &dp, nil); err != nil {
			t.Errorf("Convert_Slice_string_To_v1_DeletionPropagation(%#v): %v", tc.Input, err)
			continue
		}
		if !apiequality.Semantic.DeepEqual(dp, tc.Output) {
			t.Errorf("slice string to DeletionPropagation conversion failed: got %v; want %v", dp, tc.Output)
		}
	}
}
