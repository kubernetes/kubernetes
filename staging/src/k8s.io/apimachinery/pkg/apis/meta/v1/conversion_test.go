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
	"time"

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
		if err := v1.Convert_Map_string_To_string_To_v1_LabelSelector(&in, ls, nil); err != nil {
			t.Errorf("Convert_Map_string_To_string_To_v1_LabelSelector(%#v): %v", in, err)
			continue
		}
		out := map[string]string{}
		if err := v1.Convert_v1_LabelSelector_To_Map_string_To_string(ls, &out, nil); err != nil {
			t.Errorf("Convert_v1_LabelSelector_To_Map_string_To_string(%#v): %v", ls, err)
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
		var dpPtr *v1.DeletionPropagation
		if err := v1.Convert_Slice_string_To_Pointer_v1_DeletionPropagation(&tc.Input, &dpPtr, nil); err != nil {
			t.Errorf("Convert_Slice_string_To_Pointer_v1_DeletionPropagation(%#v): %v", tc.Input, err)
			continue
		}
		if !apiequality.Semantic.DeepEqual(dpPtr, &tc.Output) {
			t.Errorf("slice string to DeletionPropagation conversion failed: got %v; want %v", *dpPtr, tc.Output)
		}
	}
}

func TestConvertSliceStringToPointerTime(t *testing.T) {
	t1 := v1.Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC)
	t1String := t1.Format(time.RFC3339)
	t2 := v1.Date(2000, time.June, 6, 6, 6, 6, 0, time.UTC)
	t2String := t2.Format(time.RFC3339)

	tcs := []struct {
		Input  []string
		Output *v1.Time
	}{
		{
			Input:  []string{},
			Output: &v1.Time{},
		},
		{
			Input:  []string{""},
			Output: &v1.Time{},
		},
		{
			Input:  []string{t1String},
			Output: &t1,
		},
		{
			Input:  []string{t1String, t2String},
			Output: &t1,
		},
	}

	for _, tc := range tcs {
		var timePtr *v1.Time
		if err := v1.Convert_Slice_string_To_Pointer_v1_Time(&tc.Input, &timePtr, nil); err != nil {
			t.Errorf("Convert_Slice_string_To_Pointer_v1_Time(%#v): %v", tc.Input, err)
			continue
		}
		if !apiequality.Semantic.DeepEqual(timePtr, tc.Output) {
			t.Errorf("slice string to *v1.Time conversion failed: got %#v; want %#v", timePtr, tc.Output)
		}
	}
}
