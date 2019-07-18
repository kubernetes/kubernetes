/*
Copyright 2018 The Kubernetes Authors.

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

package conversion

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/validation"
)

func TestRestoreObjectMeta(t *testing.T) {
	tests := []struct {
		name          string
		original      map[string]interface{}
		converted     map[string]interface{}
		expected      map[string]interface{}
		expectedError bool
	}{
		{"no converted metadata",
			map[string]interface{}{"metadata": map[string]interface{}{}, "spec": map[string]interface{}{}},
			map[string]interface{}{"spec": map[string]interface{}{}},
			map[string]interface{}{"spec": map[string]interface{}{}},
			true,
		},
		{"invalid converted metadata",
			map[string]interface{}{"metadata": map[string]interface{}{}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": []interface{}{"foo"}},
			map[string]interface{}{"metadata": []interface{}{"foo"}},
			true,
		},
		{"no original metadata",
			map[string]interface{}{"spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{}, "spec": map[string]interface{}{}},
			false,
		},
		{"invalid original metadata",
			map[string]interface{}{"metadata": []interface{}{"foo"}},
			map[string]interface{}{"metadata": map[string]interface{}{}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": []interface{}{"foo"}, "spec": map[string]interface{}{}},
			true,
		},
		{"changed label, annotations and non-label",
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      map[string]interface{}{"a": "A", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "2"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "abc",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			false,
		},
		{"added labels and annotations",
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo": "bar",
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "abc",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			false,
		},
		{"added labels and annotations, with nil before",
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      nil,
				"annotations": nil,
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "abc",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			false,
		},
		{"removed labels and annotations",
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo": "abc",
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo": "bar",
			}, "spec": map[string]interface{}{}},
			false,
		},
		{"nil'ed labels and annotations",
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "abc",
				"labels":      nil,
				"annotations": nil,
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo": "bar",
			}, "spec": map[string]interface{}{}},
			false,
		},
		{"added labels and annotations",
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo": "bar",
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "abc",
				"labels":      map[string]interface{}{"a": nil, "b": "B"},
				"annotations": map[string]interface{}{"a": nil, "b": "22"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo": "bar",
			}, "spec": map[string]interface{}{}},
			true,
		},
		{"invalid label key",
			map[string]interface{}{"metadata": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{"labels": map[string]interface{}{"some/non-qualified/label": "x"}}},
			map[string]interface{}{"metadata": map[string]interface{}{}},
			true,
		},
		{"invalid annotation key",
			map[string]interface{}{"metadata": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{"labels": map[string]interface{}{"some/non-qualified/label": "x"}}},
			map[string]interface{}{"metadata": map[string]interface{}{}},
			true,
		},
		{"invalid label value",
			map[string]interface{}{"metadata": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{"labels": map[string]interface{}{"foo": "üäö"}}},
			map[string]interface{}{"metadata": map[string]interface{}{}},
			true,
		},
		{"too big label value",
			map[string]interface{}{"metadata": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{"labels": map[string]interface{}{"foo": strings.Repeat("x", validation.LabelValueMaxLength+1)}}},
			map[string]interface{}{"metadata": map[string]interface{}{}},
			true,
		},
		{"too big annotation value",
			map[string]interface{}{"metadata": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{"annotations": map[string]interface{}{"foo": strings.Repeat("x", 256*(1<<10)+1)}}},
			map[string]interface{}{"metadata": map[string]interface{}{}},
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := restoreObjectMeta(&unstructured.Unstructured{Object: tt.original}, &unstructured.Unstructured{Object: tt.converted}); err == nil && tt.expectedError {
				t.Fatalf("expected error, but didn't get one")
			} else if err != nil && !tt.expectedError {
				t.Fatalf("unexpected error: %v", err)
			}

			if !reflect.DeepEqual(tt.converted, tt.expected) {
				t.Errorf("unexpected result: %s", diff.ObjectDiff(tt.expected, tt.converted))
			}
		})
	}
}
