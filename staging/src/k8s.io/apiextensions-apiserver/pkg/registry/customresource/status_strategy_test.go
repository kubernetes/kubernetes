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

package customresource

import (
	"context"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestPrepareForUpdate(t *testing.T) {
	strategy := statusStrategy{}
	tcs := []struct {
		old      *unstructured.Unstructured
		obj      *unstructured.Unstructured
		expected *unstructured.Unstructured
	}{
		{
			// changes to spec are ignored
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "new",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "old",
				},
			},
		},
		{
			// changes to other places are also ignored
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"new": "new",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "old",
				},
			},
		},
		{
			// delete status
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec":   "old",
					"status": "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "old",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "old",
				},
			},
		},
		{
			// update status
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec":   "old",
					"status": "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"status": "new",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec":   "old",
					"status": "new",
				},
			},
		},
		{
			// update status and other parts
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec":   "old",
					"status": "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec":   "new",
					"new":    "new",
					"status": "new",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec":   "old",
					"status": "new",
				},
			},
		},
	}
	for index, tc := range tcs {
		strategy.PrepareForUpdate(context.TODO(), tc.obj, tc.old)
		if !reflect.DeepEqual(tc.obj, tc.expected) {
			t.Errorf("test %d failed: expected: %v, got %v", index, tc.expected, tc.obj)
		}
	}
}
