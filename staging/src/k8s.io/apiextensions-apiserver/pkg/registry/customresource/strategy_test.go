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

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func generation1() map[string]interface{} {
	return map[string]interface{}{
		"generation": int64(1),
	}
}

func generation2() map[string]interface{} {
	return map[string]interface{}{
		"generation": int64(2),
	}
}

func TestStrategyPrepareForUpdate(t *testing.T) {
	strategy := customResourceStrategy{}
	tcs := []struct {
		name          string
		old           *unstructured.Unstructured
		obj           *unstructured.Unstructured
		statusEnabled bool
		expected      *unstructured.Unstructured
	}{
		{
			name:          "/status is enabled, spec changes increment generation",
			statusEnabled: true,
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation1(),
					"spec":     "old",
					"status":   "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation1(),
					"spec":     "new",
					"status":   "old",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation2(),
					"spec":     "new",
					"status":   "old",
				},
			},
		},
		{
			name:          "/status is enabled, status changes do not increment generation, status changes removed",
			statusEnabled: true,
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation1(),
					"spec":     "old",
					"status":   "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation1(),
					"spec":     "old",
					"status":   "new",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation1(),
					"spec":     "old",
					"status":   "old",
				},
			},
		},
		{
			name:          "/status is enabled, metadata changes do not increment generation",
			statusEnabled: true,
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{
						"generation": int64(1),
						"other":      "old",
					},
					"spec":   "old",
					"status": "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{
						"generation": int64(1),
						"other":      "new",
					},
					"spec":   "old",
					"status": "old",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{
						"generation": int64(1),
						"other":      "new",
					},
					"spec":   "old",
					"status": "old",
				},
			},
		},
		{
			name:          "/status is disabled, spec changes increment generation",
			statusEnabled: false,
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation1(),
					"spec":     "old",
					"status":   "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation1(),
					"spec":     "new",
					"status":   "old",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation2(),
					"spec":     "new",
					"status":   "old",
				},
			},
		},
		{
			name:          "/status is disabled, status changes increment generation",
			statusEnabled: false,
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation1(),
					"spec":     "old",
					"status":   "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation1(),
					"spec":     "old",
					"status":   "new",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation2(),
					"spec":     "old",
					"status":   "new",
				},
			},
		},
		{
			name:          "/status is disabled, other top-level field changes increment generation",
			statusEnabled: false,
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation1(),
					"spec":     "old",
					"image":    "old",
					"status":   "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation1(),
					"spec":     "old",
					"image":    "new",
					"status":   "old",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": generation2(),
					"spec":     "old",
					"image":    "new",
					"status":   "old",
				},
			},
		},
		{
			name:          "/status is disabled, metadata changes do not increment generation",
			statusEnabled: false,
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{
						"generation": int64(1),
						"other":      "old",
					},
					"spec":   "old",
					"status": "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{
						"generation": int64(1),
						"other":      "new",
					},
					"spec":   "old",
					"status": "old",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{
						"generation": int64(1),
						"other":      "new",
					},
					"spec":   "old",
					"status": "old",
				},
			},
		},
	}
	for _, tc := range tcs {
		if tc.statusEnabled {
			strategy.status = &apiextensions.CustomResourceSubresourceStatus{}
		} else {
			strategy.status = nil
		}
		strategy.PrepareForUpdate(context.TODO(), tc.obj, tc.old)
		if !reflect.DeepEqual(tc.obj, tc.expected) {
			t.Errorf("test %q failed: expected: %v, got %v", tc.name, tc.expected, tc.obj)
		}
	}
}
