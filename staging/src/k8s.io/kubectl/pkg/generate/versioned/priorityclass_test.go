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

package versioned

import (
	apiv1 "k8s.io/api/core/v1"
	scheduling "k8s.io/api/scheduling/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"reflect"
	"testing"
)

func TestPriorityClassV1Generator(t *testing.T) {
	var (
		preemptLowerPriority = apiv1.PreemptLowerPriority
		preemptNever         = apiv1.PreemptNever
	)
	tests := []struct {
		name      string
		params    map[string]interface{}
		expected  *scheduling.PriorityClass
		expectErr bool
	}{
		{
			name: "test valid case",
			params: map[string]interface{}{
				"name":              "foo",
				"value":             int32(1000),
				"global-default":    false,
				"description":       "high priority class",
				"preemption-policy": "PreemptLowerPriority",
			},
			expected: &scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Value:            int32(1000),
				GlobalDefault:    false,
				Description:      "high priority class",
				PreemptionPolicy: &preemptLowerPriority,
			},
			expectErr: false,
		},
		{
			name: "test valid case that field non-preempting is set",
			params: map[string]interface{}{
				"name":              "foo",
				"value":             int32(1000),
				"global-default":    false,
				"description":       "high priority class",
				"preemption-policy": "Never",
			},
			expected: &scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Value:            int32(1000),
				GlobalDefault:    false,
				Description:      "high priority class",
				PreemptionPolicy: &preemptNever,
			},
			expectErr: false,
		},
		{
			name: "test valid case that as default priority",
			params: map[string]interface{}{
				"name":              "foo",
				"value":             int32(1000),
				"global-default":    true,
				"description":       "high priority class",
				"preemption-policy": "PreemptLowerPriority",
			},
			expected: &scheduling.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Value:            int32(1000),
				GlobalDefault:    true,
				Description:      "high priority class",
				PreemptionPolicy: &preemptLowerPriority,
			},
			expectErr: false,
		},
		{
			name: "test missing required param",
			params: map[string]interface{}{
				"name":              "foo",
				"global-default":    true,
				"description":       "high priority class",
				"preemption-policy": "PreemptLowerPriority",
			},
			expectErr: true,
		},
	}

	generator := PriorityClassV1Generator{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := generator.Generate(tt.params)
			if !tt.expectErr && err != nil {
				t.Errorf("%s: unexpected error: %v", tt.name, err)
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(obj.(*scheduling.PriorityClass), tt.expected) {
				t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", tt.name, tt.expected, obj.(*scheduling.PriorityClass))
			}
		})
	}
}
