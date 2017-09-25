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

package kubectl

import (
	"reflect"
	"testing"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestHPAGenerate(t *testing.T) {
	tests := []struct {
		name      string
		params    map[string]interface{}
		expected  *autoscalingv1.HorizontalPodAutoscaler
		expectErr bool
	}{
		{
			name: "valid case",
			params: map[string]interface{}{
				"name":                "foo",
				"min":                 "1",
				"max":                 "10",
				"cpu-percent":         "80",
				"scaleRef-kind":       "kind",
				"scaleRef-name":       "name",
				"scaleRef-apiVersion": "apiVersion",
			},
			expected: &autoscalingv1.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
					TargetCPUUtilizationPercentage: newInt32(80),
					ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
						Kind:       "kind",
						Name:       "name",
						APIVersion: "apiVersion",
					},
					MaxReplicas: int32(10),
					MinReplicas: newInt32(1),
				},
			},
			expectErr: false,
		},
		{
			name: "'name' is a required parameter",
			params: map[string]interface{}{
				"scaleRef-kind":       "kind",
				"scaleRef-name":       "name",
				"scaleRef-apiVersion": "apiVersion",
			},
			expectErr: true,
		},
		{
			name: "'max' is a required parameter",
			params: map[string]interface{}{
				"default-name":        "foo",
				"scaleRef-kind":       "kind",
				"scaleRef-name":       "name",
				"scaleRef-apiVersion": "apiVersion",
			},
			expectErr: true,
		},
		{
			name: "'max' must be greater than or equal to 'min'",
			params: map[string]interface{}{
				"name":                "foo",
				"min":                 "10",
				"max":                 "1",
				"scaleRef-kind":       "kind",
				"scaleRef-name":       "name",
				"scaleRef-apiVersion": "apiVersion",
			},
			expectErr: true,
		},
		{
			name: "cpu-percent must be an integer if specified",
			params: map[string]interface{}{
				"name":                "foo",
				"min":                 "1",
				"max":                 "10",
				"cpu-percent":         "",
				"scaleRef-kind":       "kind",
				"scaleRef-name":       "name",
				"scaleRef-apiVersion": "apiVersion",
			},
			expectErr: true,
		},
		{
			name: "'min' must be an integer if specified",
			params: map[string]interface{}{
				"name":                "foo",
				"min":                 "foo",
				"max":                 "10",
				"cpu-percent":         "60",
				"scaleRef-kind":       "kind",
				"scaleRef-name":       "name",
				"scaleRef-apiVersion": "apiVersion",
			},
			expectErr: true,
		},
		{
			name: "'max' must be an integer if specified",
			params: map[string]interface{}{
				"name":                "foo",
				"min":                 "1",
				"max":                 "bar",
				"cpu-percent":         "90",
				"scaleRef-kind":       "kind",
				"scaleRef-name":       "name",
				"scaleRef-apiVersion": "apiVersion",
			},
			expectErr: true,
		},
	}
	generator := HorizontalPodAutoscalerV1{}
	for _, test := range tests {
		obj, err := generator.Generate(test.params)
		if test.expectErr && err != nil {
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("[%s] unexpected error: %v", test.name, err)
		}
		if test.expectErr && err == nil {
			t.Errorf("[%s] expect error, got nil", test.name)
		}
		if !reflect.DeepEqual(obj.(*autoscalingv1.HorizontalPodAutoscaler), test.expected) {
			t.Errorf("[%s] want:\n%#v\ngot:\n%#v", test.name, test.expected, obj.(*autoscalingv1.HorizontalPodAutoscaler))
		}
	}
}

func newInt32(value int) *int32 {
	v := int32(value)
	return &v
}
