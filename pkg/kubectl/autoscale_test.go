/*
Copyright 2015 The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"reflect"
	"testing"
)

func TestAutoScaleGenerate(t *testing.T) {
	testCases := []struct {
		params    map[string]interface{}
		expected  autoscaling.HorizontalPodAutoscaler
		expectErr bool
	}{
		{
			params: map[string]interface{}{
				"min":                 "10",
				"max":                 "20",
				"scaleRef-kind":       "scaleRefKind",
				"scaleRef-name":       "scaleRefName",
				"scaleRef-apiVersion": "scaleRefApiVersion",
			},
			expected:  autoscaling.HorizontalPodAutoscaler{},
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"name":                "scaler",
				"min":                 "10",
				"scaleRef-kind":       "scaleRefKind",
				"scaleRef-name":       "scaleRefName",
				"scaleRef-apiVersion": "scaleRefApiVersion",
			},
			expected:  autoscaling.HorizontalPodAutoscaler{},
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"name":                "scaler",
				"min":                 "30",
				"max":                 "20",
				"scaleRef-kind":       "scaleRefKind",
				"scaleRef-name":       "scaleRefName",
				"scaleRef-apiVersion": "scaleRefApiVersion",
			},
			expected:  autoscaling.HorizontalPodAutoscaler{},
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"name":                "scaler",
				"min":                 "10",
				"max":                 "20",
				"scaleRef-kind":       "scaleRefKind",
				"scaleRef-name":       "scaleRefName",
				"scaleRef-apiVersion": "scaleRefApiVersion",
			},
			expected: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{
					Name: "scaler",
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Kind:       "scaleRefKind",
						Name:       "scaleRefName",
						APIVersion: "scaleRefApiVersion",
					},
					MinReplicas: newInt32(10),
					MaxReplicas: int32(20),
				},
			},
			expectErr: false,
		},
		{
			params: map[string]interface{}{
				"name":                "scaler",
				"min":                 "10",
				"max":                 "20",
				"scaleRef-kind":       "scaleRefKind",
				"scaleRef-name":       "scaleRefName",
				"scaleRef-apiVersion": "scaleRefApiVersion",
				"cpu-percent":         "50",
			},
			expected: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{
					Name: "scaler",
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Kind:       "scaleRefKind",
						Name:       "scaleRefName",
						APIVersion: "scaleRefApiVersion",
					},
					Metrics: []autoscaling.MetricSpec{
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricSource{
								Name: api.ResourceCPU,
								TargetAverageUtilization: newInt32(50),
							},
						},
					},
					MinReplicas: newInt32(10),
					MaxReplicas: int32(20),
				},
			},
			expectErr: false,
		},
	}

	generator := HorizontalPodAutoscalerV1{}
	for i, test := range testCases {

		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("case %d, unexpected error: %v", i, err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj, &test.expected) {
			t.Errorf("\ncase %d, expected:\n%#v\nsaw:\n%#v", i, &test.expected, obj)
		}
	}
}

func newInt32(val int32) *int32 {
	p := new(int32)
	*p = val
	return p
}
