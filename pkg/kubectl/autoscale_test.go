/*
Copyright 2016 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/util"
)

func TestGenerateHPA(t *testing.T) {
	tests := []struct {
		params    map[string]interface{}
		expected  *autoscaling.HorizontalPodAutoscaler
		expectErr bool
	}{
		{
			params: map[string]interface{}{
				"name":                "foo",
				"default-name":        "foo-default",
				"min":                 "2",
				"max":                 "4",
				"cpu-percent":         "80",
				"scaleRef-kind":       "ReplicationController",
				"scaleRef-name":       "rc",
				"scaleRef-apiVersion": "v1",
			},
			expected: &autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Kind:       "ReplicationController",
						Name:       "rc",
						APIVersion: "v1",
					},
					MinReplicas:                    util.Int32Ptr(2),
					MaxReplicas:                    4,
					TargetCPUUtilizationPercentage: util.Int32Ptr(80),
				},
			},
		},
		{
			params: map[string]interface{}{
				"default-name": "foo-default",
				"min":          "2",
				"max":          "4",
				"cpu-percent":  "80",
			},
			expected: &autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name: "foo-default",
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					MinReplicas:                    util.Int32Ptr(2),
					MaxReplicas:                    4,
					TargetCPUUtilizationPercentage: util.Int32Ptr(80),
				},
			},
		},
		{
			params: map[string]interface{}{
				"name":        "foo",
				"max":         "4",
				"cpu-percent": "80",
			},
			expected: &autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					MaxReplicas:                    4,
					TargetCPUUtilizationPercentage: util.Int32Ptr(80),
				},
			},
		},
		{
			params: map[string]interface{}{
				"name": "foo",
				"max":  "4",
			},
			expected: &autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					MaxReplicas: 4,
				},
			},
		},
		{
			params: map[string]interface{}{
				"min":         "2",
				"max":         "4",
				"cpu-percent": "80",
			},
			expectErr: true,
		},
		{
			params: map[string]interface{}{
				"name":        "foo",
				"min":         "5",
				"max":         "4",
				"cpu-percent": "80",
			},
			expectErr: true,
		},
	}
	generator := HorizontalPodAutoscalerV1{}
	for _, test := range tests {
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*autoscaling.HorizontalPodAutoscaler), test.expected) {
			t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*autoscaling.HorizontalPodAutoscaler))
		}
	}
}
