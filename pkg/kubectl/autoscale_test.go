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
	utilpointer "k8s.io/kubernetes/pkg/util/pointer"
)

func TestHPAGenerate(t *testing.T) {
	tests := []struct {
		name               string
		HPAName            string
		scaleRefKind       string
		scaleRefName       string
		scaleRefApiVersion string
		minReplicas        int32
		maxReplicas        int32
		CPUPercent         int32
		expected           *autoscalingv1.HorizontalPodAutoscaler
		expectErr          bool
	}{
		{
			name:               "valid case",
			HPAName:            "foo",
			minReplicas:        1,
			maxReplicas:        10,
			CPUPercent:         80,
			scaleRefKind:       "kind",
			scaleRefName:       "name",
			scaleRefApiVersion: "apiVersion",
			expected: &autoscalingv1.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
					TargetCPUUtilizationPercentage: utilpointer.Int32Ptr(80),
					ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
						Kind:       "kind",
						Name:       "name",
						APIVersion: "apiVersion",
					},
					MaxReplicas: int32(10),
					MinReplicas: utilpointer.Int32Ptr(1),
				},
			},
			expectErr: false,
		},
		{
			name:               "'name' is a required parameter",
			scaleRefKind:       "kind",
			scaleRefName:       "name",
			scaleRefApiVersion: "apiVersion",
			expectErr:          true,
		},
		{
			name:               "'max' is a required parameter",
			HPAName:            "foo",
			scaleRefKind:       "kind",
			scaleRefName:       "name",
			scaleRefApiVersion: "apiVersion",
			expectErr:          true,
		},
		{
			name:               "'max' must be greater than or equal to 'min'",
			HPAName:            "foo",
			minReplicas:        10,
			maxReplicas:        1,
			scaleRefKind:       "kind",
			scaleRefName:       "name",
			scaleRefApiVersion: "apiVersion",
			expectErr:          true,
		},
		{
			name:               "'max' must be at least 1",
			HPAName:            "foo",
			minReplicas:        1,
			maxReplicas:        -10,
			scaleRefKind:       "kind",
			scaleRefName:       "name",
			scaleRefApiVersion: "apiVersion",
			expectErr:          true,
		},
	}

	for _, test := range tests {
		generator := HorizontalPodAutoscalerGeneratorV1{
			Name:               test.HPAName,
			ScaleRefKind:       test.scaleRefKind,
			ScaleRefName:       test.scaleRefName,
			ScaleRefApiVersion: test.scaleRefApiVersion,
			MinReplicas:        test.minReplicas,
			MaxReplicas:        test.maxReplicas,
			CPUPercent:         test.CPUPercent,
		}
		obj, err := generator.StructuredGenerate()
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
