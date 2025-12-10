/*
Copyright 2025 The Kubernetes Authors.

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

package v2beta2

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	autoscalingv2beta2 "k8s.io/api/autoscaling/v2beta2"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

func TestConvertRoundTrip(t *testing.T) {
	tolerance1 := resource.MustParse("0.1")
	tolerance2 := resource.MustParse("0.2")
	tests := []struct {
		name        string
		internalHPA *autoscaling.HorizontalPodAutoscaler
	}{
		{
			"Complete HPA with scale-up tolerance",
			&autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: v1.ObjectMeta{
					Name:        "hpa",
					Namespace:   "hpa-ns",
					Annotations: map[string]string{"key": "value"},
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					MinReplicas: ptr.To(int32(1)),
					MaxReplicas: 3,
					Metrics: []autoscaling.MetricSpec{
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricSource{
								Name: api.ResourceCPU,
								Target: autoscaling.MetricTarget{
									Type:         autoscaling.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(300, resource.DecimalSI),
								},
							},
						},
					},
					Behavior: &autoscaling.HorizontalPodAutoscalerBehavior{
						ScaleUp: &autoscaling.HPAScalingRules{
							Policies: []autoscaling.HPAScalingPolicy{
								{
									Type:          autoscaling.PodsScalingPolicy,
									Value:         1,
									PeriodSeconds: 2,
								},
							},
							Tolerance: &tolerance1,
						},
					},
				},
			},
		},
		{
			"Scale-down and scale-up tolerances",
			&autoscaling.HorizontalPodAutoscaler{
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					MinReplicas: ptr.To(int32(1)),
					MaxReplicas: 3,
					Behavior: &autoscaling.HorizontalPodAutoscalerBehavior{
						ScaleUp: &autoscaling.HPAScalingRules{
							Tolerance: &tolerance1,
						},
						ScaleDown: &autoscaling.HPAScalingRules{
							Tolerance: &tolerance2,
						},
					},
				},
			},
		},
		{
			"Scale-down tolerance only",
			&autoscaling.HorizontalPodAutoscaler{
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					MinReplicas: ptr.To(int32(1)),
					MaxReplicas: 3,
					Behavior: &autoscaling.HorizontalPodAutoscalerBehavior{
						ScaleDown: &autoscaling.HPAScalingRules{
							Tolerance: &tolerance2,
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v2beta2HPA := &autoscalingv2beta2.HorizontalPodAutoscaler{}
			if err := Convert_autoscaling_HorizontalPodAutoscaler_To_v2beta2_HorizontalPodAutoscaler(tt.internalHPA, v2beta2HPA, nil); err != nil {
				t.Errorf("Convert_autoscaling_HorizontalPodAutoscaler_To_v2beta2_HorizontalPodAutoscaler() error = %v", err)
			}
			roundtripHPA := &autoscaling.HorizontalPodAutoscaler{}
			if err := Convert_v2beta2_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler(v2beta2HPA, roundtripHPA, nil); err != nil {
				t.Errorf("Convert_v2beta2_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler() error = %v", err)
			}
			if !apiequality.Semantic.DeepEqual(tt.internalHPA, roundtripHPA) {
				t.Errorf("HPA is not equivalent after round-trip:  mismatch (-want +got):\n%s", cmp.Diff(tt.internalHPA, roundtripHPA))
			}
		})
	}
}
