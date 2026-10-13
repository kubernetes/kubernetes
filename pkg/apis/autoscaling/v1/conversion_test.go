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

package v1

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

func TestBehaviorAnnotationVersionedSchema(t *testing.T) {
	want := &autoscaling.HorizontalPodAutoscalerBehavior{
		ScaleUp: &autoscaling.HPAScalingRules{
			StabilizationWindowSeconds: ptr.To[int32](60),
			SelectPolicy:               ptr.To(autoscaling.MaxPolicySelect),
			Policies: []autoscaling.HPAScalingPolicy{
				{Type: autoscaling.PercentScalingPolicy, Value: 100, PeriodSeconds: 60},
			},
			Tolerance: ptr.To(resource.MustParse("0.05")),
		},
		ScaleDown: &autoscaling.HPAScalingRules{SelectPolicy: ptr.To(autoscaling.DisabledPolicySelect)},
	}
	canonical := `{"scaleUp":{"stabilizationWindowSeconds":60,"selectPolicy":"Max","policies":[{"type":"Percent","value":100,"periodSeconds":60}],"tolerance":"50m"},"scaleDown":{"selectPolicy":"Disabled"}}`
	legacy := `{"ScaleUp":{"StabilizationWindowSeconds":60,"SelectPolicy":"Max","Policies":[{"Type":"Percent","Value":100,"PeriodSeconds":60}],"Tolerance":"0.05"},"ScaleDown":{"SelectPolicy":"Disabled"}}`

	t.Run("writer uses versioned field names and remains readable by legacy readers", func(t *testing.T) {
		in := &autoscaling.HorizontalPodAutoscaler{}
		in.Spec.Behavior = want
		in.Annotations = map[string]string{"keep": "value"}
		out := &autoscalingv1.HorizontalPodAutoscaler{}
		if err := Convert_autoscaling_HorizontalPodAutoscaler_To_v1_HorizontalPodAutoscaler(in, out, nil); err != nil {
			t.Fatal(err)
		}
		encoded := out.Annotations[autoscaling.BehaviorSpecsAnnotation]
		assert.JSONEq(t, canonical, encoded)
		var legacyReader autoscaling.HorizontalPodAutoscalerBehavior
		if err := json.Unmarshal([]byte(encoded), &legacyReader); err != nil {
			t.Fatal(err)
		}
		assert.True(t, apiequality.Semantic.DeepEqual(want, &legacyReader), "legacy reader lost behavior fields")
		assert.Equal(t, map[string]string{"keep": "value"}, in.Annotations)
		assert.Equal(t, "value", out.Annotations["keep"])
	})
	for name, encoded := range map[string]string{"legacy": legacy, "versioned": canonical} {
		t.Run("reader/"+name, func(t *testing.T) {
			in := &autoscalingv1.HorizontalPodAutoscaler{}
			in.Annotations = map[string]string{autoscaling.BehaviorSpecsAnnotation: encoded, "keep": "value"}
			out := &autoscaling.HorizontalPodAutoscaler{}
			if err := Convert_v1_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler(in, out, nil); err != nil {
				t.Fatal(err)
			}
			assert.True(t, apiequality.Semantic.DeepEqual(want, out.Spec.Behavior), "reader lost behavior fields")
			assert.Equal(t, "value", out.Annotations["keep"])
			assert.Contains(t, in.Annotations, autoscaling.BehaviorSpecsAnnotation)
		})
	}
}

// Test for #101370
func TestConvert_autoscaling_HorizontalPodAutoscalerSpec_To_v1_HorizontalPodAutoscalerSpec(t *testing.T) {

	type args struct {
		in        *autoscaling.HorizontalPodAutoscalerSpec
		out       *autoscalingv1.HorizontalPodAutoscalerSpec
		expectOut *autoscalingv1.HorizontalPodAutoscalerSpec
		s         conversion.Scope
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			"TestConversionWithCPUAverageValueAndUtilizationBoth1",
			args{
				in: &autoscaling.HorizontalPodAutoscalerSpec{
					MinReplicas: ptr.To[int32](1),
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
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricSource{
								Name: api.ResourceCPU,
								Target: autoscaling.MetricTarget{
									Type:               autoscaling.UtilizationMetricType,
									AverageUtilization: ptr.To[int32](70),
								},
							},
						},
					},
				},
				out: &autoscalingv1.HorizontalPodAutoscalerSpec{},
				expectOut: &autoscalingv1.HorizontalPodAutoscalerSpec{
					MinReplicas:                    ptr.To[int32](1),
					MaxReplicas:                    3,
					TargetCPUUtilizationPercentage: ptr.To[int32](70),
				},
				s: nil,
			},
			false,
		},
		{
			"TestConversionWithCPUAverageValueAndUtilizationBoth2",
			args{
				in: &autoscaling.HorizontalPodAutoscalerSpec{
					MinReplicas: ptr.To[int32](1),
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
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricSource{
								Name: api.ResourceCPU,
								Target: autoscaling.MetricTarget{
									Type:               autoscaling.UtilizationMetricType,
									AverageUtilization: ptr.To[int32](70),
								},
							},
						},
					},
				},
				out: &autoscalingv1.HorizontalPodAutoscalerSpec{
					MinReplicas:                    ptr.To[int32](2),
					MaxReplicas:                    4,
					TargetCPUUtilizationPercentage: ptr.To[int32](60),
				},
				expectOut: &autoscalingv1.HorizontalPodAutoscalerSpec{
					MinReplicas:                    ptr.To[int32](1),
					MaxReplicas:                    3,
					TargetCPUUtilizationPercentage: ptr.To[int32](70),
				},
				s: nil,
			},
			false,
		},
		{
			"TestConversionWithoutMetrics",
			args{
				in: &autoscaling.HorizontalPodAutoscalerSpec{
					MinReplicas: ptr.To[int32](1),
					MaxReplicas: 3,
					Metrics:     []autoscaling.MetricSpec{},
				},
				out: &autoscalingv1.HorizontalPodAutoscalerSpec{
					MinReplicas:                    ptr.To[int32](1),
					MaxReplicas:                    4,
					TargetCPUUtilizationPercentage: ptr.To[int32](60),
				},
				expectOut: &autoscalingv1.HorizontalPodAutoscalerSpec{
					MinReplicas:                    ptr.To[int32](1),
					MaxReplicas:                    3,
					TargetCPUUtilizationPercentage: ptr.To[int32](60),
				},
				s: nil,
			},
			false,
		},
		{
			"TestConversionWithCPUUtilizationOnly",
			args{
				in: &autoscaling.HorizontalPodAutoscalerSpec{
					MinReplicas: ptr.To[int32](1),
					MaxReplicas: 3,
					Metrics: []autoscaling.MetricSpec{
						{
							Type: autoscaling.ResourceMetricSourceType,
							Resource: &autoscaling.ResourceMetricSource{
								Name: api.ResourceCPU,
								Target: autoscaling.MetricTarget{
									Type:               autoscaling.UtilizationMetricType,
									AverageUtilization: ptr.To[int32](60),
								},
							},
						},
					},
				},
				out: &autoscalingv1.HorizontalPodAutoscalerSpec{},
				expectOut: &autoscalingv1.HorizontalPodAutoscalerSpec{
					MinReplicas:                    ptr.To[int32](1),
					MaxReplicas:                    3,
					TargetCPUUtilizationPercentage: ptr.To[int32](60),
				},
				s: nil,
			},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := Convert_autoscaling_HorizontalPodAutoscalerSpec_To_v1_HorizontalPodAutoscalerSpec(tt.args.in, tt.args.out, tt.args.s); (err != nil) != tt.wantErr {
				t.Errorf("Convert_autoscaling_HorizontalPodAutoscalerSpec_To_v1_HorizontalPodAutoscalerSpec() error = %v, wantErr %v", err, tt.wantErr)
			}

			assert.Equal(t, tt.args.expectOut, tt.args.out)
		})
	}
}
