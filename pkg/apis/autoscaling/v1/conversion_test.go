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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

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

func TestSyncPeriodSeconds_RoundTripV1(t *testing.T) {
	hpaInternal := &autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-hpa",
			Namespace: "default",
		},
		Spec: autoscaling.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscaling.CrossVersionObjectReference{
				Kind: "Deployment",
				Name: "my-deployment",
			},
			MinReplicas:       ptr.To[int32](1),
			MaxReplicas:       3,
			SyncPeriodSeconds: ptr.To[int32](45),
			Metrics: []autoscaling.MetricSpec{
				{
					Type: autoscaling.ResourceMetricSourceType,
					Resource: &autoscaling.ResourceMetricSource{
						Name: api.ResourceCPU,
						Target: autoscaling.MetricTarget{
							Type:               autoscaling.UtilizationMetricType,
							AverageUtilization: ptr.To[int32](80),
						},
					},
				},
			},
		},
	}

	hpaV1 := &autoscalingv1.HorizontalPodAutoscaler{}
	err := Convert_autoscaling_HorizontalPodAutoscaler_To_v1_HorizontalPodAutoscaler(hpaInternal, hpaV1, nil)
	require.NoError(t, err, "Conversion to v1 should not fail")
	assert.Equal(t, "45", hpaV1.Annotations[autoscaling.SyncPeriodSecondsAnnotation],
		"SyncPeriodSeconds should be serialized in annotation")

	roundTripped := &autoscaling.HorizontalPodAutoscaler{}
	err = Convert_v1_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler(hpaV1, roundTripped, nil)
	require.NoError(t, err, "Conversion back to internal should not fail")
	if assert.NotNil(t, roundTripped.Spec.SyncPeriodSeconds, "SyncPeriodSeconds should survive v1 round-trip") {
		assert.Equal(t, int32(45), *roundTripped.Spec.SyncPeriodSeconds)
	}

	_, hasSyncPeriod := roundTripped.Annotations[autoscaling.SyncPeriodSecondsAnnotation]
	assert.False(t, hasSyncPeriod, "Round-trip annotation should be dropped after conversion to internal")
}

func TestSyncPeriodSeconds_NilNotSerialized(t *testing.T) {
	hpaInternal := &autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-hpa",
			Namespace: "default",
		},
		Spec: autoscaling.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscaling.CrossVersionObjectReference{
				Kind: "Deployment",
				Name: "my-deployment",
			},
			MinReplicas: ptr.To[int32](1),
			MaxReplicas: 3,
			Metrics: []autoscaling.MetricSpec{
				{
					Type: autoscaling.ResourceMetricSourceType,
					Resource: &autoscaling.ResourceMetricSource{
						Name: api.ResourceCPU,
						Target: autoscaling.MetricTarget{
							Type:               autoscaling.UtilizationMetricType,
							AverageUtilization: ptr.To[int32](80),
						},
					},
				},
			},
		},
	}

	hpaV1 := &autoscalingv1.HorizontalPodAutoscaler{}
	err := Convert_autoscaling_HorizontalPodAutoscaler_To_v1_HorizontalPodAutoscaler(hpaInternal, hpaV1, nil)
	require.NoError(t, err)
	_, hasSyncPeriod := hpaV1.Annotations[autoscaling.SyncPeriodSecondsAnnotation]
	assert.False(t, hasSyncPeriod, "Nil SyncPeriodSeconds should not produce an annotation")
}
