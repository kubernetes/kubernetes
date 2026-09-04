/*
Copyright The Kubernetes Authors.

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

package v2

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/require"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	field "k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingv1internal "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	"k8s.io/kubernetes/pkg/apis/autoscaling/validation"
	"k8s.io/utils/ptr"
)

// baseHPA returns a v2 HPA with the required fields populated. Tests mutate it
// further to exercise specific scenarios.
func baseHPA(name string) *autoscalingv2.HorizontalPodAutoscaler {
	return &autoscalingv2.HorizontalPodAutoscaler{
		TypeMeta: metav1.TypeMeta{
			Kind:       "HorizontalPodAutoscaler",
			APIVersion: "autoscaling/v2",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				Kind:       "Deployment",
				Name:       "my-deployment",
				APIVersion: "apps/v1",
			},
			MinReplicas: ptr.To[int32](1),
			MaxReplicas: 3,
		},
	}
}

// roundTripV2 runs the full conversion cycle:
//
//	v2 -> internal -> v1 -> JSON v1 -> internal -> v2
func roundTripV2(hpaV2 *autoscalingv2.HorizontalPodAutoscaler) (internalFromV2, internalFromV1 *autoscaling.HorizontalPodAutoscaler, finalV2 *autoscalingv2.HorizontalPodAutoscaler, err error) {
	internalFromV2 = &autoscaling.HorizontalPodAutoscaler{}
	if err := Convert_v2_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler(hpaV2, internalFromV2, nil); err != nil {
		return nil, nil, nil, fmt.Errorf("convert v2 to internal: %w", err)
	}

	v1fromInternal := &autoscalingv1.HorizontalPodAutoscaler{}
	if err := autoscalingv1internal.Convert_autoscaling_HorizontalPodAutoscaler_To_v1_HorizontalPodAutoscaler(internalFromV2, v1fromInternal, nil); err != nil {
		return nil, nil, nil, fmt.Errorf("convert internal to v1: %w", err)
	}

	v1JSON, err := json.Marshal(v1fromInternal)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("marshal v1: %w", err)
	}
	var v1FromJSON autoscalingv1.HorizontalPodAutoscaler
	if err := json.Unmarshal(v1JSON, &v1FromJSON); err != nil {
		return nil, nil, nil, fmt.Errorf("unmarshal v1: %w", err)
	}

	internalFromV1 = &autoscaling.HorizontalPodAutoscaler{}
	if err := autoscalingv1internal.Convert_v1_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler(&v1FromJSON, internalFromV1, nil); err != nil {
		return nil, nil, nil, fmt.Errorf("convert v1 to internal: %w", err)
	}

	finalV2 = &autoscalingv2.HorizontalPodAutoscaler{}
	if err := Convert_autoscaling_HorizontalPodAutoscaler_To_v2_HorizontalPodAutoscaler(internalFromV1, finalV2, nil); err != nil {
		return nil, nil, nil, fmt.Errorf("convert internal to v2: %w", err)
	}

	finalV2.SetGroupVersionKind(SchemeGroupVersion.WithKind("HorizontalPodAutoscaler"))

	// internalFromV2: V2 -> Internal
	// internalFromV1: V2 -> Internal -> V1 -> V1 JSON -> Internal
	// finalV2: V2 -> Internal -> V1 -> V1 JSON -> Internal -> V2

	return internalFromV2, internalFromV1, finalV2, nil
}

// validateInternal returns the validation errors for an internal HPA, or nil.
func validateInternal(hpa *autoscaling.HorizontalPodAutoscaler) field.ErrorList {
	return validation.ValidateHorizontalPodAutoscaler(hpa, validation.HorizontalPodAutoscalerSpecValidationOptions{})
}

func diffInternal[T any](a, b *T) string {
	return cmp.Diff(a, b, cmpopts.EquateEmpty())
}

// requireRoundTrip asserts that hpaV2 survives the conversion cycle without
// error, that both internal forms pass validation, and that they are equal.
func requireRoundTrip(t *testing.T, hpaV2 *autoscalingv2.HorizontalPodAutoscaler) {
	t.Helper()
	internalFromV2, internalFromV1, finalV2, err := roundTripV2(hpaV2)
	require.NoError(t, err)
	require.Empty(t, validateInternal(internalFromV2), "validation of internalFromV2")
	require.Empty(t, validateInternal(internalFromV1), "validation of internalFromV1")
	require.Empty(t, diffInternal(internalFromV2, internalFromV1), "round-trip mismatch")
	require.Empty(t, diffInternal(hpaV2, finalV2), "round-trip mismatch")
}

// TestBehavior_RoundTrip_V2V1 covers Spec.Behavior shapes, which v1 carries
// through an annotation.
func TestHorizontalPodAutoscalerBehavior_RoundTrip_V2V1(t *testing.T) {
	type scalingRulesCase struct {
		name  string
		rules *autoscalingv2.HPAScalingRules
	}

	tests := []scalingRulesCase{
		{
			name:  "empty",
			rules: nil,
		},
		{
			name: "single pods policy with stabilization window",
			rules: &autoscalingv2.HPAScalingRules{
				StabilizationWindowSeconds: ptr.To[int32](300),
				Policies: []autoscalingv2.HPAScalingPolicy{{
					Type:          autoscalingv2.PodsScalingPolicy,
					Value:         2,
					PeriodSeconds: 60,
				}},
			},
		},
		{
			name: "select policy min with multiple policies",
			rules: &autoscalingv2.HPAScalingRules{
				SelectPolicy:               ptr.To(autoscalingv2.MinChangePolicySelect),
				StabilizationWindowSeconds: ptr.To[int32](0),
				Policies: []autoscalingv2.HPAScalingPolicy{
					{
						Type:          autoscalingv2.PercentScalingPolicy,
						Value:         100,
						PeriodSeconds: 30,
					},
					{
						Type:          autoscalingv2.PodsScalingPolicy,
						Value:         4,
						PeriodSeconds: 30,
					},
				},
			},
		},
		{
			name: "select policy max with multiple policies",
			rules: &autoscalingv2.HPAScalingRules{
				SelectPolicy:               ptr.To(autoscalingv2.MaxChangePolicySelect),
				StabilizationWindowSeconds: ptr.To[int32](0),
				Policies: []autoscalingv2.HPAScalingPolicy{
					{
						Type:          autoscalingv2.PercentScalingPolicy,
						Value:         100,
						PeriodSeconds: 30,
					},
					{
						Type:          autoscalingv2.PodsScalingPolicy,
						Value:         4,
						PeriodSeconds: 30,
					},
				},
			},
		},
		{
			name: "select policy disabled",
			rules: &autoscalingv2.HPAScalingRules{
				SelectPolicy: ptr.To(autoscalingv2.DisabledPolicySelect),
				Policies: []autoscalingv2.HPAScalingPolicy{{
					Type:          autoscalingv2.PercentScalingPolicy,
					Value:         100,
					PeriodSeconds: 60,
				}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hpa := baseHPA("behavior-hpa")
			hpa.Spec.Metrics = []autoscalingv2.MetricSpec{{
				Type: autoscalingv2.ResourceMetricSourceType,
				Resource: &autoscalingv2.ResourceMetricSource{
					Name: corev1.ResourceMemory,
					Target: autoscalingv2.MetricTarget{
						Type:               autoscalingv2.UtilizationMetricType,
						AverageUtilization: ptr.To[int32](80),
					},
				},
			}}

			// Test ScaleUp behavior
			hpa.Spec.Behavior = &autoscalingv2.HorizontalPodAutoscalerBehavior{
				ScaleUp: tt.rules,
			}
			requireRoundTrip(t, hpa)

			// Test ScaleDown behavior
			hpa.Spec.Behavior = &autoscalingv2.HorizontalPodAutoscalerBehavior{
				ScaleUp:   nil,
				ScaleDown: tt.rules,
			}
			requireRoundTrip(t, hpa)

			// Test both ScaleUp and ScaleDown behavior
			hpa.Spec.Behavior = &autoscalingv2.HorizontalPodAutoscalerBehavior{
				ScaleUp:   tt.rules,
				ScaleDown: tt.rules,
			}
			requireRoundTrip(t, hpa)
		})
	}
}

// TestMetricSpec_RoundTrip_V2V1 exercised one metric per HPA followed by a combination of all metrics
func TestMetricSpec_RoundTrip_V2V1(t *testing.T) {
	describedObject := autoscalingv2.CrossVersionObjectReference{
		Kind:       "Deployment",
		Name:       "my-deployment",
		APIVersion: "apps/v1",
	}
	resourceMetric := func(name corev1.ResourceName, target autoscalingv2.MetricTarget) autoscalingv2.MetricSpec {
		return autoscalingv2.MetricSpec{
			Type: autoscalingv2.ResourceMetricSourceType,
			Resource: &autoscalingv2.ResourceMetricSource{
				Name:   name,
				Target: target,
			},
		}
	}
	containerResourceMetric := func(name corev1.ResourceName, target autoscalingv2.MetricTarget) autoscalingv2.MetricSpec {
		return autoscalingv2.MetricSpec{
			Type: autoscalingv2.ContainerResourceMetricSourceType,
			ContainerResource: &autoscalingv2.ContainerResourceMetricSource{
				Name:      name,
				Container: "app",
				Target:    target,
			},
		}
	}
	utilizationTarget := func(v int32) autoscalingv2.MetricTarget {
		return autoscalingv2.MetricTarget{
			Type:               autoscalingv2.UtilizationMetricType,
			AverageUtilization: ptr.To[int32](v),
		}
	}
	averageValueTarget := func(q string) autoscalingv2.MetricTarget {
		return autoscalingv2.MetricTarget{
			Type:         autoscalingv2.AverageValueMetricType,
			AverageValue: ptr.To(resource.MustParse(q)),
		}
	}
	valueTarget := func(q string) autoscalingv2.MetricTarget {
		return autoscalingv2.MetricTarget{
			Type:  autoscalingv2.ValueMetricType,
			Value: ptr.To(resource.MustParse(q)),
		}
	}

	tests := []struct {
		name   string
		metric autoscalingv2.MetricSpec
	}{

		// Pods: AverageValue only.
		{
			name: "Pods AverageValue",
			metric: autoscalingv2.MetricSpec{
				Type: autoscalingv2.PodsMetricSourceType,
				Pods: &autoscalingv2.PodsMetricSource{
					Metric: autoscalingv2.MetricIdentifier{Name: "http_requests_per_second"},
					Target: averageValueTarget("1k"),
				},
			},
		},

		// Object: AverageValue and Value.
		{
			name: "Object AverageValue",
			metric: autoscalingv2.MetricSpec{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					Metric:          autoscalingv2.MetricIdentifier{Name: "requests-per-second"},
					DescribedObject: describedObject,
					Target:          averageValueTarget("100"),
				},
			},
		},
		{
			name: "Object Value",
			metric: autoscalingv2.MetricSpec{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricSource{
					Metric:          autoscalingv2.MetricIdentifier{Name: "errors-per-second"},
					DescribedObject: describedObject,
					Target:          valueTarget("200"),
				},
			},
		},

		// External: AverageValue and Value.
		{
			name: "External AverageValue",
			metric: autoscalingv2.MetricSpec{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricSource{
					Metric: autoscalingv2.MetricIdentifier{Name: "queue_depth", Selector: &metav1.LabelSelector{}},
					Target: averageValueTarget("50"),
				},
			},
		},
		{
			name: "External Value",
			metric: autoscalingv2.MetricSpec{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricSource{
					Metric: autoscalingv2.MetricIdentifier{Name: "queue_length", Selector: &metav1.LabelSelector{}},
					Target: valueTarget("30"),
				},
			},
		},
		// ContainerResource: CPU and Memory x Utilization and AverageValue.
		{
			name:   "ContainerResource CPU Utilization",
			metric: containerResourceMetric(corev1.ResourceCPU, utilizationTarget(60)),
		},
		{
			name:   "ContainerResource CPU AverageValue",
			metric: containerResourceMetric(corev1.ResourceCPU, averageValueTarget("250m")),
		},
		{
			name:   "ContainerResource Memory Utilization",
			metric: containerResourceMetric(corev1.ResourceMemory, utilizationTarget(70)),
		},
		{
			name:   "ContainerResource Memory AverageValue",
			metric: containerResourceMetric(corev1.ResourceMemory, averageValueTarget("128Mi")),
		},

		// Resource: CPU and Memory x Utilization and AverageValue.
		{
			name:   "Resource Memory Utilization",
			metric: resourceMetric(corev1.ResourceMemory, utilizationTarget(80)),
		},
		{
			name:   "Resource Memory AverageValue",
			metric: resourceMetric(corev1.ResourceMemory, averageValueTarget("256Mi")),
		},
		{
			name:   "Resource CPU AverageValue",
			metric: resourceMetric(corev1.ResourceCPU, averageValueTarget("500m")),
		},
		// CPU with Utilization target gets hoisted to v1's and back into the end of the list in v2
		// Which is why it needs to be last in this list. TestCPUMetricHoistedToEnd_RoundTrip_V2V1 will test that behaviour.
		{
			name:   "Resource CPU Utilization",
			metric: resourceMetric(corev1.ResourceCPU, utilizationTarget(75)),
		},
	}

	// Test each metric individually in its own HPA
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hpa := baseHPA("metric-source-hpa")
			hpa.Spec.Metrics = []autoscalingv2.MetricSpec{tt.metric}
			requireRoundTrip(t, hpa)
		})
	}

	// Combine all metrics into a single HPA
	t.Run("CombinedSingleHPA", func(t *testing.T) {
		hpa := baseHPA("metric-source-combined-hpa")
		hpa.Spec.Metrics = make([]autoscalingv2.MetricSpec, 0, len(tests))
		for _, tt := range tests {
			hpa.Spec.Metrics = append(hpa.Spec.Metrics, tt.metric)
		}
		requireRoundTrip(t, hpa)
	})
}

// TestMinMaxReplicas_RoundTrip_V2V1 covers MinReplicas and MaxReplicas
func TestMinMaxReplicas_RoundTrip_V2V1(t *testing.T) {
	tests := []struct {
		name           string
		minReplicas    *int32
		maxReplicas    int32
		scaleTargetRef autoscalingv2.CrossVersionObjectReference
	}{
		{
			name:        "defaults",
			minReplicas: ptr.To[int32](1),
			maxReplicas: 3,
		},
		{
			name:        "nil MinReplicas",
			minReplicas: nil,
			maxReplicas: 3,
		},
		{
			name:        "Min equals Max",
			minReplicas: ptr.To[int32](4),
			maxReplicas: 4,
		},
		{
			name:        "large MaxReplicas",
			minReplicas: ptr.To[int32](1),
			maxReplicas: 1000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hpa := baseHPA("replicas-target-hpa")
			hpa.Spec.Metrics = []autoscalingv2.MetricSpec{{
				Type: autoscalingv2.ResourceMetricSourceType,
				Resource: &autoscalingv2.ResourceMetricSource{
					Name: corev1.ResourceMemory,
					Target: autoscalingv2.MetricTarget{
						Type:               autoscalingv2.UtilizationMetricType,
						AverageUtilization: ptr.To[int32](75),
					},
				},
			}}
			hpa.Spec.MinReplicas = tt.minReplicas
			hpa.Spec.MaxReplicas = tt.maxReplicas

			requireRoundTrip(t, hpa)
		})
	}
}

// TestStatuses_RoundTrip_V2V1 covers various fields in the status section of an HPA
func TestStatuses_RoundTrip_V2V1(t *testing.T) {
	observedGeneration := ptr.To[int64](7)
	fixedTime := metav1.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	hpa := baseHPA("status-hpa")
	hpa.Spec.Metrics = []autoscalingv2.MetricSpec{{
		Type: autoscalingv2.ResourceMetricSourceType,
		Resource: &autoscalingv2.ResourceMetricSource{
			Name: corev1.ResourceCPU,
			Target: autoscalingv2.MetricTarget{
				Type:               autoscalingv2.UtilizationMetricType,
				AverageUtilization: ptr.To[int32](50),
			},
		},
	}}
	hpa.Status = autoscalingv2.HorizontalPodAutoscalerStatus{
		ObservedGeneration: observedGeneration,
		LastScaleTime:      &fixedTime,
		CurrentReplicas:    2,
		DesiredReplicas:    3,
		CurrentMetrics: []autoscalingv2.MetricStatus{
			{
				Type: autoscalingv2.PodsMetricSourceType,
				Pods: &autoscalingv2.PodsMetricStatus{
					Metric: autoscalingv2.MetricIdentifier{Name: "http_requests_per_second"},
					Current: autoscalingv2.MetricValueStatus{
						AverageValue: ptr.To(resource.MustParse("123")),
					},
				},
			},
			{
				Type: autoscalingv2.ExternalMetricSourceType,
				External: &autoscalingv2.ExternalMetricStatus{
					Metric: autoscalingv2.MetricIdentifier{Name: "queue_length"},
					Current: autoscalingv2.MetricValueStatus{
						Value: ptr.To(resource.MustParse("42")),
					},
				},
			},
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricStatus{
					Metric: autoscalingv2.MetricIdentifier{Name: "requests-per-second"},
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						Kind:       "Deployment",
						Name:       "my-deployment",
						APIVersion: "apps/v1",
					},
					Current: autoscalingv2.MetricValueStatus{
						AverageValue: ptr.To(resource.MustParse("0")),
					},
				},
			},
			{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricStatus{
					Metric: autoscalingv2.MetricIdentifier{Name: "requests-per-minute"},
					DescribedObject: autoscalingv2.CrossVersionObjectReference{
						Kind:       "Deployment",
						Name:       "my-deployment",
						APIVersion: "apps/v1",
					},
					Current: autoscalingv2.MetricValueStatus{
						Value: ptr.To(resource.MustParse("0")),
					},
				},
			},
			{
				Type: autoscalingv2.ContainerResourceMetricSourceType,
				ContainerResource: &autoscalingv2.ContainerResourceMetricStatus{
					Name:      corev1.ResourceMemory,
					Container: "app",
					Current: autoscalingv2.MetricValueStatus{
						AverageValue: ptr.To(resource.MustParse("256Mi")),
					},
				},
			},
		},
		Conditions: []autoscalingv2.HorizontalPodAutoscalerCondition{
			{
				Type:               autoscalingv2.ScalingActive,
				Status:             corev1.ConditionTrue,
				ObservedGeneration: observedGeneration,
				Reason:             "ScalingActive",
				Message:            "Scaling metrics available",
				LastTransitionTime: fixedTime,
			},
			{
				Type:               autoscalingv2.AbleToScale,
				Status:             corev1.ConditionTrue,
				ObservedGeneration: ptr.To[int64](3),
				Reason:             "ReadyForNewScale",
				Message:            "recommended size matches current size",
				LastTransitionTime: fixedTime,
			},
		},
	}
	requireRoundTrip(t, hpa)
}

// TestCPUMetricHoistedToEnd_RoundTrip_V2V1 documents the side effect of
// round-tripping an HPA through v1: a CPU resource metric (with
// AverageUtilization set) is hoisted into v1's TargetCPUUtilizationPercentage
// and re-emerges at the *end* of Spec.Metrics on the way back. All other
// metrics retain their relative order, carried through v1's
// autoscaling.alpha.kubernetes.io/metrics annotation.
//
// This is the reason the surrounding round-trip tests avoid mixing a CPU
// resource metric with other metrics in a single HPA.
func TestCPUMetricHoistedToEnd_RoundTrip_V2V1(t *testing.T) {
	hpa := baseHPA("cpu-hoist-hpa")
	hpa.Spec.Metrics = []autoscalingv2.MetricSpec{
		{
			Type: autoscalingv2.ResourceMetricSourceType,
			Resource: &autoscalingv2.ResourceMetricSource{
				Name: corev1.ResourceCPU,
				Target: autoscalingv2.MetricTarget{
					Type:               autoscalingv2.UtilizationMetricType,
					AverageUtilization: ptr.To[int32](80),
				},
			},
		},
		{
			Type: autoscalingv2.ExternalMetricSourceType,
			External: &autoscalingv2.ExternalMetricSource{
				Metric: autoscalingv2.MetricIdentifier{Name: "queue_length"},
				Target: autoscalingv2.MetricTarget{
					Type:  autoscalingv2.ValueMetricType,
					Value: ptr.To(resource.MustParse("30")),
				},
			},
		},
	}

	internalFromV2, internalFromV1, _, err := roundTripV2(hpa)
	require.NoError(t, err)
	require.Empty(t, validateInternal(internalFromV2), "validation of fromV2")
	require.Empty(t, validateInternal(internalFromV1), "validation of fromV1")

	metricTypes := func(metrics []autoscaling.MetricSpec) []autoscaling.MetricSourceType {
		var types []autoscaling.MetricSourceType
		for _, m := range metrics {
			types = append(types, m.Type)
		}
		return types
	}

	// Sanity: internalFromV2 preserves the v2 input order.
	require.Equal(t,
		[]autoscaling.MetricSourceType{
			autoscaling.ResourceMetricSourceType,
			autoscaling.ExternalMetricSourceType,
		},
		metricTypes(internalFromV2.Spec.Metrics),
		"internalFromV2 should preserve the v2 input order")

	// After the v1 round-trip, the CPU metric is appended at the end.
	require.Equal(t,
		[]autoscaling.MetricSourceType{
			autoscaling.ExternalMetricSourceType,
			autoscaling.ResourceMetricSourceType,
		},
		metricTypes(internalFromV1.Spec.Metrics),
		"v1 round-trip should hoist the CPU resource metric to the end of Spec.Metrics")

	last := internalFromV1.Spec.Metrics[len(internalFromV1.Spec.Metrics)-1]
	require.NotNil(t, last.Resource, "last metric should be a Resource metric")
	require.Equal(t, corev1.ResourceCPU, corev1.ResourceName(last.Resource.Name), "last metric should be CPU")
}
