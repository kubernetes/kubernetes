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

package fuzzer

import (
	"sigs.k8s.io/randfill"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/pointer"
)

// Funcs returns the fuzzer functions for the autoscaling api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(s *autoscaling.ScaleStatus, c randfill.Continue) {
			c.FillNoCustom(s) // fuzz self without calling this function again

			// ensure we have a valid selector
			metaSelector := &metav1.LabelSelector{}
			c.Fill(metaSelector)
			labelSelector, _ := metav1.LabelSelectorAsSelector(metaSelector)
			s.Selector = labelSelector.String()
		},
		func(s *autoscaling.HorizontalPodAutoscalerSpec, c randfill.Continue) {
			c.FillNoCustom(s) // fuzz self without calling this function again
			s.MinReplicas = pointer.Int32(c.Rand.Int31())

			randomQuantity := func() resource.Quantity {
				var q resource.Quantity
				c.Fill(&q)
				// precalc the string for benchmarking purposes
				_ = q.String()
				return q
			}

			var podMetricID autoscaling.MetricIdentifier
			var objMetricID autoscaling.MetricIdentifier
			c.Fill(&podMetricID)
			c.Fill(&objMetricID)

			targetUtilization := int32(c.Uint64())
			averageValue := randomQuantity()
			s.Metrics = []autoscaling.MetricSpec{
				{
					Type: autoscaling.PodsMetricSourceType,
					Pods: &autoscaling.PodsMetricSource{
						Metric: podMetricID,
						Target: autoscaling.MetricTarget{
							Type:         autoscaling.AverageValueMetricType,
							AverageValue: &averageValue,
						},
					},
				},
				{
					Type: autoscaling.ObjectMetricSourceType,
					Object: &autoscaling.ObjectMetricSource{
						Metric: objMetricID,
						Target: autoscaling.MetricTarget{
							Type:  autoscaling.ValueMetricType,
							Value: &averageValue,
						},
					},
				},
				{
					Type: autoscaling.ResourceMetricSourceType,
					Resource: &autoscaling.ResourceMetricSource{
						Name: api.ResourceCPU,
						Target: autoscaling.MetricTarget{
							Type:               autoscaling.UtilizationMetricType,
							AverageUtilization: &targetUtilization,
						},
					},
				},
			}
			stabilizationWindow := int32(c.Uint64())
			maxPolicy := autoscaling.MaxPolicySelect
			minPolicy := autoscaling.MinPolicySelect
			s.Behavior = &autoscaling.HorizontalPodAutoscalerBehavior{
				ScaleUp: &autoscaling.HPAScalingRules{
					StabilizationWindowSeconds: &stabilizationWindow,
					SelectPolicy:               &maxPolicy,
					Policies: []autoscaling.HPAScalingPolicy{
						{
							Type:          autoscaling.PodsScalingPolicy,
							Value:         int32(c.Uint64()),
							PeriodSeconds: int32(c.Uint64()),
						},
						{
							Type:          autoscaling.PercentScalingPolicy,
							Value:         int32(c.Uint64()),
							PeriodSeconds: int32(c.Uint64()),
						},
					},
				},
				ScaleDown: &autoscaling.HPAScalingRules{
					StabilizationWindowSeconds: &stabilizationWindow,
					SelectPolicy:               &minPolicy,
					Policies: []autoscaling.HPAScalingPolicy{
						{
							Type:          autoscaling.PodsScalingPolicy,
							Value:         int32(c.Uint64()),
							PeriodSeconds: int32(c.Uint64()),
						},
						{
							Type:          autoscaling.PercentScalingPolicy,
							Value:         int32(c.Uint64()),
							PeriodSeconds: int32(c.Uint64()),
						},
					},
				},
			}
		},
		func(s *autoscaling.HorizontalPodAutoscalerStatus, c randfill.Continue) {
			c.FillNoCustom(s) // fuzz self without calling this function again
			randomQuantity := func() resource.Quantity {
				var q resource.Quantity
				c.Fill(&q)
				// precalc the string for benchmarking purposes
				_ = q.String()
				return q
			}
			averageValue := randomQuantity()
			currentUtilization := int32(c.Uint64())
			s.CurrentMetrics = []autoscaling.MetricStatus{
				{
					Type: autoscaling.PodsMetricSourceType,
					Pods: &autoscaling.PodsMetricStatus{
						Metric: autoscaling.MetricIdentifier{
							Name: c.String(0),
						},
						Current: autoscaling.MetricValueStatus{
							AverageValue: &averageValue,
						},
					},
				},
				{
					Type: autoscaling.ResourceMetricSourceType,
					Resource: &autoscaling.ResourceMetricStatus{
						Name: api.ResourceCPU,
						Current: autoscaling.MetricValueStatus{
							AverageUtilization: &currentUtilization,
							AverageValue:       &averageValue,
						},
					},
				},
			}
		},
	}
}
