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
	fuzz "github.com/google/gofuzz"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// Funcs returns the fuzzer functions for the autoscaling api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(s *autoscaling.ScaleStatus, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again

			// ensure we have a valid selector
			labelSelector, _ := metav1.LabelSelectorAsSelector(randomSelector(c))
			s.Selector = labelSelector.String()
		},
		func(s *autoscaling.HorizontalPodAutoscalerSpec, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			minReplicas := int32(c.Rand.Int31())
			s.MinReplicas = &minReplicas

			targetUtilization := int32(c.RandUint64())
			s.Metrics = []autoscaling.MetricSpec{
				{
					Type: autoscaling.PodsMetricSourceType,
					Pods: &autoscaling.PodsMetricSource{
						MetricName:         c.RandString(),
						TargetAverageValue: randomQuantity(c),
					},
				},
				{
					Type: autoscaling.ResourceMetricSourceType,
					Resource: &autoscaling.ResourceMetricSource{
						Name: api.ResourceCPU,
						TargetAverageUtilization: &targetUtilization,
					},
				},
			}
		},
		func(s *autoscaling.HorizontalPodAutoscalerStatus, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			currentUtilization := int32(c.RandUint64())
			s.CurrentMetrics = []autoscaling.MetricStatus{
				{
					Type: autoscaling.PodsMetricSourceType,
					Pods: &autoscaling.PodsMetricStatus{
						MetricName:          c.RandString(),
						CurrentAverageValue: randomQuantity(c),
					},
				},
				{
					Type: autoscaling.ResourceMetricSourceType,
					Resource: &autoscaling.ResourceMetricStatus{
						Name: api.ResourceCPU,
						CurrentAverageUtilization: &currentUtilization,
					},
				},
			}
		},
		func(s *autoscaling.VerticalPodAutoscalerSpec, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			s.Selector = &metav1.LabelSelector{}
			updateModes := []autoscaling.UpdateMode{
				autoscaling.UpdateModeOff,
				autoscaling.UpdateModeInitial,
				autoscaling.UpdateModeRecreate,
				autoscaling.UpdateModeAuto,
			}
			scalingModes := []autoscaling.ContainerScalingMode{
				autoscaling.ContainerScalingModeAuto,
				autoscaling.ContainerScalingModeOff,
			}
			s.UpdatePolicy = &autoscaling.PodUpdatePolicy{UpdateMode: &updateModes[c.Rand.Intn(len(updateModes))]}
			if s.ResourcePolicy != nil {
				for i := range s.ResourcePolicy.ContainerPolicies {
					s.ResourcePolicy.ContainerPolicies[i].Mode = &scalingModes[c.Rand.Intn(len(scalingModes))]
					s.ResourcePolicy.ContainerPolicies[i].MinAllowed = randomResources(c)
					s.ResourcePolicy.ContainerPolicies[i].MaxAllowed = randomResources(c)
				}
			}
		},
		func(s *autoscaling.VerticalPodAutoscalerStatus, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			if s.Recommendation == nil {
				s.Recommendation = &autoscaling.RecommendedPodResources{}
			}
			if len(s.Recommendation.ContainerRecommendations) == 0 {
				s.Recommendation.ContainerRecommendations = []autoscaling.RecommendedContainerResources{
					{
						ContainerName: c.RandString(),
						Target:        randomResources(c),
						LowerBound:    randomResources(c),
						UpperBound:    randomResources(c),
					},
				}
			}
		},
	}
}

func randomSelector(c fuzz.Continue) *metav1.LabelSelector {
	metaSelector := &metav1.LabelSelector{}
	c.Fuzz(metaSelector)
	return metaSelector
}

func randomQuantity(c fuzz.Continue) resource.Quantity {
	var q resource.Quantity
	c.Fuzz(&q)
	// precalc the string for benchmarking purposes
	_ = q.String()
	return q
}

func randomResources(c fuzz.Continue) api.ResourceList {
	return api.ResourceList{
		api.ResourceCPU:    randomQuantity(c),
		api.ResourceMemory: randomQuantity(c),
	}
}
