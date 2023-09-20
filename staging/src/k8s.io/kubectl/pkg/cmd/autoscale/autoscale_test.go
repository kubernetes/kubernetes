/*
Copyright 2014 The Kubernetes Authors.

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

package autoscale

import (
	"testing"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestCreateHorizontalPodAutoscaler(t *testing.T) {
	refName := "test-deployment"
	mapping := &schema.GroupVersionKind{
		Group:   "apps",
		Version: "v1",
		Kind:    "Deployment",
	}

	t.Run("Create HPA with defaults", func(t *testing.T) {
		options := AutoscaleOptions{}
		hpa := options.createHorizontalPodAutoscaler(refName, mapping)

		// Checking HPA name
		expectedName := refName
		if hpa.ObjectMeta.Name != expectedName {
			t.Errorf("Expected HPA name to be %s, but got %s", expectedName, hpa.ObjectMeta.Name)
		}

		// Checking ScaleTargetRef
		expectedRef := autoscalingv2.CrossVersionObjectReference{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       refName,
		}
		if hpa.Spec.ScaleTargetRef != expectedRef {
			t.Errorf("Expected ScaleTargetRef to be %+v, but got %+v", expectedRef, hpa.Spec.ScaleTargetRef)
		}

		// Checking MaxReplicas
		if hpa.Spec.MaxReplicas != 0 {
			t.Errorf("Expected MaxReplicas to be 0, but got %d", hpa.Spec.MaxReplicas)
		}

		// Checking MinReplicas (should not be set)
		if hpa.Spec.MinReplicas != nil {
			t.Errorf("Expected MinReplicas to be nil, but got %+v", hpa.Spec.MinReplicas)
		}

		// Checking Metrics (should not be set)
		if len(hpa.Spec.Metrics) != 0 {
			t.Errorf("Expected Metrics to be empty, but got %+v", hpa.Spec.Metrics)
		}
	})

	t.Run("Create HPA with custom values", func(t *testing.T) {
		options := AutoscaleOptions{
			Min:        2,
			Max:        5,
			CPUPercent: 80,
		}
		hpa := options.createHorizontalPodAutoscaler(refName, mapping)

		// Checking MaxReplicas
		expectedMaxReplicas := int32(5)
		if hpa.Spec.MaxReplicas != expectedMaxReplicas {
			t.Errorf("Expected MaxReplicas to be %d, but got %d", expectedMaxReplicas, hpa.Spec.MaxReplicas)
		}

		// Checking MinReplicas
		expectedMinReplicas := int32(2)
		if *hpa.Spec.MinReplicas != expectedMinReplicas {
			t.Errorf("Expected MinReplicas to be %d, but got %d", expectedMinReplicas, *hpa.Spec.MinReplicas)
		}

		// Checking Metrics
		if len(hpa.Spec.Metrics) != 1 {
			t.Errorf("Expected 1 metric, but got %d", len(hpa.Spec.Metrics))
		}

		// Checking the CPU metric
		cpuMetric := hpa.Spec.Metrics[0]
		if cpuMetric.Type != autoscalingv2.ResourceMetricSourceType {
			t.Errorf("Expected metric type to be ResourceMetricSourceType, but got %s", cpuMetric.Type)
		}
		if cpuMetric.Resource.Name != corev1.ResourceCPU {
			t.Errorf("Expected resource name to be %s, but got %s", corev1.ResourceCPU, cpuMetric.Resource.Name)
		}
		if cpuMetric.Resource.Target.AverageUtilization == nil || *cpuMetric.Resource.Target.AverageUtilization != int32(80) {
			t.Errorf("Expected CPU utilization target to be 80, but got %+v", cpuMetric.Resource.Target.AverageUtilization)
		}
	})
}
