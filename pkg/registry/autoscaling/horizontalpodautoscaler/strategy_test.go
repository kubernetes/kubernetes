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

package horizontalpodautoscaler

import (
	"context"
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/pointer"
)

func makeTestContainerMetricsHPA(hasContainerMetric bool) *autoscaling.HorizontalPodAutoscaler {
	testHPA := &autoscaling.HorizontalPodAutoscaler{
		Spec: autoscaling.HorizontalPodAutoscalerSpec{
			Metrics: []autoscaling.MetricSpec{},
		},
	}
	if hasContainerMetric {
		testHPA.Spec.Metrics = append(testHPA.Spec.Metrics, autoscaling.MetricSpec{
			Type: autoscaling.ContainerResourceMetricSourceType,
			ContainerResource: &autoscaling.ContainerResourceMetricSource{
				Name:      core.ResourceCPU,
				Container: "test-container",
				Target: autoscaling.MetricTarget{
					Type:               autoscaling.UtilizationMetricType,
					AverageUtilization: pointer.Int32Ptr(30),
				},
			},
		})
	}
	return testHPA
}

func TestCreateWithFeatureEnabled(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAContainerMetrics, true)()
	testHPA := makeTestContainerMetricsHPA(true)
	Strategy.PrepareForCreate(context.Background(), testHPA)
	if testHPA.Spec.Metrics[0].ContainerResource == nil {
		t.Errorf("container metrics was set to nil")
	}
}

func TestCreateWithFeatureDisabled(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAContainerMetrics, false)()
	testHPA := makeTestContainerMetricsHPA(true)
	Strategy.PrepareForCreate(context.Background(), testHPA)
	if testHPA.Spec.Metrics[0].ContainerResource != nil {
		t.Errorf("container metrics is not nil")
	}
}

func TestAutoscalerStatusStrategy_PrepareForUpdate(t *testing.T) {
	for _, tc := range []struct {
		name           string
		featureEnabled bool
		old            bool
		expectedNew    bool
	}{
		{
			name:           "feature disabled with existing container metrics",
			featureEnabled: false,
			old:            true,
			expectedNew:    true,
		},
		{
			name:           "feature enabled with no container metrics",
			featureEnabled: true,
			old:            false,
			expectedNew:    true,
		},
		{
			name:           "feature enabled with existing container metrics",
			featureEnabled: true,
			old:            true,
			expectedNew:    true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAContainerMetrics, tc.featureEnabled)()
			oldHPA := makeTestContainerMetricsHPA(tc.old)
			newHPA := makeTestContainerMetricsHPA(true)
			Strategy.PrepareForUpdate(context.Background(), newHPA, oldHPA)
			if tc.expectedNew && newHPA.Spec.Metrics[0].ContainerResource == nil {
				t.Errorf("container metric source is nil")
			}
			if !tc.expectedNew && newHPA.Spec.Metrics[0].ContainerResource != nil {
				t.Errorf("container metric source is not nil")
			}
		})
	}
}
