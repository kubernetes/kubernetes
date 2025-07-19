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

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/autoscaling/validation"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

type toleranceSet bool
type zeroMinReplicasSet bool

const (
	withTolerance    toleranceSet       = true
	withoutTolerance                    = false
	zeroMinReplicas  zeroMinReplicasSet = true
	oneMinReplicas                      = false
)

func TestPrepareForCreateConfigurableToleranceEnabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableTolerance, true)
	hpa := prepareHPA(oneMinReplicas, withTolerance)

	Strategy.PrepareForCreate(context.Background(), &hpa)
	if hpa.Spec.Behavior.ScaleUp.Tolerance == nil {
		t.Error("Expected tolerance field, got none")
	}
}

func TestPrepareForCreateConfigurableToleranceDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableTolerance, false)
	hpa := prepareHPA(oneMinReplicas, withTolerance)

	Strategy.PrepareForCreate(context.Background(), &hpa)
	if hpa.Spec.Behavior.ScaleUp.Tolerance != nil {
		t.Errorf("Expected tolerance field wiped out, got %v", hpa.Spec.Behavior.ScaleUp.Tolerance)
	}
}

func TestPrepareForUpdateConfigurableToleranceEnabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableTolerance, true)
	newHPA := prepareHPA(oneMinReplicas, withTolerance)
	oldHPA := prepareHPA(oneMinReplicas, withTolerance)

	Strategy.PrepareForUpdate(context.Background(), &newHPA, &oldHPA)
	if newHPA.Spec.Behavior.ScaleUp.Tolerance == nil {
		t.Error("Expected tolerance field, got none")
	}
}

func TestPrepareForUpdateConfigurableToleranceDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableTolerance, false)
	newHPA := prepareHPA(oneMinReplicas, withTolerance)
	oldHPA := prepareHPA(oneMinReplicas, withoutTolerance)

	Strategy.PrepareForUpdate(context.Background(), &newHPA, &oldHPA)
	if newHPA.Spec.Behavior.ScaleUp.Tolerance != nil {
		t.Errorf("Expected tolerance field wiped out, got %v", newHPA.Spec.Behavior.ScaleUp.Tolerance)
	}

	newHPA = prepareHPA(oneMinReplicas, withTolerance)
	oldHPA = prepareHPA(oneMinReplicas, withTolerance)
	Strategy.PrepareForUpdate(context.Background(), &newHPA, &oldHPA)
	if newHPA.Spec.Behavior.ScaleUp.Tolerance == nil {
		t.Errorf("Expected tolerance field not wiped out, got nil")
	}
}

func TestValidateOptionsScaleToZeroEnabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAScaleToZero, true)
	oneReplicasHPA := prepareHPA(oneMinReplicas, withoutTolerance)

	opts := validationOptionsForHorizontalPodAutoscaler(&oneReplicasHPA, &oneReplicasHPA)
	if opts.MinReplicasLowerBound != 0 {
		t.Errorf("Expected zero minReplicasLowerBound, got %v", opts.MinReplicasLowerBound)
	}
}

func TestValidateOptionsScaleToZeroDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAScaleToZero, false)
	zeroReplicasHPA := prepareHPA(zeroMinReplicas, withoutTolerance)
	oneReplicasHPA := prepareHPA(oneMinReplicas, withoutTolerance)

	// MinReplicas should be 0 despite the gate being disabled since the old HPA
	// had MinReplicas set to 0 already.
	opts := validationOptionsForHorizontalPodAutoscaler(&zeroReplicasHPA, &zeroReplicasHPA)
	if opts.MinReplicasLowerBound != 0 {
		t.Errorf("Expected zero minReplicasLowerBound, got %v", opts.MinReplicasLowerBound)
	}

	opts = validationOptionsForHorizontalPodAutoscaler(&zeroReplicasHPA, &oneReplicasHPA)
	if opts.MinReplicasLowerBound == 0 {
		t.Errorf("Expected non-zero minReplicasLowerBound, got 0")
	}
}

func TestValidationOptionsForHorizontalPodAutoscaler(t *testing.T) {
	hpa := func(minReplicas int32, scaleTargetAPIVersion string, scaleTargetKind string, metrics []autoscaling.MetricSpec) *autoscaling.HorizontalPodAutoscaler {
		return &autoscaling.HorizontalPodAutoscaler{
			ObjectMeta: metav1.ObjectMeta{Name: "myhpa", Namespace: "default"},
			Spec: autoscaling.HorizontalPodAutoscalerSpec{
				MinReplicas: &minReplicas,
				MaxReplicas: 5,
				ScaleTargetRef: autoscaling.CrossVersionObjectReference{
					Kind:       scaleTargetKind,
					Name:       "name",
					APIVersion: scaleTargetAPIVersion,
				},
				Metrics: metrics,
			},
		}
	}

	objectMetric := func(apiVersion string) autoscaling.MetricSpec {
		return autoscaling.MetricSpec{
			Type: autoscaling.ObjectMetricSourceType,
			Object: &autoscaling.ObjectMetricSource{
				DescribedObject: autoscaling.CrossVersionObjectReference{
					Kind:       "Deployment",
					Name:       "mydeployment",
					APIVersion: apiVersion,
				},
				Metric: autoscaling.MetricIdentifier{Name: "my-metric"},
				Target: autoscaling.MetricTarget{Type: autoscaling.ValueMetricType, Value: resource.NewQuantity(10, resource.DecimalSI)},
			},
		}
	}

	podsMetric := autoscaling.MetricSpec{
		Type: autoscaling.PodsMetricSourceType,
		Pods: &autoscaling.PodsMetricSource{
			Metric: autoscaling.MetricIdentifier{Name: "my-metric"},
			Target: autoscaling.MetricTarget{Type: autoscaling.AverageValueMetricType, AverageValue: resource.NewQuantity(10, resource.DecimalSI)},
		},
	}

	testCases := []struct {
		name                               string
		newHPA                             *autoscaling.HorizontalPodAutoscaler
		oldHPA                             *autoscaling.HorizontalPodAutoscaler
		scaleToZeroEnabled                 bool
		expectMinReplicasLower             int32
		expectScaleTargetRefValidationOpts validation.CrossVersionObjectReferenceValidationOptions
		expectMetricsValidationOpts        validation.CrossVersionObjectReferenceValidationOptions
	}{
		// MinReplicasLowerBound tests
		{
			name:                   "scale to zero disabled, no old hpa",
			newHPA:                 hpa(1, "apps/v1", "Deployment", nil),
			oldHPA:                 nil,
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 1,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: false,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
		{
			name:                   "scale to zero disabled, old hpa has minReplicas=1",
			newHPA:                 hpa(1, "apps/v1", "Deployment", nil),
			oldHPA:                 hpa(1, "apps/v1", "Deployment", nil),
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 1,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: true, AllowEmptyAPIGroup: false,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
		{
			name:                   "scale to zero disabled, old hpa has minReplicas=0",
			newHPA:                 hpa(0, "apps/v1", "Deployment", nil),
			oldHPA:                 hpa(0, "apps/v1", "Deployment", nil),
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 0,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: true, AllowEmptyAPIGroup: false,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
		{
			name:                   "scale to zero enabled",
			newHPA:                 hpa(0, "apps/v1", "Deployment", nil),
			oldHPA:                 nil,
			scaleToZeroEnabled:     true,
			expectMinReplicasLower: 0,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: false,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
		// ScaleTargetRefValidationOptions tests
		{
			name:                   "ReplicationController with the legacy API Version",
			newHPA:                 hpa(1, "", "ReplicationController", nil),
			oldHPA:                 nil,
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 1,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
		{
			name:                   "scale target ref api version changed",
			newHPA:                 hpa(1, "apps/v1", "Deployment", nil),
			oldHPA:                 hpa(1, "extensions/v1beta1", "RandomCR", nil),
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 1,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: false,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
		{
			name:                   "scale target ref api version unchanged",
			newHPA:                 hpa(1, "apps/v1", "", nil),
			oldHPA:                 hpa(1, "apps/v1", "", nil),
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 1,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: true, AllowEmptyAPIGroup: false,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
		{
			name:                   "scale target ref api and Kind are changed to ReplicationController",
			newHPA:                 hpa(1, "", "ReplicationController", nil),
			oldHPA:                 hpa(1, "apps/v1", "Deployment", nil),
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 1,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
		{
			name:                   "Kind changed",
			newHPA:                 hpa(1, "apps/v1", "CronJobs", nil),
			oldHPA:                 hpa(1, "apps/v1", "Deployment", nil),
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 1,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: false,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
		// MetricsValidationOptions tests
		{
			name:                   "no metrics",
			newHPA:                 hpa(1, "apps/v1", "Deployment", nil),
			oldHPA:                 nil,
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 1,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: false,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
		{
			name:                   "non-object metric",
			newHPA:                 hpa(1, "", "", []autoscaling.MetricSpec{podsMetric}),
			oldHPA:                 nil,
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 1,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: false,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
		{
			name:                   "new object metric with valid api version",
			newHPA:                 hpa(1, "", "", []autoscaling.MetricSpec{objectMetric("apps/v1")}),
			oldHPA:                 nil,
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 1,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: false,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
		{
			name:                   "old object metric with invalid api version",
			newHPA:                 hpa(1, "apps/v1", "Deployment", []autoscaling.MetricSpec{objectMetric("apps/v1/v3")}),
			oldHPA:                 hpa(2, "apps/v1", "Deployment", []autoscaling.MetricSpec{objectMetric("apps/v1/v2")}),
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 1,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: true, AllowEmptyAPIGroup: false,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: true, AllowEmptyAPIGroup: true,
			},
		},
		{
			name:                   "new object metric with invalid api version",
			newHPA:                 hpa(1, "", "", []autoscaling.MetricSpec{objectMetric("apps/v1/v2")}),
			oldHPA:                 nil,
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 1,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: false,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
		{
			name: "nil object Metrics",
			newHPA: hpa(1, "", "", []autoscaling.MetricSpec{{
				Type:   autoscaling.ObjectMetricSourceType,
				Object: nil,
			}}),
			oldHPA:                 nil,
			scaleToZeroEnabled:     false,
			expectMinReplicasLower: 1,
			expectScaleTargetRefValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: false,
			},
			expectMetricsValidationOpts: validation.CrossVersionObjectReferenceValidationOptions{
				AllowInvalidAPIVersion: false, AllowEmptyAPIGroup: true,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAScaleToZero, tc.scaleToZeroEnabled)
			opts := validationOptionsForHorizontalPodAutoscaler(tc.newHPA, tc.oldHPA)

			if opts.MinReplicasLowerBound != tc.expectMinReplicasLower {
				t.Errorf("expected MinReplicasLowerBound %d, got %d", tc.expectMinReplicasLower, opts.MinReplicasLowerBound)
			}
			if opts.ScaleTargetRefValidationOptions != tc.expectScaleTargetRefValidationOpts {
				t.Errorf("expected ScaleTargetRefValidationOptions %v, got %v", tc.expectScaleTargetRefValidationOpts, opts.ScaleTargetRefValidationOptions)
			}
			if opts.ObjectMetricsValidationOptions != tc.expectMetricsValidationOpts {
				t.Errorf("expected ObjectMetricsValidationOptions %v, got %v", tc.expectMetricsValidationOpts, opts.ObjectMetricsValidationOptions)
			}

		})
	}
}

func prepareHPA(hasZeroMinReplicas zeroMinReplicasSet, hasTolerance toleranceSet) autoscaling.HorizontalPodAutoscaler {
	tolerance := ptr.To(resource.MustParse("0.1"))
	if !hasTolerance {
		tolerance = nil
	}

	minReplicas := int32(0)
	if !hasZeroMinReplicas {
		minReplicas = 1
	}

	return autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "myautoscaler",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "1",
		},
		Spec: autoscaling.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscaling.CrossVersionObjectReference{
				Kind: "ReplicationController",
				Name: "myrc",
			},
			MinReplicas: &minReplicas,
			MaxReplicas: 5,
			Metrics: []autoscaling.MetricSpec{{
				Type: autoscaling.ResourceMetricSourceType,
				Resource: &autoscaling.ResourceMetricSource{
					Name: api.ResourceCPU,
					Target: autoscaling.MetricTarget{
						Type:               autoscaling.UtilizationMetricType,
						AverageUtilization: ptr.To(int32(70)),
					},
				},
			}},
			Behavior: &autoscaling.HorizontalPodAutoscalerBehavior{
				ScaleUp: &autoscaling.HPAScalingRules{
					Tolerance: tolerance,
				},
			},
		},
	}
}
