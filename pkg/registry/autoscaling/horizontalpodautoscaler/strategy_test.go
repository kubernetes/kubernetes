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
