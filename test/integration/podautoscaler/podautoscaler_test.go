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

package podautoscaler

import (
	"strings"
	"testing"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

// TestHPAScaleUpToMin integration creates a Deployment and HPA, then verifies the HPA exists and
// is reconciled, by bringing the replicas of the Deployment within the HPA's min and max range.
// Given that the metrics aren't mocked, it should go up to min only.
func TestHPAScaleUpToMin(t *testing.T) {
	ctx := t.Context()

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	clients, _ := createClients(t, server.ClientConfig)
	startHPAControllerAndWaitForCaches(t, clients)

	cs := clients.apiServer
	ns := createTestNamespace(t, cs)
	deployment := createDeployment(t, cs, ns.Name, 1)

	metricSpec := autoscalingv2.MetricSpec{
		Type: autoscalingv2.ResourceMetricSourceType,
		Resource: &autoscalingv2.ResourceMetricSource{
			Name: "cpu",
			Target: autoscalingv2.MetricTarget{
				Type:               autoscalingv2.UtilizationMetricType,
				AverageUtilization: ptr.To[int32](50),
			},
		},
	}

	createHPA(t, cs, deployment, metricSpec, withHPAMinMaxReplicas(2, 10))

	err := waitForDeploymentCondition(ctx, cs, deployment, atLeastReplicas(2))
	if err != nil {
		t.Fatalf("HPA did not reconcile: %v", err)
	}
}

// TestHPAScaleToZero verifies that HPA can scale up to five, then down to zero and back again, using an external metric.
func TestHPAScaleToZero(t *testing.T) {
	ctx := t.Context()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAScaleToZero, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	clients, metricValue := createClients(t, server.ClientConfig)
	startHPAControllerAndWaitForCaches(t, clients)

	cs := clients.apiServer
	ns := createTestNamespace(t, cs)
	deployment := createDeployment(t, cs, ns.Name, 1)

	// Create HPA using an external metric.
	metricSpec := autoscalingv2.MetricSpec{
		Type: autoscalingv2.ExternalMetricSourceType,
		External: &autoscalingv2.ExternalMetricSource{
			Metric: autoscalingv2.MetricIdentifier{
				Name: externalMetricName,
			},
			Target: autoscalingv2.MetricTarget{
				Type:         autoscalingv2.ValueMetricType,
				AverageValue: new(resource.MustParse("100")),
			},
		},
	}
	createHPA(t, cs, deployment, metricSpec, withHPAMinMaxReplicas(0, 10))

	// Phase 1: metric=500, expect scale up to 5 pods.
	metricValue.Store(resource.MustParse("500"))
	err := waitForDeploymentCondition(ctx, cs, deployment, atLeastReplicas(5))
	if err != nil {
		t.Fatalf("Phase 1: HPA did not scale up: %v", err)
	}

	// Verify the ScaledToZero condition is set to false on the HPA.
	gotHPA, err := cs.AutoscalingV2().HorizontalPodAutoscalers(ns.Name).Get(ctx, hpaName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Phase 3: failed to get HPA: %v", err)
	}
	foundScaledToZero := false
	for _, c := range gotHPA.Status.Conditions {
		if c.Type == autoscalingv2.ScaledToZero {
			foundScaledToZero = true
			if c.Status != corev1.ConditionFalse {
				t.Fatalf("Phase 1: ScaledToZero condition status = %v, want False", c.Status)
			}
			break
		}
	}
	if !foundScaledToZero {
		t.Fatal("Phase 1: ScaledToZero condition not found on HPA status")
	}

	// Phase 2: metric=0, expect scale down to 0 and ScaledToZero condition.
	metricValue.Store(resource.MustParse("0"))
	err = waitForDeploymentCondition(ctx, cs, deployment, equalReplicas(0))
	if err != nil {
		t.Fatalf("Phase 2: HPA did not scale to zero: %v", err)
	}

	// Verify the ScaledToZero condition is set on the HPA.
	gotHPA, err = cs.AutoscalingV2().HorizontalPodAutoscalers(ns.Name).Get(ctx, hpaName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Phase 2: failed to get HPA: %v", err)
	}
	foundScaledToZero = false
	for _, c := range gotHPA.Status.Conditions {
		if c.Type == autoscalingv2.ScaledToZero {
			foundScaledToZero = true
			if c.Status != corev1.ConditionTrue {
				t.Fatalf("Phase 2: ScaledToZero condition status = %v, want True", c.Status)
			}
			break
		}
	}
	if !foundScaledToZero {
		t.Fatal("Phase 2: ScaledToZero condition not found on HPA status")
	}

	// Phase 3: metric=400, expect scale up again (replicas >= 4).
	metricValue.Store(resource.MustParse("400"))
	err = waitForDeploymentCondition(ctx, cs, deployment, atLeastReplicas(4))
	if err != nil {
		t.Fatalf("Phase 3: HPA did not scale back up: %v", err)
	}

	// Verify the ScaledToZero condition is set to false on the HPA.
	gotHPA, err = cs.AutoscalingV2().HorizontalPodAutoscalers(ns.Name).Get(ctx, hpaName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Phase 3: failed to get HPA: %v", err)
	}
	foundScaledToZero = false
	for _, c := range gotHPA.Status.Conditions {
		if c.Type == autoscalingv2.ScaledToZero {
			foundScaledToZero = true
			if c.Status != corev1.ConditionFalse {
				t.Fatalf("Phase 3: ScaledToZero condition status = %v, want False", c.Status)
			}
			break
		}
	}
	if !foundScaledToZero {
		t.Fatal("Phase 3: ScaledToZero condition not found on HPA status")
	}
}

// TestHPAScaleToZeroRejected verifies that scaling to zero is limited to object
// and external metrics and is guarded by the HPAScaleToZero feature gate. In
// both cases below the API server must reject an HPA with minReplicas: 0:
//   - a resource (CPU) metric is never eligible to scale to zero, even with the
//     feature gate enabled, and
//   - while the feature gate is disabled, minReplicas: 0 is forbidden even for
//     an otherwise eligible external metric.
func TestHPAScaleToZeroRejected(t *testing.T) {
	cpuMetric := autoscalingv2.MetricSpec{
		Type: autoscalingv2.ResourceMetricSourceType,
		Resource: &autoscalingv2.ResourceMetricSource{
			Name: corev1.ResourceCPU,
			Target: autoscalingv2.MetricTarget{
				Type:               autoscalingv2.UtilizationMetricType,
				AverageUtilization: ptr.To[int32](50),
			},
		},
	}
	externalMetric := autoscalingv2.MetricSpec{
		Type: autoscalingv2.ExternalMetricSourceType,
		External: &autoscalingv2.ExternalMetricSource{
			Metric: autoscalingv2.MetricIdentifier{
				Name: externalMetricName,
			},
			Target: autoscalingv2.MetricTarget{
				Type:         autoscalingv2.ValueMetricType,
				AverageValue: new(resource.MustParse("100")),
			},
		},
	}

	testCases := []struct {
		name               string
		featureGateEnabled bool
		metricSpec         autoscalingv2.MetricSpec
		wantErrSubstring   string
	}{
		{
			name:               "resource metric is not eligible to scale to zero",
			featureGateEnabled: true,
			metricSpec:         cpuMetric,
			wantErrSubstring:   "at least one Object or External metric",
		},
		{
			name:               "minReplicas zero is forbidden while the feature gate is disabled",
			featureGateEnabled: false,
			metricSpec:         externalMetric,
			wantErrSubstring:   "minReplicas",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAScaleToZero, tc.featureGateEnabled)

			server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
			t.Cleanup(server.TearDownFn)
			clients, _ := createClients(t, server.ClientConfig)

			cs := clients.apiServer
			ns := createTestNamespace(t, cs)
			deployment := createDeployment(t, cs, ns.Name, 1)

			hpa := newHPA(deployment, tc.metricSpec, withHPAMinMaxReplicas(0, 10))
			_, err := cs.AutoscalingV2().HorizontalPodAutoscalers(ns.Name).Create(t.Context(), hpa, metav1.CreateOptions{})
			if err == nil {
				t.Fatal("expected creating an HPA with minReplicas: 0 to be rejected, but it succeeded")
			}
			if !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), tc.wantErrSubstring) {
				t.Fatalf("expected an invalid error containing %q, got: %v", tc.wantErrSubstring, err)
			}
		})
	}
}
