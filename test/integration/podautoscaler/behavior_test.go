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
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	acappsv1 "k8s.io/client-go/applyconfigurations/apps/v1"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestHPAWithTolerance integration verifies that a HPA with configurable
// tolerances has the expected autoscaling behavior.
//
// The test setup composed of:
//  1. An HPA with 5% up and 50% down tolerances, scaling on a external metric
//     with a target of 100 per pod.
//  2. A Deployment with 35 initial pods
//  3. An external metric server advertising a metric value of 3500.
//
// The test runs the following test cases:
//  1. The metric value is increased to 3700 (a 6% increase), triggering a
//     scale-up to 37 replicas.
//  2. The metric value is increased to 3800 (a 3% increase), which should not
//     trigger a scale-up.
//  3. The metric value is decreased to 1000 (a 74% decrease), which should
//     trigger a scale-down to 10 replicas.
//  4. The metric value is decreased to 800 (a 20% decrease), which should not
//     trigger a scale-down.
func TestHPAWithTolerance(t *testing.T) {
	ctx := t.Context()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableTolerance, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	clients, metricValue := createClients(t, server.ClientConfig)
	startHPAControllerAndWaitForCaches(t, clients)

	cs := clients.apiServer
	ns := createTestNamespace(t, cs)
	deployment := createDeployment(t, cs, ns.Name, 35)

	metricValue.Store(resource.MustParse("3500"))

	// Create a HPA using an external metric and 5%/50% up/down tolerances.
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
	behavior := &autoscalingv2.HorizontalPodAutoscalerBehavior{
		ScaleUp: &autoscalingv2.HPAScalingRules{
			Tolerance: new(resource.MustParse("0.05")),
		},
		ScaleDown: &autoscalingv2.HPAScalingRules{
			Tolerance: new(resource.MustParse("0.5")),
		},
	}
	createHPA(t, cs, deployment, metricSpec, withHPAMinMaxReplicas(1, 1000), withHPABehavior(behavior))

	testcases := []struct {
		name       string
		metric     resource.Quantity
		expected   deploymentCondition
		unexpected deploymentCondition
	}{
		{
			name:     "scale_up",
			metric:   resource.MustParse("3700"),
			expected: atLeastReplicas(37),
		},
		{
			name:       "scale_up_in_tolerance",
			metric:     resource.MustParse("3800"),
			unexpected: atLeastReplicas(38),
		},
		{
			name:     "scale_down",
			metric:   resource.MustParse("1000"),
			expected: noMoreThanReplicas(10),
		},
		{
			name:       "scale_down_in_tolerance",
			metric:     resource.MustParse("800"),
			unexpected: noMoreThanReplicas(9),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			metricValue.Store(tc.metric)
			if tc.expected != nil {
				if err := waitForDeploymentCondition(ctx, cs, deployment, tc.expected); err != nil {
					t.Fatalf("Deployment did not reach the expected number of replicas: %v", err)
				}
			}
			if tc.unexpected != nil {
				if err := waitForDeploymentCondition(ctx, cs, deployment, tc.unexpected); err == nil {
					t.Fatal("Deployment reached an unexpected number of replicas")
				}
			}
			simulatePodsScheduled(t, cs, deployment)
		})
	}
}

// simulatePodsScheduled sets the `status.replicas` field of a deployment to
// the same value as `spec.replicas`, pretending that the deployment Pods were
// actually scheduled.
func simulatePodsScheduled(t *testing.T, cs *kubernetes.Clientset, d *appsv1.Deployment) {
	t.Helper()
	ud, err := cs.AppsV1().Deployments(d.Namespace).Get(t.Context(), d.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Cannot retrieve deployment: %v", err)
	}
	replicas := *ud.Spec.Replicas
	statusCfg := acappsv1.Deployment(d.Name, d.Namespace).
		WithStatus(acappsv1.DeploymentStatus().
			WithReplicas(replicas))
	opts := metav1.ApplyOptions{FieldManager: "podautoscaler-test"}
	if _, err = cs.AppsV1().Deployments(d.Namespace).ApplyStatus(t.Context(), statusCfg, opts); err != nil {
		t.Fatalf("Cannot set deployment status: %v", err)
	}

	// Wait to ensure HPA controller takes the update into account
	time.Sleep(hpaControllerResyncPeriod * 2)
}
