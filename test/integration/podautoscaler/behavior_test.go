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
	"fmt"
	"testing"
	"time"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
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
// The test runs in multiple phases:
//   - Phase 1: the metric value is increased to 3700 (a 6% increase),
//     triggering a scale-up to 37 replicas.
//   - Phase 2: the metric value is increased to 3800 (a 3% increase), which
//     should not trigger a scale-up.
//   - Phase 3: the metric value is decreased to 1000 (a 74% decrease), which
//     should trigger a scale-down to 10 replicas.
//   - Phase 4: the metric value is decreased to 800 (a 20% decrease), which
//     should not trigger a scale-down.
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

	// Phase 1: Increase metric from 3500 to 3700, scaling to 37 replicas.
	metricValue.Store(resource.MustParse("3700"))
	err := waitForDeploymentCondition(ctx, cs, deployment, atLeastReplicas(36))
	if err != nil {
		t.Fatalf("Phase 1: HPA did not scale up: %v", err)
	}
	setDeploymentStatusReplicas(t, cs, deployment.Name, ns.Name, 37) // Simulate replicas got scheduled.

	// Phase 2: Increase metric from 3700 to 3800 (should not scale).
	metricValue.Store(resource.MustParse("3800"))
	err = waitForDeploymentCondition(ctx, cs, deployment, atLeastReplicas(38))
	if err == nil {
		t.Fatal("Phase 2: HPA unexpectedly scaled up")
	}

	// Phase 3: Decrease metric from 3800 to 1000, scaling down to 10 replicas.
	metricValue.Store(resource.MustParse("1000"))
	err = waitForDeploymentCondition(ctx, cs, deployment, noMoreThanReplicas(10))
	if err != nil {
		t.Fatalf("Phase 3: HPA did not scale down: %v", err)
	}
	setDeploymentStatusReplicas(t, cs, deployment.Name, ns.Name, 10) // Simulate replicas got scheduled.

	// Phase 4: Decrease metric from 1000 to 800 (should not scale).
	metricValue.Store(resource.MustParse("800"))
	err = waitForDeploymentCondition(ctx, cs, deployment, noMoreThanReplicas(9))
	if err == nil {
		t.Fatal("Phase 4: HPA unexpectedly scaled up")
	}
}

// setDeploymentStatusReplicas sets the `status.replicas` field of a deployment
// designated by name and namespace.
func setDeploymentStatusReplicas(t *testing.T, cs *kubernetes.Clientset, name, namespace string, replicas int32) {
	t.Helper()
	_, err := cs.CoreV1().RESTClient().
		Patch(types.MergePatchType).
		AbsPath("/apis/apps/v1").
		Namespace(namespace).
		Resource("deployments").
		Name(name).
		SubResource("status").
		Body(fmt.Appendf(nil, `{"status":{"replicas": %d}}`, replicas)).
		Do(t.Context()).Get()
	if err != nil {
		t.Fatalf("Cannot set deployment status: %v", err)
	}
	time.Sleep(hpaControllerResyncPeriod * 2) // Wait to ensure HPA controller takes it into account
}
