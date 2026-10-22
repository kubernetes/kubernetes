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
	"context"
	"sync/atomic"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/ktesting"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	metricsv1 "k8s.io/metrics/pkg/apis/metrics/v1"
	metricsv1beta1 "k8s.io/metrics/pkg/apis/metrics/v1beta1"
)

func TestHPAResourceMetricsAPIVersionSwitch(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	tCtx := ktesting.Init(t).WithRESTConfig(server.ClientConfig)

	metricsServer := newFakeResourceMetricsServer(tCtx)

	// Initially, only metrics.k8s.io/v1beta1 is available.
	metricsServer.replaceVersions(tCtx, metricsv1beta1.SchemeGroupVersion)

	clients, _ := createClients(tCtx)
	startHPAControllerAndWaitForCaches(tCtx, clients)

	ns := createTestNamespace(tCtx)
	deployment := createDeployment(tCtx, ns.Name, 1)

	// No Deployment controller runs in this test, so update the status to make
	// the HPA observe one current replica and evaluate resource metrics.
	setDeploymentStatusReplicas(tCtx, deployment, 1)

	metricSpec := autoscalingv2.MetricSpec{
		Type: autoscalingv2.ResourceMetricSourceType,
		Resource: &autoscalingv2.ResourceMetricSource{
			Name: corev1.ResourceCPU,
			Target: autoscalingv2.MetricTarget{
				Type:               autoscalingv2.UtilizationMetricType,
				AverageUtilization: new(int32(50)),
			},
		},
	}
	createHPA(tCtx, deployment, metricSpec)

	waitForRequestCount(tCtx, &metricsServer.v1beta1Requests, 1)

	if got := metricsServer.v1Requests.Load(); got != 0 {
		t.Fatalf("got %d metrics.k8s.io/v1 requests before v1 was registered", got)
	}

	// Add v1 while keeping v1beta1 available. After the cached API version is
	// invalidated, the next metrics request should discover and prefer v1.
	metricsServer.replaceVersions(
		tCtx,
		metricsv1.SchemeGroupVersion,
		metricsv1beta1.SchemeGroupVersion,
	)

	waitForRequestCount(tCtx, &metricsServer.v1Requests, 1)

	// Remove v1 and verify fallback to v1beta1.
	v1beta1Requests := metricsServer.v1beta1Requests.Load()

	metricsServer.replaceVersions(
		tCtx,
		metricsv1beta1.SchemeGroupVersion,
	)

	waitForRequestCount(tCtx, &metricsServer.v1beta1Requests, v1beta1Requests+1)
}

func setDeploymentStatusReplicas(tCtx ktesting.TContext, deployment *appsv1.Deployment, replica int32) {
	tCtx.Helper()

	deployment = deployment.DeepCopy()
	deployment.Status.Replicas = replica

	if _, err := tCtx.Client().AppsV1().Deployments(deployment.Namespace).UpdateStatus(tCtx, deployment, metav1.UpdateOptions{}); err != nil {
		tCtx.Fatalf("failed to update status for deployment %s/%s: %v", deployment.Namespace, deployment.Name, err)
	}
}

func waitForRequestCount(tCtx ktesting.TContext, counter *atomic.Int64, want int64) {
	tCtx.Helper()

	err := wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, 10*time.Second, true,
		func(context.Context) (bool, error) {
			return counter.Load() >= want, nil
		},
	)

	if err != nil {
		tCtx.Fatalf("timed out waiting for resource metrics requests: got %d, want at least %d", counter.Load(), want)
	}
}
