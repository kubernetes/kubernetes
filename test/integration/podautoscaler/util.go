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
	"fmt"
	"math"
	"sync/atomic"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	memory "k8s.io/client-go/discovery/cached"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/ktesting"
	"k8s.io/client-go/scale"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/controller/podautoscaler"
	metricsclient "k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	"k8s.io/kubernetes/test/integration/framework"
	emapi "k8s.io/metrics/pkg/apis/external_metrics/v1beta1"
	metricsclientset "k8s.io/metrics/pkg/client/clientset/versioned"
	emfake "k8s.io/metrics/pkg/client/external_metrics/fake"
	"k8s.io/utils/ptr"
)

const (
	hpaControllerResyncPeriod     = 1 * time.Second
	downscaleStabilisationWindow  = 1 * time.Second
	tolerance                     = 0.1
	cpuInitializationPeriod       = 10 * time.Second
	delayOfInitialReadinessStatus = 10 * time.Second

	externalMetricName = "qps"
	hpaName            = "dummy-hpa"
)

type testClients struct {
	metrics         *metricsclientset.Clientset
	externalMetrics *emfake.FakeExternalMetricsClient
	scale           scale.ScalesGetter
}

// createClients creates clients connecting to API server, custom metrics API,...
//
// Creates an external metrics server to connect to, and returns a pointer to the
// metric value it's serving.
func createClients(tCtx ktesting.TContext) (testClients, *atomic.Value) {
	metrics := metricsclientset.NewForConfigOrDie(tCtx.RESTConfig())
	externalMetrics, externalMetricValue := setupFakeExternalMetrics()

	discoveryClient := memory.NewMemCacheClient(tCtx.Client().Discovery())
	scaleKindResolver := scale.NewDiscoveryScaleKindResolver(discoveryClient)

	scaleClient, err := scale.NewForConfig(tCtx.RESTConfig(), tCtx.RESTMapper(), dynamic.LegacyAPIPathResolverFunc, scaleKindResolver)
	if err != nil {
		tCtx.Fatalf("Error creating scale client: %v", err)
	}

	return testClients{
		metrics,
		externalMetrics,
		scaleClient,
	}, externalMetricValue
}

func createTestNamespace(tCtx ktesting.TContext) *corev1.Namespace {
	ns := framework.CreateNamespaceOrDie(tCtx.Client(), "podautoscaler", tCtx)
	tCtx.Cleanup(func() {
		framework.DeleteNamespaceOrDie(tCtx.Client(), ns, tCtx)
	})
	return ns
}

// setupFakeExternalMetrics returns a client to a fake metric server. The
// value advertised by the server can be updated via the second return value.
func setupFakeExternalMetrics() (*emfake.FakeExternalMetricsClient, *atomic.Value) {
	var metricValue atomic.Value

	c := &emfake.FakeExternalMetricsClient{}
	c.AddReactor("list", "*", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
		val := metricValue.Load().(resource.Quantity)
		metrics := &emapi.ExternalMetricValueList{}
		metrics.Items = append(metrics.Items, emapi.ExternalMetricValue{
			Timestamp:  metav1.Time{Time: time.Now()},
			MetricName: externalMetricName,
			Value:      val,
		})
		return true, metrics, nil
	})

	return c, &metricValue
}

type createHPAOption func(*autoscalingv2.HorizontalPodAutoscaler)

func withHPAMinMaxReplicas(minReplicas, maxReplicas int32) createHPAOption {
	return func(hpa *autoscalingv2.HorizontalPodAutoscaler) {
		hpa.Spec.MinReplicas = &minReplicas
		hpa.Spec.MaxReplicas = maxReplicas
	}
}

func withHPABehavior(behavior *autoscalingv2.HorizontalPodAutoscalerBehavior) createHPAOption {
	return func(hpa *autoscalingv2.HorizontalPodAutoscaler) {
		hpa.Spec.Behavior = behavior
	}
}

// newHPA builds an HPA targeting the given deployment with a single metric.
// It applies opts but does not create the object, so callers can also use it to
// build HPAs whose creation is expected to be rejected.
func newHPA(deployment *appsv1.Deployment, metricSpec autoscalingv2.MetricSpec, opts ...createHPAOption) *autoscalingv2.HorizontalPodAutoscaler {
	hpa := &autoscalingv2.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      hpaName,
			Namespace: deployment.Namespace,
		},
		Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       deployment.Name,
			},
			MinReplicas: ptr.To(int32(1)),
			MaxReplicas: 10,
			Metrics:     []autoscalingv2.MetricSpec{metricSpec},
		},
	}
	for _, opt := range opts {
		opt(hpa)
	}
	return hpa
}

func createHPA(tCtx ktesting.TContext, deployment *appsv1.Deployment, metricSpec autoscalingv2.MetricSpec, opts ...createHPAOption) *autoscalingv2.HorizontalPodAutoscaler {
	hpa, err := tCtx.Client().AutoscalingV2().HorizontalPodAutoscalers(deployment.Namespace).Create(tCtx, newHPA(deployment, metricSpec, opts...), metav1.CreateOptions{})
	if err != nil {
		tCtx.Fatalf("Failed to create HPA: %v", err)
	}
	return hpa
}

func startHPAControllerAndWaitForCaches(tCtx ktesting.TContext, clients testClients) {
	tCtx.Helper()

	metricsClient := metricsclient.NewRESTMetricsClient(clients.metrics.MetricsV1beta1(), nil, clients.externalMetrics)

	informerSet := informers.NewSharedInformerFactory(tCtx.Client(), 0)
	controller := podautoscaler.NewHorizontalController(
		tCtx,
		tCtx.Client().CoreV1(),
		clients.scale,
		tCtx.Client().AutoscalingV2(),
		tCtx.RESTMapper(),
		metricsClient,
		informerSet.Autoscaling().V2().HorizontalPodAutoscalers(),
		informerSet.Core().V1().Pods(),
		hpaControllerResyncPeriod,
		downscaleStabilisationWindow,
		tolerance,
		cpuInitializationPeriod,
		delayOfInitialReadinessStatus,
	)
	informerSet.Start(tCtx.Done())
	go controller.Run(tCtx, 1)

	// Since this method starts the controller in a separate goroutine
	// and the tests don't check /readyz there is no way
	// the tests can tell it is safe to call the server and requests won't be rejected
	// thus we wait until caches have synced
	informerSet.WaitForCacheSync(tCtx.Done())
}

func createDeployment(tCtx ktesting.TContext, namespace string, replicas int32) *appsv1.Deployment {
	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dummy-deployment",
			Namespace: namespace,
			Labels:    map[string]string{"app": "dummy"},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr.To(replicas),
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "dummy"}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": "dummy"}},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "fake-container",
						Image: "fake-image",
						Resources: corev1.ResourceRequirements{
							Requests: corev1.ResourceList{"cpu": resource.MustParse("100m")},
							Limits:   corev1.ResourceList{"cpu": resource.MustParse("200m")},
						},
					}},
				},
			},
		},
	}
	d, err := tCtx.Client().AppsV1().Deployments(namespace).Create(tCtx, deployment, metav1.CreateOptions{})
	if err != nil {
		tCtx.Fatalf("Failed to create deployment: %v", err)
	}
	return d
}

type deploymentCondition func(*appsv1.Deployment) error

func atLeastReplicas(minReplicas int32) deploymentCondition {
	return func(d *appsv1.Deployment) error {
		r := ptr.Deref(d.Spec.Replicas, 0)
		if r < minReplicas {
			return fmt.Errorf("got %d replicas, want at least %d", r, minReplicas)
		}
		return nil
	}
}

func equalReplicas(replicas int32) deploymentCondition {
	return func(d *appsv1.Deployment) error {
		r := ptr.Deref(d.Spec.Replicas, math.MaxInt32)
		if r != replicas {
			return fmt.Errorf("got %d replicas, want exactly %d", r, replicas)
		}
		return nil
	}
}

func noMoreThanReplicas(maxReplicas int32) deploymentCondition {
	return func(d *appsv1.Deployment) error {
		r := ptr.Deref(d.Spec.Replicas, 0)
		if r > maxReplicas {
			return fmt.Errorf("got %d replicas, want at most %d", r, maxReplicas)
		}
		return nil
	}
}

// waitForDeploymentCondition waits until a deployment matches a given condition cond.
func waitForDeploymentCondition(tCtx ktesting.TContext, d *appsv1.Deployment,
	cond deploymentCondition) error {

	// Updates shouldn't take more than 1 HPA resync period. Bump to a few more
	// to cover corner cases (e.g. slow API server).
	timeout := 10 * hpaControllerResyncPeriod

	var condErr error
	err := wait.PollUntilContextTimeout(tCtx, time.Second, timeout, false,
		func(_ context.Context) (bool, error) {
			got, err := tCtx.Client().AppsV1().Deployments(d.Namespace).Get(tCtx, d.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			condErr = cond(got)
			return condErr == nil, nil
		})
	if err != nil {
		return fmt.Errorf("condition not met: %w (last condition error: %w)", err, condErr)
	}
	return nil
}
