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
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	memory "k8s.io/client-go/discovery/cached"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
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
	apiServer       *clientset.Clientset
	metrics         *metricsclientset.Clientset
	externalMetrics *emfake.FakeExternalMetricsClient
	restMapper      meta.RESTMapper
	scale           scale.ScalesGetter
}

// createClients creates clients connecting to API server, custom metrics API,...
//
// Creates an external metrics server to connect to, and returns a pointer to the
// metric value it's serving.
func createClients(t *testing.T, config *restclient.Config) (testClients, *atomic.Value) {
	apiServer := clientset.NewForConfigOrDie(config)
	metrics := metricsclientset.NewForConfigOrDie(config)
	externalMetrics, externalMetricValue := setupFakeExternalMetrics()

	discoveryClient := memory.NewMemCacheClient(apiServer.Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)
	scaleKindResolver := scale.NewDiscoveryScaleKindResolver(discoveryClient)

	scale, err := scale.NewForConfig(config, restMapper, dynamic.LegacyAPIPathResolverFunc, scaleKindResolver)
	if err != nil {
		t.Fatalf("Error creating scale client: %v", err)
	}

	return testClients{
		apiServer,
		metrics,
		externalMetrics,
		restMapper,
		scale,
	}, externalMetricValue
}

func createTestNamespace(t *testing.T, c *clientset.Clientset) *corev1.Namespace {
	ns := framework.CreateNamespaceOrDie(c, "podautoscaler", t)
	t.Cleanup(func() {
		framework.DeleteNamespaceOrDie(c, ns, t)
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

func createHPA(t *testing.T, cs *clientset.Clientset, deployment *appsv1.Deployment, metricSpec autoscalingv2.MetricSpec, opts ...createHPAOption) *autoscalingv2.HorizontalPodAutoscaler {
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
	hpa, err := cs.AutoscalingV2().HorizontalPodAutoscalers(deployment.Namespace).Create(t.Context(), hpa, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create HPA: %v", err)
	}
	return hpa
}

func startHPAControllerAndWaitForCaches(t *testing.T, clients testClients) {
	t.Helper()

	ctx := t.Context()
	metricsClient := metricsclient.NewRESTMetricsClient(clients.metrics.MetricsV1beta1(), nil, clients.externalMetrics)

	informerSet := informers.NewSharedInformerFactory(clients.apiServer, 0)
	controller := podautoscaler.NewHorizontalController(
		ctx,
		clients.apiServer.CoreV1(),
		clients.scale,
		clients.apiServer.AutoscalingV2(),
		clients.restMapper,
		metricsClient,
		informerSet.Autoscaling().V2().HorizontalPodAutoscalers(),
		informerSet.Core().V1().Pods(),
		hpaControllerResyncPeriod,
		downscaleStabilisationWindow,
		tolerance,
		cpuInitializationPeriod,
		delayOfInitialReadinessStatus,
	)
	informerSet.Start(ctx.Done())
	go controller.Run(ctx, 1)

	// Since this method starts the controller in a separate goroutine
	// and the tests don't check /readyz there is no way
	// the tests can tell it is safe to call the server and requests won't be rejected
	// thus we wait until caches have synced
	informerSet.WaitForCacheSync(ctx.Done())
}

func createDeployment(t *testing.T, cs *clientset.Clientset, namespace string, replicas int32) *appsv1.Deployment {
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
	d, err := cs.AppsV1().Deployments(namespace).Create(t.Context(), deployment, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}
	return d
}

func atLeastReplicas(minReplicas int32) func(*appsv1.Deployment) error {
	return func(d *appsv1.Deployment) error {
		r := ptr.Deref(d.Spec.Replicas, 0)
		if r < minReplicas {
			return fmt.Errorf("got %d replicas, want at least %d", r, minReplicas)
		}
		return nil
	}
}

func equalReplicas(replicas int32) func(*appsv1.Deployment) error {
	return func(d *appsv1.Deployment) error {
		r := ptr.Deref(d.Spec.Replicas, math.MaxInt32)
		if r != replicas {
			return fmt.Errorf("got %d replicas, want exactly %d", r, replicas)
		}
		return nil
	}
}

// waitForDeploymentCondition waits until a deployment matches a given condition cond.
func waitForDeploymentCondition(ctx context.Context, cs *clientset.Clientset, d *appsv1.Deployment,
	cond func(*appsv1.Deployment) error) error {

	// Updates shouldn't take more than 1 HPA resync period. Bump to a few more
	// to cover corner cases (e.g. slow API server).
	timeout := 10 * hpaControllerResyncPeriod

	var condErr error
	err := wait.PollUntilContextTimeout(ctx, time.Second, timeout, false,
		func(_ context.Context) (bool, error) {
			got, err := cs.AppsV1().Deployments(d.Namespace).Get(ctx, d.Name, metav1.GetOptions{})
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
