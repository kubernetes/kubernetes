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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	memory "k8s.io/client-go/discovery/cached"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/scale"
	core "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/podautoscaler"
	metricsclient "k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	emapi "k8s.io/metrics/pkg/apis/external_metrics/v1beta1"
	metricsv "k8s.io/metrics/pkg/client/clientset/versioned"
	externalclient "k8s.io/metrics/pkg/client/external_metrics"
	emfake "k8s.io/metrics/pkg/client/external_metrics/fake"
	"k8s.io/utils/ptr"
)

const restConfigQPS = 10_000
const restConfigBurst = 10_000
const resyncPeriod = 1 * time.Second
const downscaleStabilisationWindow = 1 * time.Second
const tolerance = 0.1
const cpuInitializationPeriod = 10 * time.Second
const delayOfInitialReadinessStatus = 10 * time.Second

// TestHPAScaleUpToMin integration creates a Deployment and HPA, then verifies the HPA exists and
// is reconciled, by bringing the replicas of the Deployment within the HPA's min and max range.
// Given that the metris aren't mocked, it should go up to min only.
func TestHPAScaleUpToMin(t *testing.T) {
	closeFn, restConfig, cs, ns := setup(t, "hpascaleuptomin")
	t.Cleanup(closeFn)

	ctx, cancel := startHPAControllerAndWaitForCaches(t, restConfig, nil)
	t.Cleanup(func() {
		cancel()
	})

	deployment, err := createDeployment(cs, ns.Name, 1)
	if err != nil {
		t.Fatalf("failed to create deployment: %v", err)
	}

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

	_, err = createHPAWithMetricSpec(cs, "dummy-hpa", ns.Name, deployment.Name, 2, 10, metricSpec)
	if err != nil {
		t.Fatalf("failed to create HPA: %v", err)
	}

	err = wait.PollUntilContextTimeout(ctx, 2*time.Second, 30*time.Second, true,
		func(_ context.Context) (bool, error) {
			got, err := cs.AppsV1().Deployments(ns.Name).Get(ctx, deployment.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			return ptr.Deref(got.Spec.Replicas, 0) >= 2, nil // Should be within the HPA's min/max range
		})
	if err != nil {
		t.Fatalf("HPA did not reconcile: %v", err)
	}
}

// TestHPAScaleToZero verifies that HPA can scale up to five, then down to zero and back again, using an external metric.
func TestHPAScaleToZero(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAScaleToZero, true)
	closeFn, restConfig, cs, ns := setup(t, "scaletozeros")
	t.Cleanup(closeFn)

	// Use atomic.Value so the metric can be changed between test phases.
	var metricValue atomic.Value

	fakeEMClient := &emfake.FakeExternalMetricsClient{}
	fakeEMClient.AddReactor("list", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		val := metricValue.Load().(resource.Quantity)
		metrics := &emapi.ExternalMetricValueList{}
		metrics.Items = append(metrics.Items, emapi.ExternalMetricValue{
			Timestamp:  metav1.Time{Time: time.Now()},
			MetricName: "qps",
			Value:      val,
		})
		return true, metrics, nil
	})

	ctx, cancel := startHPAControllerAndWaitForCaches(t, restConfig, fakeEMClient)
	t.Cleanup(func() {
		cancel()
	})

	deployment, err := createDeployment(cs, ns.Name, 1)
	if err != nil {
		t.Fatalf("failed to create deployment: %v", err)
	}

	// Create HPA using an external metric.
	hpaName := "dummy-hpa"
	metricSpec := autoscalingv2.MetricSpec{
		Type: autoscalingv2.ExternalMetricSourceType,
		External: &autoscalingv2.ExternalMetricSource{
			Metric: autoscalingv2.MetricIdentifier{
				Name: "qps",
			},
			Target: autoscalingv2.MetricTarget{
				Type:         autoscalingv2.ValueMetricType,
				AverageValue: ptr.To(resource.MustParse("100")),
			},
		},
	}
	_, err = createHPAWithMetricSpec(cs, hpaName, ns.Name, deployment.Name, 0, 10, metricSpec)
	if err != nil {
		t.Fatalf("failed to create HPA: %v", err)
	}

	// Phase 1: metric=500, expect scale up to 5 pods.
	metricValue.Store(resource.MustParse("500"))
	err = wait.PollUntilContextTimeout(ctx, 2*time.Second, 60*time.Second, true,
		func(_ context.Context) (bool, error) {
			got, err := cs.AppsV1().Deployments(ns.Name).Get(ctx, deployment.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			return ptr.Deref(got.Spec.Replicas, 0) >= 5, nil
		})
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
	err = wait.PollUntilContextTimeout(ctx, 2*time.Second, 60*time.Second, true,
		func(_ context.Context) (bool, error) {
			got, err := cs.AppsV1().Deployments(ns.Name).Get(ctx, deployment.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			return ptr.Deref(got.Spec.Replicas, -1) == 0, nil
		})
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
	err = wait.PollUntilContextTimeout(ctx, 2*time.Second, 60*time.Second, true,
		func(_ context.Context) (bool, error) {
			got, err := cs.AppsV1().Deployments(ns.Name).Get(ctx, deployment.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			return ptr.Deref(got.Spec.Replicas, 0) >= 4, nil
		})
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

// setup starts the api server and makes a namespace
func setup(t testing.TB, nsBaseName string) (framework.TearDownFunc, *restclient.Config, *clientset.Clientset, *corev1.Namespace) {
	flags := framework.DefaultTestServerFlags()
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, flags, framework.SharedEtcd())

	config := restclient.CopyConfig(server.ClientConfig)
	config.QPS = restConfigQPS
	config.Burst = restConfigBurst
	config.Timeout = 0
	clientSet, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(clientSet, nsBaseName, t)
	closeFn := func() {
		framework.DeleteNamespaceOrDie(clientSet, ns, t)
		server.TearDownFn()
	}
	return closeFn, config, clientSet, ns
}

func startHPAControllerAndWaitForCaches(tb testing.TB, restConfig *restclient.Config, externalClient externalclient.ExternalMetricsClient, options ...informers.SharedInformerOption) (context.Context, context.CancelFunc) {
	tb.Helper()
	informerSet := informers.NewSharedInformerFactoryWithOptions(clientset.NewForConfigOrDie(restclient.AddUserAgent(restConfig, "hpa-informers")), 0, options...)
	jc, ctx, cancel := createHPAControllerWithSharedInformers(tb, restConfig, informerSet, externalClient)
	informerSet.Start(ctx.Done())
	go jc.Run(ctx, 1)

	// since this method starts the controller in a separate goroutine
	// and the tests don't check /readyz there is no way
	// the tests can tell it is safe to call the server and requests won't be rejected
	// thus we wait until caches have synced
	informerSet.WaitForCacheSync(ctx.Done())
	return ctx, cancel
}

func createHPAControllerWithSharedInformers(tb testing.TB, restConfig *restclient.Config, informerSet informers.SharedInformerFactory, externalClient externalclient.ExternalMetricsClient) (*podautoscaler.HorizontalController, context.Context, context.CancelFunc) {
	tb.Helper()
	clientSet := clientset.NewForConfigOrDie(restclient.AddUserAgent(restConfig, "hpa-controller"))
	_, ctx := ktesting.NewTestContext(tb)
	ctx, cancel := context.WithCancel(ctx)

	var hc *podautoscaler.HorizontalController
	var err error

	config := restclient.CopyConfig(restConfig)
	discoveryClient := memory.NewMemCacheClient(clientSet.Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)

	metricsClientset := metricsv.NewForConfigOrDie(config)
	metricsClient := metricsclient.NewRESTMetricsClient(metricsClientset.MetricsV1beta1(), nil, externalClient)

	scaleKindResolver := scale.NewDiscoveryScaleKindResolver(clientSet.Discovery())
	scaleClient, err := scale.NewForConfig(config, restMapper, dynamic.LegacyAPIPathResolverFunc, scaleKindResolver)
	if err != nil {
		tb.Fatalf("Error creating scaleClient: %v", err)
	}

	hc = podautoscaler.NewHorizontalController(
		ctx,
		clientSet.CoreV1(),
		scaleClient,
		clientSet.AutoscalingV2(),
		restMapper,
		metricsClient,
		informerSet.Autoscaling().V2().HorizontalPodAutoscalers(),
		informerSet.Core().V1().Pods(),
		resyncPeriod,
		downscaleStabilisationWindow,
		tolerance,
		cpuInitializationPeriod,
		delayOfInitialReadinessStatus,
	)

	return hc, ctx, cancel
}

func createDeployment(cs *clientset.Clientset, namespace string, replicas int32) (*appsv1.Deployment, error) {
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
	return cs.AppsV1().Deployments(namespace).Create(context.TODO(), deployment, metav1.CreateOptions{})
}

func createHPAWithMetricSpec(cs *clientset.Clientset, hpaName, namespace, deploymentName string, minReplicas, maxReplicas int32, metricSpec autoscalingv2.MetricSpec) (*autoscalingv2.HorizontalPodAutoscaler, error) {
	hpa := &autoscalingv2.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      hpaName,
			Namespace: namespace,
		},
		Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       deploymentName,
			},
			MinReplicas: ptr.To(minReplicas),
			MaxReplicas: maxReplicas,
			Metrics:     []autoscalingv2.MetricSpec{metricSpec},
		},
	}
	return cs.AutoscalingV2().HorizontalPodAutoscalers(namespace).Create(context.TODO(), hpa, metav1.CreateOptions{})
}
