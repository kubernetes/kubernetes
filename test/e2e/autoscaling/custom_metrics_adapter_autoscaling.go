/*
Copyright 2018 The Kubernetes Authors.

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

package autoscaling

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscaling "k8s.io/api/autoscaling/v2beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/instrumentation/monitoring"

	. "github.com/onsi/ginkgo"
)

const (
	adapterServiceName = "custom-metrics-apiserver:http"
	adapterNamespace   = "custom-metrics"
	adapterDummyPod    = "adapter-dummy-pod"
)

var _ = SIGDescribe("[HPA] Horizontal pod autoscaling (scale resource: Custom Metrics from API Server Adapter)", func() {

	f := framework.NewDefaultFramework("horizontal-pod-autoscaling")
	targetRefGVK := schema.FromAPIVersionAndKind("apps/v1", "StatefulSet")

	It("should scale down with Custom Metric of type Pod from API Server [Feature:CustomMetricsAdapterAutoscaling]", func() {
		initialReplicas := 2
		metricValue := int64(100)
		metricTarget := 2 * metricValue

		metricValues := []TestMetricValue{
			{
				resourceName: dummyDeploymentName + "-0",
				resourceType: schema.ParseGroupResource("pods"),
				metrics: map[string]int64{
					monitoring.CustomMetricName: metricValue,
				},
			},
			{
				resourceName: dummyDeploymentName + "-1",
				resourceType: schema.ParseGroupResource("pods"),
				metrics: map[string]int64{
					monitoring.CustomMetricName: metricValue,
				},
			},
		}

		tc := CustomMetricAdapterTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialPodCount: 2,
			scaledPodCount:  1,
			metricValues:    metricValues,
			deployment:      monitoring.SimpleDummyStatefulSet(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas)),
			hpa:             simplePodsHPA(f.Namespace.ObjectMeta.Name, dummyDeploymentName, targetRefGVK, metricTarget),
		}
		tc.Run()
	})

	It("should scale down with Custom Metric of type Object from API Server while ignoring other metrics [Feature:CustomMetricsAdapterAutoscaling]", func() {
		initialReplicas := 2
		metricValue := int64(100)
		metricTarget := 2 * metricValue

		metricValues := []TestMetricValue{
			{
				resourceName: adapterDummyPod,
				resourceType: schema.ParseGroupResource("pods"),
				metrics: map[string]int64{
					monitoring.CustomMetricName: metricValue,
				},
			},
			{
				resourceName: dummyDeploymentName + "-0",
				resourceType: schema.ParseGroupResource("pods"),
				metrics: map[string]int64{
					monitoring.CustomMetricName: int64(0),
				},
			},
			{
				resourceName: dummyDeploymentName + "-1",
				resourceType: schema.ParseGroupResource("pods"),
				metrics: map[string]int64{
					monitoring.CustomMetricName: int64(0),
				},
			},
		}

		tc := CustomMetricAdapterTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialPodCount: 3,
			scaledPodCount:  2,
			metricValues:    metricValues,
			deployment:      monitoring.SimpleDummyStatefulSet(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas)),
			pod:             monitoring.DummyPod(adapterDummyPod, f.Namespace.ObjectMeta.Name, adapterDummyPod, monitoring.CustomMetricName, metricValue),
			hpa:             objectHPA(f.Namespace.ObjectMeta.Name, adapterDummyPod, dummyDeploymentName, targetRefGVK, schema.ParseGroupKind("Pod"), metricTarget),
		}
		tc.Run()
	})

	It("should scale down with Custom Metric of type Object from API Server [Feature:CustomMetricsAdapterAutoscaling]", func() {
		initialReplicas := 2
		metricValue := int64(100)
		metricTarget := 2 * metricValue

		metricValues := []TestMetricValue{
			{
				resourceName: adapterDummyPod,
				resourceType: schema.ParseGroupResource("services"),
				metrics: map[string]int64{
					monitoring.CustomMetricName: metricValue,
				},
			},
		}

		// Don't actually create the object, to check the non-pod object case. Custom API should still work
		tc := CustomMetricAdapterTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialPodCount: 2,
			scaledPodCount:  1,
			metricValues:    metricValues,
			deployment:      monitoring.SimpleDummyStatefulSet(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas)),
			hpa:             objectHPA(f.Namespace.ObjectMeta.Name, adapterDummyPod, dummyDeploymentName, targetRefGVK, schema.ParseGroupKind("Service"), metricTarget),
		}
		tc.Run()
	})

	It("should scale up with Custom Metric of type Pod from API Server [Feature:CustomMetricsAdapterAutoscaling]", func() {
		initialReplicas := 1
		metricValue := int64(200)
		metricTarget := int64(0.5 * float64(metricValue))

		metricValues := []TestMetricValue{
			{
				resourceName: dummyDeploymentName + "-0",
				resourceType: schema.ParseGroupResource("pods"),
				metrics: map[string]int64{
					monitoring.CustomMetricName: metricValue,
				},
			},
		}

		tc := CustomMetricAdapterTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialPodCount: 1,
			scaledPodCount:  2,
			metricValues:    metricValues,
			deployment:      monitoring.SimpleDummyStatefulSet(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas)),
			hpa:             simplePodsHPA(f.Namespace.ObjectMeta.Name, dummyDeploymentName, targetRefGVK, metricTarget),
		}
		tc.Run()
	})

	It("should scale up with 2 Custom Metrics of type Pod from API Server [Feature:CustomMetricsAdapterAutoscaling]", func() {
		initialReplicas := 1
		// metric 1 would cause a scale down, if not for metric 2
		metric1Value := int64(100)
		metric1Target := 2 * metric1Value
		// metric2 should cause a scale up
		metric2Value := int64(200)
		metric2Target := int64(0.5 * float64(metric2Value))

		metricTargets := map[string]int64{"metric1": metric1Target, "metric2": metric2Target}

		metricValues := []TestMetricValue{
			{
				resourceName: dummyDeploymentName + "-0",
				resourceType: schema.ParseGroupResource("pods"),
				metrics: map[string]int64{
					"metric1": metric1Value,
					"metric2": metric2Value,
				},
			},
		}

		tc := CustomMetricAdapterTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialPodCount: 1,
			scaledPodCount:  2,
			metricValues:    metricValues,
			deployment:      monitoring.SimpleDummyStatefulSet(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas)),
			hpa:             podsHPA(f.Namespace.ObjectMeta.Name, dummyDeploymentName, targetRefGVK, metricTargets),
		}
		tc.Run()
	})
})

type CustomMetricAdapterTestCase struct {
	framework       *framework.Framework
	hpa             *autoscaling.HorizontalPodAutoscaler
	kubeClient      clientset.Interface
	deployment      *appsv1.StatefulSet
	pod             *corev1.Pod
	metricValues    []TestMetricValue
	initialPodCount int
	scaledPodCount  int
}

type TestMetricValue struct {
	resourceName string
	resourceType schema.GroupResource
	metrics      map[string]int64
}

func (tc *CustomMetricAdapterTestCase) WriteMetric(resource schema.GroupResource, name, metric string, value int64) {
	// Get custom-metrics adapter service
	proxyRequest, err := framework.GetServicesProxyRequest(tc.kubeClient, tc.kubeClient.CoreV1().RESTClient().Post())
	if err != nil {
		framework.Failf("Failed to make custom metric service proxy request: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
	defer cancel()

	// Build custom-metric request
	path := fmt.Sprintf("/write-metrics/namespaces/%s/%s/%s/%s", tc.framework.Namespace.ObjectMeta.Name, resource.String(), name, metric)
	body := fmt.Sprintf(`{"Value":%d}`, value)

	// Make request to custom-metrics service
	req := proxyRequest.Namespace(adapterNamespace).
		Context(ctx).
		Name(adapterServiceName).
		Suffix(path).
		SetHeader("Content-Type", "application/json").
		Body([]byte(body))
	framework.Logf("Request URL: %v", req.URL())
	_, err = req.DoRaw()
	if err != nil {
		framework.Failf("Failed to make custom metric POST request: %v", err)
	}
}

func (tc *CustomMetricAdapterTestCase) Run() {
	err := monitoring.CreateCustomAdapter(monitoring.AdapterForCustomMetrics)
	if err != nil {
		framework.Failf("Failed to set up: %v", err)
	}
	defer monitoring.CleanupCustomAdapter(monitoring.AdapterForCustomMetrics, adapterNamespace, tc.kubeClient, 15*time.Minute)

	// Run application that will be scaled based on metrics
	err = createStatefulSetsToScale(tc.framework, tc.kubeClient, tc.deployment, tc.pod)
	if err != nil {
		framework.Failf("Failed to create dummy deployment pod: %v", err)
	}
	defer cleanupStatefulSetsToScale(tc.framework, tc.kubeClient, tc.deployment, tc.pod)

	// Wait for the statefulset to run
	waitForReplicas(tc.framework.Namespace.ObjectMeta.Name, tc.kubeClient, 15*time.Minute, tc.initialPodCount, metav1.ListOptions{})

	// Write custom metric value for each resource in statefulset
	for _, testMetrics := range tc.metricValues {
		for metric, value := range testMetrics.metrics {
			tc.WriteMetric(testMetrics.resourceType, testMetrics.resourceName, metric, value)
		}
	}

	// Autoscale the deployment
	_, err = tc.kubeClient.AutoscalingV2beta1().HorizontalPodAutoscalers(tc.framework.Namespace.ObjectMeta.Name).Create(tc.hpa)
	if err != nil {
		framework.Failf("Failed to create HPA: %v", err)
	}
	defer tc.kubeClient.AutoscalingV2beta1().HorizontalPodAutoscalers(tc.framework.Namespace.ObjectMeta.Name).Delete(tc.hpa.ObjectMeta.Name, &metav1.DeleteOptions{})

	waitForReplicas(tc.framework.Namespace.ObjectMeta.Name, tc.kubeClient, 15*time.Minute, tc.scaledPodCount, metav1.ListOptions{})
}
