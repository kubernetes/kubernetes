/*
Copyright 2017 The Kubernetes Authors.

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
	"math"
	"time"

	gcm "google.golang.org/api/monitoring/v3"
	as "k8s.io/api/autoscaling/v2beta1"
	corev1 "k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/instrumentation/monitoring"

	. "github.com/onsi/ginkgo"
	"golang.org/x/oauth2/google"
)

const (
	stackdriverExporterDeployment = "stackdriver-exporter-deployment"
	dummyDeploymentName           = "dummy-deployment"
	stackdriverExporterPod        = "stackdriver-exporter-pod"
	externalMetricValue           = int64(85)
)

var _ = SIGDescribe("[HPA] Horizontal pod autoscaling (scale resource: Custom Metrics from Stackdriver)", func() {
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke")
	})

	f := framework.NewDefaultFramework("horizontal-pod-autoscaling")
	targetRefGVK := schema.FromAPIVersionAndKind("extensions/v1beta1", "Deployment")

	It("should scale down with Custom Metric of type Pod from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 2
		// metric should cause scale down
		metricValue := int64(100)
		metricTarget := 2 * metricValue
		tc := CustomMetricStackdriverTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  1,
			deployment:      monitoring.SimpleStackdriverExporterDeployment(stackdriverExporterDeployment, f.Namespace.ObjectMeta.Name, int32(initialReplicas), metricValue),
			hpa:             simplePodsHPA(f.Namespace.ObjectMeta.Name, stackdriverExporterDeployment, targetRefGVK, metricTarget)}
		tc.Run()
	})

	It("should scale down with Custom Metric of type Object from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 2
		// metric should cause scale down
		metricValue := int64(100)
		metricTarget := 2 * metricValue
		tc := CustomMetricStackdriverTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  1,
			// Metric exported by deployment is ignored
			deployment: monitoring.SimpleStackdriverExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas), 0 /* ignored */),
			pod:        monitoring.StackdriverExporterPod(stackdriverExporterPod, f.Namespace.Name, stackdriverExporterPod, monitoring.CustomMetricName, metricValue),
			hpa:        objectHPA(f.Namespace.ObjectMeta.Name, dummyDeploymentName, dummyDeploymentName, targetRefGVK, schema.ParseGroupKind("Pod"), metricTarget)}
		tc.Run()
	})

	It("should scale down with External Metric with target value from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 2
		// metric should cause scale down
		metricValue := externalMetricValue
		metricTarget := 2 * metricValue
		metricTargets := map[string]externalMetricTarget{
			"target": {
				value:     metricTarget,
				isAverage: false,
			},
		}
		tc := CustomMetricStackdriverTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  1,
			// Metric exported by deployment is ignored
			deployment: monitoring.SimpleStackdriverExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas), 0 /* ignored */),
			pod:        monitoring.StackdriverExporterPod(stackdriverExporterPod, f.Namespace.Name, stackdriverExporterPod, "target", metricValue),
			hpa:        externalHPA(f.Namespace.ObjectMeta.Name, metricTargets)}
		tc.Run()
	})

	It("should scale down with External Metric with target average value from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 2
		// metric should cause scale down
		metricValue := externalMetricValue
		metricAverageTarget := 2 * metricValue
		metricTargets := map[string]externalMetricTarget{
			"target_average": {
				value:     metricAverageTarget,
				isAverage: true,
			},
		}
		tc := CustomMetricStackdriverTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  1,
			// Metric exported by deployment is ignored
			deployment: monitoring.SimpleStackdriverExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas), 0 /* ignored */),
			pod:        monitoring.StackdriverExporterPod(stackdriverExporterPod, f.Namespace.Name, stackdriverExporterPod, "target_average", externalMetricValue),
			hpa:        externalHPA(f.Namespace.ObjectMeta.Name, metricTargets)}
		tc.Run()
	})

	It("should scale down with Custom Metric of type Pod from Stackdriver with Prometheus [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 2
		// metric should cause scale down
		metricValue := int64(100)
		metricTarget := 2 * metricValue
		tc := CustomMetricStackdriverTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  1,
			deployment:      monitoring.PrometheusExporterDeployment(stackdriverExporterDeployment, f.Namespace.ObjectMeta.Name, int32(initialReplicas), metricValue),
			hpa:             simplePodsHPA(f.Namespace.ObjectMeta.Name, stackdriverExporterDeployment, targetRefGVK, metricTarget)}
		tc.Run()
	})

	It("should scale up with two metrics of type Pod from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 1
		// metric 1 would cause a scale down, if not for metric 2
		metric1Value := int64(100)
		metric1Target := 2 * metric1Value
		// metric2 should cause a scale up
		metric2Value := int64(200)
		metric2Target := int64(0.5 * float64(metric2Value))
		containers := []monitoring.CustomMetricContainerSpec{
			{
				Name:        "stackdriver-exporter-metric1",
				MetricName:  "metric1",
				MetricValue: metric1Value,
			},
			{
				Name:        "stackdriver-exporter-metric2",
				MetricName:  "metric2",
				MetricValue: metric2Value,
			},
		}
		metricTargets := map[string]int64{"metric1": metric1Target, "metric2": metric2Target}
		tc := CustomMetricStackdriverTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  3,
			deployment:      monitoring.StackdriverExporterDeployment(stackdriverExporterDeployment, f.Namespace.ObjectMeta.Name, int32(initialReplicas), containers),
			hpa:             podsHPA(f.Namespace.ObjectMeta.Name, stackdriverExporterDeployment, targetRefGVK, metricTargets)}
		tc.Run()
	})

	It("should scale up with two External metrics from Stackdriver [Feature:CustomMetricsAutoscaling]", func() {
		initialReplicas := 1
		// metric 1 would cause a scale down, if not for metric 2
		metric1Value := externalMetricValue
		metric1Target := 2 * metric1Value
		// metric2 should cause a scale up
		metric2Value := externalMetricValue
		metric2Target := int64(math.Ceil(0.5 * float64(metric2Value)))
		metricTargets := map[string]externalMetricTarget{
			"external_metric_1": {
				value:     metric1Target,
				isAverage: false,
			},
			"external_metric_2": {
				value:     metric2Target,
				isAverage: false,
			},
		}
		containers := []monitoring.CustomMetricContainerSpec{
			{
				Name:        "stackdriver-exporter-metric1",
				MetricName:  "external_metric_1",
				MetricValue: metric1Value,
			},
			{
				Name:        "stackdriver-exporter-metric2",
				MetricName:  "external_metric_2",
				MetricValue: metric2Value,
			},
		}
		tc := CustomMetricStackdriverTestCase{
			framework:       f,
			kubeClient:      f.ClientSet,
			initialReplicas: initialReplicas,
			scaledReplicas:  3,
			deployment:      monitoring.StackdriverExporterDeployment(dummyDeploymentName, f.Namespace.ObjectMeta.Name, int32(initialReplicas), containers),
			hpa:             externalHPA(f.Namespace.ObjectMeta.Name, metricTargets)}
		tc.Run()
	})
})

type CustomMetricStackdriverTestCase struct {
	framework       *framework.Framework
	hpa             *as.HorizontalPodAutoscaler
	kubeClient      clientset.Interface
	deployment      *extensions.Deployment
	pod             *corev1.Pod
	initialReplicas int
	scaledReplicas  int
}

func (tc *CustomMetricStackdriverTestCase) Run() {
	projectId := framework.TestContext.CloudConfig.ProjectID

	ctx := context.Background()
	client, err := google.DefaultClient(ctx, gcm.CloudPlatformScope)

	// Hack for running tests locally, needed to authenticate in Stackdriver
	// If this is your use case, create application default credentials:
	// $ gcloud auth application-default login
	// and uncomment following lines:
	/*
		ts, err := google.DefaultTokenSource(oauth2.NoContext)
		framework.Logf("Couldn't get application default credentials, %v", err)
		if err != nil {
			framework.Failf("Error accessing application default credentials, %v", err)
		}
		client := oauth2.NewClient(oauth2.NoContext, ts)
	*/

	gcmService, err := gcm.New(client)
	if err != nil {
		framework.Failf("Failed to create gcm service, %v", err)
	}

	// Set up a cluster: create a custom metric and set up k8s-sd adapter
	err = monitoring.CreateDescriptors(gcmService, projectId)
	if err != nil {
		framework.Failf("Failed to create metric descriptor: %v", err)
	}
	defer monitoring.CleanupDescriptors(gcmService, projectId)

	err = monitoring.CreateStackdriverAdapter(monitoring.StackdriverAdapterDefault)
	if err != nil {
		framework.Failf("Failed to set up: %v", err)
	}
	defer monitoring.CleanupStackdriverAdapter(monitoring.StackdriverAdapterDefault)

	// Run application that exports the metric
	err = createDeploymentToScale(tc.framework, tc.kubeClient, tc.deployment, tc.pod)
	if err != nil {
		framework.Failf("Failed to create stackdriver-exporter pod: %v", err)
	}
	defer cleanupDeploymentsToScale(tc.framework, tc.kubeClient, tc.deployment, tc.pod)

	listOptions := metav1.ListOptions{
		LabelSelector: fmt.Sprintf("app=%s", tc.deployment.ObjectMeta.Name)}

	// Wait for the deployment to run
	waitForReplicas(tc.framework.Namespace.ObjectMeta.Name, tc.kubeClient, 15*time.Minute, tc.initialReplicas, listOptions)

	// Autoscale the deployment
	_, err = tc.kubeClient.AutoscalingV2beta1().HorizontalPodAutoscalers(tc.framework.Namespace.ObjectMeta.Name).Create(tc.hpa)
	if err != nil {
		framework.Failf("Failed to create HPA: %v", err)
	}
	defer tc.kubeClient.AutoscalingV2beta1().HorizontalPodAutoscalers(tc.framework.Namespace.ObjectMeta.Name).Delete(tc.hpa.ObjectMeta.Name, &metav1.DeleteOptions{})

	waitForReplicas(tc.framework.Namespace.ObjectMeta.Name, tc.kubeClient, 15*time.Minute, tc.scaledReplicas, listOptions)
}
