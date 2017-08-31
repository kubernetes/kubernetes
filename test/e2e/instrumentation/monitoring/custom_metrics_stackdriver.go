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

package monitoring

import (
	"context"
	"fmt"
	"time"

	"golang.org/x/oauth2/google"
	clientset "k8s.io/client-go/kubernetes"

	. "github.com/onsi/ginkgo"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"

	gcm "google.golang.org/api/monitoring/v3"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/client-go/discovery"
	kubeaggrcs "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	customclient "k8s.io/metrics/pkg/client/custom_metrics"
)

var _ = instrumentation.SIGDescribe("Stackdriver Monitoring", func() {
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke")
	})

	f := framework.NewDefaultFramework("stackdriver-monitoring")
	var kubeClient clientset.Interface
	var kubeAggrClient kubeaggrcs.Interface
	var customMetricsClient customclient.CustomMetricsClient
	var discoveryClient *discovery.DiscoveryClient

	It("should run Custom Metrics - Stackdriver Adapter [Feature:StackdriverMonitoring]", func() {
		kubeClient = f.ClientSet
		kubeAggrClient = f.AggregatorClient
		config, err := framework.LoadConfig()
		if err != nil {
			framework.Failf("Failed to load config: %s", err)
		}
		customMetricsClient = customclient.NewForConfigOrDie(config)
		discoveryClient = discovery.NewDiscoveryClientForConfigOrDie(config)
		testAdapter(f, kubeClient, kubeAggrClient, customMetricsClient, discoveryClient)
	})
})

func testAdapter(f *framework.Framework, kubeClient clientset.Interface, kubeAggrClient kubeaggrcs.Interface, customMetricsClient customclient.CustomMetricsClient, discoveryClient *discovery.DiscoveryClient) {
	projectId := framework.TestContext.CloudConfig.ProjectID

	ctx := context.Background()
	client, err := google.DefaultClient(ctx, gcm.CloudPlatformScope)

	// Hack for running tests locally
	// If this is your use case, create application default credentials:
	// $ gcloud auth application-default login
	// and uncomment following lines (comment out the two lines above):
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

	framework.ExpectNoError(err)

	// Set up a cluster: create a custom metric and set up k8s-sd adapter
	err = createDescriptors(gcmService, projectId)
	if err != nil {
		framework.Failf("Failed to create metric descriptor: %s", err)
	}
	defer cleanupDescriptors(gcmService, projectId)

	err = createAdapter()
	if err != nil {
		framework.Failf("Failed to set up: %s", err)
	}
	defer cleanupAdapter()

	// Run application that exports the metric
	err = createSDExporterPod(kubeClient)
	if err != nil {
		framework.Failf("Failed to create sd-exporter pod: %s", err)
	}
	defer cleanupSDExporterPod(kubeClient)

	// Wait a short amount of time to create a pod and export some metrics
	time.Sleep(60 * time.Second)

	// Verify responses from Custom Metrics API
	resources, err := discoveryClient.ServerResourcesForGroupVersion("custom.metrics.k8s.io/v1beta1")
	if err != nil {
		framework.Failf("Failed to retrieve a list of supported metrics: %s", err)
	}
	for _, resource := range resources.APIResources {
		if resource.Name != "pods/"+CustomMetricName && resource.Name != "pods/"+UnusedMetricName {
			framework.Failf("Unexpected metric %s. Only metric %s should be supported", resource.Name, CustomMetricName)
		}
	}
	value, err := customMetricsClient.NamespacedMetrics("default").GetForObject(schema.GroupKind{Group: "", Kind: "Pod"}, "sd-exporter-1", CustomMetricName)
	if err != nil {
		framework.Failf("Failed query: %s", err)
	}
	if value.Value.Value() != MetricValue1 {
		framework.Failf("Unexpected metric value for metric %s: expected %v but received %v", CustomMetricName, MetricValue1, value.Value)
	}
	filter, err := labels.NewRequirement("name", selection.Equals, []string{"sd-exporter"})
	if err != nil {
		framework.Failf("Couldn't create a label filter")
	}
	values, err := customMetricsClient.NamespacedMetrics("default").GetForObjects(schema.GroupKind{Group: "", Kind: "Pod"}, labels.NewSelector().Add(*filter), CustomMetricName)
	if err != nil {
		framework.Failf("Failed query: %s", err)
	}
	if len(values.Items) != 2 {
		framework.Failf("Expected results for exactly 2 pods, but %v results received", len(values.Items))
	}
	for _, value := range values.Items {
		if (value.DescribedObject.Name == SDExporterPod1.Name && value.Value.Value() != MetricValue1) ||
			(value.DescribedObject.Name == SDExporterPod2.Name && value.Value.Value() != MetricValue2) {
			framework.Failf("Unexpected metric value for metric %s and pod %s: %v", CustomMetricName, value.DescribedObject.Name, value.Value.Value())
		}
	}

	framework.ExpectNoError(err)
}

func createDescriptors(service *gcm.Service, projectId string) error {
	_, err := service.Projects.MetricDescriptors.Create(fmt.Sprintf("projects/%s", projectId), &gcm.MetricDescriptor{
		Name:       CustomMetricName,
		ValueType:  "INT64",
		Type:       "custom.googleapis.com/" + CustomMetricName,
		MetricKind: "GAUGE",
	}).Do()
	if err != nil {
		return err
	}
	_, err = service.Projects.MetricDescriptors.Create(fmt.Sprintf("projects/%s", projectId), &gcm.MetricDescriptor{
		Name:       UnusedMetricName,
		ValueType:  "INT64",
		Type:       "custom.googleapis.com/" + UnusedMetricName,
		MetricKind: "GAUGE",
	}).Do()
	return err
}

func createSDExporterPod(cs clientset.Interface) error {
	_, err := cs.Core().Pods("default").Create(SDExporterPod1)
	if err != nil {
		return err
	}
	_, err = cs.Core().Pods("default").Create(SDExporterPod2)
	return err
}

func createAdapter() error {
	stat, err := framework.RunKubectl("create", "-f", "https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/master/custom-metrics-stackdriver-adapter/adapter-beta.yaml")
	framework.Logf(stat)
	return err
}

func cleanupDescriptors(service *gcm.Service, projectId string) {
	_, _ = service.Projects.MetricDescriptors.Delete(fmt.Sprintf("projects/%s/metricDescriptors/custom.googleapis.com/%s", projectId, CustomMetricName)).Do()
	_, _ = service.Projects.MetricDescriptors.Delete(fmt.Sprintf("projects/%s/metricDescriptors/custom.googleapis.com/%s", projectId, UnusedMetricName)).Do()
}

func cleanupSDExporterPod(cs clientset.Interface) {
	_ = cs.Core().Pods("default").Delete("sd-exporter-1", &metav1.DeleteOptions{})
	_ = cs.Core().Pods("default").Delete("sd-exporter-2", &metav1.DeleteOptions{})
}

func cleanupAdapter() error {
	stat, err := framework.RunKubectl("delete", "-f", "https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/master/custom-metrics-stackdriver-adapter/adapter-beta.yaml")
	framework.Logf(stat)
	return err
}
