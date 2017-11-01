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
	"time"

	"golang.org/x/oauth2/google"
	clientset "k8s.io/client-go/kubernetes"

	. "github.com/onsi/ginkgo"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"

	gcm "google.golang.org/api/monitoring/v3"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/client-go/discovery"
	kubeaggrcs "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
	customclient "k8s.io/metrics/pkg/client/custom_metrics"
)

const (
	stackdriverExporterPod1  = "stackdriver-exporter-1"
	stackdriverExporterPod2  = "stackdriver-exporter-2"
	stackdriverExporterLabel = "stackdriver-exporter"
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

	It("should run Custom Metrics - Stackdriver Adapter [Feature:StackdriverCustomMetrics]", func() {
		kubeClient = f.ClientSet
		kubeAggrClient = f.AggregatorClient
		config, err := framework.LoadConfig()
		if err != nil {
			framework.Failf("Failed to load config: %s", err)
		}
		customMetricsClient = customclient.NewForConfigOrDie(config)
		discoveryClient = discovery.NewDiscoveryClientForConfigOrDie(config)
		testAdapter(f, kubeClient, customMetricsClient, discoveryClient)
	})
})

func testAdapter(f *framework.Framework, kubeClient clientset.Interface, customMetricsClient customclient.CustomMetricsClient, discoveryClient *discovery.DiscoveryClient) {
	projectId := framework.TestContext.CloudConfig.ProjectID

	ctx := context.Background()
	client, err := google.DefaultClient(ctx, gcm.CloudPlatformScope)

	// Hack for running tests locally, needed to authenticate in Stackdriver
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

	// Set up a cluster: create a custom metric and set up k8s-sd adapter
	err = CreateDescriptors(gcmService, projectId)
	if err != nil {
		framework.Failf("Failed to create metric descriptor: %s", err)
	}
	defer CleanupDescriptors(gcmService, projectId)

	err = CreateAdapter()
	if err != nil {
		framework.Failf("Failed to set up: %s", err)
	}
	defer CleanupAdapter()

	// Run application that exports the metric
	err = createSDExporterPods(f, kubeClient)
	if err != nil {
		framework.Failf("Failed to create stackdriver-exporter pod: %s", err)
	}
	defer cleanupSDExporterPod(f, kubeClient)

	// Wait a short amount of time to create a pod and export some metrics
	// TODO: add some events to wait for instead of fixed amount of time
	//       i.e. pod creation, first time series exported
	time.Sleep(60 * time.Second)

	// Verify responses from Custom Metrics API
	resources, err := discoveryClient.ServerResourcesForGroupVersion("custom.metrics.k8s.io/v1beta1")
	if err != nil {
		framework.Failf("Failed to retrieve a list of supported metrics: %s", err)
	}
	gotCustomMetric, gotUnusedMetric := false, false
	for _, resource := range resources.APIResources {
		if resource.Name == "pods/"+CustomMetricName {
			gotCustomMetric = true
		} else if resource.Name == "pods/"+UnusedMetricName {
			gotUnusedMetric = true
		} else {
			framework.Failf("Unexpected metric %s. Only metric %s should be supported", resource.Name, CustomMetricName)
		}
	}
	if !gotCustomMetric {
		framework.Failf("Metric '%s' expected but not received", CustomMetricName)
	}
	if !gotUnusedMetric {
		framework.Failf("Metric '%s' expected but not received", UnusedMetricName)
	}
	value, err := customMetricsClient.NamespacedMetrics(f.Namespace.Name).GetForObject(schema.GroupKind{Group: "", Kind: "Pod"}, stackdriverExporterPod1, CustomMetricName)
	if err != nil {
		framework.Failf("Failed query: %s", err)
	}
	if value.Value.Value() != CustomMetricValue {
		framework.Failf("Unexpected metric value for metric %s: expected %v but received %v", CustomMetricName, CustomMetricValue, value.Value)
	}
	filter, err := labels.NewRequirement("name", selection.Equals, []string{stackdriverExporterLabel})
	if err != nil {
		framework.Failf("Couldn't create a label filter")
	}
	values, err := customMetricsClient.NamespacedMetrics(f.Namespace.Name).GetForObjects(schema.GroupKind{Group: "", Kind: "Pod"}, labels.NewSelector().Add(*filter), CustomMetricName)
	if err != nil {
		framework.Failf("Failed query: %s", err)
	}
	if len(values.Items) != 1 {
		framework.Failf("Expected results for exactly 1 pod, but %v results received", len(values.Items))
	}
	if values.Items[0].DescribedObject.Name != stackdriverExporterPod1 || values.Items[0].Value.Value() != CustomMetricValue {
		framework.Failf("Unexpected metric value for metric %s and pod %s: %v", CustomMetricName, values.Items[0].DescribedObject.Name, values.Items[0].Value.Value())
	}
}

func cleanupSDExporterPod(f *framework.Framework, cs clientset.Interface) {
	err := cs.CoreV1().Pods(f.Namespace.Name).Delete(stackdriverExporterPod1, &metav1.DeleteOptions{})
	if err != nil {
		framework.Logf("Failed to delete %s pod: %v", stackdriverExporterPod1, err)
	}
	err = cs.CoreV1().Pods(f.Namespace.Name).Delete(stackdriverExporterPod2, &metav1.DeleteOptions{})
	if err != nil {
		framework.Logf("Failed to delete %s pod: %v", stackdriverExporterPod2, err)
	}
}

func createSDExporterPods(f *framework.Framework, cs clientset.Interface) error {
	_, err := cs.CoreV1().Pods(f.Namespace.Name).Create(StackdriverExporterPod(stackdriverExporterPod1, f.Namespace.Name, stackdriverExporterLabel, CustomMetricName, CustomMetricValue))
	if err != nil {
		return err
	}
	_, err = cs.CoreV1().Pods(f.Namespace.Name).Create(StackdriverExporterPod(stackdriverExporterPod2, f.Namespace.Name, stackdriverExporterLabel, UnusedMetricName, UnusedMetricValue))
	return err
}
