/*
Copyright 2015 The Kubernetes Authors.

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

package metrics

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/prometheus/common/model"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func scrapeMetrics(s *kubeapiservertesting.TestServer) (testutil.Metrics, error) {
	client, err := clientset.NewForConfig(s.ClientConfig)
	if err != nil {
		return nil, fmt.Errorf("couldn't create client")
	}

	body, err := client.RESTClient().Get().AbsPath("metrics").DoRaw(context.TODO())
	if err != nil {
		return nil, fmt.Errorf("request failed: %v", err)
	}
	metrics := testutil.NewMetrics()
	err = testutil.ParseMetrics(string(body), &metrics)
	return metrics, err
}

func checkForExpectedMetrics(t *testing.T, metrics testutil.Metrics, expectedMetrics []string) {
	for _, expected := range expectedMetrics {
		if _, found := metrics[expected]; !found {
			t.Errorf("API server metrics did not include expected metric %q", expected)
		}
	}
}

func TestAPIServerProcessMetrics(t *testing.T) {
	if runtime.GOOS == "darwin" || runtime.GOOS == "windows" {
		t.Skipf("not supported on GOOS=%s", runtime.GOOS)
	}

	s := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer s.TearDownFn()

	metrics, err := scrapeMetrics(s)
	if err != nil {
		t.Fatal(err)
	}
	checkForExpectedMetrics(t, metrics, []string{
		"process_start_time_seconds",
		"process_cpu_seconds_total",
		"process_open_fds",
		"process_resident_memory_bytes",
	})
}

func TestAPIServerStorageMetrics(t *testing.T) {
	config := framework.SharedEtcd()
	config.Transport.ServerList = []string{config.Transport.ServerList[0], config.Transport.ServerList[0]}
	s := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), config)
	defer s.TearDownFn()

	metrics, err := scrapeMetrics(s)
	if err != nil {
		t.Fatal(err)
	}

	samples, ok := metrics["apiserver_storage_size_bytes"]
	if !ok {
		t.Fatalf("apiserver_storage_size_bytes metric not exposed")
	}
	if len(samples) != 1 {
		t.Fatalf("Unexpected number of samples in apiserver_storage_size_bytes")
	}

	if samples[0].Value == -1 {
		t.Errorf("Unexpected non-zero apiserver_storage_size_bytes, got: %s", samples[0].Value)
	}
}

func TestAPIServerMetrics(t *testing.T) {
	// KUBE_APISERVER_SERVE_REMOVED_APIS_FOR_ONE_RELEASE allows for APIs pending removal to not block tests
	t.Setenv("KUBE_APISERVER_SERVE_REMOVED_APIS_FOR_ONE_RELEASE", "true")

	s := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer s.TearDownFn()

	// Make a request to the apiserver to ensure there's at least one data point
	// for the metrics we're expecting -- otherwise, they won't be exported.
	client := clientset.NewForConfigOrDie(s.ClientConfig)
	if _, err := client.CoreV1().Pods(metav1.NamespaceDefault).List(context.TODO(), metav1.ListOptions{}); err != nil {
		t.Fatalf("unexpected error getting pods: %v", err)
	}

	// Make a request to a deprecated API to ensure there's at least one data point
	if _, err := client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().List(context.TODO(), metav1.ListOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	metrics, err := scrapeMetrics(s)
	if err != nil {
		t.Fatal(err)
	}
	checkForExpectedMetrics(t, metrics, []string{
		"apiserver_requested_deprecated_apis",
		"apiserver_request_total",
		"apiserver_request_duration_seconds_sum",
		"etcd_request_duration_seconds_sum",
	})
}

func TestAPIServerMetricsLabels(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have service account controller running.
	s := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer s.TearDownFn()

	clientConfig := restclient.CopyConfig(s.ClientConfig)
	clientConfig.QPS = -1
	client, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}

	expectedMetrics := []model.Metric{}

	metricLabels := func(group, version, resource, subresource, scope, verb string) model.Metric {
		return map[model.LabelName]model.LabelValue{
			model.LabelName("group"):       model.LabelValue(group),
			model.LabelName("version"):     model.LabelValue(version),
			model.LabelName("resource"):    model.LabelValue(resource),
			model.LabelName("subresource"): model.LabelValue(subresource),
			model.LabelName("scope"):       model.LabelValue(scope),
			model.LabelName("verb"):        model.LabelValue(verb),
		}
	}

	callOrDie := func(_ interface{}, err error) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	appendExpectedMetric := func(metric model.Metric) {
		expectedMetrics = append(expectedMetrics, metric)
	}

	// Call appropriate endpoints to ensure particular metrics will be exposed

	// Namespace-scoped resource
	c := client.CoreV1().Pods(metav1.NamespaceDefault)
	makePod := func(labelValue string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": labelValue},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "container",
						Image: "image",
					},
				},
			},
		}
	}

	callOrDie(c.Create(context.TODO(), makePod("foo"), metav1.CreateOptions{}))
	appendExpectedMetric(metricLabels("", "v1", "pods", "", "resource", "POST"))
	callOrDie(c.Update(context.TODO(), makePod("bar"), metav1.UpdateOptions{}))
	appendExpectedMetric(metricLabels("", "v1", "pods", "", "resource", "PUT"))
	callOrDie(c.UpdateStatus(context.TODO(), makePod("bar"), metav1.UpdateOptions{}))
	appendExpectedMetric(metricLabels("", "v1", "pods", "status", "resource", "PUT"))
	callOrDie(c.Get(context.TODO(), "foo", metav1.GetOptions{}))
	appendExpectedMetric(metricLabels("", "v1", "pods", "", "resource", "GET"))
	callOrDie(c.List(context.TODO(), metav1.ListOptions{}))
	appendExpectedMetric(metricLabels("", "v1", "pods", "", "namespace", "LIST"))
	callOrDie(nil, c.Delete(context.TODO(), "foo", metav1.DeleteOptions{}))
	appendExpectedMetric(metricLabels("", "v1", "pods", "", "resource", "DELETE"))
	// cluster-scoped LIST of namespace-scoped resources
	callOrDie(client.CoreV1().Pods(metav1.NamespaceAll).List(context.TODO(), metav1.ListOptions{}))
	appendExpectedMetric(metricLabels("", "v1", "pods", "", "cluster", "LIST"))

	// Cluster-scoped resource
	cn := client.CoreV1().Namespaces()
	makeNamespace := func(labelValue string) *v1.Namespace {
		return &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": labelValue},
			},
		}
	}

	callOrDie(cn.Create(context.TODO(), makeNamespace("foo"), metav1.CreateOptions{}))
	appendExpectedMetric(metricLabels("", "v1", "namespaces", "", "resource", "POST"))
	callOrDie(cn.Update(context.TODO(), makeNamespace("bar"), metav1.UpdateOptions{}))
	appendExpectedMetric(metricLabels("", "v1", "namespaces", "", "resource", "PUT"))
	callOrDie(cn.UpdateStatus(context.TODO(), makeNamespace("bar"), metav1.UpdateOptions{}))
	appendExpectedMetric(metricLabels("", "v1", "namespaces", "status", "resource", "PUT"))
	callOrDie(cn.Get(context.TODO(), "foo", metav1.GetOptions{}))
	appendExpectedMetric(metricLabels("", "v1", "namespaces", "", "resource", "GET"))
	callOrDie(cn.List(context.TODO(), metav1.ListOptions{}))
	appendExpectedMetric(metricLabels("", "v1", "namespaces", "", "cluster", "LIST"))
	callOrDie(nil, cn.Delete(context.TODO(), "foo", metav1.DeleteOptions{}))
	appendExpectedMetric(metricLabels("", "v1", "namespaces", "", "resource", "DELETE"))

	// Verify if all metrics were properly exported.
	metrics, err := scrapeMetrics(s)
	if err != nil {
		t.Fatal(err)
	}

	samples, ok := metrics["apiserver_request_total"]
	if !ok {
		t.Fatalf("apiserver_request_total metric not exposed")
	}

	hasLabels := func(current, expected model.Metric) bool {
		for key, value := range expected {
			if current[key] != value {
				return false
			}
		}
		return true
	}

	for _, expectedMetric := range expectedMetrics {
		found := false
		for _, sample := range samples {
			if hasLabels(sample.Metric, expectedMetric) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("No sample found for %#v", expectedMetric)
		}
	}
}

func TestAPIServerMetricsLabelsWithAllowList(t *testing.T) {
	testCases := []struct {
		name        string
		metricName  string
		labelName   model.LabelName
		allowValues model.LabelValues
		isHistogram bool
	}{
		{
			name:        "check CounterVec metric",
			metricName:  "apiserver_request_total",
			labelName:   "code",
			allowValues: model.LabelValues{"201", "500"},
		},
		{
			name:        "check GaugeVec metric",
			metricName:  "apiserver_current_inflight_requests",
			labelName:   "request_kind",
			allowValues: model.LabelValues{"mutating"},
		},
		{
			name:        "check Histogram metric",
			metricName:  "apiserver_request_duration_seconds",
			labelName:   "verb",
			allowValues: model.LabelValues{"POST", "LIST"},
			isHistogram: true,
		},
	}

	// Assemble the allow-metric-labels flag.
	var allowMetricLabelFlagStrs []string
	for _, tc := range testCases {
		var allowValuesStr []string
		for _, allowValue := range tc.allowValues {
			allowValuesStr = append(allowValuesStr, string(allowValue))
		}
		allowMetricLabelFlagStrs = append(allowMetricLabelFlagStrs, fmt.Sprintf("\"%s,%s=%s\"", tc.metricName, tc.labelName, strings.Join(allowValuesStr, ",")))
	}
	allowMetricLabelsFlag := "--allow-metric-labels=" + strings.Join(allowMetricLabelFlagStrs, ",")

	testServerFlags := framework.DefaultTestServerFlags()
	testServerFlags = append(testServerFlags, allowMetricLabelsFlag)

	// TODO: have a better way to setup and teardown for all the other tests.
	metrics.Reset()
	defer metrics.Reset()
	metrics.ResetLabelAllowLists()
	defer metrics.ResetLabelAllowLists()
	defer compbasemetrics.ResetLabelValueAllowLists()

	// KUBE_APISERVER_SERVE_REMOVED_APIS_FOR_ONE_RELEASE allows for APIs pending removal to not block tests
	t.Setenv("KUBE_APISERVER_SERVE_REMOVED_APIS_FOR_ONE_RELEASE", "true")
	s := kubeapiservertesting.StartTestServerOrDie(t, nil, testServerFlags, framework.SharedEtcd())
	defer s.TearDownFn()

	metrics, err := scrapeMetrics(s)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			metricName := tc.metricName
			if tc.isHistogram {
				metricName += "_sum"
			}
			samples, found := metrics[metricName]
			if !found {
				t.Fatalf("metric %q not found", metricName)
			}
			for _, sample := range samples {
				if value, ok := sample.Metric[tc.labelName]; ok {
					if !slices.Contains(tc.allowValues, value) && value != "unexpected" {
						t.Fatalf("value %q is not allowed for label %q", value, tc.labelName)
					}
				}
			}
		})

	}

	t.Run("check cardinality_enforcement_unexpected_categorizations_total", func(t *testing.T) {
		samples, found := metrics["cardinality_enforcement_unexpected_categorizations_total"]
		if !found {
			t.Fatal("metric cardinality_enforcement_unexpected_categorizations_total not found")
		}
		if len(samples) != 1 {
			t.Fatalf("Unexpected number of samples in cardinality_enforcement_unexpected_categorizations_total")
		}
		if samples[0].Value <= 0 {
			t.Fatalf("Unexpected non-positive cardinality_enforcement_unexpected_categorizations_total, got: %s", samples[0].Value)
		}

	})
}

func TestAPIServerMetricsPods(t *testing.T) {
	callOrDie := func(_ interface{}, err error) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	makePod := func(labelValue string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": labelValue},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "container",
						Image: "image",
					},
				},
			},
		}
	}

	// Disable ServiceAccount admission plugin as we don't have service account controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	clientConfig := restclient.CopyConfig(server.ClientConfig)
	clientConfig.QPS = -1
	client, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}

	c := client.CoreV1().Pods(metav1.NamespaceDefault)

	for _, tc := range []struct {
		name     string
		executor func()

		want string
	}{
		{
			name: "create pod",
			executor: func() {
				callOrDie(c.Create(context.TODO(), makePod("foo"), metav1.CreateOptions{}))
			},
			want: `apiserver_request_total{code="201", component="apiserver", dry_run="", group="", resource="pods", scope="resource", subresource="", verb="POST", version="v1"}`,
		},
		{
			name: "update pod",
			executor: func() {
				callOrDie(c.Update(context.TODO(), makePod("bar"), metav1.UpdateOptions{}))
			},
			want: `apiserver_request_total{code="200", component="apiserver", dry_run="", group="", resource="pods", scope="resource", subresource="", verb="PUT", version="v1"}`,
		},
		{
			name: "update pod status",
			executor: func() {
				callOrDie(c.UpdateStatus(context.TODO(), makePod("bar"), metav1.UpdateOptions{}))
			},
			want: `apiserver_request_total{code="200", component="apiserver", dry_run="", group="", resource="pods", scope="resource", subresource="status", verb="PUT", version="v1"}`,
		},
		{
			name: "get pod",
			executor: func() {
				callOrDie(c.Get(context.TODO(), "foo", metav1.GetOptions{}))
			},
			want: `apiserver_request_total{code="200", component="apiserver", dry_run="", group="", resource="pods", scope="resource", subresource="", verb="GET", version="v1"}`,
		},
		{
			name: "list pod",
			executor: func() {
				callOrDie(c.List(context.TODO(), metav1.ListOptions{}))
			},
			want: `apiserver_request_total{code="200", component="apiserver", dry_run="", group="", resource="pods", scope="namespace", subresource="", verb="LIST", version="v1"}`,
		},
		{
			name: "delete pod",
			executor: func() {
				callOrDie(nil, c.Delete(context.TODO(), "foo", metav1.DeleteOptions{}))
			},
			want: `apiserver_request_total{code="200", component="apiserver", dry_run="", group="", resource="pods", scope="resource", subresource="", verb="DELETE", version="v1"}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {

			baseSamples, err := getSamples(server)
			if err != nil {
				t.Fatal(err)
			}

			tc.executor()

			updatedSamples, err := getSamples(server)
			if err != nil {
				t.Fatal(err)
			}

			newSamples := diffMetrics(updatedSamples, baseSamples)
			found := false

			for _, sample := range newSamples {
				if sample.Metric.String() == tc.want {
					found = true
					break
				}
			}

			if !found {
				t.Fatalf("could not find metric for API call >%s< among samples >%+v<", tc.name, newSamples)
			}
		})
	}
}

func TestAPIServerMetricsNamespaces(t *testing.T) {
	callOrDie := func(_ interface{}, err error) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	makeNamespace := func(labelValue string) *v1.Namespace {
		return &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": labelValue},
			},
		}
	}

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	clientConfig := restclient.CopyConfig(server.ClientConfig)
	clientConfig.QPS = -1
	client, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}

	c := client.CoreV1().Namespaces()

	for _, tc := range []struct {
		name     string
		executor func()

		want string
	}{
		{
			name: "create namespace",
			executor: func() {
				callOrDie(c.Create(context.TODO(), makeNamespace("foo"), metav1.CreateOptions{}))
			},
			want: `apiserver_request_total{code="201", component="apiserver", dry_run="", group="", resource="namespaces", scope="resource", subresource="", verb="POST", version="v1"}`,
		},
		{
			name: "update namespace",
			executor: func() {
				callOrDie(c.Update(context.TODO(), makeNamespace("bar"), metav1.UpdateOptions{}))
			},
			want: `apiserver_request_total{code="200", component="apiserver", dry_run="", group="", resource="namespaces", scope="resource", subresource="", verb="PUT", version="v1"}`,
		},
		{
			name: "update namespace status",
			executor: func() {
				callOrDie(c.UpdateStatus(context.TODO(), makeNamespace("bar"), metav1.UpdateOptions{}))
			},
			want: `apiserver_request_total{code="200", component="apiserver", dry_run="", group="", resource="namespaces", scope="resource", subresource="status", verb="PUT", version="v1"}`,
		},
		{
			name: "get namespace",
			executor: func() {
				callOrDie(c.Get(context.TODO(), "foo", metav1.GetOptions{}))
			},
			want: `apiserver_request_total{code="200", component="apiserver", dry_run="", group="", resource="namespaces", scope="resource", subresource="", verb="GET", version="v1"}`,
		},
		{
			name: "list namespace",
			executor: func() {
				callOrDie(c.List(context.TODO(), metav1.ListOptions{}))
			},
			want: `apiserver_request_total{code="200", component="apiserver", dry_run="", group="", resource="namespaces", scope="cluster", subresource="", verb="LIST", version="v1"}`,
		},
		{
			name: "delete namespace",
			executor: func() {
				callOrDie(nil, c.Delete(context.TODO(), "foo", metav1.DeleteOptions{}))
			},
			want: `apiserver_request_total{code="200", component="apiserver", dry_run="", group="", resource="namespaces", scope="resource", subresource="", verb="DELETE", version="v1"}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {

			baseSamples, err := getSamples(server)
			if err != nil {
				t.Fatal(err)
			}

			tc.executor()

			updatedSamples, err := getSamples(server)
			if err != nil {
				t.Fatal(err)
			}

			newSamples := diffMetrics(updatedSamples, baseSamples)
			found := false

			for _, sample := range newSamples {
				if sample.Metric.String() == tc.want {
					found = true
					break
				}
			}

			if !found {
				t.Fatalf("could not find metric for API call >%s< among samples >%+v<", tc.name, newSamples)
			}
		})
	}
}

func getSamples(s *kubeapiservertesting.TestServer) (model.Samples, error) {
	metrics, err := scrapeMetrics(s)
	if err != nil {
		return nil, err
	}

	samples, ok := metrics["apiserver_request_total"]
	if !ok {
		return nil, errors.New("apiserver_request_total doesn't exist")
	}
	return samples, nil
}

func diffMetrics(newSamples model.Samples, oldSamples model.Samples) model.Samples {
	samplesDiff := model.Samples{}
	for _, sample := range newSamples {
		if !sampleExistsInSamples(sample, oldSamples) {
			samplesDiff = append(samplesDiff, sample)
		}
	}
	return samplesDiff
}

func sampleExistsInSamples(s *model.Sample, samples model.Samples) bool {
	for _, sample := range samples {
		if sample.Equal(s) {
			return true
		}
	}
	return false
}
