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
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/prometheus/common/model"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientgofeaturegate "k8s.io/client-go/features"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/cloud-provider/fake"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2/ktesting"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/integration/util"
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

func checkForMetricsNotExist(t *testing.T, metrics testutil.Metrics, unexpectedMetrics []string) {
	for _, unexpected := range unexpectedMetrics {
		if _, found := metrics[unexpected]; found {
			t.Errorf("API server metrics include unexpected metric %q", unexpected)
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
	s := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer s.TearDownFn()

	// Make a request to the apiserver to ensure there's at least one data point
	// for the metrics we're expecting -- otherwise, they won't be exported.
	client := clientset.NewForConfigOrDie(s.ClientConfig)
	if _, err := client.CoreV1().Pods(metav1.NamespaceDefault).List(context.TODO(), metav1.ListOptions{}); err != nil {
		t.Fatalf("unexpected error getting pods: %v", err)
	}

	// Make a request to a deprecated API to ensure there's at least one data point
	if _, err := client.FlowcontrolV1beta3().FlowSchemas().List(context.TODO(), metav1.ListOptions{}); err != nil {
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

var reflectorMetrics = []string{
	"reflector_lists_total",
	"reflector_list_duration_seconds_sum",
	"reflector_list_duration_seconds_count",
	"reflector_list_duration_seconds_bucket",
	"reflector_items_per_list_bucket",
	"reflector_watchLists_total",
	"reflector_watches_total",
	"reflector_short_watches_total",
	"reflector_watch_duration_seconds_bucket",
	"reflector_items_per_watch_bucket",
	"reflector_last_resource_version",
}

func TestAPIServerReflectorMetrics(t *testing.T) {
	currentReflectorMetricsFactory := cache.SwapReflectorMetricsFactory(cache.NewFakeReflectorMetricsFactory())
	defer cache.SwapReflectorMetricsFactory(currentReflectorMetricsFactory)
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, featuregate.Feature(string(clientgofeaturegate.InformerMetrics)), false)()
	// reset default registry metrics
	legacyregistry.Reset()

	result := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--feature-gates", "InformerMetrics=true"}, framework.SharedEtcd())
	defer result.TearDownFn()

	metrics, err := scrapeMetrics(result)
	if err != nil {
		t.Fatal(err)
	}
	checkForExpectedMetrics(t, metrics, reflectorMetrics)
}

func TestAPIServerReflectorMetricsNotExist(t *testing.T) {
	// reset default registry metrics
	legacyregistry.Reset()
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{}, framework.SharedEtcd())
	defer result.TearDownFn()

	metrics, err := scrapeMetrics(result)
	if err != nil {
		t.Fatal(err)
	}
	checkForMetricsNotExist(t, metrics, reflectorMetrics)
}

func fakeCloudProviderFactory(io.Reader) (cloudprovider.Interface, error) {
	return &fake.Cloud{
		DisableRoutes: true, // disable routes for server tests, otherwise --cluster-cidr is required
	}, nil
}

func TestComponentReflectorMetrics(t *testing.T) {
	if !cloudprovider.IsCloudProvider("fake") {
		cloudprovider.RegisterCloudProvider("fake", fakeCloudProviderFactory)
	}

	// start apiserver
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{}, framework.SharedEtcd())
	defer server.TearDownFn()

	// create kubeconfig for the apiserver
	apiserverConfig, err := os.CreateTemp("", "kubeconfig")
	if err != nil {
		t.Fatal(err)
	}
	apiserverConfig.WriteString(fmt.Sprintf(`
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: %s
    certificate-authority: %s
  name: integration
contexts:
- context:
    cluster: integration
    user: controller-manager
  name: default-context
current-context: default-context
users:
- name: controller-manager
  user:
    token: %s
`, server.ClientConfig.Host, server.ServerOpts.SecureServing.ServerCert.CertKey.CertFile, server.ClientConfig.BearerToken))
	apiserverConfig.Close()

	tests := []struct {
		name       string
		tester     util.ComponentTester
		extraFlags []string
	}{
		{"kube-controller-manager", util.NewKubeControllerManagerTester("daemonset-controller"), nil},
		{"cloud-controller-manager", util.NewCloudControllerManagerTester(), []string{"--cloud-provider=fake", "--webhook-secure-port=0"}},
		{"kube-scheduler", util.NewKubeSchedulerTester(), nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			currentReflectorMetricsFactory := cache.SwapReflectorMetricsFactory(cache.NewFakeReflectorMetricsFactory())
			defer cache.SwapReflectorMetricsFactory(currentReflectorMetricsFactory)
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, featuregate.Feature(string(clientgofeaturegate.InformerMetrics)), false)()
			// reset default registry metrics
			legacyregistry.Reset()
			testComponentReflectorsMetricsWithSecureServing(t, tt.tester, apiserverConfig.Name(), tt.extraFlags)
		})
	}
}

func testComponentReflectorsMetricsWithSecureServing(t *testing.T, tester util.ComponentTester, kubeconfig string, extraFlags []string) {
	flags := []string{
		"--authorization-always-allow-paths", "/healthz,/metrics",
		"--kubeconfig", kubeconfig,
		"--leader-elect=false",
		"--feature-gates=InformerMetrics=true",
	}
	_, ctx := ktesting.NewTestContext(t)
	_, secureInfo, tearDownFn, err := tester.StartTestServer(ctx, append(flags, extraFlags...))
	if tearDownFn != nil {
		defer tearDownFn()
	}
	if err != nil {
		t.Fatalf("StartTestServer() error = %v", err)
	}
	url := fmt.Sprintf("https://%s/metrics", secureInfo.Listener.Addr().String())
	url = strings.Replace(url, "[::]", "127.0.0.1", -1) // switch to IPv4 because the self-signed cert does not support [::]

	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,
		},
	}

	client := &http.Client{Transport: tr}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		t.Fatal(err)
	}
	r, err := client.Do(req)
	if err != nil {
		t.Fatalf("failed to GET metrics from component: %v", err)
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("failed to read response body: %v", err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("failed to GET metrics from component, got: %d %q", r.StatusCode, string(body))
	}
	metrics := testutil.NewMetrics()
	err = testutil.ParseMetrics(string(body), &metrics)
	if err != nil {
		t.Fatal(err)
	}
	checkForExpectedMetrics(t, metrics, reflectorMetrics)
}
