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

package nativehistograms

import (
	"context"
	"fmt"
	"net/http"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	metricsfeatures "k8s.io/component-base/metrics/features"
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
		return nil, fmt.Errorf("request failed: %w", err)
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

// TestAPIServerNativeHistogramMetrics verifies that native histogram metrics are properly
// exposed in apiserver when the NativeHistograms feature gate is enabled.
func TestAPIServerNativeHistogramMetrics(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, metricsfeatures.NativeHistograms, true)
	s := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer s.TearDownFn()

	// Make a request to ensure histogram metrics have data
	client := clientset.NewForConfigOrDie(s.ClientConfig)
	if _, err := client.CoreV1().Pods(metav1.NamespaceDefault).List(context.TODO(), metav1.ListOptions{}); err != nil {
		t.Fatalf("unexpected error getting pods: %v", err)
	}

	transport, err := restclient.TransportFor(s.ClientConfig)
	if err != nil {
		t.Fatalf("failed to create transport: %v", err)
	}
	httpClient := &http.Client{Transport: transport}

	// Scrape metrics in protobuf format to get native histogram data
	histogramMetric := "apiserver_request_duration_seconds"
	metrics, err := testutil.ScrapeMetricsProto(s.ClientConfig.Host+"/metrics", httpClient)
	if err != nil {
		t.Fatalf("failed to scrape metrics: %v", err)
	}

	mf, ok := metrics[histogramMetric]
	if !ok {
		t.Fatalf("metric %q not found", histogramMetric)
	}

	// Verify native histogram data is present
	testutil.AssertHasNativeHistogram(t, mf, nil)

	// Verify classic histogram buckets are still exposed
	textMetrics, err := scrapeMetrics(s)
	if err != nil {
		t.Fatalf("failed to scrape text metrics: %v", err)
	}
	checkForExpectedMetrics(t, textMetrics, []string{
		"apiserver_request_duration_seconds_bucket",
		"apiserver_request_duration_seconds_sum",
		"apiserver_request_duration_seconds_count",
	})
}
