/*
Copyright 2024 The Kubernetes Authors.

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

package top

import (
	"bytes"
	"io"
	"net/http"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/rest/fake"
	core "k8s.io/client-go/testing"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
	metricsv1beta1api "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
)

func TestEdgeCases(t *testing.T) {
	t.Run("EmptyNodeName", func(t *testing.T) {
		selector, err := fields.ParseSelector("spec.nodeName=")
		if err != nil {
			t.Fatalf("Failed to parse field selector: %v", err)
		}
		nodeNameValue, _ := extractNodeNameFieldSelector(selector)
		if nodeNameValue != "" {
			t.Errorf("Expected empty node name, got %q", nodeNameValue)
		}
	})

	t.Run("NodeNameWithSpaces", func(t *testing.T) {
		selector, err := fields.ParseSelector("spec.nodeName=worker node 1")
		if err != nil {
			t.Fatalf("Failed to parse field selector: %v", err)
		}
		nodeNameValue, _ := extractNodeNameFieldSelector(selector)
		if nodeNameValue != "worker node 1" {
			t.Errorf("Expected 'worker node 1', got %q", nodeNameValue)
		}
	})

	t.Run("MultipleSpecNodeName", func(t *testing.T) {
		// This should not happen in normal usage but let's test robustness
		selector, err := fields.ParseSelector("spec.nodeName=worker-1,spec.nodeName=worker-2")
		if err != nil {
			t.Fatalf("Failed to parse field selector: %v", err)
		}
		nodeNameValue, _ := extractNodeNameFieldSelector(selector)
		// Should return one of them (behavior is undefined but should not crash)
		if nodeNameValue != "worker-1" && nodeNameValue != "worker-2" {
			t.Errorf("Expected either 'worker-1' or 'worker-2', got %q", nodeNameValue)
		}
	})

	t.Run("CombinedWithNamespace", func(t *testing.T) {
		selector, err := fields.ParseSelector("spec.nodeName=worker-1,metadata.namespace=default")
		if err != nil {
			t.Fatalf("Failed to parse field selector: %v", err)
		}
		nodeNameValue, filteredSelector := extractNodeNameFieldSelector(selector)
		if nodeNameValue != "worker-1" {
			t.Errorf("Expected 'worker-1', got %q", nodeNameValue)
		}
		if !strings.Contains(filteredSelector.String(), "metadata.namespace=default") {
			t.Errorf("Expected filtered selector to contain namespace filter, got %q", filteredSelector.String())
		}
	})
}

func TestTopPodWithNodeNameComplexScenarios(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	
	// Test scenario with pods that don't exist (deleted after metrics were retrieved)
	testMetrics := []metricsv1beta1api.PodMetrics{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "existing-pod", Namespace: "test"},
			Containers: []metricsv1beta1api.ContainerMetrics{{
				Name: "container1",
				Usage: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("100m"),
					corev1.ResourceMemory: resource.MustParse("200Mi"),
				},
			}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "deleted-pod", Namespace: "test"},
			Containers: []metricsv1beta1api.ContainerMetrics{{
				Name: "container2",
				Usage: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("150m"),
					corev1.ResourceMemory: resource.MustParse("250Mi"),
				},
			}},
		},
	}

	testPods := []corev1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "existing-pod", Namespace: "test"},
			Spec: corev1.PodSpec{NodeName: "worker-1"},
		},
		// deleted-pod is intentionally missing to test error handling
	}

	fakemetricsClientset := &metricsfake.Clientset{}
	fakemetricsClientset.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		res := &metricsv1beta1api.PodMetricsList{
			ListMeta: metav1.ListMeta{
				ResourceVersion: "2",
			},
			Items: testMetrics,
		}
		return true, res, nil
	})

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	ns := scheme.Codecs.WithoutConversion()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
			case p == "/apis":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(apisbodyWithMetrics)))}, nil
			case p == "/api/v1/namespaces/test/pods/existing-pod" && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &testPods[0])}, nil
			case p == "/api/v1/namespaces/test/pods/deleted-pod" && m == "GET":
				// Return 404 for deleted pod
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(`{"kind":"Status","message":"pods \"deleted-pod\" not found"}`)))}, nil
			default:
				t.Logf("unexpected request: %s %s", m, p)
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte{}))}, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdTopPod(tf, nil, streams)

	cmdOptions := &TopPodOptions{
		FieldSelector: "spec.nodeName=worker-1",
		IOStreams:     streams,
	}

	if err := cmdOptions.Complete(tf, cmd, []string{}); err != nil {
		t.Fatal(err)
	}
	
	cmdOptions.MetricsClient = fakemetricsClientset
	
	if err := cmdOptions.Validate(); err != nil {
		t.Fatal(err)
	}
	
	if err := cmdOptions.RunTopPod(); err != nil {
		t.Fatal(err)
	}

	output := buf.String()
	t.Logf("Command output:\n%s", output)
	
	// Should contain only existing-pod (deleted-pod should be filtered out due to 404)
	if !strings.Contains(output, "existing-pod") {
		t.Errorf("Expected output to contain existing-pod")
	}
	if strings.Contains(output, "deleted-pod") {
		t.Errorf("Expected output to NOT contain deleted-pod (should be filtered due to 404)")
	}
}

func TestFilterMetricsByNodeNameEdgeCases(t *testing.T) {
	// Test empty input
	result, err := filterMetricsByNodeName([]metricsv1beta1api.PodMetrics{}, "worker-1", "test", nil)
	if err != nil {
		t.Errorf("Unexpected error for empty input: %v", err)
	}
	if len(result) != 0 {
		t.Errorf("Expected empty result for empty input, got %d items", len(result))
	}

	// Test empty node name
	testMetrics := []metricsv1beta1api.PodMetrics{
		{ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "test"}},
	}
	result, err = filterMetricsByNodeName(testMetrics, "", "test", nil)
	if err != nil {
		t.Errorf("Unexpected error for empty node name: %v", err)
	}
	if len(result) != 1 {
		t.Errorf("Expected 1 item for empty node name (should return all), got %d items", len(result))
	}
}
