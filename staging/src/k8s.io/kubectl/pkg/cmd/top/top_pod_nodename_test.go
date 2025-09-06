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

func TestExtractNodeNameFieldSelector(t *testing.T) {
	tests := []struct {
		name           string
		fieldSelector  string
		expectedNode   string
		expectedOther  string
	}{
		{
			name:           "spec.nodeName only",
			fieldSelector:  "spec.nodeName=worker-1",
			expectedNode:   "worker-1",
			expectedOther:  "",
		},
		{
			name:           "spec.nodeName with other field",
			fieldSelector:  "spec.nodeName=worker-1,metadata.namespace=default",
			expectedNode:   "worker-1",
			expectedOther:  "metadata.namespace=default",
		},
		{
			name:           "no spec.nodeName",
			fieldSelector:  "metadata.namespace=default",
			expectedNode:   "",
			expectedOther:  "metadata.namespace=default",
		},
		{
			name:           "empty selector",
			fieldSelector:  "",
			expectedNode:   "",
			expectedOther:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var selector fields.Selector
			var err error
			
			if tt.fieldSelector == "" {
				selector = fields.Everything()
			} else {
				selector, err = fields.ParseSelector(tt.fieldSelector)
				if err != nil {
					t.Fatalf("Failed to parse field selector: %v", err)
				}
			}

			nodeNameValue, filteredSelector := extractNodeNameFieldSelector(selector)
			
			if nodeNameValue != tt.expectedNode {
				t.Errorf("Expected node name %q, got %q", tt.expectedNode, nodeNameValue)
			}

			filteredSelectorString := filteredSelector.String()
			if tt.expectedOther == "" && filteredSelectorString != "" {
				t.Errorf("Expected empty filtered selector, got %q", filteredSelectorString)
			} else if tt.expectedOther != "" && !strings.Contains(filteredSelectorString, tt.expectedOther) {
				t.Errorf("Expected filtered selector to contain %q, got %q", tt.expectedOther, filteredSelectorString)
			}
		})
	}
}

func TestTopPodWithNodeNameFieldSelector(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	
	// Define test metrics - pods on different nodes
	testMetrics := []metricsv1beta1api.PodMetrics{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod-on-worker1", Namespace: "test"},
			Containers: []metricsv1beta1api.ContainerMetrics{{
				Name: "container1",
				Usage: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("100m"),
					corev1.ResourceMemory: resource.MustParse("200Mi"),
				},
			}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod-on-worker2", Namespace: "test"},
			Containers: []metricsv1beta1api.ContainerMetrics{{
				Name: "container2",
				Usage: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("150m"),
					corev1.ResourceMemory: resource.MustParse("250Mi"),
				},
			}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "another-pod-on-worker1", Namespace: "test"},
			Containers: []metricsv1beta1api.ContainerMetrics{{
				Name: "container3",
				Usage: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("120m"),
					corev1.ResourceMemory: resource.MustParse("180Mi"),
				},
			}},
		},
	}

	// Create test pods with nodeName specs
	testPods := []corev1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod-on-worker1", Namespace: "test"},
			Spec: corev1.PodSpec{NodeName: "worker-1"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod-on-worker2", Namespace: "test"},
			Spec: corev1.PodSpec{NodeName: "worker-2"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "another-pod-on-worker1", Namespace: "test"},
			Spec: corev1.PodSpec{NodeName: "worker-1"},
		},
	}

	// Create fake metrics client
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

	// Create test factory
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	ns := scheme.Codecs.WithoutConversion()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	// Create fake REST client to handle pod GET requests
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
			case p == "/apis":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(apisbodyWithMetrics)))}, nil
			case p == "/api/v1/namespaces/test/pods/pod-on-worker1" && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &testPods[0])}, nil
			case p == "/api/v1/namespaces/test/pods/pod-on-worker2" && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &testPods[1])}, nil  
			case p == "/api/v1/namespaces/test/pods/another-pod-on-worker1" && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &testPods[2])}, nil
			default:
				t.Logf("unexpected request: %s %s", m, p)
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte{}))}, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	// Create streams for output
	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	
	// Create command
	cmd := NewCmdTopPod(tf, nil, streams)

	// Set up options for field selector test
	cmdOptions := &TopPodOptions{
		FieldSelector: "spec.nodeName=worker-1",
		IOStreams:     streams,
	}

	// Complete options
	if err := cmdOptions.Complete(tf, cmd, []string{}); err != nil {
		t.Fatal(err)
	}
	
	// Set metrics client
	cmdOptions.MetricsClient = fakemetricsClientset
	
	// Validate options
	if err := cmdOptions.Validate(); err != nil {
		t.Fatal(err)
	}
	
	// Run the command
	if err := cmdOptions.RunTopPod(); err != nil {
		t.Fatal(err)
	}

	// Check output
	output := buf.String()
	t.Logf("Command output:\n%s", output)
	
	// Should contain pods on worker-1 only
	if !strings.Contains(output, "pod-on-worker1") {
		t.Errorf("Expected output to contain pod-on-worker1")
	}
	if !strings.Contains(output, "another-pod-on-worker1") {
		t.Errorf("Expected output to contain another-pod-on-worker1")
	}
	// Should NOT contain pod on worker-2
	if strings.Contains(output, "pod-on-worker2") {
		t.Errorf("Expected output to NOT contain pod-on-worker2")
	}
}
