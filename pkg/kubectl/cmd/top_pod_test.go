/*
Copyright 2016 The Kubernetes Authors.

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

package cmd

import (
	"bytes"
	"fmt"
	"net/http"
	"strings"
	"testing"

	metrics_api "k8s.io/heapster/metrics/apis/metrics/v1alpha1"
	"k8s.io/kubernetes/pkg/client/restclient/fake"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"net/url"
)

func TestTopPodAllNamespacesMetrics(t *testing.T) {
	initTestErrorHandler(t)
	metrics := testPodMetricsData()
	firstTestNamespace := "testnamespace"
	secondTestNamespace := "secondtestns"
	thirdTestNamespace := "thirdtestns"
	metrics.Items[0].Namespace = firstTestNamespace
	metrics.Items[1].Namespace = secondTestNamespace
	metrics.Items[2].Namespace = thirdTestNamespace

	expectedPath := fmt.Sprintf("%s/%s/pods", baseMetricsAddress, metricsApiVersion)

	f, tf, _, ns := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == expectedPath && m == "GET":
				body, err := marshallBody(metrics)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\nExpected path: %#v", req, req.URL, expectedPath)
				return nil, nil
			}
		}),
	}
	tf.Namespace = firstTestNamespace
	tf.ClientConfig = defaultClientConfig()
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdTopPod(f, buf)
	cmd.Flags().Set("all-namespaces", "true")
	cmd.Run(cmd, []string{})

	// Check the presence of pod names and namespaces in the output.
	result := buf.String()
	for _, m := range metrics.Items {
		if !strings.Contains(result, m.Name) {
			t.Errorf("missing metrics for %s: \n%s", m.Name, result)
		}
		if !strings.Contains(result, m.Namespace) {
			t.Errorf("missing metrics for %s/%s: \n%s", m.Namespace, m.Name, result)
		}
	}
}

func TestTopPodAllInNamespaceMetrics(t *testing.T) {
	initTestErrorHandler(t)
	metrics := testPodMetricsData()
	testNamespace := "testnamespace"
	nonTestNamespace := "anothernamespace"
	expectedMetrics := metrics_api.PodMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[0:2],
	}
	for _, m := range expectedMetrics.Items {
		m.Namespace = testNamespace
	}
	nonExpectedMetrics := metrics_api.PodMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[2:],
	}
	for _, m := range nonExpectedMetrics.Items {
		m.Namespace = nonTestNamespace
	}
	expectedPath := fmt.Sprintf("%s/%s/namespaces/%s/pods", baseMetricsAddress, metricsApiVersion, testNamespace)

	f, tf, _, ns := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == expectedPath && m == "GET":
				body, err := marshallBody(expectedMetrics)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\nExpected path: %#v", req, req.URL, expectedPath)
				return nil, nil
			}
		}),
	}
	tf.Namespace = testNamespace
	tf.ClientConfig = defaultClientConfig()
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdTopPod(f, buf)
	cmd.Run(cmd, []string{})

	// Check the presence of pod names in the output.
	result := buf.String()
	for _, m := range expectedMetrics.Items {
		if !strings.Contains(result, m.Name) {
			t.Errorf("missing metrics for %s: \n%s", m.Name, result)
		}
	}
	for _, m := range nonExpectedMetrics.Items {
		if strings.Contains(result, m.Name) {
			t.Errorf("unexpected metrics for %s: \n%s", m.Name, result)
		}
	}
}

func TestTopPodWithNameMetrics(t *testing.T) {
	initTestErrorHandler(t)
	metrics := testPodMetricsData()
	expectedMetrics := metrics.Items[0]
	nonExpectedMetrics := metrics_api.PodMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[1:],
	}
	testNamespace := "testnamespace"
	expectedMetrics.Namespace = testNamespace
	expectedPath := fmt.Sprintf("%s/%s/namespaces/%s/pods/%s", baseMetricsAddress, metricsApiVersion, testNamespace, expectedMetrics.Name)

	f, tf, _, ns := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == expectedPath && m == "GET":
				body, err := marshallBody(expectedMetrics)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\nExpected path: %#v", req, req.URL, expectedPath)
				return nil, nil
			}
		}),
	}
	tf.Namespace = testNamespace
	tf.ClientConfig = defaultClientConfig()
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdTopPod(f, buf)
	cmd.Run(cmd, []string{expectedMetrics.Name})

	// Check the presence of pod names in the output.
	result := buf.String()
	if !strings.Contains(result, expectedMetrics.Name) {
		t.Errorf("missing metrics for %s: \n%s", expectedMetrics.Name, result)
	}
	for _, m := range nonExpectedMetrics.Items {
		if strings.Contains(result, m.Name) {
			t.Errorf("unexpected metrics for %s: \n%s", m.Name, result)
		}
	}
}

func TestTopPodWithLabelSelectorMetrics(t *testing.T) {
	initTestErrorHandler(t)
	metrics := testPodMetricsData()
	expectedMetrics := metrics_api.PodMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[0:2],
	}
	nonExpectedMetrics := metrics_api.PodMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[2:],
	}
	label := "key=value"
	testNamespace := "testnamespace"
	expectedPath := fmt.Sprintf("%s/%s/namespaces/%s/pods", baseMetricsAddress, metricsApiVersion, testNamespace)
	expectedQuery := fmt.Sprintf("labelSelector=%s", url.QueryEscape(label))

	f, tf, _, ns := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m, q := req.URL.Path, req.Method, req.URL.RawQuery; {
			case p == expectedPath && m == "GET" && q == expectedQuery:
				body, err := marshallBody(expectedMetrics)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\nExpected path: %#v", req, req.URL, expectedPath)
				return nil, nil
			}
		}),
	}
	tf.Namespace = testNamespace
	tf.ClientConfig = defaultClientConfig()
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdTopPod(f, buf)
	cmd.Flags().Set("selector", label)
	cmd.Run(cmd, []string{})

	// Check the presence of pod names in the output.
	result := buf.String()
	for _, m := range expectedMetrics.Items {
		if !strings.Contains(result, m.Name) {
			t.Errorf("missing metrics for %s: \n%s", m.Name, result)
		}
	}
	for _, m := range nonExpectedMetrics.Items {
		if strings.Contains(result, m.Name) {
			t.Errorf("unexpected metrics for %s: \n%s", m.Name, result)
		}
	}
}

func TestTopPodWithContainersMetrics(t *testing.T) {
	initTestErrorHandler(t)
	metrics := testPodMetricsData()
	expectedMetrics := metrics.Items[0]
	nonExpectedMetrics := metrics_api.PodMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[1:],
	}
	testNamespace := "testnamespace"
	expectedMetrics.Namespace = testNamespace
	expectedPath := fmt.Sprintf("%s/%s/namespaces/%s/pods/%s", baseMetricsAddress, metricsApiVersion, testNamespace, expectedMetrics.Name)

	f, tf, _, ns := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == expectedPath && m == "GET":
				body, err := marshallBody(expectedMetrics)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\nExpected path: %#v", req, req.URL, expectedPath)
				return nil, nil
			}
		}),
	}
	tf.Namespace = testNamespace
	tf.ClientConfig = defaultClientConfig()
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdTopPod(f, buf)
	cmd.Flags().Set("containers", "true")
	cmd.Run(cmd, []string{expectedMetrics.Name})

	// Check the presence of pod names in the output.
	result := buf.String()
	if !strings.Contains(result, expectedMetrics.Name) {
		t.Errorf("missing metrics for %s: \n%s", expectedMetrics.Name, result)
	}
	for _, m := range expectedMetrics.Containers {
		if !strings.Contains(result, m.Name) {
			t.Errorf("missing metrics for container %s: \n%s", m.Name, result)
		}
	}
	for _, m := range nonExpectedMetrics.Items {
		if strings.Contains(result, m.Name) {
			t.Errorf("unexpected metrics for %s: \n%s", m.Name, result)
		}
	}
}
