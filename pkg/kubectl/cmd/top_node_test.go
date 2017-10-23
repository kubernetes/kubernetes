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

	"net/url"

	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/metrics/pkg/apis/metrics/v1alpha1"
)

const (
	apiPrefix  = "api"
	apiVersion = "v1"
)

func TestTopNodeAllMetrics(t *testing.T) {
	initTestErrorHandler(t)
	metrics, nodes := testNodeMetricsData()
	expectedMetricsPath := fmt.Sprintf("%s/%s/nodes", baseMetricsAddress, metricsApiVersion)
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)

	f, tf, codec, ns := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		GroupVersion:         legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == expectedMetricsPath && m == "GET":
				body, err := marshallBody(metrics)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
			case p == expectedNodePath && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, nodes)}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\nExpected path: %#v", req, req.URL, expectedMetricsPath)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	tf.ClientConfig = defaultClientConfig()
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdTopNode(f, nil, buf)
	cmd.Run(cmd, []string{})

	// Check the presence of node names in the output.
	result := buf.String()
	for _, m := range metrics.Items {
		if !strings.Contains(result, m.Name) {
			t.Errorf("missing metrics for %s: \n%s", m.Name, result)
		}
	}
}

func TestTopNodeAllMetricsCustomDefaults(t *testing.T) {
	customBaseHeapsterServiceAddress := "/api/v1/namespaces/custom-namespace/services/https:custom-heapster-service:/proxy"
	customBaseMetricsAddress := customBaseHeapsterServiceAddress + "/apis/metrics"

	initTestErrorHandler(t)
	metrics, nodes := testNodeMetricsData()
	expectedMetricsPath := fmt.Sprintf("%s/%s/nodes", customBaseMetricsAddress, metricsApiVersion)
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)

	f, tf, codec, ns := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		GroupVersion:         legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == expectedMetricsPath && m == "GET":
				body, err := marshallBody(metrics)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
			case p == expectedNodePath && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, nodes)}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\nExpected path: %#v", req, req.URL, expectedMetricsPath)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	tf.ClientConfig = defaultClientConfig()
	buf := bytes.NewBuffer([]byte{})

	opts := &TopNodeOptions{
		HeapsterOptions: HeapsterTopOptions{
			Namespace: "custom-namespace",
			Scheme:    "https",
			Service:   "custom-heapster-service",
		},
	}
	cmd := NewCmdTopNode(f, opts, buf)
	cmd.Run(cmd, []string{})

	// Check the presence of node names in the output.
	result := buf.String()
	for _, m := range metrics.Items {
		if !strings.Contains(result, m.Name) {
			t.Errorf("missing metrics for %s: \n%s", m.Name, result)
		}
	}
}

func TestTopNodeWithNameMetrics(t *testing.T) {
	initTestErrorHandler(t)
	metrics, nodes := testNodeMetricsData()
	expectedMetrics := metrics.Items[0]
	expectedNode := nodes.Items[0]
	nonExpectedMetrics := v1alpha1.NodeMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[1:],
	}
	expectedPath := fmt.Sprintf("%s/%s/nodes/%s", baseMetricsAddress, metricsApiVersion, expectedMetrics.Name)
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes/%s", apiPrefix, apiVersion, expectedMetrics.Name)

	f, tf, codec, ns := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		GroupVersion:         legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == expectedPath && m == "GET":
				body, err := marshallBody(expectedMetrics)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
			case p == expectedNodePath && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &expectedNode)}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\nExpected path: %#v", req, req.URL, expectedPath)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	tf.ClientConfig = defaultClientConfig()
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdTopNode(f, nil, buf)
	cmd.Run(cmd, []string{expectedMetrics.Name})

	// Check the presence of node names in the output.
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

func TestTopNodeWithLabelSelectorMetrics(t *testing.T) {
	initTestErrorHandler(t)
	metrics, nodes := testNodeMetricsData()
	expectedMetrics := v1alpha1.NodeMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[0:1],
	}
	expectedNodes := api.NodeList{
		ListMeta: nodes.ListMeta,
		Items:    nodes.Items[0:1],
	}
	nonExpectedMetrics := v1alpha1.NodeMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[1:],
	}
	label := "key=value"
	expectedPath := fmt.Sprintf("%s/%s/nodes", baseMetricsAddress, metricsApiVersion)
	expectedQuery := fmt.Sprintf("labelSelector=%s", url.QueryEscape(label))
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)

	f, tf, codec, ns := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		GroupVersion:         legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m, q := req.URL.Path, req.Method, req.URL.RawQuery; {
			case p == expectedPath && m == "GET" && q == expectedQuery:
				body, err := marshallBody(expectedMetrics)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
			case p == expectedNodePath && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &expectedNodes)}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\nExpected path: %#v", req, req.URL, expectedPath)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	tf.ClientConfig = defaultClientConfig()
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdTopNode(f, nil, buf)
	cmd.Flags().Set("selector", label)
	cmd.Run(cmd, []string{})

	// Check the presence of node names in the output.
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
