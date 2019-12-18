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

package top

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"net/url"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/rest/fake"
	core "k8s.io/client-go/testing"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
	metricsv1alpha1api "k8s.io/metrics/pkg/apis/metrics/v1alpha1"
	metricsv1beta1api "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
)

const (
	apiPrefix  = "api"
	apiVersion = "v1"
)

func TestTopNodeAllMetrics(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	metrics, nodes := testNodeV1alpha1MetricsData()
	expectedMetricsPath := fmt.Sprintf("%s/%s/nodes", baseMetricsAddress, metricsAPIVersion)
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	ns := scheme.Codecs.WithoutConversion()

	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
			case p == "/apis":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apisbody)))}, nil
			case p == expectedMetricsPath && m == "GET":
				body, err := marshallBody(metrics)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
			case p == expectedNodePath && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, nodes)}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\nExpected path: %#v", req, req.URL, expectedMetricsPath)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()

	cmd := NewCmdTopNode(tf, nil, streams)
	cmd.Flags().Set("no-headers", "true")
	cmd.Run(cmd, []string{})

	// Check the presence of node names in the output.
	result := buf.String()
	for _, m := range metrics.Items {
		if !strings.Contains(result, m.Name) {
			t.Errorf("missing metrics for %s: \n%s", m.Name, result)
		}
	}
	if strings.Contains(result, "MEMORY") {
		t.Errorf("should not print headers with --no-headers option set:\n%s\n", result)
	}
}

func TestTopNodeAllMetricsCustomDefaults(t *testing.T) {
	customBaseHeapsterServiceAddress := "/api/v1/namespaces/custom-namespace/services/https:custom-heapster-service:/proxy"
	customBaseMetricsAddress := customBaseHeapsterServiceAddress + "/apis/metrics"

	cmdtesting.InitTestErrorHandler(t)
	metrics, nodes := testNodeV1alpha1MetricsData()
	expectedMetricsPath := fmt.Sprintf("%s/%s/nodes", customBaseMetricsAddress, metricsAPIVersion)
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	ns := scheme.Codecs.WithoutConversion()

	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
			case p == "/apis":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apisbody)))}, nil
			case p == expectedMetricsPath && m == "GET":
				body, err := marshallBody(metrics)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
			case p == expectedNodePath && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, nodes)}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\nExpected path: %#v", req, req.URL, expectedMetricsPath)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()

	opts := &TopNodeOptions{
		HeapsterOptions: HeapsterTopOptions{
			Namespace: "custom-namespace",
			Scheme:    "https",
			Service:   "custom-heapster-service",
		},
		IOStreams: streams,
	}
	cmd := NewCmdTopNode(tf, opts, streams)
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
	cmdtesting.InitTestErrorHandler(t)
	metrics, nodes := testNodeV1alpha1MetricsData()
	expectedMetrics := metrics.Items[0]
	expectedNode := nodes.Items[0]
	nonExpectedMetrics := metricsv1alpha1api.NodeMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[1:],
	}
	expectedPath := fmt.Sprintf("%s/%s/nodes/%s", baseMetricsAddress, metricsAPIVersion, expectedMetrics.Name)
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes/%s", apiPrefix, apiVersion, expectedMetrics.Name)

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	ns := scheme.Codecs.WithoutConversion()

	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
			case p == "/apis":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apisbody)))}, nil
			case p == expectedPath && m == "GET":
				body, err := marshallBody(expectedMetrics)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
			case p == expectedNodePath && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &expectedNode)}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\nExpected path: %#v", req, req.URL, expectedPath)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()

	cmd := NewCmdTopNode(tf, nil, streams)
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
	cmdtesting.InitTestErrorHandler(t)
	metrics, nodes := testNodeV1alpha1MetricsData()
	expectedMetrics := metricsv1alpha1api.NodeMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[0:1],
	}
	expectedNodes := v1.NodeList{
		ListMeta: nodes.ListMeta,
		Items:    nodes.Items[0:1],
	}
	nonExpectedMetrics := metricsv1alpha1api.NodeMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[1:],
	}
	label := "key=value"
	expectedPath := fmt.Sprintf("%s/%s/nodes", baseMetricsAddress, metricsAPIVersion)
	expectedQuery := fmt.Sprintf("labelSelector=%s", url.QueryEscape(label))
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	ns := scheme.Codecs.WithoutConversion()

	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m, q := req.URL.Path, req.Method, req.URL.RawQuery; {
			case p == "/api":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
			case p == "/apis":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apisbody)))}, nil
			case p == expectedPath && m == "GET" && q == expectedQuery:
				body, err := marshallBody(expectedMetrics)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
			case p == expectedNodePath && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &expectedNodes)}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\nExpected path: %#v", req, req.URL, expectedPath)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()

	cmd := NewCmdTopNode(tf, nil, streams)
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

func TestTopNodeAllMetricsFromMetricsServer(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	expectedMetrics, nodes := testNodeV1beta1MetricsData()
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	ns := scheme.Codecs.WithoutConversion()

	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
			case p == "/apis":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apisbodyWithMetrics)))}, nil
			case p == expectedNodePath && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, nodes)}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\n", req, req.URL)
				return nil, nil
			}
		}),
	}
	fakemetricsClientset := &metricsfake.Clientset{}
	fakemetricsClientset.AddReactor("list", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return true, expectedMetrics, nil
	})
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()

	cmd := NewCmdTopNode(tf, nil, streams)

	// TODO in the long run, we want to test most of our commands like this. Wire the options struct with specific mocks
	// TODO then check the particular Run functionality and harvest results from fake clients
	cmdOptions := &TopNodeOptions{
		IOStreams: streams,
	}
	if err := cmdOptions.Complete(tf, cmd, []string{}); err != nil {
		t.Fatal(err)
	}
	cmdOptions.MetricsClient = fakemetricsClientset
	if err := cmdOptions.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := cmdOptions.RunTopNode(); err != nil {
		t.Fatal(err)
	}

	// Check the presence of node names in the output.
	result := buf.String()
	for _, m := range expectedMetrics.Items {
		if !strings.Contains(result, m.Name) {
			t.Errorf("missing metrics for %s: \n%s", m.Name, result)
		}
	}
}

func TestTopNodeWithNameMetricsFromMetricsServer(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	metrics, nodes := testNodeV1beta1MetricsData()
	expectedMetrics := metrics.Items[0]
	expectedNode := nodes.Items[0]
	nonExpectedMetrics := metricsv1beta1api.NodeMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[1:],
	}
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes/%s", apiPrefix, apiVersion, expectedMetrics.Name)

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	ns := scheme.Codecs.WithoutConversion()

	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
			case p == "/apis":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apisbodyWithMetrics)))}, nil
			case p == expectedNodePath && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &expectedNode)}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\n", req, req.URL)
				return nil, nil
			}
		}),
	}
	fakemetricsClientset := &metricsfake.Clientset{}
	fakemetricsClientset.AddReactor("get", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return true, &expectedMetrics, nil
	})
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()

	cmd := NewCmdTopNode(tf, nil, streams)

	// TODO in the long run, we want to test most of our commands like this. Wire the options struct with specific mocks
	// TODO then check the particular Run functionality and harvest results from fake clients
	cmdOptions := &TopNodeOptions{
		IOStreams: streams,
	}
	if err := cmdOptions.Complete(tf, cmd, []string{expectedMetrics.Name}); err != nil {
		t.Fatal(err)
	}
	cmdOptions.MetricsClient = fakemetricsClientset
	if err := cmdOptions.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := cmdOptions.RunTopNode(); err != nil {
		t.Fatal(err)
	}

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

func TestTopNodeWithLabelSelectorMetricsFromMetricsServer(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	metrics, nodes := testNodeV1beta1MetricsData()
	expectedMetrics := &metricsv1beta1api.NodeMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[0:1],
	}
	expectedNodes := v1.NodeList{
		ListMeta: nodes.ListMeta,
		Items:    nodes.Items[0:1],
	}
	nonExpectedMetrics := &metricsv1beta1api.NodeMetricsList{
		ListMeta: metrics.ListMeta,
		Items:    metrics.Items[1:],
	}
	label := "key=value"
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	ns := scheme.Codecs.WithoutConversion()

	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m, _ := req.URL.Path, req.Method, req.URL.RawQuery; {
			case p == "/api":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
			case p == "/apis":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apisbodyWithMetrics)))}, nil
			case p == expectedNodePath && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &expectedNodes)}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\n", req, req.URL)
				return nil, nil
			}
		}),
	}

	fakemetricsClientset := &metricsfake.Clientset{}
	fakemetricsClientset.AddReactor("list", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return true, expectedMetrics, nil
	})
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()

	cmd := NewCmdTopNode(tf, nil, streams)
	cmd.Flags().Set("selector", label)

	// TODO in the long run, we want to test most of our commands like this. Wire the options struct with specific mocks
	// TODO then check the particular Run functionality and harvest results from fake clients
	cmdOptions := &TopNodeOptions{
		IOStreams: streams,
	}
	if err := cmdOptions.Complete(tf, cmd, []string{}); err != nil {
		t.Fatal(err)
	}
	cmdOptions.MetricsClient = fakemetricsClientset
	if err := cmdOptions.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := cmdOptions.RunTopNode(); err != nil {
		t.Fatal(err)
	}

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
