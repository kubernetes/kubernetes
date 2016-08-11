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

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
	"net/url"
)

func TestTopPodAllInNamespaceMetrics(t *testing.T) {
	initTestErrorHandler(t)
	// TODO(magorzata): refactor to pods/ path after updating heapster version
	metrics := testPodMetricsData()
	testNamespace := "testnamespace"
	nonTestNamespace := "anothernamespace"
	expectedMetrics := metrics[0:2]
	for _, m := range expectedMetrics {
		m.Namespace = testNamespace
	}
	nonExpectedMetrics := metrics[2:]
	for _, m := range expectedMetrics {
		m.Namespace = nonTestNamespace
	}
	expectedPath := fmt.Sprintf("%s/%s/namespaces/%s/pods", baseMetricsAddress, metricsApiVersion, testNamespace)

	f, tf, _, ns := NewAPIFactory()
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
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &unversioned.GroupVersion{Version: "v1"}}}
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdTopPod(f, buf)
	cmd.Run(cmd, []string{})

	// Check the presence of pod names in the output.
	result := buf.String()
	for _, m := range expectedMetrics {
		if !strings.Contains(result, m.Name) {
			t.Errorf("missing metrics for %s: \n%s", m.Name, result)
		}
	}
	for _, m := range nonExpectedMetrics {
		if strings.Contains(result, m.Name) {
			t.Errorf("unexpected metrics for %s: \n%s", m.Name, result)
		}
	}
}

func TestTopPodWithNameMetrics(t *testing.T) {
	initTestErrorHandler(t)
	metrics := testPodMetricsData()
	expectedMetrics := metrics[0]
	nonExpectedMetrics := metrics[1:]
	testNamespace := "testnamespace"
	expectedMetrics.Namespace = testNamespace
	expectedPath := fmt.Sprintf("%s/%s/namespaces/%s/pods/%s", baseMetricsAddress, metricsApiVersion, testNamespace, expectedMetrics.Name)

	f, tf, _, ns := NewAPIFactory()
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
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &unversioned.GroupVersion{Version: "v1"}}}
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdTopPod(f, buf)
	cmd.Run(cmd, []string{expectedMetrics.Name})

	// Check the presence of pod names in the output.
	result := buf.String()
	if !strings.Contains(result, expectedMetrics.Name) {
		t.Errorf("missing metrics for %s: \n%s", expectedMetrics.Name, result)
	}
	for _, m := range nonExpectedMetrics {
		if strings.Contains(result, m.Name) {
			t.Errorf("unexpected metrics for %s: \n%s", m.Name, result)
		}
	}
}

func TestTopPodWithLabelSelectorMetrics(t *testing.T) {
	initTestErrorHandler(t)
	metrics := testPodMetricsData()
	expectedMetrics := metrics[0:2]
	nonExpectedMetrics := metrics[2:]
	label := "key=value"
	testNamespace := "testnamespace"
	expectedPath := fmt.Sprintf("%s/%s/namespaces/%s/pods", baseMetricsAddress, metricsApiVersion, testNamespace)
	expectedQuery := fmt.Sprintf("labelSelector=%s", url.QueryEscape(label))

	f, tf, _, ns := NewAPIFactory()
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
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &unversioned.GroupVersion{Version: "v1"}}}
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdTopPod(f, buf)
	cmd.Flags().Set("selector", label)
	cmd.Run(cmd, []string{})

	// Check the presence of pod names in the output.
	result := buf.String()
	for _, m := range expectedMetrics {
		if !strings.Contains(result, m.Name) {
			t.Errorf("missing metrics for %s: \n%s", m.Name, result)
		}
	}
	for _, m := range nonExpectedMetrics {
		if strings.Contains(result, m.Name) {
			t.Errorf("unexpected metrics for %s: \n%s", m.Name, result)
		}
	}
}

func TestTopPodWithContainersMetrics(t *testing.T) {
	initTestErrorHandler(t)
	metrics := testPodMetricsData()
	expectedMetrics := metrics[0]
	nonExpectedMetrics := metrics[1:]
	testNamespace := "testnamespace"
	expectedMetrics.Namespace = testNamespace
	expectedPath := fmt.Sprintf("%s/%s/namespaces/%s/pods/%s", baseMetricsAddress, metricsApiVersion, testNamespace, expectedMetrics.Name)

	f, tf, _, ns := NewAPIFactory()
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
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &unversioned.GroupVersion{Version: "v1"}}}
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
	for _, m := range nonExpectedMetrics {
		if strings.Contains(result, m.Name) {
			t.Errorf("unexpected metrics for %s: \n%s", m.Name, result)
		}
	}
}
