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
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"net/url"
)

func TestTopPod(t *testing.T) {
	testNS := "testnamespace"
	testCases := []struct {
		name            string
		namespace       string
		flags           map[string]string
		args            []string
		expectedPath    string
		expectedQuery   string
		expectedMetrics map[int]string
		containers      bool
		listsNamespaces bool
	}{
		{
			name:         "all namespaces",
			flags:        map[string]string{"all-namespaces": "true"},
			expectedPath: "pods",
			expectedMetrics: map[int]string{
				0: testNS,
				1: "secondtestns",
				3: "thirdtestns",
			},
			listsNamespaces: true,
		},
		{
			name:            "all in namespace",
			expectedPath:    "namespaces/testnamespace/pods",
			expectedMetrics: map[int]string{0: testNS, 1: testNS},
		},
		{
			name:            "pod with name",
			args:            []string{"pod1"},
			expectedPath:    "namespaces/testnamespace/pods/pod1",
			expectedMetrics: map[int]string{0: testNS},
		},
		{
			name:            "pod with label selector",
			flags:           map[string]string{"selector": "key=value"},
			expectedPath:    "namespaces/testnamespace/pods",
			expectedQuery:   "labelSelector=" + url.QueryEscape("key=value"),
			expectedMetrics: map[int]string{0: testNS, 1: testNS},
		},
		{
			name:            "pod with container metrics",
			flags:           map[string]string{"containers": "true"},
			args:            []string{"pod1"},
			expectedPath:    "namespaces/testnamespace/pods/pod1",
			expectedMetrics: map[int]string{0: testNS},
			containers:      true,
		},
	}
	initTestErrorHandler(t)
	for _, testCase := range testCases {
		t.Logf("Running test case %s", testCase.name)
		metricsList := testPodMetricsData()
		var expectedMetrics []metrics_api.PodMetrics
		var expectedContainerNames, nonExpectedMetricsNames []string
		for n, m := range metricsList.Items {
			if namespace, found := testCase.expectedMetrics[n]; found {
				m.Namespace = namespace
				expectedMetrics = append(expectedMetrics, m)
				for _, c := range m.Containers {
					expectedContainerNames = append(expectedContainerNames, c.Name)
				}
			} else {
				nonExpectedMetricsNames = append(nonExpectedMetricsNames, m.Name)
			}
		}

		var response interface{}
		if len(expectedMetrics) == 1 {
			response = expectedMetrics[0]
		} else {
			response = metrics_api.PodMetricsList{
				ListMeta: metricsList.ListMeta,
				Items:    expectedMetrics,
			}
		}

		f, tf, _, ns := cmdtesting.NewAPIFactory()
		tf.Printer = &testPrinter{}
		expectedPath := fmt.Sprintf("%s/%s/%s", baseMetricsAddress, metricsApiVersion, testCase.expectedPath)
		tf.Client = &fake.RESTClient{
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m, q := req.URL.Path, req.Method, req.URL.RawQuery; {
				case p == expectedPath && m == "GET" && (testCase.expectedQuery == "" || q == testCase.expectedQuery):
					body, err := marshallBody(response)
					if err != nil {
						t.Errorf("%s: unexpected error: %v", testCase.name, err)
					}
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
				default:
					t.Fatalf("%s: unexpected request: %#v\nGot URL: %#v\nExpected path: %#v\nExpected query: %#v", testCase.name, req, req.URL, expectedPath, testCase.expectedQuery)
					return nil, nil
				}
			}),
		}
		tf.Namespace = testNS
		tf.ClientConfig = defaultClientConfig()
		buf := bytes.NewBuffer([]byte{})

		cmd := NewCmdTopPod(f, buf)
		for name, value := range testCase.flags {
			cmd.Flags().Set(name, value)
		}
		cmd.Run(cmd, testCase.args)

		// Check the presence of pod names&namespaces/container names in the output.
		result := buf.String()
		if testCase.containers {
			for _, containerName := range expectedContainerNames {
				if !strings.Contains(result, containerName) {
					t.Errorf("%s: missing metrics for container %s: \n%s", testCase.name, containerName, result)
				}
			}
		}
		for _, m := range expectedMetrics {
			if !strings.Contains(result, m.Name) {
				t.Errorf("%s: missing metrics for %s: \n%s", testCase.name, m.Name, result)
			}
			if testCase.listsNamespaces && !strings.Contains(result, m.Namespace) {
				t.Errorf("%s: missing metrics for %s/%s: \n%s", testCase.name, m.Namespace, m.Name, result)
			}
		}
		for _, name := range nonExpectedMetricsNames {
			if strings.Contains(result, name) {
				t.Errorf("%s: unexpected metrics for %s: \n%s", testCase.name, name, result)
			}
		}
	}
}
