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
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/rest/fake"
	core "k8s.io/client-go/testing"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
	metricsv1api "k8s.io/metrics/pkg/apis/metrics/v1"
	metricsv1beta1api "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
)

const (
	apiPrefix  = "api"
	apiVersion = "v1"
)

func TestTopNodeAllMetricsFrom(t *testing.T) {
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)

	t.Run("v1beta1", func(t *testing.T) {
		metrics, nodes := testNodeV1beta1MetricsData()
		fakeMetrics := &metricsfake.Clientset{}
		fakeMetrics.AddReactor("list", "nodes", func(action core.Action) (bool, runtime.Object, error) {
			return true, metrics, nil
		})
		result := runTopNodeTest(t, runTopNodeOpts{
			apisBody:     apisV1beta1BodyWithMetrics,
			expectedPath: expectedNodePath,
			nodeBody:     nodes,
			fakeMetrics:  fakeMetrics,
		})
		for _, m := range metrics.Items {
			if !strings.Contains(result, m.Name) {
				t.Errorf("missing metrics for %s: \n%s", m.Name, result)
			}
		}
	})

	t.Run("v1", func(t *testing.T) {
		metrics, nodes := testNodeV1MetricsData()
		fakeMetrics := &metricsfake.Clientset{}
		fakeMetrics.AddReactor("list", "nodes", func(action core.Action) (bool, runtime.Object, error) {
			return true, metrics, nil
		})
		result := runTopNodeTest(t, runTopNodeOpts{
			apisBody:     apisV1BodyWithMetrics,
			expectedPath: expectedNodePath,
			nodeBody:     nodes,
			fakeMetrics:  fakeMetrics,
		})
		for _, m := range metrics.Items {
			if !strings.Contains(result, m.Name) {
				t.Errorf("missing metrics for %s: \n%s", m.Name, result)
			}
		}
	})
}

func TestTopNodeWithNameMetricsFrom(t *testing.T) {
	t.Run("v1beta1", func(t *testing.T) {
		metrics, nodes := testNodeV1beta1MetricsData()
		expectedMetrics := metrics.Items[0]
		expectedNode := nodes.Items[0]
		nonExpectedMetrics := metrics.Items[1:]
		expectedNodePath := fmt.Sprintf("/%s/%s/nodes/%s", apiPrefix, apiVersion, expectedMetrics.Name)

		fakeMetrics := &metricsfake.Clientset{}
		fakeMetrics.AddReactor("get", "nodes", func(action core.Action) (bool, runtime.Object, error) {
			return true, &expectedMetrics, nil
		})
		result := runTopNodeTest(t, runTopNodeOpts{
			apisBody:     apisV1beta1BodyWithMetrics,
			expectedPath: expectedNodePath,
			nodeBody:     &expectedNode,
			fakeMetrics:  fakeMetrics,
			cmdArgs:      []string{expectedMetrics.Name},
		})
		if !strings.Contains(result, expectedMetrics.Name) {
			t.Errorf("missing metrics for %s: \n%s", expectedMetrics.Name, result)
		}
		for _, m := range nonExpectedMetrics {
			if strings.Contains(result, m.Name) {
				t.Errorf("unexpected metrics for %s: \n%s", m.Name, result)
			}
		}
	})

	t.Run("v1", func(t *testing.T) {
		metrics, nodes := testNodeV1MetricsData()
		expectedMetrics := metrics.Items[0]
		expectedNode := nodes.Items[0]
		nonExpectedMetrics := metrics.Items[1:]
		expectedNodePath := fmt.Sprintf("/%s/%s/nodes/%s", apiPrefix, apiVersion, expectedMetrics.Name)

		fakeMetrics := &metricsfake.Clientset{}
		fakeMetrics.AddReactor("get", "nodes", func(action core.Action) (bool, runtime.Object, error) {
			return true, &expectedMetrics, nil
		})
		result := runTopNodeTest(t, runTopNodeOpts{
			apisBody:     apisV1BodyWithMetrics,
			expectedPath: expectedNodePath,
			nodeBody:     &expectedNode,
			fakeMetrics:  fakeMetrics,
			cmdArgs:      []string{expectedMetrics.Name},
		})
		if !strings.Contains(result, expectedMetrics.Name) {
			t.Errorf("missing metrics for %s: \n%s", expectedMetrics.Name, result)
		}
		for _, m := range nonExpectedMetrics {
			if strings.Contains(result, m.Name) {
				t.Errorf("unexpected metrics for %s: \n%s", m.Name, result)
			}
		}
	})
}

func TestTopNodeWithLabelSelectorMetricsFrom(t *testing.T) {
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)
	label := "key=value"

	t.Run("v1beta1", func(t *testing.T) {
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
		fakeMetrics := &metricsfake.Clientset{}
		fakeMetrics.AddReactor("list", "nodes", func(action core.Action) (bool, runtime.Object, error) {
			return true, expectedMetrics, nil
		})
		result := runTopNodeTest(t, runTopNodeOpts{
			apisBody:     apisV1beta1BodyWithMetrics,
			expectedPath: expectedNodePath,
			nodeBody:     &expectedNodes,
			fakeMetrics:  fakeMetrics,
			selector:     label,
		})
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
	})

	t.Run("v1", func(t *testing.T) {
		metrics, nodes := testNodeV1MetricsData()
		expectedMetrics := &metricsv1api.NodeMetricsList{
			ListMeta: metrics.ListMeta,
			Items:    metrics.Items[0:1],
		}
		expectedNodes := v1.NodeList{
			ListMeta: nodes.ListMeta,
			Items:    nodes.Items[0:1],
		}
		nonExpectedMetrics := &metricsv1api.NodeMetricsList{
			ListMeta: metrics.ListMeta,
			Items:    metrics.Items[1:],
		}
		fakeMetrics := &metricsfake.Clientset{}
		fakeMetrics.AddReactor("list", "nodes", func(action core.Action) (bool, runtime.Object, error) {
			return true, expectedMetrics, nil
		})
		result := runTopNodeTest(t, runTopNodeOpts{
			apisBody:     apisV1BodyWithMetrics,
			expectedPath: expectedNodePath,
			nodeBody:     &expectedNodes,
			fakeMetrics:  fakeMetrics,
			selector:     label,
		})
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
	})
}

func TestTopNodeWithSortByCpuMetricsFrom(t *testing.T) {
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)

	t.Run("v1beta1", func(t *testing.T) {
		metrics, nodes := testNodeV1beta1MetricsData()
		expectedMetrics := &metricsv1beta1api.NodeMetricsList{
			ListMeta: metrics.ListMeta,
			Items:    metrics.Items,
		}
		expectedNodes := v1.NodeList{
			ListMeta: nodes.ListMeta,
			Items:    nodes.Items,
		}
		expectedNodesNames := []string{"node2", "node3", "node1"}
		fakemetrics := &metricsfake.Clientset{}
		fakemetrics.AddReactor("list", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			return true, expectedMetrics, nil
		})

		// Check the presence of node names in the output.
		result := runTopNodeTest(t, runTopNodeOpts{
			apisBody:     apisV1beta1BodyWithMetrics,
			expectedPath: expectedNodePath,
			nodeBody:     &expectedNodes,
			fakeMetrics:  fakemetrics,
			sortBy:       "cpu",
		})

		for _, m := range expectedMetrics.Items {
			if !strings.Contains(result, m.Name) {
				t.Errorf("missing metrics for %s: \n%s", m.Name, result)
			}
		}

		resultNodes := extractNodeNamesFromTopOutput(result)

		if !reflect.DeepEqual(resultNodes, expectedNodesNames) {
			t.Errorf("kinds not matching:\n\texpectedKinds: %v\n\tgotKinds: %v\n", expectedNodesNames, resultNodes)
		}
	})

	t.Run("v1", func(t *testing.T) {
		metrics, nodes := testNodeV1MetricsData()
		expectedMetrics := &metricsv1api.NodeMetricsList{
			ListMeta: metrics.ListMeta,
			Items:    metrics.Items,
		}
		expectedNodes := v1.NodeList{
			ListMeta: nodes.ListMeta,
			Items:    nodes.Items,
		}
		expectedNodesNames := []string{"node2", "node3", "node1"}
		fakemetrics := &metricsfake.Clientset{}
		fakemetrics.AddReactor("list", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			return true, expectedMetrics, nil
		})

		// Check the presence of node names in the output.
		result := runTopNodeTest(t, runTopNodeOpts{
			apisBody:     apisV1BodyWithMetrics,
			expectedPath: expectedNodePath,
			nodeBody:     &expectedNodes,
			fakeMetrics:  fakemetrics,
			sortBy:       "cpu",
		})

		for _, m := range expectedMetrics.Items {
			if !strings.Contains(result, m.Name) {
				t.Errorf("missing metrics for %s: \n%s", m.Name, result)
			}
		}

		resultNodes := extractNodeNamesFromTopOutput(result)

		if !reflect.DeepEqual(resultNodes, expectedNodesNames) {
			t.Errorf("kinds not matching:\n\texpectedKinds: %v\n\tgotKinds: %v\n", expectedNodesNames, resultNodes)
		}
	})
}

func TestTopNodeWithSortByMemoryMetricsFrom(t *testing.T) {
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)

	t.Run("v1beta1", func(t *testing.T) {
		metrics, nodes := testNodeV1beta1MetricsData()
		expectedMetrics := &metricsv1beta1api.NodeMetricsList{
			ListMeta: metrics.ListMeta,
			Items:    metrics.Items,
		}
		expectedNodes := v1.NodeList{
			ListMeta: nodes.ListMeta,
			Items:    nodes.Items,
		}
		expectedNodesNames := []string{"node2", "node3", "node1"}
		fakemetrics := &metricsfake.Clientset{}
		fakemetrics.AddReactor("list", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			return true, expectedMetrics, nil
		})

		// Check the presence of node names in the output.
		result := runTopNodeTest(t, runTopNodeOpts{
			apisBody:     apisV1beta1BodyWithMetrics,
			expectedPath: expectedNodePath,
			nodeBody:     &expectedNodes,
			fakeMetrics:  fakemetrics,
			sortBy:       "memory",
		})

		for _, m := range expectedMetrics.Items {
			if !strings.Contains(result, m.Name) {
				t.Errorf("missing metrics for %s: \n%s", m.Name, result)
			}
		}

		resultNodes := extractNodeNamesFromTopOutput(result)

		if !reflect.DeepEqual(resultNodes, expectedNodesNames) {
			t.Errorf("kinds not matching:\n\texpectedKinds: %v\n\tgotKinds: %v\n", expectedNodesNames, resultNodes)
		}
	})

	t.Run("v1", func(t *testing.T) {
		metrics, nodes := testNodeV1MetricsData()
		expectedMetrics := &metricsv1api.NodeMetricsList{
			ListMeta: metrics.ListMeta,
			Items:    metrics.Items,
		}
		expectedNodes := v1.NodeList{
			ListMeta: nodes.ListMeta,
			Items:    nodes.Items,
		}
		expectedNodesNames := []string{"node2", "node3", "node1"}
		fakemetrics := &metricsfake.Clientset{}
		fakemetrics.AddReactor("list", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			return true, expectedMetrics, nil
		})

		// Check the presence of node names in the output.
		result := runTopNodeTest(t, runTopNodeOpts{
			apisBody:     apisV1BodyWithMetrics,
			expectedPath: expectedNodePath,
			nodeBody:     &expectedNodes,
			fakeMetrics:  fakemetrics,
			sortBy:       "memory",
		})

		for _, m := range expectedMetrics.Items {
			if !strings.Contains(result, m.Name) {
				t.Errorf("missing metrics for %s: \n%s", m.Name, result)
			}
		}

		resultNodes := extractNodeNamesFromTopOutput(result)

		if !reflect.DeepEqual(resultNodes, expectedNodesNames) {
			t.Errorf("kinds not matching:\n\texpectedKinds: %v\n\tgotKinds: %v\n", expectedNodesNames, resultNodes)
		}
	})
}

func TestTopNodeWithSwap(t *testing.T) {
	expectedNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)

	swapCases := []struct {
		name                  string
		isSwapDisabledOnNodes bool
	}{
		{
			name:                  "nodes with swap",
			isSwapDisabledOnNodes: false,
		},
		{
			name:                  "nodes without swap",
			isSwapDisabledOnNodes: true,
		},
	}

	t.Run("v1beta1", func(t *testing.T) {
		for _, tc := range swapCases {
			t.Run(tc.name, func(t *testing.T) {
				expectedMetrics, nodes := testNodeV1beta1MetricsData()
				if tc.isSwapDisabledOnNodes {
					for i := range expectedMetrics.Items {
						delete(expectedMetrics.Items[i].Usage, "swap")
					}
					for i := range nodes.Items {
						nodes.Items[i].Status.NodeInfo.Swap = nil
					}
				}
				fakeMetrics := &metricsfake.Clientset{}
				fakeMetrics.AddReactor("list", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
					return true, expectedMetrics, nil
				})

				result := runTopNodeTest(t, runTopNodeOpts{
					apisBody:     apisV1beta1BodyWithMetrics,
					expectedPath: expectedNodePath,
					nodeBody:     nodes,
					fakeMetrics:  fakeMetrics,
					showSwap:     true,
				})

				if !strings.Contains(result, "SWAP(bytes)") {
					t.Errorf("missing SWAP(bytes) header: \n%s", result)
				}
				if !strings.Contains(result, "SWAP(%)") {
					t.Errorf("missing SWAP(%%) header: \n%s", result)
				}

				if tc.isSwapDisabledOnNodes {
					if !strings.Contains(result, "<unknown>") {
						t.Errorf("expected swap to be <unknown>: \n%s", result)
					}
				}

				for _, m := range expectedMetrics.Items {
					if !strings.Contains(result, m.Name) {
						t.Errorf("missing metrics for %s: \n%s", m.Name, result)
					}
					if _, foundSwapMetric := m.Usage["swap"]; foundSwapMetric != !tc.isSwapDisabledOnNodes {
						t.Errorf("missing swap metric for %s: \n%s", m.Name, result)
					}
				}
			})
		}
	})

	t.Run("v1", func(t *testing.T) {
		for _, tc := range swapCases {
			t.Run(tc.name, func(t *testing.T) {
				expectedMetrics, nodes := testNodeV1MetricsData()
				if tc.isSwapDisabledOnNodes {
					for i := range expectedMetrics.Items {
						delete(expectedMetrics.Items[i].Usage, "swap")
					}
					for i := range nodes.Items {
						nodes.Items[i].Status.NodeInfo.Swap = nil
					}
				}
				fakeMetrics := &metricsfake.Clientset{}
				fakeMetrics.AddReactor("list", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
					return true, expectedMetrics, nil
				})

				result := runTopNodeTest(t, runTopNodeOpts{
					apisBody:     apisV1BodyWithMetrics,
					expectedPath: expectedNodePath,
					nodeBody:     nodes,
					fakeMetrics:  fakeMetrics,
					showSwap:     true,
				})

				if !strings.Contains(result, "SWAP(bytes)") {
					t.Errorf("missing SWAP(bytes) header: \n%s", result)
				}
				if !strings.Contains(result, "SWAP(%)") {
					t.Errorf("missing SWAP(%%) header: \n%s", result)
				}

				if tc.isSwapDisabledOnNodes {
					if !strings.Contains(result, "<unknown>") {
						t.Errorf("expected swap to be <unknown>: \n%s", result)
					}
				}

				for _, m := range expectedMetrics.Items {
					if !strings.Contains(result, m.Name) {
						t.Errorf("missing metrics for %s: \n%s", m.Name, result)
					}
					if _, foundSwapMetric := m.Usage["swap"]; foundSwapMetric != !tc.isSwapDisabledOnNodes {
						t.Errorf("missing swap metric for %s: \n%s", m.Name, result)
					}
				}
			})
		}
	})
}

func extractNodeNamesFromTopOutput(result string) []string {
	lines := strings.Split(result, "\n")
	names := make([]string, len(lines)-2) // don't process first (header) and last (empty) line
	for i, line := range lines[1 : len(lines)-1] {
		names[i] = strings.Split(line, " ")[0]
	}
	return names
}

func runTopNodeTest(t *testing.T, opts runTopNodeOpts) string {
	cmdtesting.InitTestErrorHandler(t)

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	ns := scheme.Codecs.WithoutConversion()

	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
			case p == "/apis":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(opts.apisBody)))}, nil
			case p == opts.expectedPath && m == "GET":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, opts.nodeBody)}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v\n", req, req.URL)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
	streams, _, buf, _ := genericiooptions.NewTestIOStreams()

	cmd := NewCmdTopNode(tf, nil, streams)
	cmdOptions := &TopNodeOptions{
		IOStreams: streams,
		Selector:  opts.selector,
		SortBy:    opts.sortBy,
		ShowSwap:  opts.showSwap,
	}
	if err := cmdOptions.Complete(tf, cmd, opts.cmdArgs); err != nil {
		t.Fatal(err)
	}
	cmdOptions.MetricsClient = opts.fakeMetrics
	if err := cmdOptions.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := cmdOptions.RunTopNode(); err != nil {
		t.Fatal(err)
	}
	return buf.String()
}

type runTopNodeOpts struct {
	apisBody     string
	expectedPath string
	nodeBody     runtime.Object // *v1.NodeList or *v1.Node — passed to cmdtesting.ObjBody
	fakeMetrics  *metricsfake.Clientset
	cmdArgs      []string
	selector     string
	sortBy       string
	showSwap     bool
}
