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
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
)

const (
	apiPrefix  = "api"
	apiVersion = "v1"
)

func TestTopNode(t *testing.T) {
	listNodePath := fmt.Sprintf("/%s/%s/nodes", apiPrefix, apiVersion)

	testCases := []struct {
		name                string
		args                []string
		selector            string
		sortBy              string
		showSwap            bool
		noSwapOnNodes       bool
		nodeIndices         []int // fixture items served by the fakes; nil means all
		expectedNodes       []string
		nonExpectedNodes    []string
		expectedSortedNodes []string
	}{
		{
			name:          "all nodes",
			expectedNodes: []string{"node1", "node2", "node3"},
		},
		{
			name:             "node with name",
			args:             []string{"node1"},
			nodeIndices:      []int{0},
			expectedNodes:    []string{"node1"},
			nonExpectedNodes: []string{"node2", "node3"},
		},
		{
			name:             "node with label selector",
			selector:         "key=value",
			nodeIndices:      []int{0},
			expectedNodes:    []string{"node1"},
			nonExpectedNodes: []string{"node2", "node3"},
		},
		{
			name:                "node sort by cpu",
			sortBy:              "cpu",
			expectedNodes:       []string{"node1", "node2", "node3"},
			expectedSortedNodes: []string{"node2", "node3", "node1"},
		},
		{
			name:                "node sort by memory",
			sortBy:              "memory",
			expectedNodes:       []string{"node1", "node2", "node3"},
			expectedSortedNodes: []string{"node2", "node3", "node1"},
		},
		{
			name:          "nodes with swap",
			showSwap:      true,
			expectedNodes: []string{"node1", "node2", "node3"},
		},
		{
			name:          "nodes without swap",
			showSwap:      true,
			noSwapOnNodes: true,
			expectedNodes: []string{"node1", "node2", "node3"},
		},
	}

	for _, version := range metricsAPIVersions {
		t.Run(version.name, func(t *testing.T) {
			for _, testCase := range testCases {
				t.Run(testCase.name, func(t *testing.T) {
					metrics, nodes := testNodeMetricsData()
					if testCase.nodeIndices != nil {
						var metricsItems []metricsapi.NodeMetrics
						var nodeItems []v1.Node
						for _, i := range testCase.nodeIndices {
							metricsItems = append(metricsItems, metrics.Items[i])
							nodeItems = append(nodeItems, nodes.Items[i])
						}
						metrics.Items = metricsItems
						nodes.Items = nodeItems
					}
					if testCase.noSwapOnNodes {
						for i := range metrics.Items {
							delete(metrics.Items[i].Usage, "swap")
						}
						for i := range nodes.Items {
							nodes.Items[i].Status.NodeInfo.Swap = nil
						}
					}

					metricsList, firstMetrics := versionedNodeMetricsList(t, version.name, metrics)
					fakeMetrics := &metricsfake.Clientset{}
					fakeMetrics.AddReactor("get", "nodes", func(action core.Action) (bool, runtime.Object, error) {
						return true, firstMetrics, nil
					})
					fakeMetrics.AddReactor("list", "nodes", func(action core.Action) (bool, runtime.Object, error) {
						return true, metricsList, nil
					})

					expectedPath := listNodePath
					var nodeBody runtime.Object = nodes
					if len(testCase.args) > 0 {
						expectedPath = fmt.Sprintf("%s/%s", listNodePath, testCase.args[0])
						nodeBody = &nodes.Items[0]
					}

					result := runTopNodeTest(t, runTopNodeOpts{
						apisBody:     version.apisBody,
						expectedPath: expectedPath,
						nodeBody:     nodeBody,
						fakeMetrics:  fakeMetrics,
						cmdArgs:      testCase.args,
						selector:     testCase.selector,
						sortBy:       testCase.sortBy,
						showSwap:     testCase.showSwap,
					})

					assertTopNodeOutput(t, topNodeAssertion{
						result:              result,
						expectedNodes:       testCase.expectedNodes,
						nonExpectedNodes:    testCase.nonExpectedNodes,
						expectedSortedNodes: testCase.expectedSortedNodes,
						showSwap:            testCase.showSwap,
						expectUnknownSwap:   testCase.noSwapOnNodes,
					})
				})
			}
		})
	}
}

type topNodeAssertion struct {
	result              string
	expectedNodes       []string // names that should appear in the output
	nonExpectedNodes    []string // names that should not appear in the output
	expectedSortedNodes []string // sort assertion (column 0)
	showSwap            bool
	expectUnknownSwap   bool
}

func assertTopNodeOutput(t *testing.T, a topNodeAssertion) {
	t.Helper()
	for _, name := range a.expectedNodes {
		if !strings.Contains(a.result, name) {
			t.Errorf("missing metrics for %s: \n%s", name, a.result)
		}
	}
	for _, name := range a.nonExpectedNodes {
		if strings.Contains(a.result, name) {
			t.Errorf("unexpected metrics for %s: \n%s", name, a.result)
		}
	}
	if a.expectedSortedNodes != nil {
		resultNodes := extractNodeNamesFromTopOutput(a.result)
		if !reflect.DeepEqual(a.expectedSortedNodes, resultNodes) {
			t.Errorf("nodes not matching:\n\texpectedNodes: %v\n\tresultNodes: %v\n", a.expectedSortedNodes, resultNodes)
		}
	}
	if a.showSwap {
		if !strings.Contains(a.result, "SWAP(bytes)") {
			t.Errorf("missing SWAP(bytes) header: \n%s", a.result)
		}
		if !strings.Contains(a.result, "SWAP(%)") {
			t.Errorf("missing SWAP(%%) header: \n%s", a.result)
		}
	}
	if a.expectUnknownSwap && !strings.Contains(a.result, "<unknown>") {
		t.Errorf("expected swap to be <unknown>: \n%s", a.result)
	}
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
