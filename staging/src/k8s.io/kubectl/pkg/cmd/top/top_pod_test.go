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
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/rest/fake"
	core "k8s.io/client-go/testing"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
	metricsv1beta1api "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
)

const (
	apibody = `{
	"kind": "APIVersions",
	"versions": [
		"v1"
	],
	"serverAddressByClientCIDRs": [
		{
			"clientCIDR": "0.0.0.0/0",
			"serverAddress": "10.0.2.15:8443"
		}
	]
}`

	apisV1BodyWithMetrics = `{
	"kind": "APIGroupList",
	"apiVersion": "v1",
	"groups": [
		{
			"name":"metrics.k8s.io",
			"versions":[
				{
					"groupVersion":"metrics.k8s.io/v1",
					"version":"v1"
				},
				{
					"groupVersion":"metrics.k8s.io/v1beta1",
					"version":"v1beta1"
				}
			],
			"preferredVersion":{
				"groupVersion":"metrics.k8s.io/v1",
				"version":"v1"
			},
			"serverAddressByClientCIDRs":null
		}
	]
}`

	apisV1beta1BodyWithMetrics = `{
	"kind": "APIGroupList",
	"apiVersion": "v1",
	"groups": [
		{
			"name":"metrics.k8s.io",
			"versions":[
				{
					"groupVersion":"metrics.k8s.io/v1beta1",
					"version":"v1beta1"
				}
			],
			"preferredVersion":{
				"groupVersion":"metrics.k8s.io/v1beta1",
				"version":"v1beta1"
			},
			"serverAddressByClientCIDRs":null
		}
	]
}`
	testNS = "testns"
)

func TestTopPod(t *testing.T) {
	testCases := []struct {
		name               string
		options            *TopPodOptions
		args               []string
		expectedPods       []string
		expectedContainers []string
		namespaces         []string
		containers         bool
		listsNamespaces    bool
		// extraPaths overrides the response of the pod list endpoint, e.g. to
		// simulate server-side filtering by the core API.
		extraPaths func(req *http.Request) (*http.Response, error)
		// expectedInOutput/nonExpectedInOutput override the pod names derived
		// from namespaces, for cases where the printed pods differ from what
		// the metrics API returned (e.g. client-side field selector filtering).
		expectedInOutput    []string
		nonExpectedInOutput []string
		expectedSwapBytes   map[string]string // exact swap column values per pod
		expectedOutput      *string           // exact stdout match
		expectedErr         *string           // exact stderr match
	}{
		{
			name:            "all namespaces",
			options:         &TopPodOptions{AllNamespaces: true},
			namespaces:      []string{testNS, "secondtestns", "thirdtestns"},
			listsNamespaces: true,
		},
		{
			name:       "all in namespace",
			namespaces: []string{testNS, testNS},
		},
		{
			name:       "pod with name",
			args:       []string{"pod1"},
			namespaces: []string{testNS},
		},
		{
			name:       "pod with label selector",
			options:    &TopPodOptions{LabelSelector: "key=value"},
			namespaces: []string{testNS, testNS},
		},
		{
			name:       "pod with field selector",
			options:    &TopPodOptions{FieldSelector: "key=value"},
			namespaces: []string{testNS, testNS},
		},
		{
			name:    "pod with container metrics",
			options: &TopPodOptions{PrintContainers: true},
			args:    []string{"pod1"},
			expectedContainers: []string{
				"container1-1",
				"container1-2",
			},
			namespaces: []string{testNS},
			containers: true,
		},
		{
			name:         "pod sort by cpu",
			options:      &TopPodOptions{SortBy: "cpu"},
			expectedPods: []string{"pod2", "pod3", "pod1"},
			namespaces:   []string{testNS, testNS, testNS},
		},
		{
			name:         "pod sort by memory",
			options:      &TopPodOptions{SortBy: "memory"},
			expectedPods: []string{"pod2", "pod3", "pod1"},
			namespaces:   []string{testNS, testNS, testNS},
		},
		{
			name:    "container sort by cpu",
			options: &TopPodOptions{PrintContainers: true, SortBy: "cpu"},
			expectedContainers: []string{
				"container2-3",
				"container2-2",
				"container2-1",
				"container3-1",
				"container1-2",
				"container1-1",
			},
			namespaces: []string{testNS, testNS, testNS},
			containers: true,
		},
		{
			name:    "container sort by memory",
			options: &TopPodOptions{PrintContainers: true, SortBy: "memory"},
			expectedContainers: []string{
				"container2-3",
				"container2-2",
				"container2-1",
				"container3-1",
				"container1-2",
				"container1-1",
			},
			namespaces: []string{testNS, testNS, testNS},
			containers: true,
		},
		{
			name:            "with swap",
			options:         &TopPodOptions{AllNamespaces: true, ShowSwap: true},
			namespaces:      []string{testNS, "secondtestns", "thirdtestns"},
			listsNamespaces: true,
		},
		{
			name:              "swap values",
			options:           &TopPodOptions{ShowSwap: true},
			namespaces:        []string{testNS, testNS, testNS},
			expectedSwapBytes: map[string]string{"pod1": "4Mi", "pod2": "0Mi", "pod3": "3Mi"},
		},
		{
			// The metrics API returns all three pods, while the pod list
			// endpoint only returns pod1, so the client must drop pod2 and pod3.
			name:                "pod with field selector filtering metrics",
			options:             &TopPodOptions{FieldSelector: "spec.nodeName=node-a"},
			namespaces:          []string{testNS, testNS, testNS},
			extraPaths:          onlyPod1ListResponse,
			expectedInOutput:    []string{"pod1"},
			nonExpectedInOutput: []string{"pod2", "pod3"},
		},
		{
			name:           "no resources found in all namespaces",
			options:        &TopPodOptions{AllNamespaces: true},
			extraPaths:     emptyPodListResponse,
			expectedOutput: new(""),
			expectedErr:    new("No resources found\n"),
		},
		{
			name:           "no resources found in namespace",
			extraPaths:     emptyPodListResponse,
			expectedOutput: new(""),
			expectedErr:    new("No resources found in " + testNS + " namespace.\n"),
		},
	}
	cmdtesting.InitTestErrorHandler(t)

	for _, version := range metricsAPIVersions {
		t.Run(version.name, func(t *testing.T) {
			for _, testCase := range testCases {
				t.Run(testCase.name, func(t *testing.T) {
					metricsItems := testPodMetricsData()
					var expectedMetrics []metricsapi.PodMetrics
					var expectedPodNames, expectedPodNamespaces []string
					var expectedContainerNames, nonExpectedMetricsNames []string
					for n, m := range metricsItems {
						if n < len(testCase.namespaces) {
							m.Namespace = testCase.namespaces[n]
							expectedMetrics = append(expectedMetrics, m)
							expectedPodNames = append(expectedPodNames, m.Name)
							expectedPodNamespaces = append(expectedPodNamespaces, m.Namespace)
							for _, c := range m.Containers {
								expectedContainerNames = append(expectedContainerNames, c.Name)
							}
						} else {
							nonExpectedMetricsNames = append(nonExpectedMetricsNames, m.Name)
						}
					}
					if testCase.expectedInOutput != nil || testCase.nonExpectedInOutput != nil {
						expectedPodNames = testCase.expectedInOutput
						nonExpectedMetricsNames = testCase.nonExpectedInOutput
					}

					metricsList, firstMetrics := versionedPodMetricsList(t, version.name, &metricsapi.PodMetricsList{
						ListMeta: metav1.ListMeta{ResourceVersion: "2"},
						Items:    expectedMetrics,
					})
					fakemetricsClientset := &metricsfake.Clientset{}
					fakemetricsClientset.AddReactor("get", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, firstMetrics, nil
					})
					fakemetricsClientset.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
						return true, metricsList, nil
					})

					result, stderr := runTopPodTest(t, runTopPodOpts{
						apisBody:    version.apisBody,
						fakeMetrics: fakemetricsClientset,
						options:     testCase.options,
						cmdArgs:     testCase.args,
						extraPaths:  testCase.extraPaths,
					})

					assertTopPodOutput(t, topPodAssertion{
						result:                   result,
						expectedPodNames:         expectedPodNames,
						expectedPodNamespaces:    expectedPodNamespaces,
						nonExpectedMetricsNames:  nonExpectedMetricsNames,
						expectedContainerNames:   expectedContainerNames,
						expectedSortedPods:       testCase.expectedPods,
						expectedSortedContainers: testCase.expectedContainers,
						showContainers:           testCase.containers,
						listsNamespaces:          testCase.listsNamespaces,
						showSwap:                 testCase.options != nil && testCase.options.ShowSwap,
					})
					if testCase.expectedSwapBytes != nil {
						assertSwapBytesInTopOutput(t, result, testCase.expectedSwapBytes)
					}
					if testCase.expectedOutput != nil && *testCase.expectedOutput != result {
						t.Errorf("Unexpected output:\nExpected:\n%v\nActual:\n%v", *testCase.expectedOutput, result)
					}
					if testCase.expectedErr != nil && *testCase.expectedErr != stderr {
						t.Errorf("Unexpected error:\nExpected:\n%v\nActual:\n%v", *testCase.expectedErr, stderr)
					}
				})
			}
		})
	}
}

type topPodAssertion struct {
	result                   string
	expectedPodNames         []string // names that should appear; parallel to expectedPodNamespaces
	expectedPodNamespaces    []string
	nonExpectedMetricsNames  []string
	expectedContainerNames   []string
	expectedSortedPods       []string // sort assertion (column 0)
	expectedSortedContainers []string // sort assertion (column 1)
	showContainers           bool
	listsNamespaces          bool
	showSwap                 bool
}

func assertTopPodOutput(t *testing.T, a topPodAssertion) {
	t.Helper()
	if a.showContainers {
		for _, name := range a.expectedContainerNames {
			if !strings.Contains(a.result, name) {
				t.Errorf("missing metrics for container %s: \n%s", name, a.result)
			}
		}
	}
	for i, name := range a.expectedPodNames {
		if !strings.Contains(a.result, name) {
			t.Errorf("missing metrics for %s: \n%s", name, a.result)
		}
		if a.listsNamespaces && !strings.Contains(a.result, a.expectedPodNamespaces[i]) {
			t.Errorf("missing metrics for %s/%s: \n%s", a.expectedPodNamespaces[i], name, a.result)
		}
	}
	for _, name := range a.nonExpectedMetricsNames {
		if strings.Contains(a.result, name) {
			t.Errorf("unexpected metrics for %s: \n%s", name, a.result)
		}
	}
	if a.expectedSortedPods != nil {
		resultPods := getResultColumnValues(a.result, 0)
		if !reflect.DeepEqual(a.expectedSortedPods, resultPods) {
			t.Errorf("pods not matching:\n\texpectedPods: %v\n\tresultPods: %v\n", a.expectedSortedPods, resultPods)
		}
	}
	if a.expectedSortedContainers != nil {
		resultContainers := getResultColumnValues(a.result, 1)
		if !reflect.DeepEqual(a.expectedSortedContainers, resultContainers) {
			t.Errorf("containers not matching:\n\texpectedContainers: %v\n\tresultContainers: %v\n", a.expectedSortedContainers, resultContainers)
		}
	}
	if a.showSwap && !strings.Contains(a.result, "SWAP(bytes)") {
		t.Errorf("missing SWAP(bytes) header: \n%s", a.result)
	}
}

func assertSwapBytesInTopOutput(t *testing.T, stdout string, expected map[string]string) {
	t.Helper()
	actual := map[string]string{}
	for _, line := range strings.Split(stdout, "\n")[1:] {
		fields := strings.Fields(line)
		if len(fields) < 4 {
			continue
		}
		actual[fields[0]] = fields[3]
	}
	for podName, expectedBytes := range expected {
		actualBytes, found := actual[podName]
		if !found {
			t.Errorf("missing swap metrics for pod %s", podName)
			continue
		}
		if actualBytes != expectedBytes {
			t.Errorf("unexpected swap metrics for pod %s: expected %s, got %s", podName, expectedBytes, actualBytes)
		}
	}
}

func getResultColumnValues(result string, columnIndex int) []string {
	resultLines := strings.Split(result, "\n")
	values := make([]string, len(resultLines)-2) // don't process first (header) and last (empty) line

	for i, line := range resultLines[1 : len(resultLines)-1] { // don't process first (header) and last (empty) line
		value := strings.Fields(line)[columnIndex]
		values[i] = value
	}

	return values
}

func emptyPodListResponse(req *http.Request) (*http.Response, error) {
	body, _ := marshallBody(metricsv1beta1api.PodMetricsList{
		ListMeta: metav1.ListMeta{ResourceVersion: "2"},
		Items:    nil,
	})
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     cmdtesting.DefaultHeader(),
		Body:       body,
	}, nil
}

type runTopPodOpts struct {
	apisBody    string
	fakeMetrics *metricsfake.Clientset
	options     *TopPodOptions
	cmdArgs     []string
	extraPaths  func(req *http.Request) (*http.Response, error)
}

func runTopPodTest(t *testing.T, opts runTopPodOpts) (stdout string, stderr string) {
	tf := cmdtesting.NewTestFactory().WithNamespace(testNS)
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	ns := scheme.Codecs.WithoutConversion()

	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/api":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
			case "/apis":
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(opts.apisBody)))}, nil
			case "/api/v1/namespaces/" + testNS + "/pods":
				if opts.extraPaths != nil {
					return opts.extraPaths(req)
				}
				// The command lists pods directly (e.g. to resolve a field
				// selector, or to check pod age when no metrics are returned),
				// so serve the standard pod list by default.
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, defaultPodList())}, nil
			default:
				t.Fatalf("unexpected request: %#v\nGot URL: %#v", req, req.URL)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
	streams, _, buf, errbuf := genericiooptions.NewTestIOStreams()

	cmd := NewCmdTopPod(tf, nil, streams)
	var cmdOptions *TopPodOptions
	if opts.options != nil {
		cmdOptions = opts.options
	} else {
		cmdOptions = &TopPodOptions{}
	}
	cmdOptions.IOStreams = streams

	if err := cmdOptions.Complete(tf, cmd, opts.cmdArgs); err != nil {
		t.Fatal(err)
	}
	cmdOptions.MetricsClient = opts.fakeMetrics
	if err := cmdOptions.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := cmdOptions.RunTopPod(); err != nil {
		t.Fatal(err)
	}
	return buf.String(), errbuf.String()
}

func defaultPodList() *v1.PodList {
	return &v1.PodList{
		Items: []v1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: testNS},
				Spec:       v1.PodSpec{NodeName: "node-a"},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "pod2", Namespace: testNS},
				Spec:       v1.PodSpec{NodeName: "node-b"},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "pod3", Namespace: testNS},
				Spec:       v1.PodSpec{NodeName: "node-c"},
			},
		},
	}
}

// onlyPod1ListResponse serves a pod list containing only pod1, simulating the
// core API filtering pods by a field selector.
func onlyPod1ListResponse(req *http.Request) (*http.Response, error) {
	body, _ := marshallBody(v1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "2"},
		Items: []v1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: testNS},
				Spec:       v1.PodSpec{NodeName: "node-a"},
			},
		},
	})
	return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
}

func testPodMetricsData() []metricsapi.PodMetrics {
	return []metricsapi.PodMetrics{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "test", ResourceVersion: "10", Labels: map[string]string{"key": "value"}},
			Window:     metav1.Duration{Duration: time.Minute},
			Containers: []metricsapi.ContainerMetrics{
				{
					Name: "container1-1",
					Usage: v1.ResourceList{
						v1.ResourceCPU:     *resource.NewMilliQuantity(1, resource.DecimalSI),
						v1.ResourceMemory:  *resource.NewQuantity(2*(1024*1024), resource.DecimalSI),
						"swap":             *resource.NewQuantity(1*(1024*1024), resource.DecimalSI),
						v1.ResourceStorage: *resource.NewQuantity(3*(1024*1024), resource.DecimalSI),
					},
				},
				{
					Name: "container1-2",
					Usage: v1.ResourceList{
						v1.ResourceCPU:     *resource.NewMilliQuantity(4, resource.DecimalSI),
						v1.ResourceMemory:  *resource.NewQuantity(5*(1024*1024), resource.DecimalSI),
						"swap":             *resource.NewQuantity(3*(1024*1024), resource.DecimalSI),
						v1.ResourceStorage: *resource.NewQuantity(6*(1024*1024), resource.DecimalSI),
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod2", Namespace: "test", ResourceVersion: "11", Labels: map[string]string{"key": "value"}},
			Window:     metav1.Duration{Duration: time.Minute},
			Containers: []metricsapi.ContainerMetrics{
				{
					Name: "container2-1",
					Usage: v1.ResourceList{
						v1.ResourceCPU:     *resource.NewMilliQuantity(7, resource.DecimalSI),
						v1.ResourceMemory:  *resource.NewQuantity(8*(1024*1024), resource.DecimalSI),
						v1.ResourceStorage: *resource.NewQuantity(9*(1024*1024), resource.DecimalSI),
					},
				},
				{
					Name: "container2-2",
					Usage: v1.ResourceList{
						v1.ResourceCPU:     *resource.NewMilliQuantity(10, resource.DecimalSI),
						v1.ResourceMemory:  *resource.NewQuantity(11*(1024*1024), resource.DecimalSI),
						v1.ResourceStorage: *resource.NewQuantity(12*(1024*1024), resource.DecimalSI),
					},
				},
				{
					Name: "container2-3",
					Usage: v1.ResourceList{
						v1.ResourceCPU:     *resource.NewMilliQuantity(13, resource.DecimalSI),
						v1.ResourceMemory:  *resource.NewQuantity(14*(1024*1024), resource.DecimalSI),
						v1.ResourceStorage: *resource.NewQuantity(15*(1024*1024), resource.DecimalSI),
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod3", Namespace: "test", ResourceVersion: "12"},
			Window:     metav1.Duration{Duration: time.Minute},
			Containers: []metricsapi.ContainerMetrics{
				{
					Name: "container3-1",
					Usage: v1.ResourceList{
						v1.ResourceCPU:     *resource.NewMilliQuantity(7, resource.DecimalSI),
						v1.ResourceMemory:  *resource.NewQuantity(8*(1024*1024), resource.DecimalSI),
						"swap":             *resource.NewQuantity(3*(1024*1024), resource.DecimalSI),
						v1.ResourceStorage: *resource.NewQuantity(9*(1024*1024), resource.DecimalSI),
					},
				},
			},
		},
	}
}
