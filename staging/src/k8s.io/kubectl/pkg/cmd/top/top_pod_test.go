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
	"net/url"
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
	metricsv1alpha1api "k8s.io/metrics/pkg/apis/metrics/v1alpha1"
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

	apisbodyWithMetrics = `{
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
)

func TestTopPod(t *testing.T) {
	testNS := "testns"
	testCases := []struct {
		name               string
		namespace          string
		options            *TopPodOptions
		args               []string
		expectedQuery      string
		expectedPods       []string
		expectedContainers []string
		namespaces         []string
		containers         bool
		listsNamespaces    bool
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
			name:          "pod with label selector",
			options:       &TopPodOptions{LabelSelector: "key=value"},
			expectedQuery: "labelSelector=" + url.QueryEscape("key=value"),
			namespaces:    []string{testNS, testNS},
		},
		{
			name:          "pod with field selector",
			options:       &TopPodOptions{FieldSelector: "key=value"},
			expectedQuery: "fieldSelector=" + url.QueryEscape("key=value"),
			namespaces:    []string{testNS, testNS},
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
	}
	cmdtesting.InitTestErrorHandler(t)
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			metricsList := testV1beta1PodMetricsData()
			var expectedMetrics []metricsv1beta1api.PodMetrics
			var expectedContainerNames, nonExpectedMetricsNames []string
			for n, m := range metricsList {
				if n < len(testCase.namespaces) {
					m.Namespace = testCase.namespaces[n]
					expectedMetrics = append(expectedMetrics, m)
					for _, c := range m.Containers {
						expectedContainerNames = append(expectedContainerNames, c.Name)
					}
				} else {
					nonExpectedMetricsNames = append(nonExpectedMetricsNames, m.Name)
				}
			}

			fakemetricsClientset := &metricsfake.Clientset{}

			if len(expectedMetrics) == 1 {
				fakemetricsClientset.AddReactor("get", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
					return true, &expectedMetrics[0], nil
				})
			} else {
				fakemetricsClientset.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
					res := &metricsv1beta1api.PodMetricsList{
						ListMeta: metav1.ListMeta{
							ResourceVersion: "2",
						},
						Items: expectedMetrics,
					}
					return true, res, nil
				})
			}

			tf := cmdtesting.NewTestFactory().WithNamespace(testNS)
			defer tf.Cleanup()

			ns := scheme.Codecs.WithoutConversion()

			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p := req.URL.Path; {
					case p == "/api":
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
					case p == "/apis":
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(apisbodyWithMetrics)))}, nil
					default:
						t.Fatalf("%s: unexpected request: %#v\nGot URL: %#v",
							testCase.name, req, req.URL)
						return nil, nil
					}
				}),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
			streams, _, buf, _ := genericiooptions.NewTestIOStreams()

			cmd := NewCmdTopPod(tf, nil, streams)
			var cmdOptions *TopPodOptions
			if testCase.options != nil {
				cmdOptions = testCase.options
			} else {
				cmdOptions = &TopPodOptions{}
			}
			cmdOptions.IOStreams = streams

			// TODO in the long run, we want to test most of our commands like this. Wire the options struct with specific mocks
			// TODO then check the particular Run functionality and harvest results from fake clients.  We probably end up skipping the factory altogether.
			if err := cmdOptions.Complete(tf, cmd, testCase.args); err != nil {
				t.Fatal(err)
			}
			cmdOptions.MetricsClient = fakemetricsClientset
			if err := cmdOptions.Validate(); err != nil {
				t.Fatal(err)
			}
			if err := cmdOptions.RunTopPod(); err != nil {
				t.Fatal(err)
			}

			// Check the presence of pod names&namespaces/container names in the output.
			result := buf.String()
			if testCase.containers {
				for _, containerName := range expectedContainerNames {
					if !strings.Contains(result, containerName) {
						t.Errorf("missing metrics for container %s: \n%s", containerName, result)
					}
				}
			}
			for _, m := range expectedMetrics {
				if !strings.Contains(result, m.Name) {
					t.Errorf("missing metrics for %s: \n%s", m.Name, result)
				}
				if testCase.listsNamespaces && !strings.Contains(result, m.Namespace) {
					t.Errorf("missing metrics for %s/%s: \n%s", m.Namespace, m.Name, result)
				}
			}
			for _, name := range nonExpectedMetricsNames {
				if strings.Contains(result, name) {
					t.Errorf("unexpected metrics for %s: \n%s", name, result)
				}
			}
			if testCase.expectedPods != nil {
				resultPods := getResultColumnValues(result, 0)
				if !reflect.DeepEqual(testCase.expectedPods, resultPods) {
					t.Errorf("pods not matching:\n\texpectedPods: %v\n\tresultPods: %v\n", testCase.expectedPods, resultPods)
				}
			}
			if testCase.expectedContainers != nil {
				resultContainers := getResultColumnValues(result, 1)
				if !reflect.DeepEqual(testCase.expectedContainers, resultContainers) {
					t.Errorf("containers not matching:\n\texpectedContainers: %v\n\tresultContainers: %v\n", testCase.expectedContainers, resultContainers)
				}
			}
		})
	}
}

func TestTopPodPrintFlags(t *testing.T) {
	testNS := "test"
	pods := []v1.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: testNS}, Spec: v1.PodSpec{NodeName: "node1"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "pod2", Namespace: testNS}, Spec: v1.PodSpec{NodeName: "node2"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "pod3", Namespace: testNS}, Spec: v1.PodSpec{NodeName: "node3"}},
	}
	metrics := testV1beta1PodMetricsData()
	testCases := []struct {
		name string

		sortBy          string
		printContainers bool
		allNamespaces   bool
		noHeaders       bool
		sum             bool

		outputFormat string
		resourceName string

		expectedOutput string
		expectedErr    string
	}{
		{
			name: "output",
			expectedOutput: `NAME   CPU(cores)   MEMORY(bytes)   
pod1   5m           7Mi             
pod2   30m          33Mi            
pod3   7m           8Mi             
`,
		},
		{
			name:         "output with spec pod name",
			resourceName: "pod2",
			expectedOutput: `NAME   CPU(cores)   MEMORY(bytes)   
pod2   30m          33Mi            
`,
		},
		{
			name:      "output with no headers",
			noHeaders: true,
			expectedOutput: `pod1   5m    7Mi    
pod2   30m   33Mi   
pod3   7m    8Mi    
`,
		},
		{
			name: "output with sum",
			sum:  true,
			expectedOutput: `NAME   CPU(cores)   MEMORY(bytes)   
pod1   5m           7Mi             
pod2   30m          33Mi            
pod3   7m           8Mi             
       ________     ________        
       42m          48Mi            
`,
		},
		{
			name:   "sort by cpu",
			sortBy: "cpu",
			expectedOutput: `NAME   CPU(cores)   MEMORY(bytes)   
pod2   30m          33Mi            
pod3   7m           8Mi             
pod1   5m           7Mi             
`,
		},
		{
			name:          "output with namespace",
			allNamespaces: true,
			expectedOutput: `NAMESPACE   NAME   CPU(cores)   MEMORY(bytes)   
test        pod1   5m           7Mi             
test        pod2   30m          33Mi            
test        pod3   7m           8Mi             
`,
		},
		{
			name:            "output with containers",
			printContainers: true,
			expectedOutput: `POD    NAME           CPU(cores)   MEMORY(bytes)   
pod1   container1-1   1m           2Mi             
pod1   container1-2   4m           5Mi             
pod2   container2-1   7m           8Mi             
pod2   container2-2   10m          11Mi            
pod2   container2-3   13m          14Mi            
pod3   container3-1   7m           8Mi             
`,
		},
		{
			name:         "sort by cpu and wide",
			sortBy:       "cpu",
			outputFormat: "wide",
			expectedOutput: `NAME   NODE    CPU(cores)   MEMORY(bytes)   
pod2   node2   30m          33Mi            
pod3   node3   7m           8Mi             
pod1   node1   5m           7Mi             
`,
		},
		{
			name:            "output with containers and wide and sort by cpu",
			printContainers: true,
			outputFormat:    "wide",
			sortBy:          "cpu",
			expectedOutput: `POD    NAME           NODE    CPU(cores)   MEMORY(bytes)   
pod2   container2-3   node2   13m          14Mi            
pod2   container2-2   node2   10m          11Mi            
pod2   container2-1   node2   7m           8Mi             
pod3   container3-1   node3   7m           8Mi             
pod1   container1-2   node1   4m           5Mi             
pod1   container1-1   node1   1m           2Mi             
`,
		},
		{
			name:            "output with namespace and containers and wide and sort by cpu",
			allNamespaces:   true,
			printContainers: true,
			outputFormat:    "wide",
			sortBy:          "cpu",
			expectedOutput: `NAMESPACE   POD    NAME           NODE    CPU(cores)   MEMORY(bytes)   
test        pod2   container2-3   node2   13m          14Mi            
test        pod2   container2-2   node2   10m          11Mi            
test        pod2   container2-1   node2   7m           8Mi             
test        pod3   container3-1   node3   7m           8Mi             
test        pod1   container1-2   node1   4m           5Mi             
test        pod1   container1-1   node1   1m           2Mi             
`,
		},
		{
			name:            "output with namespace and containers and wide and sort by cpu and sum",
			allNamespaces:   true,
			printContainers: true,
			outputFormat:    "wide",
			sortBy:          "cpu",
			sum:             true,
			expectedOutput: `NAMESPACE   POD    NAME           NODE    CPU(cores)   MEMORY(bytes)   
test        pod2   container2-3   node2   13m          14Mi            
test        pod2   container2-2   node2   10m          11Mi            
test        pod2   container2-1   node2   7m           8Mi             
test        pod3   container3-1   node3   7m           8Mi             
test        pod1   container1-2   node1   4m           5Mi             
test        pod1   container1-1   node1   1m           2Mi             
                                          ________     ________        
                                          42m          48Mi            
`,
		},
		{
			name:            "output with namespace and containers and wide and sort by cpu and sum and no-headers",
			allNamespaces:   true,
			printContainers: true,
			outputFormat:    "wide",
			sortBy:          "cpu",
			noHeaders:       true,
			sum:             true,
			expectedOutput: `test   pod2   container2-3   node2   13m        14Mi       
test   pod2   container2-2   node2   10m        11Mi       
test   pod2   container2-1   node2   7m         8Mi        
test   pod3   container3-1   node3   7m         8Mi        
test   pod1   container1-2   node1   4m         5Mi        
test   pod1   container1-1   node1   1m         2Mi        
                                     ________   ________   
                                     42m        48Mi       
`,
		},
		{
			name:            "output with spec pod name and namespace and containers and wide and sort by cpu and sum",
			resourceName:    "pod2",
			allNamespaces:   true,
			printContainers: true,
			outputFormat:    "wide",
			sortBy:          "cpu",
			sum:             true,
			expectedOutput: `NAMESPACE   POD    NAME           NODE    CPU(cores)   MEMORY(bytes)   
test        pod2   container2-3   node2   13m          14Mi            
test        pod2   container2-2   node2   10m          11Mi            
test        pod2   container2-1   node2   7m           8Mi             
                                          ________     ________        
                                          30m          33Mi            
`,
		},
		{
			name:            "output with spec pod name and namespace and containers and wide and sort by cpu and sum and no-headers",
			resourceName:    "pod2",
			allNamespaces:   true,
			printContainers: true,
			noHeaders:       true,
			outputFormat:    "wide",
			sortBy:          "cpu",
			sum:             true,
			expectedOutput: `test   pod2   container2-3   node2   13m        14Mi       
test   pod2   container2-2   node2   10m        11Mi       
test   pod2   container2-1   node2   7m         8Mi        
                                     ________   ________   
                                     30m        33Mi       
`,
		},
	}
	cmdtesting.InitTestErrorHandler(t)
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			fakemetricsClientset := &metricsfake.Clientset{}
			fakemetricsClientset.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
				res := &metricsv1beta1api.PodMetricsList{
					ListMeta: metav1.ListMeta{
						ResourceVersion: "2",
					},
					Items: metrics,
				}
				return true, res, nil
			})

			resourceName := "test"
			pod := v1.Pod{}
			if len(testCase.resourceName) > 0 {
				resourceName = testCase.resourceName
				for _, m := range metrics {
					if m.Name == resourceName {
						fakemetricsClientset.AddReactor("get", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
							return true, &m, nil
						})
						break
					}
				}
				for _, p := range pods {
					if p.Name == resourceName {
						pod = p
						break
					}
				}
			}

			tf := cmdtesting.NewTestFactory().WithNamespace(testNS)
			defer tf.Cleanup()

			ns := scheme.Codecs.WithoutConversion()
			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p := req.URL.Path; {
					case p == "/api":
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
					case p == "/apis":
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(apisbodyWithMetrics)))}, nil
					case p == "/api/v1/pods":
						fallthrough
					case p == "/api/v1/namespaces/"+testNS+"/pods":
						body, _ := marshallBody(v1.PodList{
							ListMeta: metav1.ListMeta{
								ResourceVersion: "2",
							},
							Items: pods,
						})
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					case p == "/api/v1/namespaces/"+testNS+"/pods/"+resourceName:
						body, _ := marshallBody(pod)
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					default:
						t.Fatalf("%s: unexpected request: %#v\nGot URL: %#v",
							testCase.name, req, req.URL)
						return nil, nil
					}
				}),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
			streams, _, buf, errbuf := genericiooptions.NewTestIOStreams()

			cmd := NewCmdTopPod(tf, nil, streams)
			cmdOptions := &TopPodOptions{IOStreams: streams}
			if err := cmdOptions.Complete(tf, cmd, nil); err != nil {
				t.Fatal(err)
			}
			cmdOptions.MetricsClient = fakemetricsClientset
			if err := cmdOptions.Validate(); err != nil {
				t.Fatal(err)
			}

			cmdOptions.PrintFlags.OutputFormat = &testCase.outputFormat
			if len(testCase.resourceName) > 0 {
				cmdOptions.ResourceName = pod.Name
				cmdOptions.Namespace = pod.Namespace
			}

			cmdOptions.SortBy = testCase.sortBy
			cmdOptions.PrintContainers = testCase.printContainers
			cmdOptions.AllNamespaces = testCase.allNamespaces
			cmdOptions.NoHeaders = testCase.noHeaders
			cmdOptions.Sum = testCase.sum

			if err := cmdOptions.RunTopPod(); err != nil {
				t.Fatal(err)
			}

			if e, a := testCase.expectedOutput, buf.String(); e != a {
				t.Errorf("Unexpected output:\nExpected:\n%v\nActual:\n%v", e, a)
			}
			if e, a := testCase.expectedErr, errbuf.String(); e != a {
				t.Errorf("Unexpected error:\nExpected:\n%v\nActual:\n%v", e, a)
			}
		})
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

func TestTopPodNoResourcesFound(t *testing.T) {
	testNS := "testns"
	testCases := []struct {
		name           string
		options        *TopPodOptions
		namespace      string
		expectedOutput string
		expectedErr    string
	}{
		{
			name:           "all namespaces",
			options:        &TopPodOptions{AllNamespaces: true},
			expectedOutput: "",
			expectedErr:    "No resources found\n",
		},
		{
			name:           "all in namespace",
			namespace:      testNS,
			expectedOutput: "",
			expectedErr:    "No resources found in " + testNS + " namespace.\n",
		},
	}
	cmdtesting.InitTestErrorHandler(t)
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			fakemetricsClientset := &metricsfake.Clientset{}
			fakemetricsClientset.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
				res := &metricsv1beta1api.PodMetricsList{
					ListMeta: metav1.ListMeta{
						ResourceVersion: "2",
					},
					Items: nil, // No metrics found
				}
				return true, res, nil
			})

			tf := cmdtesting.NewTestFactory().WithNamespace(testNS)
			defer tf.Cleanup()

			ns := scheme.Codecs.WithoutConversion()

			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p := req.URL.Path; {
					case p == "/api":
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
					case p == "/apis":
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: io.NopCloser(bytes.NewReader([]byte(apisbodyWithMetrics)))}, nil
					case p == "/api/v1/namespaces/"+testNS+"/pods":
						// Top Pod calls this endpoint to check if there are pods whenever it gets no metrics,
						// so we need to return no pods for this test scenario
						body, _ := marshallBody(metricsv1alpha1api.PodMetricsList{
							ListMeta: metav1.ListMeta{
								ResourceVersion: "2",
							},
							Items: nil, // No pods found
						})
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					default:
						t.Fatalf("%s: unexpected request: %#v\nGot URL: %#v",
							testCase.name, req, req.URL)
						return nil, nil
					}
				}),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
			streams, _, buf, errbuf := genericiooptions.NewTestIOStreams()

			cmd := NewCmdTopPod(tf, nil, streams)
			var cmdOptions *TopPodOptions
			if testCase.options != nil {
				cmdOptions = testCase.options
			} else {
				cmdOptions = &TopPodOptions{}
			}
			cmdOptions.IOStreams = streams

			if err := cmdOptions.Complete(tf, cmd, nil); err != nil {
				t.Fatal(err)
			}
			cmdOptions.MetricsClient = fakemetricsClientset
			if err := cmdOptions.Validate(); err != nil {
				t.Fatal(err)
			}
			if err := cmdOptions.RunTopPod(); err != nil {
				t.Fatal(err)
			}

			if e, a := testCase.expectedOutput, buf.String(); e != a {
				t.Errorf("Unexpected output:\nExpected:\n%v\nActual:\n%v", e, a)
			}
			if e, a := testCase.expectedErr, errbuf.String(); e != a {
				t.Errorf("Unexpected error:\nExpected:\n%v\nActual:\n%v", e, a)
			}
		})
	}
}

func testV1beta1PodMetricsData() []metricsv1beta1api.PodMetrics {
	return []metricsv1beta1api.PodMetrics{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "test", ResourceVersion: "10", Labels: map[string]string{"key": "value"}},
			Window:     metav1.Duration{Duration: time.Minute},
			Containers: []metricsv1beta1api.ContainerMetrics{
				{
					Name: "container1-1",
					Usage: v1.ResourceList{
						v1.ResourceCPU:     *resource.NewMilliQuantity(1, resource.DecimalSI),
						v1.ResourceMemory:  *resource.NewQuantity(2*(1024*1024), resource.DecimalSI),
						v1.ResourceStorage: *resource.NewQuantity(3*(1024*1024), resource.DecimalSI),
					},
				},
				{
					Name: "container1-2",
					Usage: v1.ResourceList{
						v1.ResourceCPU:     *resource.NewMilliQuantity(4, resource.DecimalSI),
						v1.ResourceMemory:  *resource.NewQuantity(5*(1024*1024), resource.DecimalSI),
						v1.ResourceStorage: *resource.NewQuantity(6*(1024*1024), resource.DecimalSI),
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod2", Namespace: "test", ResourceVersion: "11", Labels: map[string]string{"key": "value"}},
			Window:     metav1.Duration{Duration: time.Minute},
			Containers: []metricsv1beta1api.ContainerMetrics{
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
			Containers: []metricsv1beta1api.ContainerMetrics{
				{
					Name: "container3-1",
					Usage: v1.ResourceList{
						v1.ResourceCPU:     *resource.NewMilliQuantity(7, resource.DecimalSI),
						v1.ResourceMemory:  *resource.NewQuantity(8*(1024*1024), resource.DecimalSI),
						v1.ResourceStorage: *resource.NewQuantity(9*(1024*1024), resource.DecimalSI),
					},
				},
			},
		},
	}
}
