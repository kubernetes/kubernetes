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
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/googleapis/gnostic/OpenAPIv2"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	apiversion "k8s.io/apimachinery/pkg/version"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	core "k8s.io/client-go/testing"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	metricsv1alpha1api "k8s.io/metrics/pkg/apis/metrics/v1alpha1"
	metricsv1beta1api "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
)

const (
	topPathPrefix           = baseMetricsAddress + "/" + metricsApiVersion
	topMetricsAPIPathPrefix = "/apis/metrics.k8s.io/v1beta1"
	apibody                 = `{
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
	// This is not the full output one would usually get, just a trimmed down version.
	apisbody = `{
	"kind": "APIGroupList",
	"apiVersion": "v1",
	"groups": [{}]
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
		name            string
		flags           map[string]string
		args            []string
		expectedPath    string
		expectedQuery   string
		namespaces      []string
		containers      bool
		listsNamespaces bool
	}{
		{
			name:            "all namespaces",
			flags:           map[string]string{"all-namespaces": "true"},
			expectedPath:    topPathPrefix + "/pods",
			namespaces:      []string{testNS, "secondtestns", "thirdtestns"},
			listsNamespaces: true,
		},
		{
			name:         "all in namespace",
			expectedPath: topPathPrefix + "/namespaces/" + testNS + "/pods",
			namespaces:   []string{testNS, testNS},
		},
		{
			name:         "pod with name",
			args:         []string{"pod1"},
			expectedPath: topPathPrefix + "/namespaces/" + testNS + "/pods/pod1",
			namespaces:   []string{testNS},
		},
		{
			name:          "pod with label selector",
			flags:         map[string]string{"selector": "key=value"},
			expectedPath:  topPathPrefix + "/namespaces/" + testNS + "/pods",
			expectedQuery: "labelSelector=" + url.QueryEscape("key=value"),
			namespaces:    []string{testNS, testNS},
		},
		{
			name:         "pod with container metrics",
			flags:        map[string]string{"containers": "true"},
			args:         []string{"pod1"},
			expectedPath: topPathPrefix + "/namespaces/" + testNS + "/pods/pod1",
			namespaces:   []string{testNS},
			containers:   true,
		},
		{
			name:         "no-headers set",
			flags:        map[string]string{"containers": "true", "no-headers": "true"},
			args:         []string{"pod1"},
			expectedPath: topPathPrefix + "/namespaces/" + testNS + "/pods/pod1",
			namespaces:   []string{testNS},
			containers:   true,
		},
	}
	cmdtesting.InitTestErrorHandler(t)
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Logf("Running test case: %s", testCase.name)
			metricsList := testPodMetricsData()
			var expectedMetrics []metricsv1alpha1api.PodMetrics
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

			var response interface{}
			if len(expectedMetrics) == 1 {
				response = expectedMetrics[0]
			} else {
				response = metricsv1alpha1api.PodMetricsList{
					ListMeta: metav1.ListMeta{
						ResourceVersion: "2",
					},
					Items: expectedMetrics,
				}
			}

			tf := cmdtesting.NewTestFactory().WithNamespace(testNS)
			defer tf.Cleanup()

			ns := scheme.Codecs

			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m, q := req.URL.Path, req.Method, req.URL.RawQuery; {
					case p == "/api":
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
					case p == "/apis":
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apisbody)))}, nil
					case p == testCase.expectedPath && m == "GET" && (testCase.expectedQuery == "" || q == testCase.expectedQuery):
						body, err := marshallBody(response)
						if err != nil {
							t.Errorf("%s: unexpected error: %v", testCase.name, err)
						}
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					default:
						t.Fatalf("%s: unexpected request: %#v\nGot URL: %#v\nExpected path: %#v\nExpected query: %#v",
							testCase.name, req, req.URL, testCase.expectedPath, testCase.expectedQuery)
						return nil, nil
					}
				}),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
			streams, _, buf, _ := genericclioptions.NewTestIOStreams()

			cmd := NewCmdTopPod(tf, nil, streams)
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
			if cmdutil.GetFlagBool(cmd, "no-headers") && strings.Contains(result, "MEMORY") {
				t.Errorf("%s: unexpected headers with no-headers option set: \n%s", testCase.name, result)
			}
		})
	}
}

func TestTopPodWithMetricsServer(t *testing.T) {
	testNS := "testns"
	testCases := []struct {
		name            string
		namespace       string
		options         *TopPodOptions
		args            []string
		expectedPath    string
		expectedQuery   string
		namespaces      []string
		containers      bool
		listsNamespaces bool
	}{
		{
			name:            "all namespaces",
			options:         &TopPodOptions{AllNamespaces: true},
			expectedPath:    topMetricsAPIPathPrefix + "/pods",
			namespaces:      []string{testNS, "secondtestns", "thirdtestns"},
			listsNamespaces: true,
		},
		{
			name:         "all in namespace",
			expectedPath: topMetricsAPIPathPrefix + "/namespaces/" + testNS + "/pods",
			namespaces:   []string{testNS, testNS},
		},
		{
			name:         "pod with name",
			args:         []string{"pod1"},
			expectedPath: topMetricsAPIPathPrefix + "/namespaces/" + testNS + "/pods/pod1",
			namespaces:   []string{testNS},
		},
		{
			name:          "pod with label selector",
			options:       &TopPodOptions{Selector: "key=value"},
			expectedPath:  topMetricsAPIPathPrefix + "/namespaces/" + testNS + "/pods",
			expectedQuery: "labelSelector=" + url.QueryEscape("key=value"),
			namespaces:    []string{testNS, testNS},
		},
		{
			name:         "pod with container metrics",
			options:      &TopPodOptions{PrintContainers: true},
			args:         []string{"pod1"},
			expectedPath: topMetricsAPIPathPrefix + "/namespaces/" + testNS + "/pods/pod1",
			namespaces:   []string{testNS},
			containers:   true,
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

			ns := scheme.Codecs

			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p := req.URL.Path; {
					case p == "/api":
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
					case p == "/apis":
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apisbodyWithMetrics)))}, nil
					default:
						t.Fatalf("%s: unexpected request: %#v\nGot URL: %#v",
							testCase.name, req, req.URL)
						return nil, nil
					}
				}),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
			streams, _, buf, _ := genericclioptions.NewTestIOStreams()

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
		})
	}
}

type fakeDiscovery struct{}

// ServerGroups returns the supported groups, with information like supported versions and the
// preferred version.
func (d *fakeDiscovery) ServerGroups() (*metav1.APIGroupList, error) {
	return nil, nil
}

// ServerResourcesForGroupVersion returns the supported resources for a group and version.
func (d *fakeDiscovery) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	return nil, nil
}

// ServerResources returns the supported resources for all groups and versions.
func (d *fakeDiscovery) ServerResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

// ServerPreferredResources returns the supported resources with the version preferred by the
// server.
func (d *fakeDiscovery) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

// ServerPreferredNamespacedResources returns the supported namespaced resources with the
// version preferred by the server.
func (d *fakeDiscovery) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

// ServerVersion retrieves and parses the server's version (git version).
func (d *fakeDiscovery) ServerVersion() (*apiversion.Info, error) {
	return nil, nil
}

// OpenAPISchema retrieves and parses the swagger API schema the server supports.
func (d *fakeDiscovery) OpenAPISchema() (*openapi_v2.Document, error) {
	return nil, nil
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (d *fakeDiscovery) RESTClient() restclient.Interface {
	return nil
}

func TestTopPodCustomDefaults(t *testing.T) {
	customBaseHeapsterServiceAddress := "/api/v1/namespaces/custom-namespace/services/https:custom-heapster-service:/proxy"
	customBaseMetricsAddress := customBaseHeapsterServiceAddress + "/apis/metrics"
	customTopPathPrefix := customBaseMetricsAddress + "/" + metricsApiVersion

	testNS := "custom-namespace"
	testCases := []struct {
		name            string
		flags           map[string]string
		args            []string
		expectedPath    string
		expectedQuery   string
		namespaces      []string
		containers      bool
		listsNamespaces bool
	}{
		{
			name:            "all namespaces",
			flags:           map[string]string{"all-namespaces": "true"},
			expectedPath:    customTopPathPrefix + "/pods",
			namespaces:      []string{testNS, "secondtestns", "thirdtestns"},
			listsNamespaces: true,
		},
		{
			name:         "all in namespace",
			expectedPath: customTopPathPrefix + "/namespaces/" + testNS + "/pods",
			namespaces:   []string{testNS, testNS},
		},
		{
			name:         "pod with name",
			args:         []string{"pod1"},
			expectedPath: customTopPathPrefix + "/namespaces/" + testNS + "/pods/pod1",
			namespaces:   []string{testNS},
		},
		{
			name:          "pod with label selector",
			flags:         map[string]string{"selector": "key=value"},
			expectedPath:  customTopPathPrefix + "/namespaces/" + testNS + "/pods",
			expectedQuery: "labelSelector=" + url.QueryEscape("key=value"),
			namespaces:    []string{testNS, testNS},
		},
		{
			name:         "pod with container metrics",
			flags:        map[string]string{"containers": "true"},
			args:         []string{"pod1"},
			expectedPath: customTopPathPrefix + "/namespaces/" + testNS + "/pods/pod1",
			namespaces:   []string{testNS},
			containers:   true,
		},
	}
	cmdtesting.InitTestErrorHandler(t)
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Logf("Running test case: %s", testCase.name)
			metricsList := testPodMetricsData()
			var expectedMetrics []metricsv1alpha1api.PodMetrics
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

			var response interface{}
			if len(expectedMetrics) == 1 {
				response = expectedMetrics[0]
			} else {
				response = metricsv1alpha1api.PodMetricsList{
					ListMeta: metav1.ListMeta{
						ResourceVersion: "2",
					},
					Items: expectedMetrics,
				}
			}

			tf := cmdtesting.NewTestFactory().WithNamespace(testNS)
			defer tf.Cleanup()

			ns := scheme.Codecs

			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m, q := req.URL.Path, req.Method, req.URL.RawQuery; {
					case p == "/api":
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apibody)))}, nil
					case p == "/apis":
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(apisbody)))}, nil
					case p == testCase.expectedPath && m == "GET" && (testCase.expectedQuery == "" || q == testCase.expectedQuery):
						body, err := marshallBody(response)
						if err != nil {
							t.Errorf("%s: unexpected error: %v", testCase.name, err)
						}
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					default:
						t.Fatalf("%s: unexpected request: %#v\nGot URL: %#v\nExpected path: %#v\nExpected query: %#v",
							testCase.name, req, req.URL, testCase.expectedPath, testCase.expectedQuery)
						return nil, nil
					}
				}),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
			streams, _, buf, _ := genericclioptions.NewTestIOStreams()

			opts := &TopPodOptions{
				HeapsterOptions: HeapsterTopOptions{
					Namespace: "custom-namespace",
					Scheme:    "https",
					Service:   "custom-heapster-service",
				},
				DiscoveryClient: &fakeDiscovery{},
				IOStreams:       streams,
			}
			cmd := NewCmdTopPod(tf, opts, streams)
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

func testPodMetricsData() []metricsv1alpha1api.PodMetrics {
	return []metricsv1alpha1api.PodMetrics{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "test", ResourceVersion: "10"},
			Window:     metav1.Duration{Duration: time.Minute},
			Containers: []metricsv1alpha1api.ContainerMetrics{
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
			ObjectMeta: metav1.ObjectMeta{Name: "pod2", Namespace: "test", ResourceVersion: "11"},
			Window:     metav1.Duration{Duration: time.Minute},
			Containers: []metricsv1alpha1api.ContainerMetrics{
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
			Containers: []metricsv1alpha1api.ContainerMetrics{
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
