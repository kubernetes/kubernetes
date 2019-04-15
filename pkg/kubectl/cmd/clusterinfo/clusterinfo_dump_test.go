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

package clusterinfo

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"testing"

	flag "github.com/spf13/pflag"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
)

func TestSetupOutputWriterNoOp(t *testing.T) {
	tests := []string{"", "-"}
	for _, test := range tests {
		_, _, buf, _ := genericclioptions.NewTestIOStreams()
		f := cmdtesting.NewTestFactory()
		defer f.Cleanup()

		writer := setupOutputWriter(test, buf, "/some/file/that/should/be/ignored")
		if writer != buf {
			t.Errorf("expected: %v, saw: %v", buf, writer)
		}
	}
}

func TestSetupOutputWriterFile(t *testing.T) {
	file := "output.json"
	dir, err := ioutil.TempDir(os.TempDir(), "out")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fullPath := path.Join(dir, file)
	defer os.RemoveAll(dir)

	_, _, buf, _ := genericclioptions.NewTestIOStreams()
	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()

	writer := setupOutputWriter(dir, buf, file)
	if writer == buf {
		t.Errorf("expected: %v, saw: %v", buf, writer)
	}
	output := "some data here"
	writer.Write([]byte(output))

	data, err := ioutil.ReadFile(fullPath)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if string(data) != output {
		t.Errorf("expected: %v, saw: %v", output, data)
	}
}

func TestCmdClusterInfoDumpCustomNamespaces(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	tests := map[string]struct {
		expectedNamespacesRequests []string
		populateCmdFlags           func(f *flag.FlagSet) error
	}{
		"should dump default namespaces": {
			expectedNamespacesRequests: []string{"kube-system", "default"},

			populateCmdFlags: func(f *flag.FlagSet) error {
				// do not populate any flags, use default options
				return nil
			},
		},
		"should dump requested namespaces": {
			expectedNamespacesRequests: []string{"qa", "production"},

			populateCmdFlags: func(f *flag.FlagSet) error {
				return f.Set("namespaces", "qa,production")
			},
		},
	}

	for testName, testCase := range tests {
		t.Run(testName, func(t *testing.T) {
			expectedCalls := map[string]io.ReadCloser{
				"/api/v1/nodes": dummyResponse(),
			}

			for _, ns := range testCase.expectedNamespacesRequests {
				expectedCalls[fmt.Sprintf("/api/v1/namespaces/%s/events", ns)] = dummyResponse()
				expectedCalls[fmt.Sprintf("/api/v1/namespaces/%s/replicationcontrollers", ns)] = dummyResponse()
				expectedCalls[fmt.Sprintf("/api/v1/namespaces/%s/services", ns)] = dummyResponse()
				expectedCalls[fmt.Sprintf("/api/v1/namespaces/%s/pods", ns)] = dummyResponse()

				expectedCalls[fmt.Sprintf("/apis/apps/v1/namespaces/%s/daemonsets", ns)] = dummyResponse()
				expectedCalls[fmt.Sprintf("/apis/apps/v1/namespaces/%s/deployments", ns)] = dummyResponse()
				expectedCalls[fmt.Sprintf("/apis/apps/v1/namespaces/%s/replicasets", ns)] = dummyResponse()
			}

			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
			tf.ClientConfigVal.Transport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
				responseBody, isExpectedCall := expectedCalls[req.URL.Path]
				if !isExpectedCall || req.Method != "GET" {
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}

				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: responseBody}, nil
			})

			ioStreams, _, _, _ := genericclioptions.NewTestIOStreams()
			cmd := NewCmdClusterInfoDump(tf, ioStreams)

			err := testCase.populateCmdFlags(cmd.Flags())
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			cmd.Run(cmd, []string{})
		})
	}
}

func TestCmdClusterInfoDumpAllNamespaces(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	namespacesAvailableInCluster := []string{"qa", "production", "test", "kube-system"}
	var items []corev1.Namespace
	for _, name := range namespacesAvailableInCluster {
		items = append(items, corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}})
	}
	expectedCalls := map[string]io.ReadCloser{
		"/api/v1/namespaces": cmdtesting.ObjBody(codec, &corev1.NamespaceList{Items: items}),
		"/api/v1/nodes":      dummyResponse(),
	}

	for _, ns := range namespacesAvailableInCluster {
		expectedCalls[fmt.Sprintf("/api/v1/namespaces/%s/events", ns)] = dummyResponse()
		expectedCalls[fmt.Sprintf("/api/v1/namespaces/%s/replicationcontrollers", ns)] = dummyResponse()
		expectedCalls[fmt.Sprintf("/api/v1/namespaces/%s/services", ns)] = dummyResponse()
		expectedCalls[fmt.Sprintf("/api/v1/namespaces/%s/pods", ns)] = dummyResponse()

		expectedCalls[fmt.Sprintf("/apis/apps/v1/namespaces/%s/daemonsets", ns)] = dummyResponse()
		expectedCalls[fmt.Sprintf("/apis/apps/v1/namespaces/%s/deployments", ns)] = dummyResponse()
		expectedCalls[fmt.Sprintf("/apis/apps/v1/namespaces/%s/replicasets", ns)] = dummyResponse()
	}

	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
	tf.ClientConfigVal.Transport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		responseBody, isExpectedCall := expectedCalls[req.URL.Path]
		if !isExpectedCall || req.Method != "GET" {
			t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
			return nil, nil
		}

		return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: responseBody}, nil
	})

	ioStreams, _, _, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdClusterInfoDump(tf, ioStreams)
	err := cmd.Flags().Set("all-namespaces", "true")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	cmd.Run(cmd, []string{})
}

func dummyResponse() io.ReadCloser {
	return cmdtesting.StringBody(`{
		"metadata": {
			"name": "dummyObject"
		}
	}`)
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
