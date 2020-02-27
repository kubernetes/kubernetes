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
	"encoding/json"
	"fmt"
	"gopkg.in/yaml.v2"
	"io/ioutil"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubectl/pkg/scheme"
	"net/http"
	"os"
	"path"
	"strings"
	"testing"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestSetupOutputWriterNoOp(t *testing.T) {
	tests := []string{"", "-"}
	for _, test := range tests {
		_, _, buf, _ := genericclioptions.NewTestIOStreams()
		f := cmdtesting.NewTestFactory()
		defer f.Cleanup()

		writer := setupOutputWriter(test, buf, "/some/file/that/should/be/ignored", "")
		if writer != buf {
			t.Errorf("expected: %v, saw: %v", buf, writer)
		}
	}
}

func TestSetupOutputWriterFile(t *testing.T) {
	file := "output"
	extension := ".json"
	dir, err := ioutil.TempDir(os.TempDir(), "out")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fullPath := path.Join(dir, file) + extension
	defer os.RemoveAll(dir)

	_, _, buf, _ := genericclioptions.NewTestIOStreams()
	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()

	writer := setupOutputWriter(dir, buf, file, extension)
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

func TestClusterInfoDump(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)

	tempOutputDirectory, err := ioutil.TempDir(os.TempDir(), "TestClusterInfoDump")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	defer os.RemoveAll(tempOutputDirectory)

	testCases := map[string]struct {
		outputFlag                string
		outputDirectoryFlag       string
		expectedStdout            string
		expectedStdoutNotContains string
		expectStdoutIsValidJson   bool
		expectStdoutIsValidYaml   bool
	}{
		"should output valid json when json output format is specified": {
			outputFlag:              "json",
			expectStdoutIsValidJson: true,
		},
		"should output valid yaml when yaml output format is specified": {
			outputFlag:              "yaml",
			expectStdoutIsValidYaml: true,
		},
		"should output valid json when no output format is specified": {
			expectStdoutIsValidJson: true,
		},
		"should not output message indicating where output was written if output directory is -": {
			outputDirectoryFlag:       "-",
			expectedStdoutNotContains: "Cluster info dumped to ",
		},
		"should not output message indicating where output was written if an output directory not specified": {
			expectedStdoutNotContains: "Cluster info dumped to ",
		},
		"should output message indicating where output was written if an output directory is specified": {
			outputDirectoryFlag: tempOutputDirectory,
			expectedStdout:      "Cluster info dumped to test-directory",
		},
	}

	// Define the expected requests and the response for each when it is called
	expectedRequests := map[string]runtime.Object{
		"GET /api/v1/nodes":                                         testNodeList(),
		"GET /api/v1/namespaces/kube-system/events":                 testEventList(),
		"GET /api/v1/namespaces/kube-system/replicationcontrollers": testReplicationControllerList(),
		"GET /api/v1/namespaces/kube-system/services":               testServiceList(),
		"GET /api/v1/namespaces/kube-system/pods":                   testPodList(),
		"GET /apis/apps/v1/namespaces/kube-system/daemonsets":       testDaemonSetList(),
		"GET /apis/apps/v1/namespaces/kube-system/deployments":      testDeploymentList(),
		"GET /apis/apps/v1/namespaces/kube-system/replicasets":      testReplicaSetList(),
		"GET /api/v1/namespaces/default/events":                     testEventList(),
		"GET /api/v1/namespaces/default/replicationcontrollers":     testReplicationControllerList(),
		"GET /api/v1/namespaces/default/services":                   testServiceList(),
		"GET /api/v1/namespaces/default/pods":                       testPodList(),
		"GET /apis/apps/v1/namespaces/default/daemonsets":           testDaemonSetList(),
		"GET /apis/apps/v1/namespaces/default/deployments":          testDeploymentList(),
		"GET /apis/apps/v1/namespaces/default/replicasets":          testReplicaSetList(),
	}

	for k, testCase := range testCases {
		t.Run(k, func(t *testing.T) {
			var actualRequests []string

			tf := cmdtesting.NewTestFactory()
			defer tf.Cleanup()

			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					requestKey := fmt.Sprintf("%s %s", req.Method, req.URL.Path)
					actualRequests = append(actualRequests, requestKey)
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, expectedRequests[requestKey])}, nil
				}),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			streams, _, buf, _ := genericclioptions.NewTestIOStreams()
			cmd := NewCmdClusterInfoDump(tf, streams)

			if len(testCase.outputDirectoryFlag) > 0 {
				_ = cmd.Flags().Set("output-directory", testCase.outputDirectoryFlag)
			}
			if len(testCase.outputFlag) > 0 {
				_ = cmd.Flags().Set("output", testCase.outputFlag)
			}
			cmd.Run(cmd, []string{})

			if len(testCase.expectedStdout) > 0 && buf.String() == testCase.expectedStdout {
				t.Fatalf("expected output of %v but got %v", testCase.expectedStdout, buf.String())
			}
			if len(testCase.expectedStdoutNotContains) > 0 && strings.Contains(buf.String(), testCase.expectedStdoutNotContains) {
				t.Fatalf("expected output to not to contain %v", testCase.expectedStdoutNotContains)
			}
			if testCase.expectStdoutIsValidJson && !json.Valid(buf.Bytes()) {
				t.Fatalf("output is not valid json")
			}
			if testCase.expectStdoutIsValidYaml {
				temp := make(map[interface{}]interface{})
				err = yaml.Unmarshal(buf.Bytes(), &temp)
				if err != nil {
					t.Fatalf("output is not valid yaml")
				}
			}

			// Make sure all expected requests were performed
			for expectedRequest := range expectedRequests {
				found := false
				for _, actualRequest := range actualRequests {
					if actualRequest == expectedRequest {
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("expected request was not made: %v", expectedRequest)
				}
			}

			// Make sure no unexpected requests were performed
			for _, actualRequest := range actualRequests {
				found := false
				for expectedRequest := range expectedRequests {
					if actualRequest == expectedRequest {
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("unexpected request was made: %v", actualRequest)
				}
			}
		})
	}
}

func testNodeList() *corev1.NodeList {
	return &corev1.NodeList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "List",
			APIVersion: "v1",
		},
		Items: []corev1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: corev1.NodeSpec{
					PodCIDR:  "10.244.0.0/24",
					PodCIDRs: []string{"10.244.0.0/24"},
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar",
				},
				Spec: corev1.NodeSpec{
					PodCIDR:  "10.243.0.0/24",
					PodCIDRs: []string{"10.244.0.0/24"},
				},
			},
		},
	}
}

func testEventList() *corev1.EventList {
	return &corev1.EventList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "List",
			APIVersion: "v1",
		},
		Items: []corev1.Event{
			{
				ObjectMeta:          metav1.ObjectMeta{},
				InvolvedObject:      corev1.ObjectReference{},
				Reason:              "foo",
				Message:             "bar",
				Source:              corev1.EventSource{},
				FirstTimestamp:      metav1.Time{},
				LastTimestamp:       metav1.Time{},
				Count:               0,
				Type:                "",
				EventTime:           metav1.MicroTime{},
				Series:              nil,
				Action:              "",
				Related:             nil,
				ReportingController: "",
				ReportingInstance:   "",
			},
		},
	}
}

func testReplicationControllerList() *corev1.ReplicationControllerList {
	return &corev1.ReplicationControllerList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "List",
			APIVersion: "v1",
		},
		Items: []corev1.ReplicationController{
			{
				ObjectMeta: metav1.ObjectMeta{},
				Spec:       corev1.ReplicationControllerSpec{},
				Status:     corev1.ReplicationControllerStatus{},
			},
		},
	}
}

func testServiceList() *corev1.ServiceList {
	return &corev1.ServiceList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "List",
			APIVersion: "v1",
		},
		Items: []corev1.Service{
			{
				ObjectMeta: metav1.ObjectMeta{},
				Spec:       corev1.ServiceSpec{},
				Status:     corev1.ServiceStatus{},
			},
		},
	}
}

func testPodList() *corev1.PodList {
	return &corev1.PodList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "List",
			APIVersion: "v1",
		},
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{},
				Spec:       corev1.PodSpec{},
				Status:     corev1.PodStatus{},
			},
		},
	}
}

func testDaemonSetList() *appsv1.DaemonSetList {
	return &appsv1.DaemonSetList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "List",
			APIVersion: "apps/v1",
		},
		Items: []appsv1.DaemonSet{
			{
				ObjectMeta: metav1.ObjectMeta{},
				Spec:       appsv1.DaemonSetSpec{},
				Status:     appsv1.DaemonSetStatus{},
			},
		},
	}
}

func testDeploymentList() *appsv1.DeploymentList {
	return &appsv1.DeploymentList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "List",
			APIVersion: "apps/v1",
		},
		Items: []appsv1.Deployment{
			{
				ObjectMeta: metav1.ObjectMeta{},
				Spec:       appsv1.DeploymentSpec{},
				Status:     appsv1.DeploymentStatus{},
			},
		},
	}
}

func testReplicaSetList() *appsv1.ReplicaSetList {
	return &appsv1.ReplicaSetList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "List",
			APIVersion: "apps/v1",
		},
		Items: []appsv1.ReplicaSet{
			{
				ObjectMeta: metav1.ObjectMeta{},
				Spec:       appsv1.ReplicaSetSpec{},
				Status:     appsv1.ReplicaSetStatus{},
			},
		},
	}
}
