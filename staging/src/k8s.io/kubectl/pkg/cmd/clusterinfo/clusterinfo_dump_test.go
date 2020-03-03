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
	"gopkg.in/yaml.v2"
	"io/ioutil"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"os"
	"path"
	"strings"
	"testing"
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
		"should not output message indicating where output was written if output directory is -": {
			outputFlag:                "json",
			outputDirectoryFlag:       "-",
			expectedStdoutNotContains: "Cluster info dumped to ",
		},
		"should not output message indicating where output was written if an output directory not specified": {
			outputFlag:                "json",
			expectedStdoutNotContains: "Cluster info dumped to ",
		},
		"should output message indicating where output was written if an output directory is specified": {
			outputFlag:          "json",
			outputDirectoryFlag: tempOutputDirectory,
			expectedStdout:      "Cluster info dumped to test-directory",
		},
	}

	for k, testCase := range testCases {
		t.Run(k, func(t *testing.T) {

			fakeClientset := &clientsetfake.Clientset{}
			fakeClientset.AddReactor("list", "*", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
				switch action.GetResource().String() {
				case "/v1, Resource=nodes":
					return true, testNodeList, nil
				case "/v1, Resource=events":
					return true, testEventList, nil
				case "/v1, Resource=replicationcontrollers":
					return true, testReplicationControllerList, nil
				case "/v1, Resource=services":
					return true, testServiceList, nil
				case "/v1, Resource=pods":
					return true, testPodList, nil
				case "apps/v1, Resource=daemonsets":
					return true, testDaemonSetList, nil
				case "apps/v1, Resource=deployments":
					return true, testDeploymentList, nil
				case "apps/v1, Resource=replicasets":
					return true, testReplicatSetList, nil
				}
				t.Fatalf("unexpected action: %v", action.GetResource().String())
				return false, nil, nil
			})

			tf := cmdtesting.NewTestFactory()
			defer tf.Cleanup()

			streams, _, buf, _ := genericclioptions.NewTestIOStreams()
			cmd := NewCmdClusterInfoDump(tf, streams)

			options := &ClusterInfoDumpOptions{
				PrintFlags: &genericclioptions.PrintFlags{
					JSONYamlPrintFlags: &genericclioptions.JSONYamlPrintFlags{},
					OutputFormat:       &testCase.outputFlag,
				},
				OutputDir: testCase.outputDirectoryFlag,
				IOStreams: streams,
			}

			if err := options.Complete(tf, cmd); err != nil {
				t.Fatalf("unexcpted Complete error: %v", err)
			}

			options.CoreClient = fakeClientset.CoreV1()
			options.AppsClient = fakeClientset.AppsV1()

			if err := options.Run(); err != nil {
				t.Fatalf("unexcpted Run error: %v", err)
			}

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

			expectAction(t, *fakeClientset, "list", "", "v1", "nodes")

			expectAction(t, *fakeClientset, "list", "kube-system", "v1", "events")
			expectAction(t, *fakeClientset, "list", "kube-system", "v1", "replicationcontrollers")
			expectAction(t, *fakeClientset, "list", "kube-system", "v1", "services")
			expectAction(t, *fakeClientset, "list", "kube-system", "v1", "pods")
			expectAction(t, *fakeClientset, "list", "kube-system", "v1", "daemonsets")
			expectAction(t, *fakeClientset, "list", "kube-system", "v1", "deployments")
			expectAction(t, *fakeClientset, "list", "kube-system", "v1", "replicasets")

			expectAction(t, *fakeClientset, "list", "default", "v1", "events")
			expectAction(t, *fakeClientset, "list", "default", "v1", "replicationcontrollers")
			expectAction(t, *fakeClientset, "list", "default", "v1", "services")
			expectAction(t, *fakeClientset, "list", "default", "v1", "pods")
			expectAction(t, *fakeClientset, "list", "default", "v1", "daemonsets")
			expectAction(t, *fakeClientset, "list", "default", "v1", "deployments")
			expectAction(t, *fakeClientset, "list", "default", "v1", "replicasets")
		})
	}
}

func expectAction(t *testing.T, clientset clientsetfake.Clientset, verb, namespace, version, resource string) {
	for _, action := range clientset.Actions() {
		if action.GetVerb() == verb &&
			action.GetNamespace() == namespace &&
			action.GetResource().Version == version &&
			action.GetResource().Resource == resource {
			return
		}
	}
	t.Fatalf("expected %s action was not performed for %s/%s in namespace %s", verb, version, resource, namespace)
}

var testNodeList = &corev1.NodeList{
	TypeMeta: metav1.TypeMeta{Kind: "List", APIVersion: "v1"},
	ListMeta: metav1.ListMeta{},
	Items: []corev1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: corev1.NodeSpec{
				PodCIDR:  "10.244.0.0/24",
				PodCIDRs: []string{"10.244.0.0/24"},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "bar"},
			Spec: corev1.NodeSpec{
				PodCIDR:  "10.243.0.0/24",
				PodCIDRs: []string{"10.244.0.0/24"},
			},
		},
	},
}

var testEventList = &corev1.EventList{
	TypeMeta: metav1.TypeMeta{Kind: "List", APIVersion: "v1"},
	Items: []corev1.Event{
		{
			ObjectMeta: metav1.ObjectMeta{},
			Reason:     "foo",
			Message:    "bar",
		},
	},
}

var testReplicationControllerList = &corev1.ReplicationControllerList{
	TypeMeta: metav1.TypeMeta{Kind: "List", APIVersion: "v1"},
	Items: []corev1.ReplicationController{
		{
			ObjectMeta: metav1.ObjectMeta{},
			Spec:       corev1.ReplicationControllerSpec{},
			Status:     corev1.ReplicationControllerStatus{},
		},
	},
}

var testServiceList = &corev1.ServiceList{
	TypeMeta: metav1.TypeMeta{Kind: "List", APIVersion: "v1"},
	Items: []corev1.Service{
		{
			ObjectMeta: metav1.ObjectMeta{},
			Spec:       corev1.ServiceSpec{},
			Status:     corev1.ServiceStatus{},
		},
	},
}

var testPodList = &corev1.PodList{
	TypeMeta: metav1.TypeMeta{Kind: "List", APIVersion: "v1"},
	Items: []corev1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{},
			Spec:       corev1.PodSpec{},
			Status:     corev1.PodStatus{},
		},
	},
}

var testDaemonSetList = &appsv1.DaemonSetList{
	TypeMeta: metav1.TypeMeta{Kind: "List", APIVersion: "apps/v1"},
	Items: []appsv1.DaemonSet{
		{
			ObjectMeta: metav1.ObjectMeta{},
			Spec:       appsv1.DaemonSetSpec{},
			Status:     appsv1.DaemonSetStatus{},
		},
	},
}

var testDeploymentList = &appsv1.DeploymentList{
	TypeMeta: metav1.TypeMeta{Kind: "List", APIVersion: "apps/v1"},
	Items: []appsv1.Deployment{
		{
			ObjectMeta: metav1.ObjectMeta{},
			Spec:       appsv1.DeploymentSpec{},
			Status:     appsv1.DeploymentStatus{},
		},
	},
}

var testReplicatSetList = &appsv1.ReplicaSetList{
	TypeMeta: metav1.TypeMeta{Kind: "List", APIVersion: "apps/v1"},
	Items: []appsv1.ReplicaSet{
		{
			ObjectMeta: metav1.ObjectMeta{},
			Spec:       appsv1.ReplicaSetSpec{},
			Status:     appsv1.ReplicaSetStatus{},
		},
	},
}
