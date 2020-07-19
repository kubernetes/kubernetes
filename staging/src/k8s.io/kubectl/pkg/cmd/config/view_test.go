/*
Copyright 2017 The Kubernetes Authors.

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

package config

import (
	"io/ioutil"
	"os"
	"testing"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
)

type viewClusterTest struct {
	description string
	config      clientcmdapi.Config //initiate kubectl config
	flags       []string            //kubectl config viw flags
	expected    string              //expect out
}

func TestViewCluster(t *testing.T) {
	conf := clientcmdapi.Config{
		Kind:       "Config",
		APIVersion: "v1",
		Clusters: map[string]*clientcmdapi.Cluster{
			"minikube":   {Server: "https://192.168.99.100:8443"},
			"my-cluster": {Server: "https://192.168.0.1:3434"},
		},
		Contexts: map[string]*clientcmdapi.Context{
			"minikube":   {AuthInfo: "minikube", Cluster: "minikube"},
			"my-cluster": {AuthInfo: "mu-cluster", Cluster: "my-cluster"},
		},
		CurrentContext: "minikube",
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"minikube":   {Token: "REDACTED"},
			"mu-cluster": {Token: "REDACTED"},
		},
	}

	test := viewClusterTest{
		description: "Testing for kubectl config view",
		config:      conf,
		expected: `apiVersion: v1
clusters:
- cluster:
    server: https://192.168.99.100:8443
  name: minikube
- cluster:
    server: https://192.168.0.1:3434
  name: my-cluster
contexts:
- context:
    cluster: minikube
    user: minikube
  name: minikube
- context:
    cluster: my-cluster
    user: mu-cluster
  name: my-cluster
current-context: minikube
kind: Config
preferences: {}
users:
- name: minikube
  user:
    token: REDACTED
- name: mu-cluster
  user:
    token: REDACTED` + "\n",
	}

	test.run(t)

}

func TestViewClusterMinify(t *testing.T) {
	conf := clientcmdapi.Config{
		Kind:       "Config",
		APIVersion: "v1",
		Clusters: map[string]*clientcmdapi.Cluster{
			"minikube":   {Server: "https://192.168.99.100:8443"},
			"my-cluster": {Server: "https://192.168.0.1:3434"},
		},
		Contexts: map[string]*clientcmdapi.Context{
			"minikube":   {AuthInfo: "minikube", Cluster: "minikube"},
			"my-cluster": {AuthInfo: "mu-cluster", Cluster: "my-cluster"},
		},
		CurrentContext: "minikube",
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"minikube":   {Token: "REDACTED"},
			"mu-cluster": {Token: "REDACTED"},
		},
	}

	testCases := []struct {
		description string
		config      clientcmdapi.Config
		flags       []string
		expected    string
	}{
		{
			description: "Testing for kubectl config view --minify=true",
			config:      conf,
			flags:       []string{"--minify=true"},
			expected: `apiVersion: v1
clusters:
- cluster:
    server: https://192.168.99.100:8443
  name: minikube
contexts:
- context:
    cluster: minikube
    user: minikube
  name: minikube
current-context: minikube
kind: Config
preferences: {}
users:
- name: minikube
  user:
    token: REDACTED` + "\n",
		},
		{
			description: "Testing for kubectl config view --minify=true --context=my-cluster",
			config:      conf,
			flags:       []string{"--minify=true", "--context=my-cluster"},
			expected: `apiVersion: v1
clusters:
- cluster:
    server: https://192.168.0.1:3434
  name: my-cluster
contexts:
- context:
    cluster: my-cluster
    user: mu-cluster
  name: my-cluster
current-context: my-cluster
kind: Config
preferences: {}
users:
- name: mu-cluster
  user:
    token: REDACTED` + "\n",
		},
	}

	for _, test := range testCases {
		cmdTest := viewClusterTest{
			description: test.description,
			config:      test.config,
			flags:       test.flags,
			expected:    test.expected,
		}
		cmdTest.run(t)
	}
}

func (test viewClusterTest) run(t *testing.T) {
	fakeKubeFile, err := ioutil.TempFile(os.TempDir(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(fakeKubeFile.Name())
	err = clientcmd.WriteToFile(test.config, fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdConfigView(cmdutil.NewFactory(genericclioptions.NewTestConfigFlags()), streams, pathOptions)
	// "context" is a global flag, inherited from base kubectl command in the real world
	cmd.Flags().String("context", "", "The name of the kubeconfig context to use")
	cmd.Flags().Parse(test.flags)

	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v,kubectl config view flags: %v", err, test.flags)
	}
	if len(test.expected) != 0 {
		if buf.String() != test.expected {
			t.Errorf("Failed in %q\n expected %v\n but got %v\n", test.description, test.expected, buf.String())
		}
	}
}
