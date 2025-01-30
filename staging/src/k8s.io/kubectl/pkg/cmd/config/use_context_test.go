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
	"bytes"
	"os"
	"testing"

	utiltesting "k8s.io/client-go/util/testing"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type useContextTest struct {
	description    string
	config         clientcmdapi.Config //initiate kubectl config
	args           []string            //kubectl config use-context args
	expected       string              //expect out
	expectedConfig clientcmdapi.Config //expect kubectl config
}

func TestUseContext(t *testing.T) {
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
	}
	test := useContextTest{
		description: "Testing for kubectl config use-context",
		config:      conf,
		args:        []string{"my-cluster"},
		expected:    `Switched to context "my-cluster".` + "\n",
		expectedConfig: clientcmdapi.Config{
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
			CurrentContext: "my-cluster",
		},
	}
	test.run(t)
}

func (test useContextTest) run(t *testing.T) {
	fakeKubeFile, err := os.CreateTemp(os.TempDir(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer utiltesting.CloseAndRemove(t, fakeKubeFile)
	err = clientcmd.WriteToFile(test.config, fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""
	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdConfigUseContext(buf, pathOptions)
	cmd.SetArgs(test.args)
	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v,kubectl config use-context args: %v", err, test.args)
	}
	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}
	if len(test.expected) != 0 {
		if buf.String() != test.expected {
			t.Errorf("Failed in :%q\n expected %v\n, but got %v\n", test.description, test.expected, buf.String())
		}
	}
	if test.expectedConfig.CurrentContext != config.CurrentContext {
		t.Errorf("Failed in :%q\n expected config %v, but found %v\n in kubeconfig\n", test.description, test.expectedConfig, config)
	}
}
