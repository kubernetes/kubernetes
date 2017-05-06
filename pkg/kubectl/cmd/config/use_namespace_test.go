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
	"io/ioutil"
	"os"
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type useNamespaceTest struct {
	description    string
	config         clientcmdapi.Config //initiate kubectl config
	args           []string            //kubectl config use-context args
	expected       string              //expect out
	expectedConfig clientcmdapi.Config //expect kubectl config
}

func TestUseNamespace(t *testing.T) {
	conf := clientcmdapi.Config{
		Kind:       "Config",
		APIVersion: "v1",
		Clusters: map[string]*clientcmdapi.Cluster{
			"my-cluster": {Server: "https://192.168.0.1:3434"},
		},
		Contexts: map[string]*clientcmdapi.Context{
			"my-cluster": {AuthInfo: "mu-cluster", Cluster: "my-cluster", Namespace: "my-namespace"},
		},
		CurrentContext: "my-cluster",
	}
	test := useNamespaceTest{
		description: "Testing for kubectl config use-ns",
		config:      conf,
		args:        []string{"new-namespace"},
		expected:    `Switched to namespace "new-namespace".` + "\n",
		expectedConfig: clientcmdapi.Config{
			Kind:       "Config",
			APIVersion: "v1",
			Clusters: map[string]*clientcmdapi.Cluster{
				"my-cluster": {Server: "https://192.168.0.1:3434"},
			},
			Contexts: map[string]*clientcmdapi.Context{
				"my-cluster": {AuthInfo: "mu-cluster", Cluster: "my-cluster", Namespace: "new-namespace"},
			},
			CurrentContext: "my-cluster",
		},
	}
	test.run(t)
}

func (test useNamespaceTest) run(t *testing.T) {
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
	if test.expectedConfig.Contexts["my-cluster"].Namespace != config.Contexts["my-cluster"].Namespace {
		t.Errorf("Failed in :%q\n expected config %v, but found %v\n in kubeconfig\n", test.description, test.expectedConfig, config)
	}
}
