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

type unsetConfigTest struct {
	description string
	config      clientcmdapi.Config
	args        []string
	expected    string
	expectedErr string
}

func TestUnsetConfigString(t *testing.T) {
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
	test := unsetConfigTest{
		description: "Testing for kubectl config unset a value",
		config:      conf,
		args:        []string{"current-context"},
		expected:    `Property "current-context" unset.` + "\n",
	}
	test.run(t)
}

func TestUnsetConfigMap(t *testing.T) {
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
	test := unsetConfigTest{
		description: "Testing for kubectl config unset a map",
		config:      conf,
		args:        []string{"clusters"},
		expected:    `Property "clusters" unset.` + "\n",
	}
	test.run(t)
}

func TestUnsetUnexistConfig(t *testing.T) {
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

	test := unsetConfigTest{
		description: "Testing for kubectl config unset a unexist map key",
		config:      conf,
		args:        []string{"contexts.foo.namespace"},
		expectedErr: "current map key `foo` is invalid",
	}
	test.run(t)

}

func (test unsetConfigTest) run(t *testing.T) {
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
	cmd := NewCmdConfigUnset(buf, pathOptions)
	opts := &unsetOptions{configAccess: pathOptions}
	err = opts.complete(cmd, test.args)
	if err == nil {
		err = opts.run(buf)
	}
	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}

	if err != nil && err.Error() != test.expectedErr {
		t.Fatalf("expected error:\n %v\nbut got error:\n%v", test.expectedErr, err)
	}
	if len(test.expected) != 0 {
		if buf.String() != test.expected {
			t.Errorf("Failed in :%q\n expected %v\n but got %v", test.description, test.expected, buf.String())
		}
	}
	if test.args[0] == "current-context" {
		if config.CurrentContext != "" {
			t.Errorf("Failed in :%q\n expected current-context nil,but got %v", test.description, config.CurrentContext)
		}
	} else if test.args[0] == "clusters" {
		if len(config.Clusters) != 0 {
			t.Errorf("Failed in :%q\n expected clusters nil map, but got %v", test.description, config.Clusters)
		}
	}
}
