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
	"reflect"
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type unsetConfigTest struct {
	config         clientcmdapi.Config
	args           []string
	expected       string
	expectedConfig clientcmdapi.Config
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
		config:   conf,
		args:     []string{"current-context"},
		expected: `Property "current-context" unset.` + "\n",
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
		},
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
		config:   conf,
		args:     []string{"clusters"},
		expected: `Property "clusters" unset.` + "\n",
		expectedConfig: clientcmdapi.Config{
			Kind:       "Config",
			APIVersion: "v1",
			Contexts: map[string]*clientcmdapi.Context{
				"minikube":   {AuthInfo: "minikube", Cluster: "minikube"},
				"my-cluster": {AuthInfo: "mu-cluster", Cluster: "my-cluster"},
			},
			CurrentContext: "minikube",
		},
	}
	test.run(t)
}

func (test unsetConfigTest) run(t *testing.T) {
	fakeKubeFile, _ := ioutil.TempFile("", "")
	defer os.Remove(fakeKubeFile.Name())
	err := clientcmd.WriteToFile(test.config, fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""
	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdConfigUnset(buf, pathOptions)
	cmd.SetArgs(test.args)
	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v", err)
	}
	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}
	if len(test.expected) != 0 {
		if buf.String() != test.expected {
			t.Errorf("expected %v, but got %v", test.expected, buf.String())
		}
		return
	}
	if !reflect.DeepEqual(test.expectedConfig, &config) {
		t.Errorf("expected clusters %v, but found %v in kubeconfig", test.expectedConfig, config)
	}
}
