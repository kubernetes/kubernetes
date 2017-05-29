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
			"minikube":  {AuthInfo: "minikube", Cluster: "minikube"},
			"my-cluser": {AuthInfo: "mu-cluster", Cluster: "my-cluster"},
		},
		CurrentContext: "minikube",
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"minikube":  {Token: "minikube-token"},
			"my-cluser": {Token: "minikube-token"},
		},
	}
	test := viewClusterTest{
		description: "Testing for kubectl config view",
		config:      conf,
		flags:       []string{},
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
  name: my-cluser
current-context: minikube
kind: Config
preferences: {}
users:
- name: minikube
  user:
    token: minikube-token
- name: my-cluser
  user:
    token: minikube-token` + "\n"}
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
			"minikube":  {AuthInfo: "minikube", Cluster: "minikube"},
			"my-cluser": {AuthInfo: "mu-cluster", Cluster: "my-cluster"},
		},
		CurrentContext: "minikube",
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"minikube":  {Token: "minikube-token"},
			"my-cluser": {Token: "minikube-token"},
		},
	}
	test := viewClusterTest{
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
    token: minikube-token` + "\n"}
	test.run(t)
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
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})
	cmd := NewCmdConfigView(buf, errBuf, pathOptions)
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
