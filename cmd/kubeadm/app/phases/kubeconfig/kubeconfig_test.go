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

package kubeconfig

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

const (
	configOut1 = `apiVersion: v1
clusters:
- cluster:
    server: ""
  name: k8s
contexts:
- context:
    cluster: k8s
    user: user1
  name: user1@k8s
current-context: user1@k8s
kind: Config
preferences: {}
users:
- name: user1
  user:
    token: abc
`
	configOut2 = `apiVersion: v1
clusters:
- cluster:
    server: localhost:8080
  name: kubernetes
contexts:
- context:
    cluster: kubernetes
    user: user2
  name: user2@kubernetes
current-context: user2@kubernetes
kind: Config
preferences: {}
users:
- name: user2
  user:
    token: cba
`
)

type configClient struct {
	clusterName string
	userName    string
	serverURL   string
	caCert      []byte
}

type configClientWithCerts struct {
	clientKey  []byte
	clientCert []byte
}

type configClientWithToken struct {
	token string
}

func TestMakeClientConfigWithCerts(t *testing.T) {
	var createBasicTest = []struct {
		cc          configClient
		ccWithCerts configClientWithCerts
		expected    string
	}{
		{configClient{}, configClientWithCerts{}, ""},
		{configClient{clusterName: "kubernetes"}, configClientWithCerts{}, ""},
	}
	for _, rt := range createBasicTest {
		cwc := MakeClientConfigWithCerts(
			rt.cc.serverURL,
			rt.cc.clusterName,
			rt.cc.userName,
			rt.cc.caCert,
			rt.ccWithCerts.clientKey,
			rt.ccWithCerts.clientCert,
		)
		if cwc.Kind != rt.expected {
			t.Errorf(
				"failed MakeClientConfigWithCerts:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				cwc.Kind,
			)
		}
	}
}

func TestMakeClientConfigWithToken(t *testing.T) {
	var createBasicTest = []struct {
		cc          configClient
		ccWithToken configClientWithToken
		expected    string
	}{
		{configClient{}, configClientWithToken{}, ""},
		{configClient{clusterName: "kubernetes"}, configClientWithToken{}, ""},
	}
	for _, rt := range createBasicTest {
		cwc := MakeClientConfigWithToken(
			rt.cc.serverURL,
			rt.cc.clusterName,
			rt.cc.userName,
			rt.cc.caCert,
			rt.ccWithToken.token,
		)
		if cwc.Kind != rt.expected {
			t.Errorf(
				"failed MakeClientConfigWithCerts:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				cwc.Kind,
			)
		}
	}
}

func TestWriteKubeconfigToDisk(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.Remove(tmpdir)

	// set up tmp GlobalEnvParams values for testing
	oldEnv := kubeadmapi.GlobalEnvParams
	kubeadmapi.GlobalEnvParams = kubeadmapi.SetEnvParams()
	kubeadmapi.GlobalEnvParams.KubernetesDir = fmt.Sprintf("%s/etc/kubernetes", tmpdir)
	defer func() { kubeadmapi.GlobalEnvParams = oldEnv }()

	var writeConfig = []struct {
		name        string
		cc          configClient
		ccWithToken configClientWithToken
		expected    error
		file        []byte
	}{
		{"test1", configClient{clusterName: "k8s", userName: "user1"}, configClientWithToken{token: "abc"}, nil, []byte(configOut1)},
		{"test2", configClient{clusterName: "kubernetes", userName: "user2", serverURL: "localhost:8080"}, configClientWithToken{token: "cba"}, nil, []byte(configOut2)},
	}
	for _, rt := range writeConfig {
		c := MakeClientConfigWithToken(
			rt.cc.serverURL,
			rt.cc.clusterName,
			rt.cc.userName,
			rt.cc.caCert,
			rt.ccWithToken.token,
		)
		configPath := filepath.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, fmt.Sprintf("%s.conf", rt.name))
		err := WriteKubeconfigToDisk(configPath, c)
		if err != rt.expected {
			t.Errorf(
				"failed WriteKubeconfigToDisk with an error:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				err,
			)
		}
		newFile, err := ioutil.ReadFile(configPath)
		if !bytes.Equal(newFile, rt.file) {
			t.Errorf(
				"failed WriteKubeconfigToDisk config write:\n\texpected: %s\n\t  actual: %s",
				rt.file,
				newFile,
			)
		}
	}
}
