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

package util

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
)

const (
	configOut1 = `apiVersion: v1
clusters:
- cluster:
    server: ""
  name: ""
contexts: []
current-context: ""
kind: Config
preferences: {}
users: []
`
	configOut2 = `apiVersion: v1
clusters:
- cluster:
    server: ""
  name: kubernetes
contexts: []
current-context: ""
kind: Config
preferences: {}
users: []
`
)

type configClient struct {
	c  string
	s  string
	ca []byte
}

type configClientWithCerts struct {
	c           *clientcmdapi.Config
	clusterName string
	userName    string
	clientKey   []byte
	clientCert  []byte
}

type configClientWithToken struct {
	c           *clientcmdapi.Config
	clusterName string
	userName    string
	token       string
}

func TestCreateBasicClientConfig(t *testing.T) {
	var createBasicTest = []struct {
		cc       configClient
		expected string
	}{
		{configClient{}, ""},
		{configClient{c: "kubernetes"}, ""},
	}
	for _, rt := range createBasicTest {
		c := CreateBasicClientConfig(rt.cc.c, rt.cc.s, rt.cc.ca)
		if c.Kind != rt.expected {
			t.Errorf(
				"failed CreateBasicClientConfig:\n\texpected: %s\n\t  actual: %s",
				c.Kind,
				rt.expected,
			)
		}
	}
}

func TestMakeClientConfigWithCerts(t *testing.T) {
	var createBasicTest = []struct {
		cc          configClient
		ccWithCerts configClientWithCerts
		expected    string
	}{
		{configClient{}, configClientWithCerts{}, ""},
		{configClient{c: "kubernetes"}, configClientWithCerts{}, ""},
	}
	for _, rt := range createBasicTest {
		c := CreateBasicClientConfig(rt.cc.c, rt.cc.s, rt.cc.ca)
		rt.ccWithCerts.c = c
		cwc := MakeClientConfigWithCerts(
			rt.ccWithCerts.c,
			rt.ccWithCerts.clusterName,
			rt.ccWithCerts.userName,
			rt.ccWithCerts.clientKey,
			rt.ccWithCerts.clientCert,
		)
		if cwc.Kind != rt.expected {
			t.Errorf(
				"failed MakeClientConfigWithCerts:\n\texpected: %s\n\t  actual: %s",
				c.Kind,
				rt.expected,
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
		{configClient{c: "kubernetes"}, configClientWithToken{}, ""},
	}
	for _, rt := range createBasicTest {
		c := CreateBasicClientConfig(rt.cc.c, rt.cc.s, rt.cc.ca)
		rt.ccWithToken.c = c
		cwc := MakeClientConfigWithToken(
			rt.ccWithToken.c,
			rt.ccWithToken.clusterName,
			rt.ccWithToken.userName,
			rt.ccWithToken.token,
		)
		if cwc.Kind != rt.expected {
			t.Errorf(
				"failed MakeClientConfigWithCerts:\n\texpected: %s\n\t  actual: %s",
				c.Kind,
				rt.expected,
			)
		}
	}
}

func TestWriteKubeconfigIfNotExists(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.Remove(tmpdir)

	// set up tmp GlobalEnvParams values for testing
	oldEnv := kubeadmapi.GlobalEnvParams
	kubeadmapi.GlobalEnvParams = kubeadmapi.SetEnvParams()
	kubeadmapi.GlobalEnvParams.KubernetesDir = fmt.Sprintf("%s/etc/kubernetes", tmpdir)
	kubeadmapi.GlobalEnvParams.HostPKIPath = fmt.Sprintf("%s/etc/kubernetes/pki", tmpdir)
	kubeadmapi.GlobalEnvParams.HostEtcdPath = fmt.Sprintf("%s/var/lib/etcd", tmpdir)
	kubeadmapi.GlobalEnvParams.DiscoveryImage = fmt.Sprintf("%s/var/lib/etcd", tmpdir)
	defer func() { kubeadmapi.GlobalEnvParams = oldEnv }()

	var writeConfig = []struct {
		name     string
		cc       configClient
		expected error
		file     []byte
	}{
		{"test1", configClient{}, nil, []byte(configOut1)},
		{"test2", configClient{c: "kubernetes"}, nil, []byte(configOut2)},
	}
	for _, rt := range writeConfig {
		c := CreateBasicClientConfig(rt.cc.c, rt.cc.s, rt.cc.ca)
		err := WriteKubeconfigIfNotExists(rt.name, c)
		if err != rt.expected {
			t.Errorf(
				"failed WriteKubeconfigIfNotExists with an error:\n\texpected: %s\n\t  actual: %s",
				err,
				rt.expected,
			)
		}
		configPath := filepath.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, fmt.Sprintf("%s.conf", rt.name))
		newFile, err := ioutil.ReadFile(configPath)
		if !bytes.Equal(newFile, rt.file) {
			t.Errorf(
				"failed WriteKubeconfigIfNotExists config write:\n\texpected: %s\n\t  actual: %s",
				newFile,
				rt.file,
			)
		}
	}
}
