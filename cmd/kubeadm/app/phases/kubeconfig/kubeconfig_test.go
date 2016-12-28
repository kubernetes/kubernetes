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
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
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
    server: "localhost:8080"
  name: kubernetes
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
		{configClient{c: "kubernetes"}, configClientWithCerts{}, ""},
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
	defer func() { kubeadmapi.GlobalEnvParams = oldEnv }()

	var writeConfig = []struct {
		name        string
		cc          configClient
		ccWithToken configClientWithToken
		expected    error
		file        []byte
	}{
		{"test1", configClientWithToken{clusterName: "k8s", userName: "user1", token: "abc"}, nil, []byte(configOut1)},
		{"test2", configClientWithToken{clusterName: "kubernetes", userName: "user2", serverURL: "localhost:8080", token: "cba"}, nil, []byte(configOut2)},
	}
	for _, rt := range writeConfig {
		c := MakeClientConfigWithToken(
			rt.cc.serverURL,
			rt.cc.clusterName,
			rt.cc.userName,
			rt.cc.caCert,
			rt.ccWithToken.token,
		)
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

// TODO: Update this test
func TestCreateCertsAndConfigForClients(t *testing.T) {
	var tests = []struct {
		a         kubeadmapi.API
		cn        []string
		caKeySize int
		expected  bool
	}{
		{
			a:         kubeadmapi.API{AdvertiseAddresses: []string{"foo"}},
			cn:        []string{"localhost"},
			caKeySize: 128,
			expected:  false,
		},
		{
			a:         kubeadmapi.API{AdvertiseAddresses: []string{"foo"}},
			cn:        []string{},
			caKeySize: 128,
			expected:  true,
		},
		{
			a:         kubeadmapi.API{AdvertiseAddresses: []string{"foo"}},
			cn:        []string{"localhost"},
			caKeySize: 2048,
			expected:  true,
		},
	}

	for _, rt := range tests {
		caKey, err := rsa.GenerateKey(rand.Reader, rt.caKeySize)
		if err != nil {
			t.Fatalf("Couldn't create rsa Private Key")
		}
		caCert := &x509.Certificate{}
		_, actual := CreateCertsAndConfigForClients(rt.a, rt.cn, caKey, caCert)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed CreateCertsAndConfigForClients:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
	}
}
