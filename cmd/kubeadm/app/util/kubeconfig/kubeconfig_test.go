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

package kubeconfig

import (
	"bytes"
	"fmt"
	"os"
	"reflect"
	"testing"

	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
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

func TestCreateWithCerts(t *testing.T) {
	var createBasicTest = []struct {
		name        string
		cc          configClient
		ccWithCerts configClientWithCerts
		expected    string
	}{
		{"empty config", configClient{}, configClientWithCerts{}, ""},
		{"clusterName kubernetes", configClient{clusterName: "kubernetes"}, configClientWithCerts{}, ""},
	}
	for _, rt := range createBasicTest {
		t.Run(rt.name, func(t *testing.T) {
			cwc := CreateWithCerts(
				rt.cc.serverURL,
				rt.cc.clusterName,
				rt.cc.userName,
				rt.cc.caCert,
				rt.ccWithCerts.clientKey,
				rt.ccWithCerts.clientCert,
			)
			if cwc.Kind != rt.expected {
				t.Errorf(
					"failed CreateWithCerts:\n\texpected: %s\n\t  actual: %s",
					rt.expected,
					cwc.Kind,
				)
			}
		})
	}
}

func TestCreateWithToken(t *testing.T) {
	var createBasicTest = []struct {
		name        string
		cc          configClient
		ccWithToken configClientWithToken
		expected    string
	}{
		{"empty config", configClient{}, configClientWithToken{}, ""},
		{"clusterName kubernetes", configClient{clusterName: "kubernetes"}, configClientWithToken{}, ""},
	}
	for _, rt := range createBasicTest {
		t.Run(rt.name, func(t *testing.T) {
			cwc := CreateWithToken(
				rt.cc.serverURL,
				rt.cc.clusterName,
				rt.cc.userName,
				rt.cc.caCert,
				rt.ccWithToken.token,
			)
			if cwc.Kind != rt.expected {
				t.Errorf(
					"failed CreateWithToken:\n\texpected: %s\n\t  actual: %s",
					rt.expected,
					cwc.Kind,
				)
			}
		})
	}
}

func TestWriteKubeconfigToDisk(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

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
		t.Run(rt.name, func(t *testing.T) {
			c := CreateWithToken(
				rt.cc.serverURL,
				rt.cc.clusterName,
				rt.cc.userName,
				rt.cc.caCert,
				rt.ccWithToken.token,
			)
			configPath := fmt.Sprintf("%s/etc/kubernetes/%s.conf", tmpdir, rt.name)
			err := WriteToDisk(configPath, c)
			if err != rt.expected {
				t.Errorf(
					"failed WriteToDisk with an error:\n\texpected: %s\n\t  actual: %s",
					rt.expected,
					err,
				)
			}
			newFile, _ := os.ReadFile(configPath)
			if !bytes.Equal(newFile, rt.file) {
				t.Errorf(
					"failed WriteToDisk config write:\n\texpected: %s\n\t  actual: %s",
					rt.file,
					newFile,
				)
			}
		})
	}
}

func TestGetCurrentAuthInfo(t *testing.T) {
	var testCases = []struct {
		name     string
		config   *clientcmdapi.Config
		expected bool
	}{
		{
			name:     "nil context",
			config:   nil,
			expected: false,
		},
		{
			name:     "no CurrentContext value",
			config:   &clientcmdapi.Config{},
			expected: false,
		},
		{
			name:     "no CurrentContext object",
			config:   &clientcmdapi.Config{CurrentContext: "kubernetes"},
			expected: false,
		},
		{
			name: "CurrentContext object with bad contents",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"NOTkubernetes": {}},
			},
			expected: false,
		},
		{
			name: "no AuthInfo value",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"kubernetes": {}},
			},
			expected: false,
		},
		{
			name: "no AuthInfo object",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"kubernetes": {AuthInfo: "kubernetes"}},
			},
			expected: false,
		},
		{
			name: "AuthInfo object with bad contents",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"kubernetes": {AuthInfo: "kubernetes"}},
				AuthInfos:      map[string]*clientcmdapi.AuthInfo{"NOTkubernetes": {}},
			},
			expected: false,
		},
		{
			name: "valid AuthInfo",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"kubernetes": {AuthInfo: "kubernetes"}},
				AuthInfos:      map[string]*clientcmdapi.AuthInfo{"kubernetes": {}},
			},
			expected: true,
		},
	}
	for _, rt := range testCases {
		t.Run(rt.name, func(t *testing.T) {
			r := getCurrentAuthInfo(rt.config)
			if rt.expected != (r != nil) {
				t.Errorf(
					"failed TestHasCredentials:\n\texpected: %v\n\t  actual: %v",
					rt.expected,
					r,
				)
			}
		})
	}
}

func TestHasCredentials(t *testing.T) {
	var testCases = []struct {
		name     string
		config   *clientcmdapi.Config
		expected bool
	}{
		{
			name:     "no authInfo",
			config:   nil,
			expected: false,
		},
		{
			name: "no credentials",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"kubernetes": {AuthInfo: "kubernetes"}},
				AuthInfos:      map[string]*clientcmdapi.AuthInfo{"kubernetes": {}},
			},
			expected: false,
		},
		{
			name: "token authentication credentials",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"kubernetes": {AuthInfo: "kubernetes"}},
				AuthInfos:      map[string]*clientcmdapi.AuthInfo{"kubernetes": {Token: "123"}},
			},
			expected: true,
		},
		{
			name: "basic authentication credentials",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"kubernetes": {AuthInfo: "kubernetes"}},
				AuthInfos:      map[string]*clientcmdapi.AuthInfo{"kubernetes": {Username: "A", Password: "B"}},
			},
			expected: true,
		},
		{
			name: "X509 authentication credentials",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"kubernetes": {AuthInfo: "kubernetes"}},
				AuthInfos:      map[string]*clientcmdapi.AuthInfo{"kubernetes": {ClientKey: "A", ClientCertificate: "B"}},
			},
			expected: true,
		},
		{
			name: "exec authentication credentials",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"kubernetes": {AuthInfo: "kubernetes"}},
				AuthInfos:      map[string]*clientcmdapi.AuthInfo{"kubernetes": {Exec: &clientcmdapi.ExecConfig{Command: "command"}}},
			},
			expected: true,
		},
		{
			name: "authprovider authentication credentials",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"kubernetes": {AuthInfo: "kubernetes"}},
				AuthInfos:      map[string]*clientcmdapi.AuthInfo{"kubernetes": {AuthProvider: &clientcmdapi.AuthProviderConfig{Name: "A"}}},
			},
			expected: true,
		},
	}
	for _, rt := range testCases {
		t.Run(rt.name, func(t *testing.T) {
			r := HasAuthenticationCredentials(rt.config)
			if rt.expected != r {
				t.Errorf(
					"failed TestHasCredentials:\n\texpected: %v\n\t  actual: %v",
					rt.expected,
					r,
				)
			}
		})
	}
}

func TestGetClusterFromKubeConfig(t *testing.T) {
	tests := []struct {
		name                string
		config              *clientcmdapi.Config
		expectedClusterName string
		expectedCluster     *clientcmdapi.Cluster
	}{
		{
			name: "cluster is empty",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
			},
			expectedClusterName: "",
			expectedCluster:     nil,
		},
		{
			name: "cluster and currentContext are not empty",
			config: &clientcmdapi.Config{
				CurrentContext: "foo",
				Contexts: map[string]*clientcmdapi.Context{
					"foo": {AuthInfo: "foo", Cluster: "foo"},
					"bar": {AuthInfo: "bar", Cluster: "bar"},
				},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {Server: "http://foo:8080"},
					"bar": {Server: "https://bar:16443"},
				},
			},
			expectedClusterName: "foo",
			expectedCluster: &clientcmdapi.Cluster{
				Server: "http://foo:8080",
			},
		},
		{
			name: "cluster is not empty and currentContext is not in contexts",
			config: &clientcmdapi.Config{
				CurrentContext: "foo",
				Contexts: map[string]*clientcmdapi.Context{
					"bar": {AuthInfo: "bar", Cluster: "bar"},
				},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {Server: "http://foo:8080"},
					"bar": {Server: "https://bar:16443"},
				},
			},
			expectedClusterName: "",
			expectedCluster:     nil,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			clusterName, cluster := GetClusterFromKubeConfig(rt.config)
			if clusterName != rt.expectedClusterName {
				t.Errorf("got cluster name = %s, expected %s", clusterName, rt.expectedClusterName)
			}
			if !reflect.DeepEqual(cluster, rt.expectedCluster) {
				t.Errorf("got cluster = %+v, expected %+v", cluster, rt.expectedCluster)
			}
		})
	}
}

func TestEnsureAuthenticationInfoAreEmbedded(t *testing.T) {
	file, err := os.CreateTemp("", t.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(file.Name())
	defer file.Close()

	tests := []struct {
		name    string
		config  *clientcmdapi.Config
		wantErr bool
	}{
		{
			name: "get data from file",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"kubernetes": {AuthInfo: "kubernetes"}},
				AuthInfos: map[string]*clientcmdapi.AuthInfo{"kubernetes": {
					ClientCertificate: file.Name(),
					ClientKey:         file.Name(),
					TokenFile:         file.Name(),
				},
				},
			},
			wantErr: false,
		},
		{
			name: "get data from config",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"kubernetes": {AuthInfo: "kubernetes"}},
				AuthInfos: map[string]*clientcmdapi.AuthInfo{"kubernetes": {
					ClientCertificateData: []byte{'f', 'o', 'o'},
					ClientKeyData:         []byte{'b', 'a', 'r'},
					Token:                 "k8s",
				},
				},
			},
			wantErr: false,
		},
		{
			name:    "invalid authInfo: no authInfo",
			config:  nil,
			wantErr: true,
		},
		{
			name: "get data from file but the file doesn't exist",
			config: &clientcmdapi.Config{
				CurrentContext: "kubernetes",
				Contexts:       map[string]*clientcmdapi.Context{"kubernetes": {AuthInfo: "kubernetes"}},
				AuthInfos: map[string]*clientcmdapi.AuthInfo{"kubernetes": {
					ClientCertificate: "unknownfile",
					ClientKey:         "unknownfile",
					TokenFile:         "unknownfile",
				},
				},
			},
			wantErr: true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			if err := EnsureAuthenticationInfoAreEmbedded(rt.config); (err != nil) != rt.wantErr {
				t.Errorf("error = %v, wantErr %v", err, rt.wantErr)
			}
		})
	}
}

func TestEnsureCertificateAuthorityIsEmbedded(t *testing.T) {
	file, err := os.CreateTemp("", t.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(file.Name())
	defer file.Close()

	tests := []struct {
		name    string
		cluster *clientcmdapi.Cluster
		wantErr bool
	}{
		{
			name: "get data from file",
			cluster: &clientcmdapi.Cluster{
				CertificateAuthority: file.Name(),
			},
			wantErr: false,
		},
		{
			name: "get data from config",
			cluster: &clientcmdapi.Cluster{
				CertificateAuthorityData: []byte{'f', 'o', 'o'},
			},
			wantErr: false,
		},
		{
			name:    "cluster is nil",
			cluster: nil,
			wantErr: true,
		},
		{
			name: "get data from file but the file doesn't exist",
			cluster: &clientcmdapi.Cluster{
				CertificateAuthority: "unknownfile",
			},
			wantErr: true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			if err := EnsureCertificateAuthorityIsEmbedded(rt.cluster); (err != nil) != rt.wantErr {
				t.Errorf("error = %v, wantErr %v", err, rt.wantErr)
			}
		})
	}
}
