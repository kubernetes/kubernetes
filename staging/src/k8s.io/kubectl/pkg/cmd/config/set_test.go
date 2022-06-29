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
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
	"reflect"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type setConfigTest struct {
	name           string
	description    string
	config         clientcmdapi.Config
	args           []string
	expectedErr    string
	expectedConfig clientcmdapi.Config
}

func TestSetFromNewConfig(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	tests := []setConfigTest{
		{
			name:        "Set with bad key",
			description: "Testing for kubectl config set users.foo.exec.fake.key to test",
			config:      conf,
			args:        []string{"users.foo.exec.fake.key", "test"},
			expectedErr: "unable to parse one or more field values of users.foo.exec.fake.key",
		},
		{
			name:        "SetAuthInfoUsername",
			description: "Testing for kubectl config set users.foo.username to test1",
			config:      conf,
			args:        []string{"users.foo.username", "test1"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Username:   "test1",
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoPassword",
			description: "Testing for kubectl config set users.foo.password to abadpassword",
			config:      conf,
			args:        []string{"users.foo.password", "abadpassword"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Password:   "abadpassword",
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoClientCertificate",
			description: "Testing for kubectl config set users.foo.client-certificate to ./file/path",
			config:      conf,
			args:        []string{"users.foo.client-certificate", "./file/path"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ClientCertificate: "./file/path",
						Extensions:        map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoClientCertificateData",
			description: "Testing for kubectl config set users.foo.client-certificate-data to not real cert data",
			config:      conf,
			args:        []string{"users.foo.client-certificate-data", "not real cert data", "--set-raw-bytes=true"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ClientCertificateData: []byte("not real cert data"),
						Extensions:            map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoClientKey",
			description: "Testing for kubectl config set users.foo.client-key to ./file/path",
			config:      conf,
			args:        []string{"users.foo.client-key", "./file/path"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ClientKey:  "./file/path",
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoClientKeyData",
			description: "Testing for kubectl config set users.foo.client-key-data to not real key data",
			config:      conf,
			args:        []string{"users.foo.client-key-data", "not real key data", "--set-raw-bytes=true"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ClientKeyData: []byte("not real key data"),
						Extensions:    map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoToken",
			description: "Testing for kubectl config set users.foo.token to fake token data",
			config:      conf,
			args:        []string{"users.foo.token", "fake token data"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Token:      "fake token data",
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoTokenFile",
			description: "Testing for kubectl config set users.foo.tokenFile to ./file/path",
			config:      conf,
			args:        []string{"users.foo.tokenFile", "./file/path"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						TokenFile:  "./file/path",
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
	}
	for _, test := range tests {
		test.run(t)
	}
}

func TestSetFromExistingConfig(t *testing.T) {
	conf := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				Exec: &clientcmdapi.ExecConfig{
					Args: []string{
						"test1",
						"test2",
						"test3",
					},
					Env: []clientcmdapi.ExecEnvVar{
						{
							Name:  "test",
							Value: "value1",
						},
					},
				},
				Extensions: map[string]runtime.Object{},
				ImpersonateGroups: []string{
					"test1",
					"test2",
					"test3",
				},
				ImpersonateUserExtra: map[string][]string{
					"test1": {
						"val1",
						"val2",
						"val3",
					},
				},
			},
		},
		Clusters:       map[string]*clientcmdapi.Cluster{},
		Contexts:       map[string]*clientcmdapi.Context{},
		CurrentContext: "minikube",
		Extensions:     map[string]runtime.Object{},
		Preferences: clientcmdapi.Preferences{
			Colors:     false,
			Extensions: map[string]runtime.Object{},
		},
	}
	tests := []setConfigTest{
		{
			name:        "SetCurrentContext",
			description: "Testing for kubectl config set current-context",
			config:      conf,
			args:        []string{"current-context", "my-cluster"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							Args: []string{
								"test1",
								"test2",
								"test3",
							},
							Env: []clientcmdapi.ExecEnvVar{
								{
									Name:  "test",
									Value: "value1",
								},
							},
						},
						Extensions: map[string]runtime.Object{},
						ImpersonateGroups: []string{
							"test1",
							"test2",
							"test3",
						},
						ImpersonateUserExtra: map[string][]string{
							"test1": {
								"val1",
								"val2",
								"val3",
							},
						},
					},
				},
				Clusters:       map[string]*clientcmdapi.Cluster{},
				Contexts:       map[string]*clientcmdapi.Context{},
				CurrentContext: "my-cluster",
				Extensions:     map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
	}
	for _, test := range tests {
		test.run(t)
	}
}

func TestSetFromNewConfigJson(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	tests := []setConfigTest{
		{
			name:        "Set with bad key",
			description: "Testing for kubectl config set {users.foo.exec.fake.key} to test",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"test\")].user.exec.fake.key}", "test"},
			expectedErr: "unable to parse path \"{.users[?(@.name==\\\"test\\\")].user.exec.fake.key}\" at \"fake\"",
		},
		{
			name:        "SetAuthInfoClientCertificate",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.client-certificate} to ./file/path",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.client-certificate}", "./file/path"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ClientCertificate: "./file/path",
						Extensions:        map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoClientCertificateData",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.client-certificate-data} to not real cert data",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.client-certificate-data}", "not real cert data", "--set-raw-bytes=true"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ClientCertificateData: []byte("not real cert data"),
						Extensions:            map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoClientKey",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.client-key} to ./file/path",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.client-key}", "./file/path"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ClientKey:  "./file/path",
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoClientKeyData",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.client-key-data} to not real key data",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.client-key-data}", "not real key data", "--set-raw-bytes=true"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ClientKeyData: []byte("not real key data"),
						Extensions:    map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoToken",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.token} to fake token data",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.token}", "fake token data"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Token:      "fake token data",
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoTokenFile",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.tokenFile} to ./file/path",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.tokenFile}", "./file/path"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						TokenFile:  "./file/path",
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoActAsUser",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.foo.as}",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.as}", "test1"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Impersonate: "test1",
						Extensions:  map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoActAsUid",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.foo.as-uid} to 1000",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.as-uid}", "1000"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ImpersonateUID: "1000",
						Extensions:     map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoActAsGroups",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.foo.as-groups} to test1,test2,test3",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.as-groups}", "test1,test2,test3"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ImpersonateGroups: []string{
							"test1",
							"test2",
							"test3",
						},
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoActAsUserExtra",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.foo.as-user-extra} to test1,test2,test3",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.as-user-extra.test}", "test1,test2,test3"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ImpersonateUserExtra: map[string][]string{
							"test": {
								"test1",
								"test2",
								"test3",
							},
						},
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoUsername",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.username} to test1",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.username}", "test1"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Username:   "test1",
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthInfoPassword",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.password} to abadpassword",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.password}", "abadpassword"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Password:   "abadpassword",
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthProviderConfigName",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.auth-provider.name} to oidc",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.auth-provider.name}", "oidc"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						AuthProvider: &clientcmdapi.AuthProviderConfig{
							Name: "oidc",
						},
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetAuthProviderConfigName",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.auth-provider.config.refresh-token} to tokenData",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.auth-provider.config.refresh-token}", "tokenData"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						AuthProvider: &clientcmdapi.AuthProviderConfig{
							Config: map[string]string{
								"refresh-token": "tokenData",
							},
						},
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetExecConfigCommand",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.exec.command} to curl",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.exec.command}", "curl"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							Command: "curl",
						},
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetExecConfigArgs",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.exec.args} to test1,test2,test3",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.exec.args}", "test1,test2,test3"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							Args: []string{
								"test1",
								"test2",
								"test3",
							},
						},
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetExecConfigEnv",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.exec.env[?(@.name==\"test\")].value} to test",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.exec.env[?(@.name==\"test\")].value}", "test"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							Env: []clientcmdapi.ExecEnvVar{
								{
									Name:  "test",
									Value: "test",
								},
							},
						},
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetExecConfigApiVersion",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.exec.apiVersion} to v2",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.exec.apiVersion}", "v2"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							APIVersion: "v2",
						},
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetExecConfigInstallHint",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.exec.installHint} to fake install hint text",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.exec.installHint}", "fake install hint text"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							InstallHint: "fake install hint text",
						},
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetExecConfigProvideClusterInfo",
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.exec.provideClusterInfo} to true",
			config:      conf,
			args:        []string{"{.users[?(@.name==\"foo\")].user.exec.provideClusterInfo}", "true"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							ProvideClusterInfo: true,
						},
						Extensions: map[string]runtime.Object{},
					},
				},
				Clusters:   map[string]*clientcmdapi.Cluster{},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetClusterServer",
			description: "Testing for kubectl config set {.clusters[?(@.name==\"foo\")].cluster.server} to https://1.2.3.4",
			config:      conf,
			args:        []string{"{.clusters[?(@.name==\"foo\")].cluster.server}", "https://1.2.3.4"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {
						Server:     "https://1.2.3.4",
						Extensions: map[string]runtime.Object{},
					},
				},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetClusterTLSServerName",
			description: "Testing for kubectl config set {.clusters[?(@.name==\"foo\")].cluster.tls-server-name} to test.server.com",
			config:      conf,
			args:        []string{"{.clusters[?(@.name==\"foo\")].cluster.tls-server-name}", "test.server.com"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {
						TLSServerName: "test.server.com",
						Extensions:    map[string]runtime.Object{},
					},
				},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetClusterInsecureSkipTLSVerify",
			description: "Testing for kubectl config set {.clusters[?(@.name==\"foo\")].cluster.insecure-skip-tls-verify} to true",
			config:      conf,
			args:        []string{"{.clusters[?(@.name==\"foo\")].cluster.insecure-skip-tls-verify}", "true"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {
						InsecureSkipTLSVerify: true,
						Extensions:            map[string]runtime.Object{},
					},
				},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetClusterCertificateAuthority",
			description: "Testing for kubectl config set {.clusters[?(@.name==\"foo\")].cluster.certificate-authority} to ./test.ca",
			config:      conf,
			args:        []string{"{.clusters[?(@.name==\"foo\")].cluster.certificate-authority}", "./test.ca"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {
						CertificateAuthority: "./test.ca",
						Extensions:           map[string]runtime.Object{},
					},
				},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetClusterCertificateAuthorityData",
			description: "Testing for kubectl config set {.clusters[?(@.name==\"foo\")].cluster.certificate-authority-data} to fake ca data",
			config:      conf,
			args:        []string{"{.clusters[?(@.name==\"foo\")].cluster.certificate-authority-data}", "fake ca data", "--set-raw-bytes=true"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {
						CertificateAuthorityData: []byte("fake ca data"),
						Extensions:               map[string]runtime.Object{},
					},
				},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetClusterProxyURL",
			description: "Testing for kubectl config set {.clusters[?(@.name==\"foo\")].cluster.proxy-url} to https://4.3.2.1",
			config:      conf,
			args:        []string{"{.clusters[?(@.name==\"foo\")].cluster.proxy-url}", "https://4.3.2.1"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {
						ProxyURL:   "https://4.3.2.1",
						Extensions: map[string]runtime.Object{},
					},
				},
				Contexts:   map[string]*clientcmdapi.Context{},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetContextCluster",
			description: "Testing for kubectl config set {.contexts[?(@.name==\"foo\")].context.cluster} to test-cluster",
			config:      conf,
			args:        []string{"{.contexts[?(@.name==\"foo\")].context.cluster}", "test-cluster"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters:  map[string]*clientcmdapi.Cluster{},
				Contexts: map[string]*clientcmdapi.Context{
					"foo": {
						Cluster:    "test-cluster",
						Extensions: map[string]runtime.Object{},
					},
				},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetContextAuthInfo",
			description: "Testing for kubectl config set {.contexts[?(@.name==\"foo\")].context.user} to test-user",
			config:      conf,
			args:        []string{"{.contexts[?(@.name==\"foo\")].context.user}", "test-user"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters:  map[string]*clientcmdapi.Cluster{},
				Contexts: map[string]*clientcmdapi.Context{
					"foo": {
						AuthInfo:   "test-user",
						Extensions: map[string]runtime.Object{},
					},
				},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetContextNamespace",
			description: "Testing for kubectl config set {.contexts[?(@.name==\"foo\")].context.namespace} to test-ns",
			config:      conf,
			args:        []string{"{.contexts[?(@.name==\"foo\")].context.namespace}", "test-ns"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters:  map[string]*clientcmdapi.Cluster{},
				Contexts: map[string]*clientcmdapi.Context{
					"foo": {
						Namespace:  "test-ns",
						Extensions: map[string]runtime.Object{},
					},
				},
				Extensions: map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
		{
			name:        "SetCurrentContext",
			description: "Testing for kubectl config set {.current-context} to test-context",
			config:      conf,
			args:        []string{"{.current-context}", "test-context"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos:      map[string]*clientcmdapi.AuthInfo{},
				Clusters:       map[string]*clientcmdapi.Cluster{},
				Contexts:       map[string]*clientcmdapi.Context{},
				CurrentContext: "test-context",
				Extensions:     map[string]runtime.Object{},
				Preferences: clientcmdapi.Preferences{
					Colors:     false,
					Extensions: map[string]runtime.Object{},
				},
			},
		},
	}
	for _, test := range tests {
		test.run(t)
	}
}

func (test setConfigTest) run(t *testing.T) {
	// We must define two sets of path options to get proper test coverage
	// Define path options for cmd.Execute() run
	fakeKubeFile, err := ioutil.TempFile(os.TempDir(), "")
	if err != nil {
		t.Errorf("Failed in: %q\n unexpected error: %v", test.name, err)
	}
	defer func(name string) {
		err := os.Remove(name)
		if err != nil {
			t.Error("Failed to remove test file")
		}
	}(fakeKubeFile.Name())
	err = clientcmd.WriteToFile(test.config, fakeKubeFile.Name())
	if err != nil {
		t.Errorf("Failed in: %q\n unexpected error: %v", test.name, err)
	}
	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""
	streams, _, _, _ := genericclioptions.NewTestIOStreams()

	opts := &setOptions{
		configAccess:  pathOptions,
		propertyName:  test.args[0],
		propertyValue: test.args[1],
		streams:       streams,
	}
	if sets.NewString(test.args...).Has("--set-raw-bytes=true") {
		opts.setRawBytes = 1
	}
	if sets.NewString(test.args...).Has("--deduplicate") {
		opts.deduplicate = true
	}
	if string(test.args[0][0]) == "{" {
		opts.jsonPath = true
	}

	// Must use opts.run to get error outputs
	if err := opts.run(); test.expectedErr == "" && err != nil {
		t.Errorf("Failed in: %q\n unexpected error: %v", test.name, err)
	}

	// Check for expected error message
	if test.expectedErr != "" {
		if err != nil && err.Error() != test.expectedErr {
			t.Errorf("Failed in: %q\n expected error:\n %v\nbut got error:\n%v", test.name, test.expectedErr, err)
		}
		return
	}

	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Errorf("Failed in: %q\n unexpected error loading kubeconfig file: %v", test.name, err)
	}

	// Must manually set LocationOfOrigin field of AuthInfos if they exists
	if len(config.AuthInfos) > 0 {
		for k := range config.AuthInfos {
			if test.expectedConfig.AuthInfos[k] != nil && config.AuthInfos[k] != nil {
				test.expectedConfig.AuthInfos[k].LocationOfOrigin = config.AuthInfos[k].LocationOfOrigin
			} else {
				t.Errorf("Failed in: %q\n cannot find key %v in AuthInfos map for expectedConfig and/or config", test.name, k)
			}
		}
	}
	if len(config.Clusters) > 0 {
		for k := range config.Clusters {
			if test.expectedConfig.Clusters[k] != nil && config.Clusters[k] != nil {
				test.expectedConfig.Clusters[k].LocationOfOrigin = config.Clusters[k].LocationOfOrigin
			} else {
				t.Errorf("Failed in: %q\n cannot find key %v in Clusters map for expectedConfig and/or config", test.name, k)
			}
		}
	}
	if len(config.Contexts) > 0 {
		for k := range config.Contexts {
			if test.expectedConfig.Contexts[k] != nil && config.Contexts[k] != nil {
				test.expectedConfig.Contexts[k].LocationOfOrigin = config.Contexts[k].LocationOfOrigin
			} else {
				t.Errorf("Failed in: %q\n cannot find key %v in Contexts map for expectedConfig and/or config", test.name, k)
			}
		}
	}

	if !reflect.DeepEqual(*config, test.expectedConfig) {
		t.Errorf("Failed in: %q\nconfig want/got mismatch (-want +got):\n%s", test.name, cmp.Diff(test.expectedConfig, *config))
	}
}
