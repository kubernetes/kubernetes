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
	expected       string
	expectedErr    string
	expectedConfig clientcmdapi.Config
}

func TestFromNewConfig(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	tests := []setConfigTest{
		{
			name:        "Set with bad key",
			description: "Testing for kubectl config set users.foo.exec.fake.key to test",
			config:      conf,
			args:        []string{"users.foo.exec.fake.key", "test"},
			expectedErr: "unable to locate path fake",
		},
		{
			name:        "SetAuthProviderName",
			description: "Testing for kubectl config set users.foo.auth-provider.name to oidc",
			config:      conf,
			args:        []string{"users.foo.auth-provider.name", "oidc"},
			expected:    `Property "users.foo.auth-provider.name" set.` + "\n",
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
			name:        "SetAuthProviderConfigMap",
			description: "Testing for kubectl config set users.foo.auth-provider.config.refresh-token to test",
			config:      conf,
			args:        []string{"users.foo.auth-provider.config.refresh-token", "test"},
			expected:    `Property "users.foo.auth-provider.config.refresh-token" set.` + "\n",
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						AuthProvider: &clientcmdapi.AuthProviderConfig{
							Name: "",
							Config: map[string]string{
								"refresh-token": "test",
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
			name:        "SetAuthProviderConfigMapMalformed",
			description: "Testing for kubectl config set users.foo.auth-provider.config to test",
			config:      conf,
			args:        []string{"users.foo.auth-provider.config", "test"},
			expectedErr: `empty key provided for map`,
		},
		{
			name:        "SetAuthInfoExecCommand",
			description: "Testing for kubectl config set users.foo.exec.command to test",
			config:      conf,
			args:        []string{"users.foo.exec.command", "test"},
			expected:    `Property "users.foo.exec.command" set.` + "\n",
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							Command: "test",
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
			name:        "SetAuthInfoExecArgs",
			description: "Testing for kubectl config set users.foo.exec.args to test3,test2,test1",
			config:      conf,
			args:        []string{"users.foo.exec.args", "test3,test2,test1"},
			expected:    `Property "users.foo.exec.args" set.` + "\n",
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							Args: []string{
								"test3",
								"test2",
								"test1",
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
			name:        "SetAuthInfoExecArgsDeduplicated",
			description: "Testing for kubectl config set users.foo.exec.args to test3,test2,test1,test3",
			config:      conf,
			args:        []string{"users.foo.exec.args", "--deduplicate", "test3,test2,test1,test3"},
			expected:    `Property "users.foo.exec.args" set.` + "\n",
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
			name:        "SetAuthInfoProvideClusterInfo",
			description: "Testing for kubectl config set users.foo.exec.provideClusterInfo to true",
			config:      conf,
			args:        []string{"users.foo.exec.provideClusterInfo", "true"},
			expected:    `Property "users.foo.exec.provideClusterInfo" set.` + "\n",
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
			name:        "SetAuthInfoExecEnvVar",
			description: "Testing for kubectl config set users.foo.exec.env.name.test to value:val1",
			config:      conf,
			args:        []string{"users.foo.exec.env.name.test", "value:val1"},
			expected:    `Property "users.foo.exec.env.name.test" set.` + "\n",
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							Env: []clientcmdapi.ExecEnvVar{
								{
									Name:  "test",
									Value: "val1",
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
			name:        "SetAuthInfoExecEnvVarMalformed",
			description: "Testing for kubectl config set users.foo.exec.env.name.test to value2",
			config:      conf,
			args:        []string{"users.foo.exec.env.name.test", "value2"},
			expectedErr: `error parsing field name for value, should be of format fieldName:fieldValue`,
		},
		{
			name:        "SetAuthInfoExecInstallHint",
			description: "Testing for kubectl config set users.foo.exec.installHint to test",
			config:      conf,
			args:        []string{"users.foo.exec.installHint", "test"},
			expected:    `Property "users.foo.exec.installHint" set.` + "\n",
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							InstallHint: "test",
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
			name:        "SetAuthInfoActAsUser",
			description: "Testing for kubectl config set users.foo.act-as to test1",
			config:      conf,
			args:        []string{"users.foo.act-as", "test1"},
			expected:    `Property "users.foo.act-as" set.` + "\n",
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
			name:        "SetAuthInfoActUid",
			description: "Testing for kubectl config set users.foo.act-as-uid to 1000",
			config:      conf,
			args:        []string{"users.foo.act-as-uid", "1000"},
			expected:    `Property "users.foo.act-as-uid" set.` + "\n",
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
			description: "Testing for kubectl config set users.foo.act-as-groups to test3,test2,test1",
			config:      conf,
			args:        []string{"users.foo.act-as-groups", "test3,test2,test1"},
			expected:    `Property "users.foo.act-as-groups" set.` + "\n",
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ImpersonateGroups: []string{
							"test3",
							"test2",
							"test1",
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
			name:        "SetAuthInfoActAsGroupsDeduplicated",
			description: "Testing for kubectl config set users.foo.act-as-groups to test3,test2,test1,test3",
			config:      conf,
			args:        []string{"users.foo.act-as-groups", "--deduplicate", "test3,test2,test1,test3"},
			expected:    `Property "users.foo.act-as-groups" set.` + "\n",
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
			description: "Testing for kubectl config set users.foo.act-as-user-extra.test1 to val3,val2,val1",
			config:      conf,
			args:        []string{"users.foo.act-as-user-extra.test1", "val3,val2,val1"},
			expected:    `Property "users.foo.act-as-user-extra.test1" set.` + "\n",
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ImpersonateUserExtra: map[string][]string{
							"test1": {
								"val3",
								"val2",
								"val1",
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
			name:        "SetAuthInfoActAsUserExtraDeduplicate",
			description: "Testing for kubectl config set users.foo.act-as-user-extra.test1 to val3,val2,val1,val3",
			config:      conf,
			args:        []string{"users.foo.act-as-user-extra.test1", "--deduplicate", "val3,val2,val1,val3"},
			expected:    `Property "users.foo.act-as-user-extra.test1" set.` + "\n",
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						ImpersonateUserExtra: map[string][]string{
							"test1": {
								"val1",
								"val2",
								"val3",
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
			description: "Testing for kubectl config set users.foo.username to test1",
			config:      conf,
			args:        []string{"users.foo.username", "test1"},
			expected:    `Property "users.foo.username" set.` + "\n",
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
			expected:    `Property "users.foo.password" set.` + "\n",
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
			expected:    `Property "users.foo.client-certificate" set.` + "\n",
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
			args:        []string{"users.foo.client-certificate-data", "--set-raw-bytes=true", "not real cert data"},
			expected:    `Property "users.foo.client-certificate-data" set.` + "\n",
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
			expected:    `Property "users.foo.client-key" set.` + "\n",
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
			args:        []string{"users.foo.client-key-data", "--set-raw-bytes=true", "not real key data"},
			expected:    `Property "users.foo.client-key-data" set.` + "\n",
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
			expected:    `Property "users.foo.token" set.` + "\n",
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
			expected:    `Property "users.foo.tokenFile" set.` + "\n",
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
			name:        "SetAuthProviderMalformed",
			description: "Testing for kubectl config set users.foo.auth-provider to test error",
			config:      conf,
			args:        []string{"users.foo.auth-provider", "test"},
			expectedErr: `unable to locate path auth-provider`,
		},
	}
	for _, test := range tests {
		test.run(t)
	}
}

func TestFromExistingConfig(t *testing.T) {
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
			expected:    `Property "current-context" set.` + "\n",
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
		{
			name:        "AddExecArg",
			description: "Testing for kubectl config set users.foo.exec.args to test4+",
			config:      conf,
			args:        []string{"users.foo.exec.args", "test4+"},
			expected:    `Property "users.foo.exec.args" set.` + "\n",
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							Args: []string{
								"test1",
								"test2",
								"test3",
								"test4",
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
			},
		},
		{
			name:        "AddExecArgDeduplicate",
			description: "Testing for kubectl config set users.foo.exec.args to test3,test4+",
			config:      conf,
			args:        []string{"users.foo.exec.args", "test3,test4+", "--deduplicate"},
			expected:    `Property "users.foo.exec.args" set.` + "\n",
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							Args: []string{
								"test1",
								"test2",
								"test3",
								"test4",
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
			},
		},
		{
			name:        "DeleteExecArg",
			description: "Testing for kubectl config set users.foo.exec.args to test2-",
			config:      conf,
			args:        []string{"users.foo.exec.args", "test2-"},
			expected:    `Property "users.foo.exec.args" set.` + "\n",
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							Args: []string{
								"test1",
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
			},
		},
		{
			name:        "UpdateExecArgs",
			description: "Testing for kubectl config set users.foo.exec.args to test5,test4",
			config:      conf,
			args:        []string{"users.foo.exec.args", "test5,test4"},
			expected:    `Property "users.foo.exec.args" set.` + "\n",
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							Args: []string{
								"test5",
								"test4",
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
			},
		},
		{
			name:        "UpdateExecArgsDeduplication",
			description: "Testing for kubectl config set users.foo.exec.args to test5,test4,test5",
			config:      conf,
			args:        []string{"users.foo.exec.args", "test5,test4,test5", "--deduplicate"},
			expected:    `Property "users.foo.exec.args" set.` + "\n",
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							Args: []string{
								"test4",
								"test5",
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
			},
		},
		{
			name:        "UpdateExecEnvVar",
			description: "Testing for kubectl config set users.foo.exec.env.name.test to value:value2",
			config:      conf,
			args:        []string{"users.foo.exec.env.name.test", "value:value2"},
			expected:    `Property "users.foo.exec.env.name.test" set.` + "\n",
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
									Value: "value2",
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
			},
		},
		{
			name:        "AddNewExecEnvVar",
			description: "Testing for kubectl config set users.foo.exec.env.name.test1 to value:value2",
			config:      conf,
			args:        []string{"users.foo.exec.env.name.test1", "value:value2"},
			expected:    `Property "users.foo.exec.env.name.test1" set.` + "\n",
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
								{
									Name:  "test1",
									Value: "value2",
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
			},
		},
		{
			name:        "AddActAsGroups",
			description: "Testing for kubectl config set users.foo.act-as-groups to test4+",
			config:      conf,
			args:        []string{"users.foo.act-as-groups", "test4+"},
			expected:    `Property "users.foo.act-as-groups" set.` + "\n",
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
							"test4",
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
			},
		},
		{
			name:        "AddActAsGroupsDeduplicate",
			description: "Testing for kubectl config set users.foo.act-as-groups to test3,test4+",
			config:      conf,
			args:        []string{"users.foo.act-as-groups", "test3,test4+", "--deduplicate"},
			expected:    `Property "users.foo.act-as-groups" set.` + "\n",
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
							"test4",
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
			},
		},
		{
			name:        "DeleteActAsGroups",
			description: "Testing for kubectl config set users.foo.act-as-groups to test3-",
			config:      conf,
			args:        []string{"users.foo.act-as-groups", "test3-"},
			expected:    `Property "users.foo.act-as-groups" set.` + "\n",
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
			},
		},
		{
			name:        "UpdateActAsGroups",
			description: "Testing for kubectl config set users.foo.act-as-groups to test8,test9",
			config:      conf,
			args:        []string{"users.foo.act-as-groups", "test8,test9"},
			expected:    `Property "users.foo.act-as-groups" set.` + "\n",
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
							"test8",
							"test9",
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
			},
		},
		{
			name:        "AddActAsUserExtra",
			description: "Testing for kubectl config set users.foo.act-as-user-extra.test1 to val4+",
			config:      conf,
			args:        []string{"users.foo.act-as-user-extra.test1", "val4+"},
			expected:    `Property "users.foo.act-as-user-extra.test1" set.` + "\n",
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
								"val4",
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
			},
		},
		{
			name:        "AddActAsUserExtraDeduplicate",
			description: "Testing for kubectl config set users.foo.act-as-user-extra.test1 to val3,val4+",
			config:      conf,
			args:        []string{"users.foo.act-as-user-extra.test1", "val3,val4+", "--deduplicate"},
			expected:    `Property "users.foo.act-as-user-extra.test1" set.` + "\n",
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
								"val4",
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
			},
		},
		{
			name:        "DeleteActAsUserExtra",
			description: "Testing for kubectl config set users.foo.act-as-user-extra.test1 to val3-",
			config:      conf,
			args:        []string{"users.foo.act-as-user-extra.test1", "val3-"},
			expected:    `Property "users.foo.act-as-user-extra.test1" set.` + "\n",
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
			},
		},
		{
			name:        "UpdateActAsUserExtra",
			description: "Testing for kubectl config set users.foo.act-as-user-extra.test1 to val5,val6,val7",
			config:      conf,
			args:        []string{"users.foo.act-as-user-extra.test1", "val5,val6,val7"},
			expected:    `Property "users.foo.act-as-user-extra.test1" set.` + "\n",
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
								"val5",
								"val6",
								"val7",
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
			},
		},
		{
			name:        "AddNewActAsUserExtra",
			description: "Testing for kubectl config set users.foo.act-as-user-extra.test2 to val1",
			config:      conf,
			args:        []string{"users.foo.act-as-user-extra.test2", "val1"},
			expected:    `Property "users.foo.act-as-user-extra.test2" set.` + "\n",
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
							"test2": {
								"val1",
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
	fakeKubeFileCmd, err := ioutil.TempFile(os.TempDir(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(fakeKubeFileCmd.Name())
	err = clientcmd.WriteToFile(test.config, fakeKubeFileCmd.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pathOptionsCmd := clientcmd.NewDefaultPathOptions()
	pathOptionsCmd.GlobalFile = fakeKubeFileCmd.Name()
	pathOptionsCmd.EnvVar = ""

	// Define path options for opts.run() execution
	fakeKubeFileOpts, err := ioutil.TempFile(os.TempDir(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(fakeKubeFileOpts.Name())
	err = clientcmd.WriteToFile(test.config, fakeKubeFileOpts.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pathOptionsOpts := clientcmd.NewDefaultPathOptions()
	pathOptionsOpts.GlobalFile = fakeKubeFileOpts.Name()
	pathOptionsOpts.EnvVar = ""

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdConfigSet(buf, pathOptionsCmd)
	cmd.SetArgs(test.args)
	opts := &setOptions{
		configAccess:  pathOptionsOpts,
		propertyName:  test.args[0],
		propertyValue: test.args[1],
	}
	if sets.NewString(test.args...).Has("--set-raw-bytes=true") {
		opts.setRawBytes = 1
	}
	if sets.NewString(test.args...).Has("--deduplicate") {
		opts.deduplicate = true
	}

	// Must use opts.run to get error outputs
	err = opts.run()
	if test.expectedErr == "" && err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check for expected error message
	if test.expectedErr != "" {
		if err != nil && err.Error() != test.expectedErr {
			t.Fatalf("expected error:\n %v\nbut got error:\n%v", test.expectedErr, err)
		}
		return
	}

	// Must use cmd.Execute to get stdout output
	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v", err)
	}
	config, err := clientcmd.LoadFromFile(fakeKubeFileCmd.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}

	// Must manually set LocationOfOrigin field of AuthInfos if they exists
	if len(config.AuthInfos) > 0 {
		for k := range config.AuthInfos {
			if test.expectedConfig.AuthInfos[k] != nil && config.AuthInfos[k] != nil {
				test.expectedConfig.AuthInfos[k].LocationOfOrigin = config.AuthInfos[k].LocationOfOrigin
			} else {
				t.Errorf("Failed in:%q\n cannot find key %v in AuthInfos map for expectedConfig and/or config", test.description, k)
			}
		}
	}

	if len(test.expected) != 0 {
		if buf.String() != test.expected {
			t.Errorf("Failed in:%q\n expected %v\n but got %v", test.description, test.expected, buf.String())
		}
	}
	if !reflect.DeepEqual(*config, test.expectedConfig) {
		t.Errorf("%v\nconfig want/got mismatch (-want +got):\n%s", test.name, cmp.Diff(test.expectedConfig, *config))
	}
}
