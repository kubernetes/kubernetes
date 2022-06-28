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
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type unsetConfigTest struct {
	name           string
	description    string
	config         clientcmdapi.Config
	args           []string
	expectedErr    string
	expectedConfig clientcmdapi.Config
}

func TestUnsetFromNewConfig(t *testing.T) {
	emptyConf := *clientcmdapi.NewConfig()
	tests := []unsetConfigTest{
		{
			name:        "Set with bad key",
			description: "Testing for kubectl config set users.foo.exec.fake.key to test",
			config:      emptyConf,
			args:        []string{"users.foo.exec.fake.key"},
			expectedErr: "unable to parse one or more field values of users.foo.exec.fake.key",
		},
		{
			name:        "SetAuthInfoUsername",
			description: "Testing for kubectl config unset users.foo.username",
			config: clientcmdapi.Config{
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
			args: []string{"users.foo.username"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			description: "Testing for kubectl config unset users.foo.password",
			config: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Password:   "test1",
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
			args: []string{"users.foo.password", "abadpassword"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			description: "Testing for kubectl config unset users.foo.client-certificate",
			config: clientcmdapi.Config{
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
			args: []string{"users.foo.client-certificate"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			name:        "SetAuthInfoClientCertificateData",
			description: "Testing for kubectl config unset users.foo.client-certificate-data",
			config: clientcmdapi.Config{
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
			args: []string{"users.foo.client-certificate-data"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			name:        "SetAuthInfoClientKey",
			description: "Testing for kubectl config unset users.foo.client-key",
			config: clientcmdapi.Config{
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
			args: []string{"users.foo.client-key"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			description: "Testing for kubectl config unset users.foo.client-key-data",
			config: clientcmdapi.Config{
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
			args: []string{"users.foo.client-key-data"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			name:        "SetAuthInfoToken",
			description: "Testing for kubectl config unset users.foo.token",
			config: clientcmdapi.Config{
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
			args: []string{"users.foo.token"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			description: "Testing for kubectl config unset users.foo.tokenFile",
			config: clientcmdapi.Config{
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
			args: []string{"users.foo.tokenFile"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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

func TestUnsetFromNewConfigJson(t *testing.T) {
	emptyConf := *clientcmdapi.NewConfig()
	tests := []unsetConfigTest{
		{
			name:        "Set with bad key",
			description: "Testing for kubectl config unset {users.foo.exec.fake.key}",
			config:      emptyConf,
			args:        []string{"{.users[?(@.name==\"test\")].user.exec.fake.key}"},
			expectedErr: "unable to parse path \"{.users[?(@.name==\\\"test\\\")].user.exec.fake.key}\" at \"fake\"",
		},
		{
			name:        "SetAuthInfoClientCertificate",
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.client-certificate}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.client-certificate}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			name:        "SetAuthInfoClientCertificateData",
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.client-certificate-data}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.client-certificate-data}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			name:        "SetAuthInfoClientKey",
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.client-key}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.client-key}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			description: "Testing for kubectl config set {.users[?(@.name==\"foo\")].user.client-key-data}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.client-key-data}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			name:        "SetAuthInfoToken",
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.token}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.token}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.tokenFile}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.tokenFile}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.foo.as}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.as}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			name:        "SetAuthInfoActAsUid",
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.foo.as-uid}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.as-uid}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			name:        "SetAuthInfoActAsGroups",
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.foo.as-groups}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.as-groups}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.foo.as-user-extra}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.as-user-extra.test}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.username}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.username}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.password}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.password}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
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
			name:        "SetAuthProviderName",
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.auth-provider.name}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.auth-provider.name}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						AuthProvider: &clientcmdapi.AuthProviderConfig{
							Name:   "",
							Config: nil,
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
			name:        "SetAuthProviderConfig",
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.auth-provider.config.refresh-token}",
			config: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						AuthProvider: &clientcmdapi.AuthProviderConfig{
							Config: map[string]string{
								"refresh-token": "tokenData",
							},
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
			args: []string{"{.users[?(@.name==\"foo\")].user.auth-provider.config.refresh-token}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						AuthProvider: &clientcmdapi.AuthProviderConfig{
							Config: map[string]string{},
							Name:   "oidc",
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
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.exec.command}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.exec.command}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec:       &clientcmdapi.ExecConfig{},
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
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.exec.args}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.exec.args}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec:       &clientcmdapi.ExecConfig{},
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
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.exec.env[?(@.name==\"test\")].value}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.exec.env[?(@.name==\"test\")]}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec: &clientcmdapi.ExecConfig{
							Env: []clientcmdapi.ExecEnvVar{{}},
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
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.exec.apiVersion}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.exec.apiVersion}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec:       &clientcmdapi.ExecConfig{},
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
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.exec.installHint}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.exec.installHint}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec:       &clientcmdapi.ExecConfig{},
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
			description: "Testing for kubectl config unset {.users[?(@.name==\"foo\")].user.exec.provideClusterInfo}",
			config: clientcmdapi.Config{
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
			args: []string{"{.users[?(@.name==\"foo\")].user.exec.provideClusterInfo}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{
					"foo": {
						Exec:       &clientcmdapi.ExecConfig{},
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
			description: "Testing for kubectl config unset {.clusters[?(@.name==\"foo\")].cluster.server}",
			config: clientcmdapi.Config{
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
			args: []string{"{.clusters[?(@.name==\"foo\")].cluster.server}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {
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
			description: "Testing for kubectl config unset {.clusters[?(@.name==\"foo\")].cluster.tls-server-name}",
			config: clientcmdapi.Config{
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
			args: []string{"{.clusters[?(@.name==\"foo\")].cluster.tls-server-name}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {
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
			name:        "SetClusterInsecureSkipTLSVerify",
			description: "Testing for kubectl config unset {.clusters[?(@.name==\"foo\")].cluster.insecure-skip-tls-verify}",
			config: clientcmdapi.Config{
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
			args: []string{"{.clusters[?(@.name==\"foo\")].cluster.insecure-skip-tls-verify}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {
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
			name:        "SetClusterCertificateAuthority",
			description: "Testing for kubectl config unset {.clusters[?(@.name==\"foo\")].cluster.certificate-authority}",
			config: clientcmdapi.Config{
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
			args: []string{"{.clusters[?(@.name==\"foo\")].cluster.certificate-authority}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {
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
			name:        "SetClusterCertificateAuthorityData",
			description: "Testing for kubectl config unset {.clusters[?(@.name==\"foo\")].cluster.certificate-authority-data}",
			config: clientcmdapi.Config{
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
			args: []string{"{.clusters[?(@.name==\"foo\")].cluster.certificate-authority-data}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {
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
			name:        "SetClusterProxyURL",
			description: "Testing for kubectl config unset {.clusters[?(@.name==\"foo\")].cluster.proxy-url}",
			config: clientcmdapi.Config{
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
			args: []string{"{.clusters[?(@.name==\"foo\")].cluster.proxy-url}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters: map[string]*clientcmdapi.Cluster{
					"foo": {
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
			description: "Testing for kubectl config unset {.contexts[?(@.name==\"foo\")].context.cluster}",
			config: clientcmdapi.Config{
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
			args: []string{"{.contexts[?(@.name==\"foo\")].context.cluster}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters:  map[string]*clientcmdapi.Cluster{},
				Contexts: map[string]*clientcmdapi.Context{
					"foo": {
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
			description: "Testing for kubectl config unset {.contexts[?(@.name==\"foo\")].context.user}",
			config: clientcmdapi.Config{
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
			args: []string{"{.contexts[?(@.name==\"foo\")].context.user}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters:  map[string]*clientcmdapi.Cluster{},
				Contexts: map[string]*clientcmdapi.Context{
					"foo": {
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
			description: "Testing for kubectl config unset {.contexts[?(@.name==\"foo\")].context.namespace}",
			config: clientcmdapi.Config{
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
			args: []string{"{.contexts[?(@.name==\"foo\")].context.namespace}"},
			expectedConfig: clientcmdapi.Config{
				AuthInfos: map[string]*clientcmdapi.AuthInfo{},
				Clusters:  map[string]*clientcmdapi.Cluster{},
				Contexts: map[string]*clientcmdapi.Context{
					"foo": {
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
			description: "Testing for kubectl config unset {.current-context}",
			config: clientcmdapi.Config{
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
			args:           []string{"{.current-context}"},
			expectedConfig: emptyConf,
		},
	}
	for _, test := range tests {
		test.run(t)
	}
}

func (test unsetConfigTest) run(t *testing.T) {
	// We must define two sets of path options to get proper test coverage
	// Define path options for cmd.Execute() run
	fakeKubeFile, err := ioutil.TempFile(os.TempDir(), "")
	if err != nil {
		t.Errorf("Failed in: %q\n unexpected error: %v", test.name, err)
	}
	defer func(name string) {
		err := os.Remove(name)
		if err != nil {
			t.Error("failed to remove test file")
		}
	}(fakeKubeFile.Name())
	err = clientcmd.WriteToFile(test.config, fakeKubeFile.Name())
	if err != nil {
		t.Errorf("Failed in: %q\n unexpected error: %v", test.name, err)
	}
	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""

	buf := bytes.NewBuffer([]byte{})
	opts := &unsetOptions{
		configAccess: pathOptions,
		propertyName: test.args[0],
	}
	if string(test.args[0][0]) == "{" {
		opts.jsonPath = true
	}

	// Must use opts.run to get error outputs
	if err := opts.run(buf); test.expectedErr == "" && err != nil {
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
	if len(test.expectedConfig.AuthInfos) > 0 {
		for k := range test.expectedConfig.AuthInfos {
			if test.expectedConfig.AuthInfos[k] != nil {
				test.expectedConfig.AuthInfos[k].LocationOfOrigin = fakeKubeFile.Name()
				test.expectedConfig.AuthInfos[k].Extensions = map[string]runtime.Object{}
			} else {
				t.Errorf("Failed in: %q\n cannot find key %v in AuthInfos map for expectedConfig and/or config", test.name, k)
			}
		}
	}
	if len(test.expectedConfig.Clusters) > 0 {
		for k := range test.expectedConfig.Clusters {
			if test.expectedConfig.Clusters[k] != nil {
				test.expectedConfig.Clusters[k].LocationOfOrigin = fakeKubeFile.Name()
				test.expectedConfig.Clusters[k].Extensions = map[string]runtime.Object{}
			}
		}
	}
	if len(test.expectedConfig.Contexts) > 0 {
		for k := range test.expectedConfig.Contexts {
			if test.expectedConfig.Contexts[k] != nil {
				test.expectedConfig.Contexts[k].LocationOfOrigin = fakeKubeFile.Name()
				test.expectedConfig.Contexts[k].Extensions = map[string]runtime.Object{}
			} else {
				t.Errorf("Failed in: %q\n cannot find key %v in Contexts map for expectedConfig and/or config", test.name, k)
			}
		}
	}

	if !reflect.DeepEqual(*config, test.expectedConfig) {
		t.Errorf("Failed in: %q\nconfig want/got mismatch (-want +got):\n%s", test.name, cmp.Diff(test.expectedConfig, *config))
	}
}
