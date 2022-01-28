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

	"reflect"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type setConfigTest struct {
	description    string
	config         clientcmdapi.Config
	args           []string
	expected       string
	expectedErr    string
	expectedConfig clientcmdapi.Config
}

func TestSetConfigCurrentContext(t *testing.T) {
	conf := clientcmdapi.Config{
		Kind:           "Config",
		APIVersion:     "v1",
		CurrentContext: "minikube",
	}
	expectedConfig := *clientcmdapi.NewConfig()
	expectedConfig.CurrentContext = "my-cluster"
	test := setConfigTest{
		description:    "Testing for kubectl config set current-context",
		config:         conf,
		args:           []string{"current-context", "my-cluster"},
		expected:       `Property "current-context" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigAuthProviderName(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.auth-provider.name to oidc",
		config:         conf,
		args:           []string{"users.foo.auth-provider.name", "oidc"},
		expected:       `Property "users.foo.auth-provider.name" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigAuthProviderConfigRefreshToken(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.auth-provider.config.refresh-token to test",
		config:         conf,
		args:           []string{"users.foo.auth-provider.config.refresh-token", "test"},
		expected:       `Property "users.foo.auth-provider.config.refresh-token" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigExecConfigCommand(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.exec.command to test",
		config:         conf,
		args:           []string{"users.foo.exec.command", "test"},
		expected:       `Property "users.foo.exec.command" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigExecArgsNew(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.exec.args to test1,test2,test3",
		config:         conf,
		args:           []string{"users.foo.exec.args", "test1,test2,test3"},
		expected:       `Property "users.foo.exec.args" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigExecArgsAdd(t *testing.T) {
	conf := clientcmdapi.Config{
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
	}
	expectedConfig := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				Exec: &clientcmdapi.ExecConfig{
					Args: []string{
						"test1",
						"test2",
						"test3",
						"test4",
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.exec.args to test4+",
		config:         conf,
		args:           []string{"users.foo.exec.args", "test4+"},
		expected:       `Property "users.foo.exec.args" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigExecArgsDelete(t *testing.T) {
	conf := clientcmdapi.Config{
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
	}
	expectedConfig := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				Exec: &clientcmdapi.ExecConfig{
					Args: []string{
						"test1",
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.exec.args to test2-",
		config:         conf,
		args:           []string{"users.foo.exec.args", "test2-"},
		expected:       `Property "users.foo.exec.args" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigExecArgsUpdate(t *testing.T) {
	conf := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				Exec: &clientcmdapi.ExecConfig{
					Args: []string{
						"test1",
						"test2",
						"test3",
						"test4",
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
	}
	expectedConfig := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				Exec: &clientcmdapi.ExecConfig{
					Args: []string{
						"test1",
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.exec.args to test1,test3",
		config:         conf,
		args:           []string{"users.foo.exec.args", "test1,test3"},
		expected:       `Property "users.foo.exec.args" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigExecProvideClusterInfo(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.exec.provideClusterInfo to true",
		config:         conf,
		args:           []string{"users.foo.exec.provideClusterInfo", "true"},
		expected:       `Property "users.foo.exec.provideClusterInfo" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigExecEnvNew(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.exec.env.name.test to value:val1",
		config:         conf,
		args:           []string{"users.foo.exec.env.name.test", "value:val1"},
		expected:       `Property "users.foo.exec.env.name.test" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigExecEnvUpdate(t *testing.T) {
	conf := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				Exec: &clientcmdapi.ExecConfig{
					Env: []clientcmdapi.ExecEnvVar{
						{
							Name:  "test",
							Value: "value1",
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
	}
	expectedConfig := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				Exec: &clientcmdapi.ExecConfig{
					Env: []clientcmdapi.ExecEnvVar{
						{
							Name:  "test",
							Value: "value2",
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.exec.env.name.test to value:value2",
		config:         conf,
		args:           []string{"users.foo.exec.env.name.test", "value:value2"},
		expected:       `Property "users.foo.exec.env.name.test" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigInstallHint(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.exec.installHint to test",
		config:         conf,
		args:           []string{"users.foo.exec.installHint", "test"},
		expected:       `Property "users.foo.exec.installHint" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigImpersonateUser(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.act-as to test1",
		config:         conf,
		args:           []string{"users.foo.act-as", "test1"},
		expected:       `Property "users.foo.act-as" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigImpersonateUID(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.act-as to 1000",
		config:         conf,
		args:           []string{"users.foo.act-as-uid", "1000"},
		expected:       `Property "users.foo.act-as-uid" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigImpersonateGroupsNew(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.act-as-groups to test1,test2,test3",
		config:         conf,
		args:           []string{"users.foo.act-as-groups", "test1,test2,test3"},
		expected:       `Property "users.foo.act-as-groups" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigImpersonateGroupsAdd(t *testing.T) {
	conf := clientcmdapi.Config{
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
	}
	expectedConfig := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				ImpersonateGroups: []string{
					"test1",
					"test2",
					"test3",
					"test4",
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.act-as-groups to test4+",
		config:         conf,
		args:           []string{"users.foo.act-as-groups", "test4+"},
		expected:       `Property "users.foo.act-as-groups" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigImpersonateGroupsDelete(t *testing.T) {
	conf := clientcmdapi.Config{
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
	}
	expectedConfig := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				ImpersonateGroups: []string{
					"test1",
					"test2",
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.act-as-groups to test3-",
		config:         conf,
		args:           []string{"users.foo.act-as-groups", "test3-"},
		expected:       `Property "users.foo.act-as-groups" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigImpersonateGroupsUpdate(t *testing.T) {
	conf := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				ImpersonateGroups: []string{
					"test1",
					"test2",
					"test3",
					"test4",
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
	}
	expectedConfig := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				ImpersonateGroups: []string{
					"test8",
					"test9",
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.act-as-groups to test8,test9",
		config:         conf,
		args:           []string{"users.foo.act-as-groups", "test8,test9"},
		expected:       `Property "users.foo.act-as-groups" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigImpersonateUserExtraNew(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.act-as-user-extra.test1 to val1,val2,val3",
		config:         conf,
		args:           []string{"users.foo.act-as-user-extra.test1", "val1,val2,val3"},
		expected:       `Property "users.foo.act-as-user-extra.test1" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigImpersonateUserExtraAdd(t *testing.T) {
	conf := clientcmdapi.Config{
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
	}
	expectedConfig := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				ImpersonateUserExtra: map[string][]string{
					"test1": {
						"val1",
						"val2",
						"val3",
						"val4",
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.act-as-user-extra.test1 to val4+",
		config:         conf,
		args:           []string{"users.foo.act-as-user-extra.test1", "val4+"},
		expected:       `Property "users.foo.act-as-user-extra.test1" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigImpersonateUserExtraDelete(t *testing.T) {
	conf := clientcmdapi.Config{
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
	}
	expectedConfig := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				ImpersonateUserExtra: map[string][]string{
					"test1": {
						"val1",
						"val2",
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.act-as-user-extra.test1 to val3-",
		config:         conf,
		args:           []string{"users.foo.act-as-user-extra.test1", "val3-"},
		expected:       `Property "users.foo.act-as-user-extra.test1" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigImpersonateUserExtraUpdate(t *testing.T) {
	conf := clientcmdapi.Config{
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
	}
	expectedConfig := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
				ImpersonateUserExtra: map[string][]string{
					"test1": {
						"val5",
						"val6",
						"val7",
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.act-as-user-extra.test1 to val5,val6,val7",
		config:         conf,
		args:           []string{"users.foo.act-as-user-extra.test1", "val5,val6,val7"},
		expected:       `Property "users.foo.act-as-user-extra.test1" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigImpersonateUserExtraNewKey(t *testing.T) {
	conf := clientcmdapi.Config{
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
	}
	expectedConfig := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"foo": {
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.act-as-user-extra.test2 to val1",
		config:         conf,
		args:           []string{"users.foo.act-as-user-extra.test2", "val1"},
		expected:       `Property "users.foo.act-as-user-extra.test2" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigUsername(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.username to test1",
		config:         conf,
		args:           []string{"users.foo.username", "test1"},
		expected:       `Property "users.foo.username" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigPassword(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.password to abadpassword",
		config:         conf,
		args:           []string{"users.foo.password", "abadpassword"},
		expected:       `Property "users.foo.password" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigClientCertificate(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.client-certificate to ./file/path",
		config:         conf,
		args:           []string{"users.foo.client-certificate", "./file/path"},
		expected:       `Property "users.foo.client-certificate" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigClientCertificateData(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.client-certificate-data to not real cert data",
		config:         conf,
		args:           []string{"users.foo.client-certificate-data", "--set-raw-bytes=true", "not real cert data"},
		expected:       `Property "users.foo.client-certificate-data" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigClientKey(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.client-key to ./file/path",
		config:         conf,
		args:           []string{"users.foo.client-key", "./file/path"},
		expected:       `Property "users.foo.client-key" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigClientKeyData(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.client-key-data to not real key data",
		config:         conf,
		args:           []string{"users.foo.client-key-data", "--set-raw-bytes=true", "not real key data"},
		expected:       `Property "users.foo.client-key-data" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigToken(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.token to fake token data",
		config:         conf,
		args:           []string{"users.foo.token", "fake token data"},
		expected:       `Property "users.foo.token" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigTokenFile(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	expectedConfig := clientcmdapi.Config{
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
	}
	expectedConfig.Extensions = map[string]runtime.Object{}
	test := setConfigTest{
		description:    "Testing for kubectl config set users.foo.tokenFile to ./file/path",
		config:         conf,
		args:           []string{"users.foo.tokenFile", "./file/path"},
		expected:       `Property "users.foo.tokenFile" set.` + "\n",
		expectedConfig: expectedConfig,
	}
	test.run(t)
}

func TestSetConfigAuthProviderError(t *testing.T) {
	conf := *clientcmdapi.NewConfig()
	test := setConfigTest{
		description: "Testing for kubectl config set users.foo.auth-provider to oidc error",
		config:      conf,
		args:        []string{"users.foo.auth-provider", "test"},
		expectedErr: `unable to locate path auth-provider`,
	}
	test.run(t)
}

func (test setConfigTest) run(t *testing.T) {
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
	cmd := NewCmdConfigSet(buf, pathOptions)
	cmd.SetArgs(test.args)
	opts := &setOptions{
		configAccess:  pathOptions,
		propertyName:  test.args[0],
		propertyValue: test.args[1],
	}
	if len(test.args) > 2 {
		opts.setRawBytes = 1
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
	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
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
		t.Errorf("config want/got mismatch (-want +got):\n%s", cmp.Diff(test.expectedConfig, *config))
	}
}
