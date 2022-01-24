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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type setConfigTest struct {
	description    string
	config         clientcmdapi.Config
	args           []string
	expected       string
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
		description:    "Testing for kubectl config set users.foo.auth-provider.config.refresh-token to oidc",
		config:         conf,
		args:           []string{"users.foo.auth-provider.config.refresh-token", "test"},
		expected:       `Property "users.foo.auth-provider.config.refresh-token" set.` + "\n",
		expectedConfig: expectedConfig,
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
				t.Errorf("Failed in:%q\n cannot find key %v in either expectedConfig or config", test.description, k)
			}
		}
	}

	if len(test.expected) != 0 {
		if buf.String() != test.expected {
			t.Errorf("Failed in:%q\n expected %v\n but got %v", test.description, test.expected, buf.String())
		}
	}
	if !reflect.DeepEqual(*config, test.expectedConfig) {
		t.Errorf("Failed in: %q\n expected %#v\n but got %#v", test.description, *config, test.expectedConfig)
	}
}
