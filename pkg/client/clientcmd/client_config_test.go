/*
Copyright 2014 Google Inc. All rights reserved.

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

package clientcmd

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
)

func createValidTestConfig() *clientcmdapi.Config {
	const (
		server = "https://anything.com:8080"
		token  = "the-token"
	)

	config := clientcmdapi.NewConfig()
	config.Clusters["clean"] = clientcmdapi.Cluster{
		Server:     server,
		APIVersion: latest.Version,
	}
	config.AuthInfos["clean"] = clientcmdapi.AuthInfo{
		Token: token,
	}
	config.Contexts["clean"] = clientcmdapi.Context{
		Cluster:  "clean",
		AuthInfo: "clean",
	}
	config.CurrentContext = "clean"

	return config
}

func TestCreateClean(t *testing.T) {
	config := createValidTestConfig()
	clientBuilder := NewNonInteractiveClientConfig(*config, "clean", &ConfigOverrides{})

	clientConfig, err := clientBuilder.ClientConfig()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	matchStringArg(config.Clusters["clean"].Server, clientConfig.Host, t)
	matchStringArg(config.Clusters["clean"].APIVersion, clientConfig.Version, t)
	matchBoolArg(config.Clusters["clean"].InsecureSkipTLSVerify, clientConfig.Insecure, t)
	matchStringArg(config.AuthInfos["clean"].Token, clientConfig.BearerToken, t)
}

func TestCreateCleanDefault(t *testing.T) {
	config := createValidTestConfig()
	clientBuilder := NewDefaultClientConfig(*config, &ConfigOverrides{})

	clientConfig, err := clientBuilder.ClientConfig()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	matchStringArg(config.Clusters["clean"].Server, clientConfig.Host, t)
	matchStringArg(config.Clusters["clean"].APIVersion, clientConfig.Version, t)
	matchBoolArg(config.Clusters["clean"].InsecureSkipTLSVerify, clientConfig.Insecure, t)
	matchStringArg(config.AuthInfos["clean"].Token, clientConfig.BearerToken, t)
}

func TestCreateMissingContext(t *testing.T) {
	const expectedErrorContains = "Context was not found for specified context"
	config := createValidTestConfig()
	clientBuilder := NewNonInteractiveClientConfig(*config, "not-present", &ConfigOverrides{})
	expectedConfig := &client.Config{Host: "http://localhost:8080"}

	clientConfig, err := clientBuilder.ClientConfig()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !reflect.DeepEqual(expectedConfig, clientConfig) {
		t.Errorf("Expected %#v, got %#v", expectedConfig, clientConfig)
	}

}

func matchBoolArg(expected, got bool, t *testing.T) {
	if expected != got {
		t.Errorf("Expected %v, got %v", expected, got)
	}
}

func matchStringArg(expected, got string, t *testing.T) {
	if expected != got {
		t.Errorf("Expected %v, got %v", expected, got)
	}
}
