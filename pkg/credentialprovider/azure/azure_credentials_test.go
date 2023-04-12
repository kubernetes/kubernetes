//go:build !providerless
// +build !providerless

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

package azure

import (
	"bytes"
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-05-01/containerregistry"
	"github.com/Azure/go-autorest/autorest/azure"
	"k8s.io/client-go/tools/cache"
	"k8s.io/utils/pointer"

	"github.com/stretchr/testify/assert"
)

func Test(t *testing.T) {
	configStr := `
    {
        "aadClientId": "foo",
        "aadClientSecret": "bar"
    }`
	result := []containerregistry.Registry{
		{
			Name: pointer.String("foo"),
			RegistryProperties: &containerregistry.RegistryProperties{
				LoginServer: pointer.String("*.azurecr.io"),
			},
		},
		{
			Name: pointer.String("bar"),
			RegistryProperties: &containerregistry.RegistryProperties{
				LoginServer: pointer.String("*.azurecr.cn"),
			},
		},
		{
			Name: pointer.String("baz"),
			RegistryProperties: &containerregistry.RegistryProperties{
				LoginServer: pointer.String("*.azurecr.de"),
			},
		},
		{
			Name: pointer.String("bus"),
			RegistryProperties: &containerregistry.RegistryProperties{
				LoginServer: pointer.String("*.azurecr.us"),
			},
		},
	}

	provider := &acrProvider{
		cache: cache.NewExpirationStore(stringKeyFunc, &acrExpirationPolicy{}),
	}
	provider.loadConfig(bytes.NewBufferString(configStr))

	creds := provider.Provide("foo.azurecr.io/nginx:v1")

	if len(creds) != len(result)+1 {
		t.Errorf("Unexpected list: %v, expected length %d", creds, len(result)+1)
	}
	for _, cred := range creds {
		if cred.Username != "" && cred.Username != "foo" {
			t.Errorf("expected 'foo' for username, saw: %v", cred.Username)
		}
		if cred.Password != "" && cred.Password != "bar" {
			t.Errorf("expected 'bar' for password, saw: %v", cred.Username)
		}
	}
	for _, val := range result {
		registryName := getLoginServer(val)
		if _, found := creds[registryName]; !found {
			t.Errorf("Missing expected registry: %s", registryName)
		}
	}
}

func TestProvide(t *testing.T) {
	testCases := []struct {
		desc                string
		image               string
		configStr           string
		expectedCredsLength int
	}{
		{
			desc:  "return multiple credentials using Service Principal",
			image: "foo.azurecr.io/bar/image:v1",
			configStr: `
    {
        "aadClientId": "foo",
        "aadClientSecret": "bar"
    }`,
			expectedCredsLength: 5,
		},
		{
			desc:  "retuen 0 credential for non-ACR image using Managed Identity",
			image: "busybox",
			configStr: `
    {
	"UseManagedIdentityExtension": true
    }`,
			expectedCredsLength: 0,
		},
	}

	for i, test := range testCases {
		provider := &acrProvider{
			cache: cache.NewExpirationStore(stringKeyFunc, &acrExpirationPolicy{}),
		}
		provider.loadConfig(bytes.NewBufferString(test.configStr))

		creds := provider.Provide(test.image)
		assert.Equal(t, test.expectedCredsLength, len(creds), "TestCase[%d]: %s", i, test.desc)
	}
}

func TestParseACRLoginServerFromImage(t *testing.T) {
	configStr := `
    {
        "aadClientId": "foo",
        "aadClientSecret": "bar"
    }`

	provider := &acrProvider{}
	provider.loadConfig(bytes.NewBufferString(configStr))
	provider.environment = &azure.Environment{
		ContainerRegistryDNSSuffix: ".azurecr.my.cloud",
	}
	tests := []struct {
		image    string
		expected string
	}{
		{
			image:    "invalidImage",
			expected: "",
		},
		{
			image:    "docker.io/library/busybox:latest",
			expected: "",
		},
		{
			image:    "foo.azurecr.io/bar/image:version",
			expected: "foo.azurecr.io",
		},
		{
			image:    "foo.azurecr.cn/bar/image:version",
			expected: "foo.azurecr.cn",
		},
		{
			image:    "foo.azurecr.de/bar/image:version",
			expected: "foo.azurecr.de",
		},
		{
			image:    "foo.azurecr.us/bar/image:version",
			expected: "foo.azurecr.us",
		},
		{
			image:    "foo.azurecr.my.cloud/bar/image:version",
			expected: "foo.azurecr.my.cloud",
		},
	}
	for _, test := range tests {
		if loginServer := provider.parseACRLoginServerFromImage(test.image); loginServer != test.expected {
			t.Errorf("function parseACRLoginServerFromImage returns \"%s\" for image %s, expected \"%s\"", loginServer, test.image, test.expected)
		}
	}
}
