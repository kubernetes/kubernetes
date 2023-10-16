/*
Copyright 2020 The Kubernetes Authors.

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

package secrets

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/credentialprovider"
)

// fakeKeyring is a fake docker auth config keyring
type fakeKeyring struct {
	auth []credentialprovider.AuthConfig
	ok   bool
}

// Lookup implements the DockerKeyring method for fetching credentials based on image name.
// Returns fake results based on the auth and ok fields in fakeKeyring
func (f *fakeKeyring) Lookup(image string) ([]credentialprovider.AuthConfig, bool) {
	return f.auth, f.ok
}

func Test_MakeDockerKeyring(t *testing.T) {
	testcases := []struct {
		name           string
		image          string
		defaultKeyring credentialprovider.DockerKeyring
		pullSecrets    []v1.Secret
		authConfigs    []credentialprovider.AuthConfig
		found          bool
	}{
		{
			name:           "with .dockerconfigjson and auth field",
			image:          "test.registry.io",
			defaultKeyring: &fakeKeyring{},
			pullSecrets: []v1.Secret{
				{
					Type: v1.SecretTypeDockerConfigJson,
					Data: map[string][]byte{
						v1.DockerConfigJsonKey: []byte(`{"auths": {"test.registry.io": {"auth": "dXNlcjpwYXNzd29yZA=="}}}`),
					},
				},
			},
			authConfigs: []credentialprovider.AuthConfig{
				{
					Username: "user",
					Password: "password",
				},
			},
			found: true,
		},
		{
			name:           "with .dockerconfig and auth field",
			image:          "test.registry.io",
			defaultKeyring: &fakeKeyring{},
			pullSecrets: []v1.Secret{
				{
					Type: v1.SecretTypeDockercfg,
					Data: map[string][]byte{
						v1.DockerConfigKey: []byte(`{"test.registry.io": {"auth": "dXNlcjpwYXNzd29yZA=="}}`),
					},
				},
			},
			authConfigs: []credentialprovider.AuthConfig{
				{
					Username: "user",
					Password: "password",
				},
			},
			found: true,
		},
		{
			name:           "with .dockerconfigjson and username/password fields",
			image:          "test.registry.io",
			defaultKeyring: &fakeKeyring{},
			pullSecrets: []v1.Secret{
				{
					Type: v1.SecretTypeDockerConfigJson,
					Data: map[string][]byte{
						v1.DockerConfigJsonKey: []byte(`{"auths": {"test.registry.io": {"username": "user", "password": "password"}}}`),
					},
				},
			},
			authConfigs: []credentialprovider.AuthConfig{
				{
					Username: "user",
					Password: "password",
				},
			},
			found: true,
		},
		{
			name:           "with .dockerconfig and username/password fields",
			image:          "test.registry.io",
			defaultKeyring: &fakeKeyring{},
			pullSecrets: []v1.Secret{
				{
					Type: v1.SecretTypeDockercfg,
					Data: map[string][]byte{
						v1.DockerConfigKey: []byte(`{"test.registry.io": {"username": "user", "password": "password"}}`),
					},
				},
			},
			authConfigs: []credentialprovider.AuthConfig{
				{
					Username: "user",
					Password: "password",
				},
			},
			found: true,
		},
		{
			name:           "with .dockerconfigjson but with wrong Secret Type",
			image:          "test.registry.io",
			defaultKeyring: &fakeKeyring{},
			pullSecrets: []v1.Secret{
				{
					Type: v1.SecretTypeDockercfg,
					Data: map[string][]byte{
						v1.DockerConfigJsonKey: []byte(`{"auths": {"test.registry.io": {"auth": "dXNlcjpwYXNzd29yZA=="}}}`),
					},
				},
			},
			authConfigs: nil,
			found:       false,
		},
		{
			name:           "with .dockerconfig but with wrong Secret Type",
			image:          "test.registry.io",
			defaultKeyring: &fakeKeyring{},
			pullSecrets: []v1.Secret{
				{
					Type: v1.SecretTypeDockerConfigJson,
					Data: map[string][]byte{
						v1.DockerConfigKey: []byte(`{"test.registry.io": {"auth": "dXNlcjpwYXNzd29yZA=="}}`),
					},
				},
			},
			authConfigs: nil,
			found:       false,
		},
		{
			name:  "with not matching .dockerconfigjson and default keyring",
			image: "test.registry.io",
			defaultKeyring: &fakeKeyring{
				auth: []credentialprovider.AuthConfig{
					{
						Username: "default-user",
						Password: "default-password",
					},
				},
			},
			pullSecrets: []v1.Secret{
				{
					Type: v1.SecretTypeDockerConfigJson,
					Data: map[string][]byte{
						v1.DockerConfigJsonKey: []byte(`{"auths": {"foobar.io": {"auth": "dXNlcjpwYXNzd29yZA=="}}}`),
					},
				},
			},
			authConfigs: []credentialprovider.AuthConfig{
				{
					Username: "default-user",
					Password: "default-password",
				},
			},
			found: true,
		},
		{
			name:  "with not matching .dockerconfig and default keyring",
			image: "test.registry.io",
			defaultKeyring: &fakeKeyring{
				auth: []credentialprovider.AuthConfig{
					{
						Username: "default-user",
						Password: "default-password",
					},
				},
			},
			pullSecrets: []v1.Secret{
				{
					Type: v1.SecretTypeDockercfg,
					Data: map[string][]byte{
						v1.DockerConfigKey: []byte(`{"foobar.io": {"auth": "dXNlcjpwYXNzd29yZA=="}}`),
					},
				},
			},
			authConfigs: []credentialprovider.AuthConfig{
				{
					Username: "default-user",
					Password: "default-password",
				},
			},
			found: true,
		},
		{
			name:  "with no pull secrets but has default keyring",
			image: "test.registry.io",
			defaultKeyring: &fakeKeyring{
				auth: []credentialprovider.AuthConfig{
					{
						Username: "default-user",
						Password: "default-password",
					},
				},
			},
			pullSecrets: []v1.Secret{},
			authConfigs: []credentialprovider.AuthConfig{
				{
					Username: "default-user",
					Password: "default-password",
				},
			},
			found: false,
		},
		{
			name:  "with pull secrets and has default keyring",
			image: "test.registry.io",
			defaultKeyring: &fakeKeyring{
				auth: []credentialprovider.AuthConfig{
					{
						Username: "default-user",
						Password: "default-password",
					},
				},
			},
			pullSecrets: []v1.Secret{
				{
					Type: v1.SecretTypeDockerConfigJson,
					Data: map[string][]byte{
						v1.DockerConfigJsonKey: []byte(`{"auths": {"test.registry.io": {"auth": "dXNlcjpwYXNzd29yZA=="}}}`),
					},
				},
			},
			authConfigs: []credentialprovider.AuthConfig{
				{
					Username: "user",
					Password: "password",
				},
				{
					Username: "default-user",
					Password: "default-password",
				},
			},
			found: true,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			keyring, err := MakeDockerKeyring(testcase.pullSecrets, testcase.defaultKeyring)
			if err != nil {
				t.Fatalf("error creating secret-based docker keyring: %v", err)
			}

			authConfigs, found := keyring.Lookup(testcase.image)
			if found != testcase.found {
				t.Logf("actual lookup status: %v", found)
				t.Logf("expected lookup status: %v", testcase.found)
				t.Errorf("unexpected lookup for image: %s", testcase.image)
			}

			if !reflect.DeepEqual(authConfigs, testcase.authConfigs) {
				t.Logf("actual auth configs: %#v", authConfigs)
				t.Logf("expected auth configs: %#v", testcase.authConfigs)
				t.Error("auth configs did not match")
			}
		})
	}
}
