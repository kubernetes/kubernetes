/*
Copyright 2015 The Kubernetes Authors.

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

package kubectl

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
)

func TestSecretForDockerRegistryGenerate(t *testing.T) {
	username, password, email, server := "test-user", "test-password", "test-user@example.org", "https://index.docker.io/v1/"
	secretData, err := handleDockercfgContent(username, password, email, server)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	secretDataNoEmail, err := handleDockercfgContent(username, password, "", server)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	tests := map[string]struct {
		params    map[string]interface{}
		expected  *api.Secret
		expectErr bool
	}{
		"test-valid-use": {
			params: map[string]interface{}{
				"name":            "foo",
				"docker-server":   server,
				"docker-username": username,
				"docker-password": password,
				"docker-email":    email,
			},
			expected: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{
					api.DockerConfigKey: secretData,
				},
				Type: api.SecretTypeDockercfg,
			},
			expectErr: false,
		},
		"test-valid-use-append-hash": {
			params: map[string]interface{}{
				"name":            "foo",
				"docker-server":   server,
				"docker-username": username,
				"docker-password": password,
				"docker-email":    email,
				"append-hash":     true,
			},
			expected: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-94759gc65b",
				},
				Data: map[string][]byte{
					api.DockerConfigKey: secretData,
				},
				Type: api.SecretTypeDockercfg,
			},
			expectErr: false,
		},
		"test-valid-use-no-email": {
			params: map[string]interface{}{
				"name":            "foo",
				"docker-server":   server,
				"docker-username": username,
				"docker-password": password,
			},
			expected: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{
					api.DockerConfigKey: secretDataNoEmail,
				},
				Type: api.SecretTypeDockercfg,
			},
			expectErr: false,
		},
		"test-missing-required-param": {
			params: map[string]interface{}{
				"name":            "foo",
				"docker-server":   server,
				"docker-password": password,
				"docker-email":    email,
			},
			expectErr: true,
		},
	}

	generator := SecretForDockerRegistryGeneratorV1{}
	for _, test := range tests {
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*api.Secret), test.expected) {
			t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*api.Secret))
		}
	}
}
