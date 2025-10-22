/*
Copyright 2025 The Kubernetes Authors.

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

package secret

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

const (
	testSecretName       = "test-secret"
	testNamespace        = "default"
	testDockerSecretName = "test-docker-secret"
	testBasicAuthName    = "test-basic-auth"
	testSSHAuthName      = "test-ssh-auth"
	testTLSName          = "test-tls"
	testImmutableName    = "test-immutable"
)

// createContext creates a context with requestInfo for testing
func createContext() context.Context {
	return genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
		Resource:   "secrets",
	})
}

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	ctx := createContext()

	testCases := map[string]struct {
		input        api.Secret
		expectedErrs field.ErrorList
	}{
		"valid opaque secret": {
			input: makeValidSecret(
				withName(testSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeOpaque),
				withData(map[string][]byte{"key": []byte("value")}),
			),
		},
		"valid dockerconfigjson secret": {
			input: makeValidSecret(
				withName(testDockerSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeDockerConfigJSON),
				withData(map[string][]byte{
					api.DockerConfigJSONKey: []byte(`{"auths":{"example.com":{"username":"user","password":"pass"}}}`),
				}),
			),
		},
		"valid basic auth secret": {
			input: makeValidSecret(
				withName(testBasicAuthName),
				withNamespace(testNamespace),
				withType(api.SecretTypeBasicAuth),
				withData(map[string][]byte{
					api.BasicAuthUsernameKey: []byte("admin"),
					api.BasicAuthPasswordKey: []byte("password"),
				}),
			),
		},
		"valid ssh auth secret": {
			input: makeValidSecret(
				withName(testSSHAuthName),
				withNamespace(testNamespace),
				withType(api.SecretTypeSSHAuth),
				withData(map[string][]byte{
					api.SSHAuthPrivateKey: []byte("-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----"),
				}),
			),
		},
		"valid tls secret": {
			input: makeValidSecret(
				withName(testTLSName),
				withNamespace(testNamespace),
				withType(api.SecretTypeTLS),
				withData(map[string][]byte{
					api.TLSCertKey:       []byte("-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----"),
					api.TLSPrivateKeyKey: []byte("-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----"),
				}),
			),
		},
		"valid immutable secret": {
			input: makeValidSecret(
				withName(testImmutableName),
				withNamespace(testNamespace),
				withType(api.SecretTypeOpaque),
				withImmutable(true),
				withData(map[string][]byte{"key": []byte("value")}),
			),
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	ctx := createContext()

	testCases := map[string]struct {
		old          api.Secret
		update       api.Secret
		expectedErrs field.ErrorList
	}{
		"no change in data - valid": {
			old: makeValidSecret(
				withName(testSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeOpaque),
				withData(map[string][]byte{"key": []byte("value")}),
			),
			update: makeValidSecret(
				withName(testSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeOpaque),
				withData(map[string][]byte{"key": []byte("value")}),
			),
		},
		"update secret data - valid": {
			old: makeValidSecret(
				withName(testSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeOpaque),
				withData(map[string][]byte{"key": []byte("old-value")}),
			),
			update: makeValidSecret(
				withName(testSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeOpaque),
				withData(map[string][]byte{"key": []byte("new-value")}),
			),
		},
		"add new key to secret - valid": {
			old: makeValidSecret(
				withName(testSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeOpaque),
				withData(map[string][]byte{"key1": []byte("value1")}),
			),
			update: makeValidSecret(
				withName(testSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeOpaque),
				withData(map[string][]byte{
					"key1": []byte("value1"),
					"key2": []byte("value2"),
				}),
			),
		},
		"make secret immutable - valid": {
			old: makeValidSecret(
				withName(testSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeOpaque),
				withData(map[string][]byte{"key": []byte("value")}),
			),
			update: makeValidSecret(
				withName(testSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeOpaque),
				withImmutable(true),
				withData(map[string][]byte{"key": []byte("value")}),
			),
		},
		"update labels - valid": {
			old: makeValidSecret(
				withName(testSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeOpaque),
				withLabels(map[string]string{"app": "v1"}),
			),
			update: makeValidSecret(
				withName(testSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeOpaque),
				withLabels(map[string]string{"app": "v2"}),
			),
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "1"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

// Helper functions to build secrets

func makeValidSecret(mutators ...func(*api.Secret)) api.Secret {
	secret := api.Secret{
		ObjectMeta: metav1.ObjectMeta{},
		Type:       api.SecretTypeOpaque,
	}
	for _, mutate := range mutators {
		mutate(&secret)
	}
	return secret
}

func withName(name string) func(*api.Secret) {
	return func(s *api.Secret) {
		s.ObjectMeta.Name = name
	}
}

func withNamespace(namespace string) func(*api.Secret) {
	return func(s *api.Secret) {
		s.ObjectMeta.Namespace = namespace
	}
}

func withType(secretType api.SecretType) func(*api.Secret) {
	return func(s *api.Secret) {
		s.Type = secretType
	}
}

func withData(data map[string][]byte) func(*api.Secret) {
	return func(s *api.Secret) {
		s.Data = data
	}
}

func withImmutable(immutable bool) func(*api.Secret) {
	return func(s *api.Secret) {
		s.Immutable = &immutable
	}
}

func withLabels(labels map[string]string) func(*api.Secret) {
	return func(s *api.Secret) {
		s.ObjectMeta.Labels = labels
	}
}
