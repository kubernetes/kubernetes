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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

var apiVersions = []string{"v1"}

const (
	testSecretName = "test-secret"
	testNamespace  = "default"
)

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidateForDeclarative(t, apiVersion)
	}
}

func testDeclarativeValidateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: apiVersion,
		Resource:   "secrets",
	})

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
		"valid tls secret": {
			input: makeValidSecret(
				withName("test-tls"),
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
				withName("test-immutable"),
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
	for _, apiVersion := range apiVersions {
		testValidateUpdateForDeclarative(t, apiVersion)
	}
}

func testValidateUpdateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: apiVersion,
		Resource:   "secrets",
	})

	testCases := map[string]struct {
		old          api.Secret
		update       api.Secret
		expectedErrs field.ErrorList
	}{
		"valid update of secret data": {
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
		"valid update to make secret immutable": {
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
		"valid update of labels": {
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
		"invalid update - type changed": {
			old: makeValidSecret(
				withName(testSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeOpaque),
				withData(map[string][]byte{"key": []byte("value")}),
			),
			update: makeValidSecret(
				withName(testSecretName),
				withNamespace(testNamespace),
				withType(api.SecretTypeTLS),
				withData(map[string][]byte{
					api.TLSCertKey:       []byte("-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----"),
					api.TLSPrivateKeyKey: []byte("-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----"),
				}),
			),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("type"), api.SecretTypeTLS, "field is immutable").WithOrigin("immutable"),
			},
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
