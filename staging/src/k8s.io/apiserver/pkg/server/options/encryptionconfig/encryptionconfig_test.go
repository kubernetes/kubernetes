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

package encryptionconfig

import (
	"bytes"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage/value"
)

const (
	sampleText = "abcdefghijklmnopqrstuvwxyz"

	sampleContextText = "0123456789"

	correctConfigWithIdentityFirst = `
kind: EncryptionConfig
apiVersion: v1
resources:
  - resources:
    - secrets
    - namespaces
    providers:
    - identity: {}
    - aes:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
`

	correctConfigWithAesFirst = `
kind: EncryptionConfig
apiVersion: v1
resources:
  - resources:
    - secrets
    providers:
    - aes:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
    - identity: {}
`

	incorrectConfigNoSecretForKey = `
kind: EncryptionConfig
apiVersion: v1
resources:
  - resources:
    - namespaces
    - secrets
    providers:
    - aes:
        keys:
        - name: key1
`

	incorrectConfigInvalidKey = `
kind: EncryptionConfig
apiVersion: v1
resources:
  - resources:
    - namespaces
    - secrets
    providers:
    - aes:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: YSBzZWNyZXQgYSBzZWNyZXQ=
`
)

func TestEncryptionProviderConfigCorrect(t *testing.T) {
	// Creates two transformers with different ordering of identity and AES transformers.
	// Transforms data using one of them, and tries to untransform using both of them.
	// Repeats this for both the possible combinations.

	identityFirstTransformerOverrides, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithIdentityFirst))
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithIdentityFirst)
	}

	aesFirstTransformerOverrides, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithAesFirst))
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithAesFirst)
	}

	// Pick the transformer for any of the returned resources.
	identityFirstTransformer := identityFirstTransformerOverrides[schema.ParseGroupResource("secrets")]
	aesFirstTransformer := aesFirstTransformerOverrides[schema.ParseGroupResource("secrets")]

	context := value.DefaultContext([]byte(sampleContextText))
	originalText := []byte(sampleText)

	testCases := []struct {
		WritingTransformer value.Transformer
		Name               string
		AesStale           bool
		IdentityStale      bool
	}{
		{aesFirstTransformer, "aesFirst", false, true},
		{identityFirstTransformer, "identityFirst", true, false},
	}

	for _, testCase := range testCases {
		transformedData, err := testCase.WritingTransformer.TransformToStorage(originalText, context)
		if err != nil {
			t.Fatalf("%s: error while transforming data to storage: %s", testCase.Name, err)
		}

		aesUntransformedData, stale, err := aesFirstTransformer.TransformFromStorage(transformedData, context)
		if err != nil {
			t.Fatalf("%s: error while reading using aesFirst transformer: %s", testCase.Name, err)
		}
		if stale != testCase.AesStale {
			t.Fatalf("%s: wrong stale information on reading using aesFirst transformer, should be %v", testCase.Name, testCase.AesStale)
		}

		identityUntransformedData, stale, err := identityFirstTransformer.TransformFromStorage(transformedData, context)
		if err != nil {
			t.Fatalf("%s: error while reading using identityFirst transformer: %s", testCase.Name, err)
		}
		if stale != testCase.IdentityStale {
			t.Fatalf("%s: wrong stale information on reading using identityFirst transformer, should be %v", testCase.Name, testCase.IdentityStale)
		}

		if bytes.Compare(aesUntransformedData, originalText) != 0 {
			t.Fatalf("%s: aesFirst transformer transformed data incorrectly. Expected: %v, got %v", testCase.Name, originalText, aesUntransformedData)
		}

		if bytes.Compare(identityUntransformedData, originalText) != 0 {
			t.Fatalf("%s: identityFirst transformer transformed data incorrectly. Expected: %v, got %v", testCase.Name, originalText, aesUntransformedData)
		}
	}
}

// Throw error if key has no secret
func TestEncryptionProviderConfigNoSecretForKey(t *testing.T) {
	if _, err := ParseEncryptionConfiguration(strings.NewReader(incorrectConfigNoSecretForKey)); err == nil {
		t.Fatalf("invalid configuration file (one key has no secret) got parsed:\n%s", incorrectConfigNoSecretForKey)
	}
}

// Throw error if invalid key for AES
func TestEncryptionProviderConfigInvalidKey(t *testing.T) {
	if _, err := ParseEncryptionConfiguration(strings.NewReader(incorrectConfigInvalidKey)); err == nil {
		t.Fatalf("invalid configuration file (bad AES key) got parsed:\n%s", incorrectConfigInvalidKey)
	}
}
