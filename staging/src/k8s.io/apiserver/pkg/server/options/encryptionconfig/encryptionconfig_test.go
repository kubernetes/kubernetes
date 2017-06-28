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
	"encoding/base64"
	"fmt"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/transformhelpers"
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
    - aesgcm:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
    - aescbc:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
    - gkms:
          projectID: an-optional-project-id
          keyRing: google-kubernetes
          cryptoKey: testCryptoKey
    - secretbox:
        keys:
        - name: key1
          secret: YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY=
`

	correctConfigWithAesGcmFirst = `
kind: EncryptionConfig
apiVersion: v1
resources:
  - resources:
    - secrets
    providers:
    - aesgcm:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
    - gkms:
          projectID: an-optional-project-id
          keyRing: google-kubernetes
          cryptoKey: testCryptoKey
    - secretbox:
        keys:
        - name: key1
          secret: YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY=
    - aescbc:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
    - identity: {}
`

	correctConfigWithAesCbcFirst = `
kind: EncryptionConfig
apiVersion: v1
resources:
  - resources:
    - secrets
    providers:
    - aescbc:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
    - gkms:
          projectID: an-optional-project-id
          keyRing: google-kubernetes
          cryptoKey: testCryptoKey
    - identity: {}
    - secretbox:
        keys:
        - name: key1
          secret: YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY=
    - aesgcm:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
`

	correctConfigWithSecretboxFirst = `
kind: EncryptionConfig
apiVersion: v1
resources:
  - resources:
    - secrets
    providers:
    - secretbox:
        keys:
        - name: key1
          secret: YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY=
    - aescbc:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
    - gkms:
          projectID: an-optional-project-id
          keyRing: google-kubernetes
          cryptoKey: testCryptoKey
    - identity: {}
    - aesgcm:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
`

	correctConfigWithKMSFirst = `
kind: EncryptionConfig
apiVersion: v1
resources:
  - resources:
    - secrets
    - namespaces
    providers:
    - gkms:
          projectID: an-optional-project-id
          keyRing: google-kubernetes
          cryptoKey: testCryptoKey
    - aesgcm:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
    - identity: {}
    - aescbc:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
    - secretbox:
        keys:
        - name: key1
          secret: YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY=
`

	incorrectConfigNoSecretForKey = `
kind: EncryptionConfig
apiVersion: v1
resources:
  - resources:
    - namespaces
    - secrets
    providers:
    - aesgcm:
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
    - aesgcm:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: YSBzZWNyZXQgYSBzZWNyZXQ=
`
)

// testKMSStorage and testKMSService are mockups for testing the KMS provider
type testKMSStorage struct {
	data *map[string]string
}

func (t *testKMSStorage) Setup(_ string) error {
	_, err := t.GetAllDEKs()
	if err != nil {
		cfg := &(map[string]string{})
		t.data = cfg
	}

	return nil
}

func (t *testKMSStorage) GetAllDEKs() (map[string]string, error) {
	if t.data == nil {
		return nil, fmt.Errorf("no data stored in storage yet")
	}
	return *(t.data), nil
}

func (t *testKMSStorage) StoreNewDEK(keyvalue string) error {
	newDEKs := map[string]string{}
	for dekname, dek := range *(t.data) {
		// Remove the identifying prefix in front of the primary key.
		if strings.HasPrefix(dekname, "-") {
			dekname = dekname[1:]
		}
		newDEKs[dekname] = dek
	}
	newDEKname := transformhelpers.GenerateName(newDEKs)
	newDEKs["-"+newDEKname] = keyvalue
	t.data = &newDEKs
	return nil
}

func (t *testKMSStorage) Snapshot() *map[string]string {
	return t.data
}

func (t *testKMSStorage) RestoreSnapshot(snapshot *map[string]string) {
	t.data = snapshot
}

type testKMSService struct {
}

func (t *testKMSService) Decrypt(data string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(data)
}

func (t *testKMSService) Encrypt(data []byte) (string, error) {
	return base64.StdEncoding.EncodeToString(data), nil
}

func (t *testKMSService) GetUniqueID() string {
	return ""
}

var _ value.KMSStorage = &testKMSStorage{}
var _ value.KMSService = &testKMSService{}

func TestEncryptionProviderConfigCorrect(t *testing.T) {
	// Create a mock kmsFactory
	kmsFactory := transformhelpers.NewKMSFactoryWithStorageAndGKMS(&testKMSStorage{}, &testKMSService{})

	// Creates compound/prefix transformers with different ordering of available transformers.
	// Transforms data using one of them, and tries to untransform using the others.
	// Repeats this for all possible combinations.
	identityFirstTransformerOverrides, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithIdentityFirst), kmsFactory)
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithIdentityFirst)
	}

	aesGcmFirstTransformerOverrides, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithAesGcmFirst), kmsFactory)
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithAesGcmFirst)
	}

	aesCbcFirstTransformerOverrides, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithAesCbcFirst), kmsFactory)
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithAesCbcFirst)
	}

	secretboxFirstTransformerOverrides, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithSecretboxFirst), kmsFactory)
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithSecretboxFirst)
	}

	kmsFirstTransformerOverrides, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithKMSFirst), kmsFactory)
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithKMSFirst)
	}

	// Pick the transformer for any of the returned resources.
	identityFirstTransformer := identityFirstTransformerOverrides[schema.ParseGroupResource("secrets")]
	aesGcmFirstTransformer := aesGcmFirstTransformerOverrides[schema.ParseGroupResource("secrets")]
	aesCbcFirstTransformer := aesCbcFirstTransformerOverrides[schema.ParseGroupResource("secrets")]
	secretboxFirstTransformer := secretboxFirstTransformerOverrides[schema.ParseGroupResource("secrets")]
	kmsFirstTransformer := kmsFirstTransformerOverrides[schema.ParseGroupResource("secrets")]

	context := value.DefaultContext([]byte(sampleContextText))
	originalText := []byte(sampleText)

	transformers := []struct {
		Transformer value.Transformer
		Name        string
	}{
		{aesGcmFirstTransformer, "aesGcmFirst"},
		{aesCbcFirstTransformer, "aesCbcFirst"},
		{secretboxFirstTransformer, "secretboxFirst"},
		{identityFirstTransformer, "identityFirst"},
		{kmsFirstTransformer, "kmsFirst"},
	}

	for _, testCase := range transformers {
		transformedData, err := testCase.Transformer.TransformToStorage(originalText, context)
		if err != nil {
			t.Fatalf("%s: error while transforming data to storage: %s", testCase.Name, err)
		}

		for _, transformer := range transformers {
			untransformedData, stale, err := transformer.Transformer.TransformFromStorage(transformedData, context)
			if err != nil {
				t.Fatalf("%s: error while reading using %s transformer: %s", testCase.Name, transformer.Name, err)
			}
			if stale != (transformer.Name != testCase.Name) {
				t.Fatalf("%s: wrong stale information on reading using %s transformer, should be %v", testCase.Name, transformer.Name, testCase.Name == transformer.Name)
			}
			if bytes.Compare(untransformedData, originalText) != 0 {
				t.Fatalf("%s: %s transformer transformed data incorrectly. Expected: %v, got %v", testCase.Name, transformer.Name, originalText, untransformedData)
			}
		}
	}

}

// Throw error if key has no secret
func TestEncryptionProviderConfigNoSecretForKey(t *testing.T) {
	if _, err := ParseEncryptionConfiguration(strings.NewReader(incorrectConfigNoSecretForKey), nil); err == nil {
		t.Fatalf("invalid configuration file (one key has no secret) got parsed:\n%s", incorrectConfigNoSecretForKey)
	}
}

// Throw error if invalid key for AES
func TestEncryptionProviderConfigInvalidKey(t *testing.T) {
	if _, err := ParseEncryptionConfiguration(strings.NewReader(incorrectConfigInvalidKey), nil); err == nil {
		t.Fatalf("invalid configuration file (bad AES key) got parsed:\n%s", incorrectConfigInvalidKey)
	}
}

func TestKMSTransformerRotate(t *testing.T) {
	// Create a mock kmsFactory
	mockStorage := testKMSStorage{}
	kmsFactory := transformhelpers.NewKMSFactoryWithStorageAndGKMS(&mockStorage, &testKMSService{})

	// We create two different transformer instances, with just the storage shared.
	// Sharing the mocked cloud is insignificant, since it does not have any state.
	kmsTransformerOverrides1, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithKMSFirst), kmsFactory)
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithKMSFirst)
	}
	kmsTransformerOverrides2, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithKMSFirst), kmsFactory)
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithKMSFirst)
	}

	// Two transformers simulate 2 masters
	kmsTransformer1 := kmsTransformerOverrides1[schema.ParseGroupResource("secrets")]
	kmsTransformer2 := kmsTransformerOverrides2[schema.ParseGroupResource("secrets")]

	context := value.DefaultContext([]byte(sampleContextText))
	originalText := []byte(sampleText)

	// Sanity check that rotating does not break encryption at any point.
	transformedData, err := kmsTransformer1.TransformToStorage(originalText, context)
	if err != nil {
		t.Fatalf("error while transforming data to storage using kmsTransformer1: %v", err)
	}
	err = kmsTransformer1.Rotate()
	if err != nil {
		t.Fatalf("error while rotating key: %v", err)
	}
	untransformedData, _, err := kmsTransformer2.TransformFromStorage(transformedData, context)
	if err != nil {
		t.Fatalf("error while transforming data from storage using kmsTransformer2 after rotation: %v", err)
	}
	if fmt.Sprintf("%v", untransformedData) != fmt.Sprintf("%v", originalText) {
		t.Fatalf("untransformed data (\"%s\") did not match original text (\"%s\")", untransformedData, originalText)
	}

	// Check that if the new key had not propagated to the other master, decrypting new data fails.
	snapshotBeforeRotation := mockStorage.Snapshot()
	err = kmsTransformer1.Rotate()
	if err != nil {
		t.Fatalf("error while rotating key: %v", err)
	}
	transformedData, err = kmsTransformer1.TransformToStorage(originalText, context)
	if err != nil {
		t.Fatalf("error while transforming data to storage using kmsTransformer1: %v", err)
	}
	mockStorage.RestoreSnapshot(snapshotBeforeRotation)
	untransformedData, _, err = kmsTransformer2.TransformFromStorage(transformedData, context)
	if err == nil {
		t.Fatalf("a transformer without the newly created key should not have been able to decrypt data encrypted with new key")
	}
}
