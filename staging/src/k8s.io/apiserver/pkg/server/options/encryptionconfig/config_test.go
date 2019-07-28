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
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	apiserverconfig "k8s.io/apiserver/pkg/apis/config"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope"
)

const (
	sampleText = "abcdefghijklmnopqrstuvwxyz"

	sampleContextText = "0123456789"

	legacyV1Config = `
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
      - kms:
          name: testprovider
          endpoint: unix:///tmp/testprovider.sock
          cachesize: 10
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

	correctConfigWithIdentityFirst = `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
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
    - kms:
        name: testprovider
        endpoint: unix:///tmp/testprovider.sock
        cachesize: 10
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

	correctConfigWithAesGcmFirst = `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
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
    - secretbox:
        keys:
        - name: key1
          secret: YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY=
    - kms:
        name: testprovider
        endpoint: unix:///tmp/testprovider.sock
        cachesize: 10
    - aescbc:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
    - identity: {}
`

	correctConfigWithAesCbcFirst = `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
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
    - kms:
        name: testprovider
        endpoint: unix:///tmp/testprovider.sock
        cachesize: 10
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
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
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
    - kms:
        name: testprovider
        endpoint: unix:///tmp/testprovider.sock
        cachesize: 10
    - identity: {}
    - aesgcm:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
`

	correctConfigWithKMSFirst = `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - kms:
        name: testprovider
        endpoint: unix:///tmp/testprovider.sock
        cachesize: 10
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
    - aesgcm:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
        - name: key2
          secret: dGhpcyBpcyBwYXNzd29yZA==
`

	incorrectConfigNoSecretForKey = `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
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
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
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

	incorrectConfigNoEndpointForKMS = `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - kms:
        name: testprovider
        cachesize: 10
`
)

// testEnvelopeService is a mock envelope service which can be used to simulate remote Envelope services
// for testing of the envelope transformer with other transformers.
type testEnvelopeService struct {
}

func (t *testEnvelopeService) Decrypt(data []byte) ([]byte, error) {
	return base64.StdEncoding.DecodeString(string(data))
}

func (t *testEnvelopeService) Encrypt(data []byte) ([]byte, error) {
	return []byte(base64.StdEncoding.EncodeToString(data)), nil
}

// The factory method to create mock envelope service.
func newMockEnvelopeService(endpoint string, timeout time.Duration) (envelope.Service, error) {
	return &testEnvelopeService{}, nil
}

func TestLegacyConfig(t *testing.T) {
	legacyConfigObject, err := loadConfig([]byte(legacyV1Config))
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, legacyV1Config)
	}

	expected := &apiserverconfig.EncryptionConfiguration{
		Resources: []apiserverconfig.ResourceConfiguration{
			{
				Resources: []string{"secrets", "namespaces"},
				Providers: []apiserverconfig.ProviderConfiguration{
					{Identity: &apiserverconfig.IdentityConfiguration{}},
					{AESGCM: &apiserverconfig.AESConfiguration{
						Keys: []apiserverconfig.Key{
							{Name: "key1", Secret: "c2VjcmV0IGlzIHNlY3VyZQ=="},
							{Name: "key2", Secret: "dGhpcyBpcyBwYXNzd29yZA=="},
						},
					}},
					{KMS: &apiserverconfig.KMSConfiguration{
						Name:      "testprovider",
						Endpoint:  "unix:///tmp/testprovider.sock",
						CacheSize: 10,
					}},
					{AESCBC: &apiserverconfig.AESConfiguration{
						Keys: []apiserverconfig.Key{
							{Name: "key1", Secret: "c2VjcmV0IGlzIHNlY3VyZQ=="},
							{Name: "key2", Secret: "dGhpcyBpcyBwYXNzd29yZA=="},
						},
					}},
					{Secretbox: &apiserverconfig.SecretboxConfiguration{
						Keys: []apiserverconfig.Key{
							{Name: "key1", Secret: "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="},
						},
					}},
				},
			},
		},
	}
	if !reflect.DeepEqual(legacyConfigObject, expected) {
		t.Fatal(diff.ObjectReflectDiff(expected, legacyConfigObject))
	}
}
func TestEncryptionProviderConfigCorrect(t *testing.T) {
	// Set factory for mock envelope service
	factory := envelopeServiceFactory
	envelopeServiceFactory = newMockEnvelopeService
	defer func() {
		envelopeServiceFactory = factory
	}()

	// Creates compound/prefix transformers with different ordering of available transformers.
	// Transforms data using one of them, and tries to untransform using the others.
	// Repeats this for all possible combinations.
	identityFirstTransformerOverrides, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithIdentityFirst))
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithIdentityFirst)
	}

	aesGcmFirstTransformerOverrides, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithAesGcmFirst))
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithAesGcmFirst)
	}

	aesCbcFirstTransformerOverrides, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithAesCbcFirst))
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithAesCbcFirst)
	}

	secretboxFirstTransformerOverrides, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithSecretboxFirst))
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithSecretboxFirst)
	}

	kmsFirstTransformerOverrides, err := ParseEncryptionConfiguration(strings.NewReader(correctConfigWithKMSFirst))
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

// Throw error if kms has no endpoint
func TestEncryptionProviderConfigNoEndpointForKMS(t *testing.T) {
	if _, err := ParseEncryptionConfiguration(strings.NewReader(incorrectConfigNoEndpointForKMS)); err == nil {
		t.Fatalf("invalid configuration file (kms has no endpoint) got parsed:\n%s", incorrectConfigNoEndpointForKMS)
	}
}

func TestKMSConfigTimeout(t *testing.T) {
	testCases := []struct {
		desc    string
		config  string
		want    time.Duration
		wantErr string
	}{
		{
			desc: "duration explicitly provided",
			config: `kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - kms:
        name: foo
        endpoint: unix:///tmp/testprovider.sock
        timeout:   15s
`,
			want: 15 * time.Second,
		},
		{
			desc: "duration explicitly provided as 0 which is an invalid value, error should be returned",
			config: `kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - kms:
        name: foo
        endpoint: unix:///tmp/testprovider.sock
        timeout:   0s
`,
			wantErr: "timeout should be a positive value",
		},
		{
			desc: "duration is not provided, default will be supplied",
			config: `kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - kms:
        name: foo
        endpoint: unix:///tmp/testprovider.sock
`,
			want: kmsPluginConnectionTimeout,
		},
		{
			desc: "duration is invalid (negative), error should be returned",
			config: `kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - kms:
        name: foo
        endpoint: unix:///tmp/testprovider.sock
        timeout:   -15s

`,
			wantErr: "timeout should be a positive value",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			// mocking envelopeServiceFactory to sense the value of the supplied timeout.
			envelopeServiceFactory = func(endpoint string, callTimeout time.Duration) (envelope.Service, error) {
				if callTimeout != tt.want {
					t.Fatalf("got timeout: %v, want %v", callTimeout, tt.want)
				}

				return newMockEnvelopeService(endpoint, callTimeout)
			}

			// mocked envelopeServiceFactory is called during ParseEncryptionConfiguration.
			if _, err := ParseEncryptionConfiguration(strings.NewReader(tt.config)); err != nil && !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("unable to parse yaml\n%s\nerror: %v", tt.config, err)
			}
		})
	}
}

func TestKMSPluginHealthz(t *testing.T) {
	service, err := envelope.NewGRPCService("unix:///tmp/testprovider.sock", kmsPluginConnectionTimeout)
	if err != nil {
		t.Fatalf("Could not initialize envelopeService, error: %v", err)
	}

	testCases := []struct {
		desc    string
		config  string
		want    []*kmsPluginProbe
		wantErr bool
	}{
		{
			desc: "Install Healthz",
			config: `kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - kms:
        name: foo
        endpoint: unix:///tmp/testprovider.sock
        timeout:   15s
`,
			want: []*kmsPluginProbe{
				{
					name:    "foo",
					Service: service,
				},
			},
		},
		{
			desc: "Install multiple healthz",
			config: `kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - kms:
        name: foo
        endpoint: unix:///tmp/testprovider.sock
        timeout:   15s
    - kms:
        name: bar
        endpoint: unix:///tmp/testprovider.sock
        timeout:   15s
`,
			want: []*kmsPluginProbe{
				{
					name:    "foo",
					Service: service,
				},
				{
					name:    "bar",
					Service: service,
				},
			},
		},
		{
			desc: "No KMS Providers",
			config: `kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - aesgcm:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
`,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			got, err := getKMSPluginProbes(strings.NewReader(tt.config))
			if err != nil && !tt.wantErr {
				t.Fatalf("got %v, want nil for error", err)
			}

			if d := cmp.Diff(tt.want, got, cmp.Comparer(serviceComparer)); d != "" {
				t.Fatalf("HealthzConfig mismatch (-want +got):\n%s", d)
			}
		})
	}
}

// As long as got and want contain envelope.Service we will return true.
// If got has an envelope.Service and want does note (or vice versa) this will return false.
func serviceComparer(_, _ envelope.Service) bool {
	return true
}
