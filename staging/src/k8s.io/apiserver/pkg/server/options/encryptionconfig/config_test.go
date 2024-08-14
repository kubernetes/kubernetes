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
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope"
	envelopekmsv2 "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2"
	kmstypes "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2/v2"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/metrics"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	kmsservice "k8s.io/kms/pkg/service"
	"k8s.io/utils/pointer"
)

const (
	sampleText        = "abcdefghijklmnopqrstuvwxyz"
	sampleContextText = "0123456789"
)

var (
	sampleInvalidKeyID = string(make([]byte, envelopekmsv2.KeyIDMaxSize+1))
)

// testEnvelopeService is a mock envelope service which can be used to simulate remote Envelope services
// for testing of the envelope transformer with other transformers.
type testEnvelopeService struct {
	err error
}

func (t *testEnvelopeService) Decrypt(data []byte) ([]byte, error) {
	if t.err != nil {
		return nil, t.err
	}
	return base64.StdEncoding.DecodeString(string(data))
}

func (t *testEnvelopeService) Encrypt(data []byte) ([]byte, error) {
	if t.err != nil {
		return nil, t.err
	}
	return []byte(base64.StdEncoding.EncodeToString(data)), nil
}

// testKMSv2EnvelopeService is a mock kmsv2 envelope service which can be used to simulate remote Envelope v2 services
// for testing of the envelope transformer with other transformers.
type testKMSv2EnvelopeService struct {
	err                error
	keyID              string
	encryptCalls       int
	encryptAnnotations map[string][]byte
}

func (t *testKMSv2EnvelopeService) Decrypt(ctx context.Context, uid string, req *kmsservice.DecryptRequest) ([]byte, error) {
	if t.err != nil {
		return nil, t.err
	}
	return base64.StdEncoding.DecodeString(string(req.Ciphertext))
}

func (t *testKMSv2EnvelopeService) Encrypt(ctx context.Context, uid string, data []byte) (*kmsservice.EncryptResponse, error) {
	t.encryptCalls++
	if t.err != nil {
		return nil, t.err
	}
	return &kmsservice.EncryptResponse{
		Ciphertext:  []byte(base64.StdEncoding.EncodeToString(data)),
		KeyID:       t.keyID,
		Annotations: t.encryptAnnotations,
	}, nil
}

func (t *testKMSv2EnvelopeService) Status(ctx context.Context) (*kmsservice.StatusResponse, error) {
	if t.err != nil {
		return nil, t.err
	}
	return &kmsservice.StatusResponse{Healthz: "ok", KeyID: t.keyID, Version: "v2beta1"}, nil
}

// The factory method to create mock envelope service.
func newMockEnvelopeService(ctx context.Context, endpoint string, timeout time.Duration) (envelope.Service, error) {
	return &testEnvelopeService{nil}, nil
}

// The factory method to create mock envelope service which always returns error.
func newMockErrorEnvelopeService(endpoint string, timeout time.Duration) (envelope.Service, error) {
	return &testEnvelopeService{errors.New("test")}, nil
}

// The factory method to create mock envelope kmsv2 service.
func newMockEnvelopeKMSv2Service(ctx context.Context, endpoint, providerName string, timeout time.Duration) (kmsservice.Service, error) {
	return &testKMSv2EnvelopeService{nil, "1", 0, nil}, nil
}

// The factory method to create mock envelope kmsv2 service which always returns error.
func newMockErrorEnvelopeKMSv2Service(endpoint string, timeout time.Duration) (kmsservice.Service, error) {
	return &testKMSv2EnvelopeService{errors.New("test"), "1", 0, nil}, nil
}

// The factory method to create mock envelope kmsv2 service that always returns invalid keyID.
func newMockInvalidKeyIDEnvelopeKMSv2Service(ctx context.Context, endpoint string, timeout time.Duration, keyID string) (kmsservice.Service, error) {
	return &testKMSv2EnvelopeService{nil, keyID, 0, nil}, nil
}

func TestLegacyConfig(t *testing.T) {
	legacyV1Config := "testdata/valid-configs/legacy.yaml"
	legacyConfigObject, _, err := loadConfig(legacyV1Config, false)
	cacheSize := int32(10)
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, legacyV1Config)
	}

	expected := &apiserver.EncryptionConfiguration{
		Resources: []apiserver.ResourceConfiguration{
			{
				Resources: []string{"secrets", "namespaces"},
				Providers: []apiserver.ProviderConfiguration{
					{Identity: &apiserver.IdentityConfiguration{}},
					{AESGCM: &apiserver.AESConfiguration{
						Keys: []apiserver.Key{
							{Name: "key1", Secret: "c2VjcmV0IGlzIHNlY3VyZQ=="},
							{Name: "key2", Secret: "dGhpcyBpcyBwYXNzd29yZA=="},
						},
					}},
					{KMS: &apiserver.KMSConfiguration{
						APIVersion: "v1",
						Name:       "testprovider",
						Endpoint:   "unix:///tmp/testprovider.sock",
						CacheSize:  &cacheSize,
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
					}},
					{AESCBC: &apiserver.AESConfiguration{
						Keys: []apiserver.Key{
							{Name: "key1", Secret: "c2VjcmV0IGlzIHNlY3VyZQ=="},
							{Name: "key2", Secret: "dGhpcyBpcyBwYXNzd29yZA=="},
						},
					}},
					{Secretbox: &apiserver.SecretboxConfiguration{
						Keys: []apiserver.Key{
							{Name: "key1", Secret: "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="},
						},
					}},
				},
			},
		},
	}
	if d := cmp.Diff(expected, legacyConfigObject); d != "" {
		t.Fatalf("EncryptionConfig mismatch (-want +got):\n%s", d)
	}
}

func TestEncryptionProviderConfigCorrect(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv1, true)

	// Set factory for mock envelope service
	factory := envelopeServiceFactory
	factoryKMSv2 := EnvelopeKMSv2ServiceFactory
	envelopeServiceFactory = newMockEnvelopeService
	EnvelopeKMSv2ServiceFactory = newMockEnvelopeKMSv2Service
	defer func() {
		envelopeServiceFactory = factory
		EnvelopeKMSv2ServiceFactory = factoryKMSv2
	}()

	ctx := testContext(t)

	// Creates compound/prefix transformers with different ordering of available transformers.
	// Transforms data using one of them, and tries to untransform using the others.
	// Repeats this for all possible combinations.
	// Math for GracePeriod is explained at - https://github.com/kubernetes/kubernetes/blob/c9ed04762f94a319d7b1fb718dc345491a32bea6/staging/src/k8s.io/apiserver/pkg/server/options/encryptionconfig/config.go#L159-L163
	expectedKMSCloseGracePeriod := 46 * time.Second
	correctConfigWithIdentityFirst := "testdata/valid-configs/identity-first.yaml"
	identityFirstEncryptionConfiguration, err := LoadEncryptionConfig(ctx, correctConfigWithIdentityFirst, false, "")
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithIdentityFirst)
	}
	if identityFirstEncryptionConfiguration.KMSCloseGracePeriod != expectedKMSCloseGracePeriod {
		t.Fatalf("KMSCloseGracePeriod mismatch (-want +got):\n%s", cmp.Diff(expectedKMSCloseGracePeriod, identityFirstEncryptionConfiguration.KMSCloseGracePeriod))
	}

	// Math for GracePeriod is explained at - https://github.com/kubernetes/kubernetes/blob/c9ed04762f94a319d7b1fb718dc345491a32bea6/staging/src/k8s.io/apiserver/pkg/server/options/encryptionconfig/config.go#L159-L163
	expectedKMSCloseGracePeriod = 32 * time.Second
	correctConfigWithAesGcmFirst := "testdata/valid-configs/aes-gcm-first.yaml"
	aesGcmFirstEncryptionConfiguration, err := LoadEncryptionConfig(ctx, correctConfigWithAesGcmFirst, false, "")
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithAesGcmFirst)
	}
	if aesGcmFirstEncryptionConfiguration.KMSCloseGracePeriod != expectedKMSCloseGracePeriod {
		t.Fatalf("KMSCloseGracePeriod mismatch (-want +got):\n%s", cmp.Diff(expectedKMSCloseGracePeriod, aesGcmFirstEncryptionConfiguration.KMSCloseGracePeriod))
	}

	invalidConfigWithAesGcm := "testdata/invalid-configs/invalid-aes-gcm.yaml"
	_, err = LoadEncryptionConfig(ctx, invalidConfigWithAesGcm, false, "")
	if !strings.Contains(errString(err), "error while parsing file") {
		t.Fatalf("should result in error while parsing configuration file: %s.\nThe file was:\n%s", err, invalidConfigWithAesGcm)
	}

	invalidConfigWithTypo := "testdata/invalid-configs/invalid-typo.yaml"
	_, err = LoadEncryptionConfig(ctx, invalidConfigWithTypo, false, "")
	if got, wantSubString := errString(err), `strict decoding error: unknown field "resources[0].providers[3].kms.pandas"`; !strings.Contains(got, wantSubString) {
		t.Fatalf("should result in strict decode error while parsing configuration file %q:\ngot: %q\nwant substring: %q", invalidConfigWithTypo, got, wantSubString)
	}

	// Math for GracePeriod is explained at - https://github.com/kubernetes/kubernetes/blob/c9ed04762f94a319d7b1fb718dc345491a32bea6/staging/src/k8s.io/apiserver/pkg/server/options/encryptionconfig/config.go#L159-L163
	expectedKMSCloseGracePeriod = 26 * time.Second
	correctConfigWithAesCbcFirst := "testdata/valid-configs/aes-cbc-first.yaml"
	aesCbcFirstEncryptionConfiguration, err := LoadEncryptionConfig(ctx, correctConfigWithAesCbcFirst, false, "")
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithAesCbcFirst)
	}
	if aesCbcFirstEncryptionConfiguration.KMSCloseGracePeriod != expectedKMSCloseGracePeriod {
		t.Fatalf("KMSCloseGracePeriod mismatch (-want +got):\n%s", cmp.Diff(expectedKMSCloseGracePeriod, aesCbcFirstEncryptionConfiguration.KMSCloseGracePeriod))
	}

	// Math for GracePeriod is explained at - https://github.com/kubernetes/kubernetes/blob/c9ed04762f94a319d7b1fb718dc345491a32bea6/staging/src/k8s.io/apiserver/pkg/server/options/encryptionconfig/config.go#L159-L163
	expectedKMSCloseGracePeriod = 14 * time.Second
	correctConfigWithSecretboxFirst := "testdata/valid-configs/secret-box-first.yaml"
	secretboxFirstEncryptionConfiguration, err := LoadEncryptionConfig(ctx, correctConfigWithSecretboxFirst, false, "")
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithSecretboxFirst)
	}
	if secretboxFirstEncryptionConfiguration.KMSCloseGracePeriod != expectedKMSCloseGracePeriod {
		t.Fatalf("KMSCloseGracePeriod mismatch (-want +got):\n%s", cmp.Diff(expectedKMSCloseGracePeriod, secretboxFirstEncryptionConfiguration.KMSCloseGracePeriod))
	}

	// Math for GracePeriod is explained at - https://github.com/kubernetes/kubernetes/blob/c9ed04762f94a319d7b1fb718dc345491a32bea6/staging/src/k8s.io/apiserver/pkg/server/options/encryptionconfig/config.go#L159-L163
	expectedKMSCloseGracePeriod = 34 * time.Second
	correctConfigWithKMSFirst := "testdata/valid-configs/kms-first.yaml"
	kmsFirstEncryptionConfiguration, err := LoadEncryptionConfig(ctx, correctConfigWithKMSFirst, false, "")
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithKMSFirst)
	}
	if kmsFirstEncryptionConfiguration.KMSCloseGracePeriod != expectedKMSCloseGracePeriod {
		t.Fatalf("KMSCloseGracePeriod mismatch (-want +got):\n%s", cmp.Diff(expectedKMSCloseGracePeriod, kmsFirstEncryptionConfiguration.KMSCloseGracePeriod))
	}

	// Math for GracePeriod is explained at - https://github.com/kubernetes/kubernetes/blob/c9ed04762f94a319d7b1fb718dc345491a32bea6/staging/src/k8s.io/apiserver/pkg/server/options/encryptionconfig/config.go#L159-L163
	expectedKMSCloseGracePeriod = 42 * time.Second
	correctConfigWithKMSv2First := "testdata/valid-configs/kmsv2-first.yaml"
	kmsv2FirstEncryptionConfiguration, err := LoadEncryptionConfig(ctx, correctConfigWithKMSv2First, false, "")
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithKMSv2First)
	}
	if kmsv2FirstEncryptionConfiguration.KMSCloseGracePeriod != expectedKMSCloseGracePeriod {
		t.Fatalf("KMSCloseGracePeriod mismatch (-want +got):\n%s", cmp.Diff(expectedKMSCloseGracePeriod, kmsv2FirstEncryptionConfiguration.KMSCloseGracePeriod))
	}

	// Pick the transformer for any of the returned resources.
	identityFirstTransformer := identityFirstEncryptionConfiguration.Transformers[schema.ParseGroupResource("secrets")]
	aesGcmFirstTransformer := aesGcmFirstEncryptionConfiguration.Transformers[schema.ParseGroupResource("secrets")]
	aesCbcFirstTransformer := aesCbcFirstEncryptionConfiguration.Transformers[schema.ParseGroupResource("secrets")]
	secretboxFirstTransformer := secretboxFirstEncryptionConfiguration.Transformers[schema.ParseGroupResource("secrets")]
	kmsFirstTransformer := kmsFirstEncryptionConfiguration.Transformers[schema.ParseGroupResource("secrets")]
	kmsv2FirstTransformer := kmsv2FirstEncryptionConfiguration.Transformers[schema.ParseGroupResource("secrets")]

	dataCtx := value.DefaultContext(sampleContextText)
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
		{kmsv2FirstTransformer, "kmvs2First"},
	}

	for _, testCase := range transformers {
		transformedData, err := testCase.Transformer.TransformToStorage(ctx, originalText, dataCtx)
		if err != nil {
			t.Fatalf("%s: error while transforming data to storage: %s", testCase.Name, err)
		}

		for _, transformer := range transformers {
			untransformedData, stale, err := transformer.Transformer.TransformFromStorage(ctx, transformedData, dataCtx)
			if err != nil {
				t.Fatalf("%s: error while reading using %s transformer: %s", testCase.Name, transformer.Name, err)
			}
			if stale != (transformer.Name != testCase.Name) {
				t.Fatalf("%s: wrong stale information on reading using %s transformer, should be %v", testCase.Name, transformer.Name, testCase.Name == transformer.Name)
			}
			if !bytes.Equal(untransformedData, originalText) {
				t.Fatalf("%s: %s transformer transformed data incorrectly. Expected: %v, got %v", testCase.Name, transformer.Name, originalText, untransformedData)
			}
		}
	}
}

func TestKMSv1Deprecation(t *testing.T) {
	testCases := []struct {
		name         string
		kmsv1Enabled bool
		expectedErr  string
	}{
		{
			name:         "config with kmsv1, KMSv1=false",
			kmsv1Enabled: false,
			expectedErr:  "KMSv1 is deprecated and will only receive security updates going forward. Use KMSv2 instead.  Set --feature-gates=KMSv1=true to use the deprecated KMSv1 feature.",
		},
		{
			name:         "config with kmsv1, KMSv1=true",
			kmsv1Enabled: true,
			expectedErr:  "",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv1, testCase.kmsv1Enabled)

			kmsv1Config := "testdata/valid-configs/kms/multiple-providers.yaml"
			_, err := LoadEncryptionConfig(testContext(t), kmsv1Config, false, "")
			if !strings.Contains(errString(err), testCase.expectedErr) {
				t.Fatalf("expected error %q, got %q", testCase.expectedErr, errString(err))
			}
		})
	}
}

func TestKMSvsEnablement(t *testing.T) {
	testCases := []struct {
		name        string
		filePath    string
		expectedErr string
	}{
		{
			name:        "config with kmsv2 and kmsv1, KMSv2=true, KMSv1=false, should fail when feature is disabled",
			filePath:    "testdata/valid-configs/kms/multiple-providers-mixed.yaml",
			expectedErr: "KMSv1 is deprecated and will only receive security updates going forward. Use KMSv2 instead",
		},
		{
			name:        "config with kmsv2, KMSv2=true, KMSv1=false",
			filePath:    "testdata/valid-configs/kms/multiple-providers-kmsv2.yaml",
			expectedErr: "",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			// only the KMSv2 feature flag is enabled
			_, err := LoadEncryptionConfig(testContext(t), testCase.filePath, false, "")

			if len(testCase.expectedErr) > 0 && !strings.Contains(errString(err), testCase.expectedErr) {
				t.Fatalf("expected error %q, got %q", testCase.expectedErr, errString(err))
			}
			if len(testCase.expectedErr) == 0 && err != nil {
				t.Fatalf("unexpected error %q", errString(err))
			}

		})
	}
	tts := []struct {
		name            string
		kmsv2Enabled    bool
		expectedErr     string
		expectedTimeout time.Duration
		config          apiserver.EncryptionConfiguration
		wantV2Used      bool
	}{
		{
			name:         "with kmsv1 and kmsv2, KMSv2=true",
			kmsv2Enabled: true,
			config: apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{"secrets"},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout: &metav1.Duration{
										Duration: 1 * time.Second,
									},
									Endpoint:  "unix:///tmp/testprovider.sock",
									CacheSize: pointer.Int32(1000),
								},
							},
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "another-kms",
									APIVersion: "v2",
									Timeout: &metav1.Duration{
										Duration: 1 * time.Second,
									},
									Endpoint:  "unix:///tmp/anothertestprovider.sock",
									CacheSize: pointer.Int32(1000),
								},
							},
						},
					},
				},
			},
			expectedErr: "",
			wantV2Used:  true,
		},
	}

	for _, tt := range tts {
		t.Run(tt.name, func(t *testing.T) {
			// Just testing KMSv2 feature flag
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv1, true)

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2, tt.kmsv2Enabled)

			ctx, cancel := context.WithCancel(context.Background())
			cancel() // cancel this upfront so the kms v2 checks do not block

			_, _, kmsUsed, err := getTransformerOverridesAndKMSPluginHealthzCheckers(ctx, &tt.config, "")
			if err == nil {
				if kmsUsed == nil || kmsUsed.v2Used != tt.wantV2Used {
					t.Fatalf("unexpected kmsUsed value, expected: %v, got: %v", tt.wantV2Used, kmsUsed)
				}

			}
			if !strings.Contains(errString(err), tt.expectedErr) {
				t.Fatalf("expecting error calling prefixTransformersAndProbes, expected: %s, got: %s", tt.expectedErr, errString(err))
			}
		})
	}
}

func TestKMSMaxTimeout(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv1, true)

	testCases := []struct {
		name            string
		expectedErr     string
		expectedTimeout time.Duration
		config          apiserver.EncryptionConfiguration
	}{
		{
			name: "config with bad provider",
			config: apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{"secrets"},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: nil,
							},
						},
					},
				},
			},
			expectedErr:     "provider does not contain any of the expected providers: KMS, AESGCM, AESCBC, Secretbox, Identity",
			expectedTimeout: 6 * time.Second,
		},
		{
			name: "default timeout",
			config: apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{"secrets"},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout: &metav1.Duration{
										// default timeout is 3s
										// this will be set automatically if not provided in config file
										Duration: 3 * time.Second,
									},
									Endpoint: "unix:///tmp/testprovider.sock",
								},
							},
						},
					},
				},
			},
			expectedErr:     "",
			expectedTimeout: 6 * time.Second,
		},
		{
			name: "with v1 provider",
			config: apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{"secrets"},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout: &metav1.Duration{
										// default timeout is 3s
										// this will be set automatically if not provided in config file
										Duration: 3 * time.Second,
									},
									Endpoint: "unix:///tmp/testprovider.sock",
								},
							},
						},
					},
					{
						Resources: []string{"configmaps"},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout: &metav1.Duration{
										// default timeout is 3s
										// this will be set automatically if not provided in config file
										Duration: 3 * time.Second,
									},
									Endpoint: "unix:///tmp/testprovider.sock",
								},
							},
						},
					},
				},
			},
			expectedErr:     "",
			expectedTimeout: 12 * time.Second,
		},
		{
			name: "with v2 provider",
			config: apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{"secrets"},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v2",
									Timeout: &metav1.Duration{
										Duration: 15 * time.Second,
									},
									Endpoint: "unix:///tmp/testprovider.sock",
								},
							},
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "new-kms",
									APIVersion: "v2",
									Timeout: &metav1.Duration{
										Duration: 5 * time.Second,
									},
									Endpoint: "unix:///tmp/anothertestprovider.sock",
								},
							},
						},
					},
					{
						Resources: []string{"configmaps"},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "another-kms",
									APIVersion: "v2",
									Timeout: &metav1.Duration{
										Duration: 10 * time.Second,
									},
									Endpoint: "unix:///tmp/testprovider.sock",
								},
							},
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "yet-another-kms",
									APIVersion: "v2",
									Timeout: &metav1.Duration{
										Duration: 2 * time.Second,
									},
									Endpoint: "unix:///tmp/anothertestprovider.sock",
								},
							},
						},
					},
				},
			},
			expectedErr:     "",
			expectedTimeout: 32 * time.Second,
		},
		{
			name: "with v1 and v2 provider",
			config: apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{"secrets"},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout: &metav1.Duration{
										Duration: 1 * time.Second,
									},
									Endpoint: "unix:///tmp/testprovider.sock",
								},
							},
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "another-kms",
									APIVersion: "v2",
									Timeout: &metav1.Duration{
										Duration: 1 * time.Second,
									},
									Endpoint: "unix:///tmp/anothertestprovider.sock",
								},
							},
						},
					},
					{
						Resources: []string{"configmaps"},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout: &metav1.Duration{
										Duration: 4 * time.Second,
									},
									Endpoint: "unix:///tmp/testprovider.sock",
								},
							},
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "yet-another-kms",
									APIVersion: "v1",
									Timeout: &metav1.Duration{
										Duration: 2 * time.Second,
									},
									Endpoint: "unix:///tmp/anothertestprovider.sock",
								},
							},
						},
					},
				},
			},
			expectedErr:     "",
			expectedTimeout: 15 * time.Second,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			cacheSize := int32(1000)
			for _, resource := range testCase.config.Resources {
				for _, provider := range resource.Providers {
					if provider.KMS != nil {
						provider.KMS.CacheSize = &cacheSize
					}
				}
			}

			ctx, cancel := context.WithCancel(context.Background())
			cancel() // cancel this upfront so the kms v2 checks do not block

			_, _, kmsUsed, err := getTransformerOverridesAndKMSPluginHealthzCheckers(ctx, &testCase.config, "")

			if !strings.Contains(errString(err), testCase.expectedErr) {
				t.Fatalf("expecting error calling prefixTransformersAndProbes, expected: %s, got: %s", testCase.expectedErr, errString(err))
			}
			if len(testCase.expectedErr) == 0 {
				if kmsUsed == nil {
					t.Fatal("kmsUsed should not be nil")
				}

				if kmsUsed.kmsTimeoutSum != testCase.expectedTimeout {
					t.Fatalf("expected timeout %v, got %v", testCase.expectedTimeout, kmsUsed.kmsTimeoutSum)
				}
			}

		})
	}
}

func TestKMSPluginHealthz(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv1, true)

	kmsv2Probe := &kmsv2PluginProbe{
		name:        "foo",
		ttl:         3 * time.Second,
		apiServerID: "",
	}
	keyID := "1"
	kmsv2Probe.state.Store(&envelopekmsv2.State{EncryptedObject: kmstypes.EncryptedObject{KeyID: keyID}})

	testCases := []struct {
		desc    string
		config  string
		want    []healthChecker
		wantErr string
		kmsv2   bool
		kmsv1   bool
	}{
		{
			desc:    "Invalid config file path",
			config:  "invalid/path",
			want:    nil,
			wantErr: `error reading encryption provider configuration file "invalid/path"`,
		},
		{
			desc:    "Empty config file content",
			config:  "testdata/invalid-configs/kms/invalid-content.yaml",
			want:    nil,
			wantErr: `encryption provider configuration file "testdata/invalid-configs/kms/invalid-content.yaml" is empty`,
		},
		{
			desc:    "Unable to decode",
			config:  "testdata/invalid-configs/kms/invalid-gvk.yaml",
			want:    nil,
			wantErr: `error decoding encryption provider configuration file`,
		},
		{
			desc:    "Unexpected config type",
			config:  "testdata/invalid-configs/kms/invalid-config-type.yaml",
			want:    nil,
			wantErr: `no kind "EncryptionConfigurations" is registered for version "apiserver.config.k8s.io/v1"`,
		},
		{
			desc:   "Install Healthz",
			config: "testdata/valid-configs/kms/default-timeout.yaml",
			want: []healthChecker{
				&kmsPluginProbe{
					name: "foo",
					ttl:  3 * time.Second,
				},
			},
			kmsv1: true,
		},
		{
			desc:   "Install multiple healthz",
			config: "testdata/valid-configs/kms/multiple-providers.yaml",
			want: []healthChecker{
				&kmsPluginProbe{
					name: "foo",
					ttl:  3 * time.Second,
				},
				&kmsPluginProbe{
					name: "bar",
					ttl:  3 * time.Second,
				},
			},
			kmsv1: true,
		},
		{
			desc:   "No KMS Providers",
			config: "testdata/valid-configs/aes/aes-gcm.yaml",
		},
		{
			desc:   "Install multiple healthz with v1 and v2",
			config: "testdata/valid-configs/kms/multiple-providers-mixed.yaml",
			want: []healthChecker{
				kmsv2Probe,
				&kmsPluginProbe{
					name: "bar",
					ttl:  3 * time.Second,
				},
			},
			kmsv2: true,
			kmsv1: true,
		},
		{
			desc:    "Invalid API version",
			config:  "testdata/invalid-configs/kms/invalid-apiversion.yaml",
			want:    nil,
			wantErr: `resources[0].providers[0].kms.apiVersion: Invalid value: "v3": unsupported apiVersion apiVersion for KMS provider, only v1 and v2 are supported`,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			config, _, err := loadConfig(tt.config, false)
			if errStr := errString(err); !strings.Contains(errStr, tt.wantErr) {
				t.Fatalf("unexpected error state got=%s want=%s", errStr, tt.wantErr)
			}
			if len(tt.wantErr) > 0 {
				return
			}

			ctx, cancel := context.WithCancel(context.Background())
			cancel() // cancel this upfront so the kms v2 healthz check poll does not run
			_, got, kmsUsed, err := getTransformerOverridesAndKMSPluginProbes(ctx, config, "")
			if err != nil {
				t.Fatal(err)
			}

			// unset fields that are not relevant to the test
			for i := range got {
				checker := got[i]
				switch p := checker.(type) {
				case *kmsPluginProbe:
					p.service = nil
					p.l = nil
					p.lastResponse = nil
				case *kmsv2PluginProbe:
					p.service = nil
					p.l = nil
					p.lastResponse = nil
					p.state.Store(kmsv2Probe.state.Load())
				default:
					t.Fatalf("unexpected probe type %T", p)
				}
			}

			if tt.kmsv2 != kmsUsed.v2Used {
				t.Errorf("incorrect kms v2 detection: want=%v got=%v", tt.kmsv2, kmsUsed.v2Used)
			}
			if tt.kmsv1 != kmsUsed.v1Used {
				t.Errorf("incorrect kms v1 detection: want=%v got=%v", tt.kmsv1, kmsUsed.v1Used)
			}

			if d := cmp.Diff(tt.want, got,
				cmp.Comparer(func(a, b *kmsPluginProbe) bool {
					return *a == *b
				}),
				cmp.Comparer(func(a, b *kmsv2PluginProbe) bool {
					return *a == *b
				}),
			); d != "" {
				t.Fatalf("HealthzConfig mismatch (-want +got):\n%s", d)
			}
		})
	}
}

// tests for masking rules
func TestWildcardMasking(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv1, true)

	testCases := []struct {
		desc          string
		config        *apiserver.EncryptionConfiguration
		expectedError string
	}{
		{
			desc: "resources masked by *. group",
			config: &apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{
							"configmaps",
							"*.",
							"secrets",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
				},
			},
			expectedError: "resource \"secrets\" is masked by earlier rule \"*.\"",
		},
		{
			desc: "*. masked by *. group",
			config: &apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{
							"*.",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
					{
						Resources: []string{
							"*.",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms2",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
				},
			},
			expectedError: "resource \"*.\" is masked by earlier rule \"*.\"",
		},
		{
			desc: "*.foo masked by *.foo",
			config: &apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{
							"*.foo",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
					{
						Resources: []string{
							"*.foo",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms2",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
				},
			},
			expectedError: "resource \"*.foo\" is masked by earlier rule \"*.foo\"",
		},
		{
			desc: "*.* masked by *.*",
			config: &apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{
							"*.*",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
					{
						Resources: []string{
							"*.*",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms2",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
				},
			},
			expectedError: "resource \"*.*\" is masked by earlier rule \"*.*\"",
		},
		{
			desc: "resources masked by *. group in multiple configurations",
			config: &apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{
							"configmaps",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
					{
						Resources: []string{
							"*.",
							"secrets",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "another-kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/another-testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
				},
			},
			expectedError: "resource \"secrets\" is masked by earlier rule \"*.\"",
		},
		{
			desc: "resources masked by *.*",
			config: &apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{
							"configmaps",
							"*.*",
							"secrets",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
				},
			},
			expectedError: "resource \"secrets\" is masked by earlier rule \"*.*\"",
		},
		{
			desc: "resources masked by *.* in multiple configurations",
			config: &apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{
							"configmaps",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
					{
						Resources: []string{
							"*.*",
							"secrets",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "another-kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/another-testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
				},
			},
			expectedError: "resource \"secrets\" is masked by earlier rule \"*.*\"",
		},
		{
			desc: "resources *. masked by *.*",
			config: &apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{
							"configmaps",
							"*.*",
							"*.",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
				},
			},
			expectedError: "resource \"*.\" is masked by earlier rule \"*.*\"",
		},
		{
			desc: "resources *. masked by *.* in multiple configurations",
			config: &apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{
							"configmaps",
							"*.*",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
					{
						Resources: []string{
							"*.",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "another-kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/another-testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
				},
			},
			expectedError: "resource \"*.\" is masked by earlier rule \"*.*\"",
		},
		{
			desc: "resources not masked by any rule",
			config: &apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{
							"configmaps",
							"secrets",
							"*.*",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
				},
			},
		},
		{
			desc: "resources not masked by any rule in multiple configurations",
			config: &apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{
							"configmaps",
							"secrets",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
					{
						Resources: []string{
							"*.*",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "another-kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/another-testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)

			_, _, _, err := getTransformerOverridesAndKMSPluginProbes(ctx, tc.config, "")
			if errString(err) != tc.expectedError {
				t.Errorf("expected error %s but got %s", tc.expectedError, errString(err))
			}
		})
	}
}

func TestWildcardStructure(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv1, true)
	testCases := []struct {
		desc                         string
		expectedResourceTransformers map[string]string
		config                       *apiserver.EncryptionConfiguration
		errorValue                   string
	}{
		{
			desc: "should not result in error",
			expectedResourceTransformers: map[string]string{
				"configmaps":       "k8s:enc:kms:v1:kms:",
				"secrets":          "k8s:enc:kms:v1:another-kms:",
				"events":           "k8s:enc:kms:v1:fancy:",
				"deployments.apps": "k8s:enc:kms:v1:kms:",
				"pods":             "k8s:enc:kms:v1:fancy:",
				"pandas":           "k8s:enc:kms:v1:fancy:",
				"pandas.bears":     "k8s:enc:kms:v1:yet-another-provider:",
				"jobs.apps":        "k8s:enc:kms:v1:kms:",
			},

			errorValue: "",
			config: &apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{
							"configmaps",
							"*.apps",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
					{
						Resources: []string{
							"secrets",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "another-kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
							{
								Identity: &apiserver.IdentityConfiguration{},
							},
						},
					},
					{
						Resources: []string{
							"*.",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "fancy",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
					{
						Resources: []string{
							"*.*",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "yet-another-provider",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
				},
			},
		},
		{
			desc:       "should result in error",
			errorValue: "resource \"secrets\" is masked by earlier rule \"*.\"",
			config: &apiserver.EncryptionConfiguration{
				Resources: []apiserver.ResourceConfiguration{
					{
						Resources: []string{
							"configmaps",
							"*.",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
						},
					},
					{
						Resources: []string{
							"*.*",
							"secrets",
						},
						Providers: []apiserver.ProviderConfiguration{
							{
								KMS: &apiserver.KMSConfiguration{
									Name:       "kms",
									APIVersion: "v1",
									Timeout:    &metav1.Duration{Duration: 3 * time.Second},
									Endpoint:   "unix:///tmp/testprovider.sock",
									CacheSize:  pointer.Int32(10),
								},
							},
							{
								Identity: &apiserver.IdentityConfiguration{},
							},
						},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)

			transformers, _, _, err := getTransformerOverridesAndKMSPluginProbes(ctx, tc.config, "")
			if errString(err) != tc.errorValue {
				t.Errorf("expected error %s but got %s", tc.errorValue, errString(err))
			}

			if len(tc.errorValue) > 0 {
				return
			}

			// check if expectedResourceTransformers are present
			for resource, expectedTransformerName := range tc.expectedResourceTransformers {
				transformer := transformerFromOverrides(transformers, schema.ParseGroupResource(resource))
				transformerName := string(
					reflect.ValueOf(transformer).
						Elem().
						FieldByName("delegate").
						Elem().
						Elem().
						FieldByName("transformers").
						Index(0).
						FieldByName("Prefix").
						Bytes(),
				)

				if transformerName != expectedTransformerName {
					t.Errorf("resource %s: expected same transformer name but got %v", resource, cmp.Diff(transformerName, expectedTransformerName))
				}
			}
		})
	}
}

func TestKMSPluginHealthzTTL(t *testing.T) {
	ctx := testContext(t)

	service, _ := newMockEnvelopeService(ctx, "unix:///tmp/testprovider.sock", 3*time.Second)
	errService, _ := newMockErrorEnvelopeService("unix:///tmp/testprovider.sock", 3*time.Second)

	testCases := []struct {
		desc    string
		probe   *kmsPluginProbe
		wantTTL time.Duration
	}{
		{
			desc: "kms provider in good state",
			probe: &kmsPluginProbe{
				name:         "test",
				ttl:          kmsPluginHealthzNegativeTTL,
				service:      service,
				l:            &sync.Mutex{},
				lastResponse: &kmsPluginHealthzResponse{},
			},
			wantTTL: kmsPluginHealthzPositiveTTL,
		},
		{
			desc: "kms provider in bad state",
			probe: &kmsPluginProbe{
				name:         "test",
				ttl:          kmsPluginHealthzPositiveTTL,
				service:      errService,
				l:            &sync.Mutex{},
				lastResponse: &kmsPluginHealthzResponse{},
			},
			wantTTL: kmsPluginHealthzNegativeTTL,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			_ = tt.probe.check()
			if tt.probe.ttl != tt.wantTTL {
				t.Fatalf("want ttl %v, got ttl %v", tt.wantTTL, tt.probe.ttl)
			}
		})
	}
}

func TestKMSv2PluginHealthzTTL(t *testing.T) {
	ctx := testContext(t)

	service, _ := newMockEnvelopeKMSv2Service(ctx, "unix:///tmp/testprovider.sock", "providerName", 3*time.Second)
	errService, _ := newMockErrorEnvelopeKMSv2Service("unix:///tmp/testprovider.sock", 3*time.Second)

	testCases := []struct {
		desc    string
		probe   *kmsv2PluginProbe
		wantTTL time.Duration
	}{
		{
			desc: "kmsv2 provider in good state",
			probe: &kmsv2PluginProbe{
				name:         "test",
				ttl:          kmsPluginHealthzNegativeTTL,
				service:      service,
				l:            &sync.Mutex{},
				lastResponse: &kmsPluginHealthzResponse{},
			},
			wantTTL: kmsPluginHealthzPositiveTTL,
		},
		{
			desc: "kmsv2 provider in bad state",
			probe: &kmsv2PluginProbe{
				name:         "test",
				ttl:          kmsPluginHealthzPositiveTTL,
				service:      errService,
				l:            &sync.Mutex{},
				lastResponse: &kmsPluginHealthzResponse{},
			},
			wantTTL: kmsPluginHealthzNegativeTTL,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			tt.probe.state.Store(&envelopekmsv2.State{})
			_ = tt.probe.check(ctx)
			if tt.probe.ttl != tt.wantTTL {
				t.Fatalf("want ttl %v, got ttl %v", tt.wantTTL, tt.probe.ttl)
			}
		})
	}
}

func TestKMSv2InvalidKeyID(t *testing.T) {
	ctx := testContext(t)
	invalidKeyIDService, _ := newMockInvalidKeyIDEnvelopeKMSv2Service(ctx, "unix:///tmp/testprovider.sock", 3*time.Second, "")
	invalidLongKeyIDService, _ := newMockInvalidKeyIDEnvelopeKMSv2Service(ctx, "unix:///tmp/testprovider.sock", 3*time.Second, sampleInvalidKeyID)
	service, _ := newMockInvalidKeyIDEnvelopeKMSv2Service(ctx, "unix:///tmp/testprovider.sock", 3*time.Second, "1")

	testCases := []struct {
		desc    string
		probe   *kmsv2PluginProbe
		metrics []string
		want    string
	}{
		{
			desc: "kmsv2 provider returns an invalid empty keyID",
			probe: &kmsv2PluginProbe{
				name:         "test",
				ttl:          kmsPluginHealthzNegativeTTL,
				service:      invalidKeyIDService,
				l:            &sync.Mutex{},
				lastResponse: &kmsPluginHealthzResponse{},
			},
			metrics: []string{
				"apiserver_envelope_encryption_invalid_key_id_from_status_total",
			},
			want: `
			# HELP apiserver_envelope_encryption_invalid_key_id_from_status_total [ALPHA] Number of times an invalid keyID is returned by the Status RPC call split by error.
			# TYPE apiserver_envelope_encryption_invalid_key_id_from_status_total counter
			apiserver_envelope_encryption_invalid_key_id_from_status_total{error="empty",provider_name="test"} 1
			`,
		},
		{
			desc: "kmsv2 provider returns a valid keyID",
			probe: &kmsv2PluginProbe{
				name:         "test",
				ttl:          kmsPluginHealthzNegativeTTL,
				service:      service,
				l:            &sync.Mutex{},
				lastResponse: &kmsPluginHealthzResponse{},
			},
			metrics: []string{
				"apiserver_envelope_encryption_invalid_key_id_from_status_total",
			},
			want: ``,
		},
		{
			desc: "kmsv2 provider returns an invalid long keyID",
			probe: &kmsv2PluginProbe{
				name:         "test",
				ttl:          kmsPluginHealthzNegativeTTL,
				service:      invalidLongKeyIDService,
				l:            &sync.Mutex{},
				lastResponse: &kmsPluginHealthzResponse{},
			},
			metrics: []string{
				"apiserver_envelope_encryption_invalid_key_id_from_status_total",
			},
			want: `
			# HELP apiserver_envelope_encryption_invalid_key_id_from_status_total [ALPHA] Number of times an invalid keyID is returned by the Status RPC call split by error.
			# TYPE apiserver_envelope_encryption_invalid_key_id_from_status_total counter
			apiserver_envelope_encryption_invalid_key_id_from_status_total{error="too_long",provider_name="test"} 1
			`,
		},
	}

	metrics.InvalidKeyIDFromStatusTotal.Reset()
	metrics.RegisterMetrics()

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			defer metrics.InvalidKeyIDFromStatusTotal.Reset()
			tt.probe.state.Store(&envelopekmsv2.State{})
			_ = tt.probe.check(ctx)
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), tt.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCBCKeyRotationWithOverlappingProviders(t *testing.T) {
	testCBCKeyRotationWithProviders(
		t,
		"testdata/valid-configs/aes/aes-cbc-multiple-providers.json",
		"k8s:enc:aescbc:v1:1:",
		"testdata/valid-configs/aes/aes-cbc-multiple-providers-reversed.json",
		"k8s:enc:aescbc:v1:2:",
	)
}

func TestCBCKeyRotationWithoutOverlappingProviders(t *testing.T) {
	testCBCKeyRotationWithProviders(
		t,
		"testdata/valid-configs/aes/aes-cbc-multiple-keys.json",
		"k8s:enc:aescbc:v1:A:",
		"testdata/valid-configs/aes/aes-cbc-multiple-keys-reversed.json",
		"k8s:enc:aescbc:v1:B:",
	)
}

func testCBCKeyRotationWithProviders(t *testing.T, firstEncryptionConfig, firstPrefix, secondEncryptionConfig, secondPrefix string) {
	p := getTransformerFromEncryptionConfig(t, firstEncryptionConfig)

	ctx := testContext(t)
	dataCtx := value.DefaultContext("authenticated_data")

	out, err := p.TransformToStorage(ctx, []byte("firstvalue"), dataCtx)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(out, []byte(firstPrefix)) {
		t.Fatalf("unexpected prefix: %q", out)
	}
	from, stale, err := p.TransformFromStorage(ctx, out, dataCtx)
	if err != nil {
		t.Fatal(err)
	}
	if stale || !bytes.Equal([]byte("firstvalue"), from) {
		t.Fatalf("unexpected data: %t %q", stale, from)
	}

	// verify changing the context fails storage
	_, _, err = p.TransformFromStorage(ctx, out, value.DefaultContext("incorrect_context"))
	if err != nil {
		t.Fatalf("CBC mode does not support authentication: %v", err)
	}

	// reverse the order, use the second key
	p = getTransformerFromEncryptionConfig(t, secondEncryptionConfig)
	from, stale, err = p.TransformFromStorage(ctx, out, dataCtx)
	if err != nil {
		t.Fatal(err)
	}
	if !stale || !bytes.Equal([]byte("firstvalue"), from) {
		t.Fatalf("unexpected data: %t %q", stale, from)
	}

	out, err = p.TransformToStorage(ctx, []byte("firstvalue"), dataCtx)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(out, []byte(secondPrefix)) {
		t.Fatalf("unexpected prefix: %q", out)
	}
	from, stale, err = p.TransformFromStorage(ctx, out, dataCtx)
	if err != nil {
		t.Fatal(err)
	}
	if stale || !bytes.Equal([]byte("firstvalue"), from) {
		t.Fatalf("unexpected data: %t %q", stale, from)
	}
}

func getTransformerFromEncryptionConfig(t *testing.T, encryptionConfigPath string) value.Transformer {
	ctx := testContext(t)

	t.Helper()
	encryptionConfiguration, err := LoadEncryptionConfig(ctx, encryptionConfigPath, false, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(encryptionConfiguration.Transformers) != 1 {
		t.Fatalf("input config does not have exactly one resource: %s", encryptionConfigPath)
	}
	for _, transformer := range encryptionConfiguration.Transformers {
		return transformer
	}
	panic("unreachable")
}

func TestIsKMSv2ProviderHealthyError(t *testing.T) {
	probe := &kmsv2PluginProbe{name: "testplugin"}

	testCases := []struct {
		desc           string
		expectedErr    string
		wantMetrics    string
		statusResponse *kmsservice.StatusResponse
	}{
		{
			desc: "healthz status is not ok",
			statusResponse: &kmsservice.StatusResponse{
				Healthz: "unhealthy",
			},
			expectedErr: "got unexpected healthz status: unhealthy, expected KMSv2 API version v2, got , got invalid KMSv2 KeyID ",
			wantMetrics: `
			# HELP apiserver_envelope_encryption_invalid_key_id_from_status_total [ALPHA] Number of times an invalid keyID is returned by the Status RPC call split by error.
			# TYPE apiserver_envelope_encryption_invalid_key_id_from_status_total counter
			apiserver_envelope_encryption_invalid_key_id_from_status_total{error="empty",provider_name="testplugin"} 1
			`,
		},
		{
			desc: "version is not v2",
			statusResponse: &kmsservice.StatusResponse{
				Version: "v1beta1",
			},
			expectedErr: "got unexpected healthz status: , expected KMSv2 API version v2, got v1beta1, got invalid KMSv2 KeyID ",
			wantMetrics: `
			# HELP apiserver_envelope_encryption_invalid_key_id_from_status_total [ALPHA] Number of times an invalid keyID is returned by the Status RPC call split by error.
			# TYPE apiserver_envelope_encryption_invalid_key_id_from_status_total counter
			apiserver_envelope_encryption_invalid_key_id_from_status_total{error="empty",provider_name="testplugin"} 1
			`,
		},
		{
			desc: "missing keyID",
			statusResponse: &kmsservice.StatusResponse{
				Healthz: "ok",
				Version: "v2beta1",
			},
			expectedErr: "got invalid KMSv2 KeyID ",
			wantMetrics: `
			# HELP apiserver_envelope_encryption_invalid_key_id_from_status_total [ALPHA] Number of times an invalid keyID is returned by the Status RPC call split by error.
			# TYPE apiserver_envelope_encryption_invalid_key_id_from_status_total counter
			apiserver_envelope_encryption_invalid_key_id_from_status_total{error="empty",provider_name="testplugin"} 1
			`,
		},
		{
			desc: "invalid long keyID",
			statusResponse: &kmsservice.StatusResponse{
				Healthz: "ok",
				Version: "v2",
				KeyID:   sampleInvalidKeyID,
			},
			expectedErr: "got invalid KMSv2 KeyID ",
			wantMetrics: `
			# HELP apiserver_envelope_encryption_invalid_key_id_from_status_total [ALPHA] Number of times an invalid keyID is returned by the Status RPC call split by error.
			# TYPE apiserver_envelope_encryption_invalid_key_id_from_status_total counter
			apiserver_envelope_encryption_invalid_key_id_from_status_total{error="too_long",provider_name="testplugin"} 1
			`,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			metrics.InvalidKeyIDFromStatusTotal.Reset()
			err := probe.isKMSv2ProviderHealthyAndMaybeRotateDEK(testContext(t), tt.statusResponse)
			if !strings.Contains(errString(err), tt.expectedErr) {
				t.Errorf("expected err %q, got %q", tt.expectedErr, errString(err))
			}
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.wantMetrics),
				"apiserver_envelope_encryption_invalid_key_id_from_status_total",
			); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// test to ensure KMSv2 API version is not changed after the first status response
func TestKMSv2SameVersionFromStatus(t *testing.T) {
	probe := &kmsv2PluginProbe{name: "testplugin"}
	service, _ := newMockEnvelopeKMSv2Service(testContext(t), "unix:///tmp/testprovider.sock", "providerName", 3*time.Second)
	probe.l = &sync.Mutex{}
	probe.state.Store(&envelopekmsv2.State{})
	probe.service = service

	testCases := []struct {
		desc        string
		expectedErr string
		newVersion  string
	}{
		{
			desc:        "version changed",
			newVersion:  "v2",
			expectedErr: "KMSv2 API version should not change",
		},
		{
			desc:        "version unchanged",
			newVersion:  "v2beta1",
			expectedErr: "",
		},
	}
	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			statusResponse := &kmsservice.StatusResponse{
				Healthz: "ok",
				Version: "v2beta1",
				KeyID:   "1",
			}
			if err := probe.isKMSv2ProviderHealthyAndMaybeRotateDEK(testContext(t), statusResponse); err != nil {
				t.Fatal(err)
			}
			statusResponse.Version = tt.newVersion
			err := probe.isKMSv2ProviderHealthyAndMaybeRotateDEK(testContext(t), statusResponse)
			if len(tt.expectedErr) > 0 && !strings.Contains(errString(err), tt.expectedErr) {
				t.Errorf("expected err %q, got %q", tt.expectedErr, errString(err))
			}
			if len(tt.expectedErr) == 0 && err != nil {
				t.Fatal(err)
			}
		})
	}
}

func testContext(t *testing.T) context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	return ctx
}

func errString(err error) string {
	if err == nil {
		return ""
	}

	return err.Error()
}

func TestComputeEncryptionConfigHash(t *testing.T) {
	// hash the empty string to be sure that sha256 is being used
	expect := "k8s:enc:unstable:1:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	sum := computeEncryptionConfigHash([]byte(""))
	if expect != sum {
		t.Errorf("expected hash %q but got %q", expect, sum)
	}
}

func Test_kmsv2PluginProbe_rotateDEKOnKeyIDChange(t *testing.T) {
	defaultUseSeed := GetKDF()

	origNowFunc := envelopekmsv2.NowFunc
	now := origNowFunc() // freeze time
	t.Cleanup(func() { envelopekmsv2.NowFunc = origNowFunc })
	envelopekmsv2.NowFunc = func() time.Time { return now }

	klog.LogToStderr(false)
	var level klog.Level
	if err := level.Set("6"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		klog.LogToStderr(true)
		if err := level.Set("0"); err != nil {
			t.Fatal(err)
		}
		klog.SetOutput(io.Discard)
	})

	tests := []struct {
		name             string
		service          *testKMSv2EnvelopeService
		state            envelopekmsv2.State
		useSeed          bool
		statusKeyID      string
		wantState        envelopekmsv2.State
		wantEncryptCalls int
		wantLogs         []string
		wantErr          string
	}{
		{
			name:        "happy path, no previous state",
			service:     &testKMSv2EnvelopeService{keyID: "1"},
			state:       envelopekmsv2.State{},
			statusKeyID: "1",
			wantState: envelopekmsv2.State{
				EncryptedObject:     kmstypes.EncryptedObject{KeyID: "1"},
				ExpirationTimestamp: now.Add(3 * time.Minute),
			},
			wantEncryptCalls: 1,
			wantLogs: []string{
				`"encrypting content using envelope service" uid="panda"`,
				fmt.Sprintf(`"successfully rotated DEK" uid="panda" useSeed=false newKeyIDHash="sha256:6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b" oldKeyIDHash="" expirationTimestamp="%s"`,
					now.Add(3*time.Minute).Format(time.RFC3339)),
			},
			wantErr: "",
		},
		{
			name:        "happy path, with previous state",
			service:     &testKMSv2EnvelopeService{err: fmt.Errorf("broken")}, // not called
			state:       validState(t, "2", now, false),
			statusKeyID: "2",
			wantState: envelopekmsv2.State{
				EncryptedObject:     kmstypes.EncryptedObject{KeyID: "2"},
				ExpirationTimestamp: now.Add(3 * time.Minute),
			},
			wantEncryptCalls: 0,
			wantLogs:         nil,
			wantErr:          "",
		},
		{
			name:        "happy path, with previous state, useSeed=true",
			service:     &testKMSv2EnvelopeService{keyID: "2"},
			state:       validState(t, "2", now, false),
			useSeed:     true,
			statusKeyID: "2",
			wantState: envelopekmsv2.State{
				EncryptedObject:     kmstypes.EncryptedObject{KeyID: "2", EncryptedDEKSourceType: kmstypes.EncryptedDEKSourceType_HKDF_SHA256_XNONCE_AES_GCM_SEED},
				ExpirationTimestamp: now.Add(3 * time.Minute),
			},
			wantEncryptCalls: 1,
			wantLogs: []string{
				`"encrypting content using envelope service" uid="panda"`,
				fmt.Sprintf(`"successfully rotated DEK" uid="panda" useSeed=true newKeyIDHash="sha256:d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35" oldKeyIDHash="sha256:d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35" expirationTimestamp="%s"`,
					now.Add(3*time.Minute).Format(time.RFC3339)),
			},
			wantErr: "",
		},
		{
			name:        "happy path, with previous useSeed=true state",
			service:     &testKMSv2EnvelopeService{keyID: "2"},
			state:       validState(t, "2", now, true),
			statusKeyID: "2",
			wantState: envelopekmsv2.State{
				EncryptedObject:     kmstypes.EncryptedObject{KeyID: "2"},
				ExpirationTimestamp: now.Add(3 * time.Minute),
			},
			wantEncryptCalls: 1,
			wantLogs: []string{
				`"encrypting content using envelope service" uid="panda"`,
				fmt.Sprintf(`"successfully rotated DEK" uid="panda" useSeed=false newKeyIDHash="sha256:d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35" oldKeyIDHash="sha256:d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35" expirationTimestamp="%s"`,
					now.Add(3*time.Minute).Format(time.RFC3339)),
			},
			wantErr: "",
		},
		{
			name:        "happy path, with previous useSeed=true state, useSeed=true",
			service:     &testKMSv2EnvelopeService{keyID: "2"},
			state:       validState(t, "2", now, true),
			useSeed:     true,
			statusKeyID: "2",
			wantState: envelopekmsv2.State{
				EncryptedObject:     kmstypes.EncryptedObject{KeyID: "2", EncryptedDEKSourceType: kmstypes.EncryptedDEKSourceType_HKDF_SHA256_XNONCE_AES_GCM_SEED},
				ExpirationTimestamp: now.Add(3 * time.Minute),
			},
			wantEncryptCalls: 0,
			wantLogs:         nil,
			wantErr:          "",
		},
		{
			name:        "happy path, with previous state, useSeed=default",
			service:     &testKMSv2EnvelopeService{keyID: "2"},
			state:       validState(t, "2", now, false),
			useSeed:     defaultUseSeed,
			statusKeyID: "2",
			wantState: envelopekmsv2.State{
				EncryptedObject:     kmstypes.EncryptedObject{KeyID: "2", EncryptedDEKSourceType: kmstypes.EncryptedDEKSourceType_HKDF_SHA256_XNONCE_AES_GCM_SEED},
				ExpirationTimestamp: now.Add(3 * time.Minute),
			},
			wantEncryptCalls: 1,
			wantLogs: []string{
				`"encrypting content using envelope service" uid="panda"`,
				fmt.Sprintf(`"successfully rotated DEK" uid="panda" useSeed=true newKeyIDHash="sha256:d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35" oldKeyIDHash="sha256:d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35" expirationTimestamp="%s"`,
					now.Add(3*time.Minute).Format(time.RFC3339)),
			},
			wantErr: "",
		},
		{
			name:        "happy path, with previous useSeed=true state, useSeed=default",
			service:     &testKMSv2EnvelopeService{keyID: "2"},
			state:       validState(t, "2", now, true),
			useSeed:     defaultUseSeed,
			statusKeyID: "2",
			wantState: envelopekmsv2.State{
				EncryptedObject:     kmstypes.EncryptedObject{KeyID: "2", EncryptedDEKSourceType: kmstypes.EncryptedDEKSourceType_HKDF_SHA256_XNONCE_AES_GCM_SEED},
				ExpirationTimestamp: now.Add(3 * time.Minute),
			},
			wantEncryptCalls: 0,
			wantLogs:         nil,
			wantErr:          "",
		},
		{
			name:        "previous state expired but key ID matches",
			service:     &testKMSv2EnvelopeService{err: fmt.Errorf("broken")}, // not called
			state:       validState(t, "3", now.Add(-time.Hour), false),
			statusKeyID: "3",
			wantState: envelopekmsv2.State{
				EncryptedObject:     kmstypes.EncryptedObject{KeyID: "3"},
				ExpirationTimestamp: now.Add(3 * time.Minute),
			},
			wantEncryptCalls: 0,
			wantLogs:         nil,
			wantErr:          "",
		},
		{
			name:        "previous state expired but key ID does not match",
			service:     &testKMSv2EnvelopeService{keyID: "4"},
			state:       validState(t, "3", now.Add(-time.Hour), false),
			statusKeyID: "4",
			wantState: envelopekmsv2.State{
				EncryptedObject:     kmstypes.EncryptedObject{KeyID: "4"},
				ExpirationTimestamp: now.Add(3 * time.Minute),
			},
			wantEncryptCalls: 1,
			wantLogs: []string{
				`"encrypting content using envelope service" uid="panda"`,
				fmt.Sprintf(`"successfully rotated DEK" uid="panda" useSeed=false newKeyIDHash="sha256:4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a" oldKeyIDHash="sha256:4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce" expirationTimestamp="%s"`,
					now.Add(3*time.Minute).Format(time.RFC3339)),
			},
			wantErr: "",
		},
		{
			name:        "service down but key ID does not match",
			service:     &testKMSv2EnvelopeService{err: fmt.Errorf("broken")},
			state:       validState(t, "4", now.Add(7*time.Minute), false),
			statusKeyID: "5",
			wantState: envelopekmsv2.State{
				EncryptedObject:     kmstypes.EncryptedObject{KeyID: "4"},
				ExpirationTimestamp: now.Add(7 * time.Minute),
			},
			wantEncryptCalls: 1,
			wantLogs: []string{
				`"encrypting content using envelope service" uid="panda"`,
			},
			wantErr: `failed to rotate DEK uid="panda", useSeed=false, ` +
				`errState=<nil>, errGen=failed to encrypt DEK, error: broken, statusKeyIDHash="sha256:ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d", ` +
				`encryptKeyIDHash="", stateKeyIDHash="sha256:4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a", expirationTimestamp=` + now.Add(7*time.Minute).Format(time.RFC3339),
		},
		{
			name:             "invalid service response, no previous state",
			service:          &testKMSv2EnvelopeService{keyID: "1", encryptAnnotations: map[string][]byte{"panda": nil}},
			state:            envelopekmsv2.State{},
			statusKeyID:      "1",
			wantState:        envelopekmsv2.State{},
			wantEncryptCalls: 1,
			wantLogs: []string{
				`"encrypting content using envelope service" uid="panda"`,
			},
			wantErr: `failed to rotate DEK uid="panda", useSeed=false, ` +
				`errState=got unexpected nil transformer, errGen=failed to validate annotations: annotations: Invalid value: "panda": ` +
				`should be a domain with at least two segments separated by dots, statusKeyIDHash="sha256:6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b", ` +
				`encryptKeyIDHash="", stateKeyIDHash="", expirationTimestamp=` + (time.Time{}).Format(time.RFC3339),
		},
		{
			name:        "invalid service response, with previous state",
			service:     &testKMSv2EnvelopeService{keyID: "3", encryptAnnotations: map[string][]byte{"panda": nil}},
			state:       validState(t, "2", now, false),
			statusKeyID: "3",
			wantState: envelopekmsv2.State{
				EncryptedObject:     kmstypes.EncryptedObject{KeyID: "2"},
				ExpirationTimestamp: now,
			},
			wantEncryptCalls: 1,
			wantLogs: []string{
				`"encrypting content using envelope service" uid="panda"`,
			},
			wantErr: `failed to rotate DEK uid="panda", useSeed=false, ` +
				`errState=<nil>, errGen=failed to validate annotations: annotations: Invalid value: "panda": ` +
				`should be a domain with at least two segments separated by dots, statusKeyIDHash="sha256:4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce", ` +
				`encryptKeyIDHash="", stateKeyIDHash="sha256:d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35", expirationTimestamp=` + now.Format(time.RFC3339),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer SetKDFForTests(tt.useSeed)()

			var buf bytes.Buffer
			klog.SetOutput(&buf)

			ctx := testContext(t)

			h := &kmsv2PluginProbe{
				name:    "panda",
				service: tt.service,
			}
			h.state.Store(&tt.state)

			err := h.rotateDEKOnKeyIDChange(ctx, tt.statusKeyID, "panda")

			klog.Flush()
			klog.SetOutput(io.Discard) // prevent further writes into buf

			if diff := cmp.Diff(tt.wantLogs, logLines(buf.String())); len(diff) > 0 {
				t.Errorf("log mismatch (-want +got):\n%s", diff)
			}

			ignoredFields := sets.NewString("Transformer", "EncryptedObject.EncryptedDEKSource", "UID", "CacheKey")

			gotState := *h.state.Load()

			if diff := cmp.Diff(tt.wantState, gotState,
				cmp.FilterPath(func(path cmp.Path) bool { return ignoredFields.Has(path.String()) }, cmp.Ignore()),
			); len(diff) > 0 {
				t.Errorf("state mismatch (-want +got):\n%s", diff)
			}

			if len(cmp.Diff(tt.wantState, gotState)) > 0 { // we only need to run this check when the state changes
				validCiphertext := len(gotState.EncryptedObject.EncryptedDEKSource) > 0
				if tt.useSeed {
					validCiphertext = validCiphertext && gotState.EncryptedObject.EncryptedDEKSourceType == kmstypes.EncryptedDEKSourceType_HKDF_SHA256_XNONCE_AES_GCM_SEED
				} else {
					validCiphertext = validCiphertext && gotState.EncryptedObject.EncryptedDEKSourceType == kmstypes.EncryptedDEKSourceType_AES_GCM_KEY
				}
				if !validCiphertext {
					t.Errorf("invalid ciphertext with useSeed=%v, encryptedDEKSourceLen=%d, encryptedDEKSourceType=%d", tt.useSeed,
						len(gotState.EncryptedObject.EncryptedDEKSource), gotState.EncryptedObject.EncryptedDEKSourceType)
				}
			}

			if tt.wantEncryptCalls != tt.service.encryptCalls {
				t.Errorf("want %d encryptCalls, got %d", tt.wantEncryptCalls, tt.service.encryptCalls)
			}

			if errString(err) != tt.wantErr {
				t.Errorf("rotateDEKOnKeyIDChange() error = %v, wantErr %v", err, tt.wantErr)
			}

			// if the old or new state is valid, we should be able to use it
			if _, stateErr := h.getCurrentState(); stateErr == nil || err == nil {
				transformer := envelopekmsv2.NewEnvelopeTransformer(
					&testKMSv2EnvelopeService{err: fmt.Errorf("broken")}, // not called
					"panda",
					h.getCurrentState,
					"",
				)

				dataCtx := value.DefaultContext(sampleContextText)
				originalText := []byte(sampleText)

				transformedData, err := transformer.TransformToStorage(ctx, originalText, dataCtx)
				if err != nil {
					t.Fatal(err)
				}

				untransformedData, stale, err := transformer.TransformFromStorage(ctx, transformedData, dataCtx)
				if err != nil {
					t.Fatal(err)
				}
				if stale {
					t.Error("unexpected stale data")
				}
				if !bytes.Equal(untransformedData, originalText) {
					t.Fatalf("incorrect transformation, want: %v, got: %v", originalText, untransformedData)
				}
			}
		})
	}
}

func validState(t *testing.T, keyID string, exp time.Time, useSeed bool) envelopekmsv2.State {
	t.Helper()

	transformer, encObject, cacheKey, err := envelopekmsv2.GenerateTransformer(testContext(t), "", &testKMSv2EnvelopeService{keyID: keyID}, useSeed)
	if err != nil {
		t.Fatal(err)
	}
	return envelopekmsv2.State{
		Transformer:         transformer,
		EncryptedObject:     *encObject,
		ExpirationTimestamp: exp,
		CacheKey:            cacheKey,
	}
}

func logLines(logs string) []string {
	if len(logs) == 0 {
		return nil
	}

	lines := strings.Split(strings.TrimSpace(logs), "\n")
	for i, line := range lines {
		lines[i] = strings.SplitN(line, "] ", 2)[1]
	}
	return lines
}

func TestGetEncryptionConfigHash(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		filepath string
		wantHash string
		wantErr  string
	}{
		{
			name:     "empty config file content",
			filepath: "testdata/invalid-configs/kms/invalid-content.yaml",
			wantHash: "",
			wantErr:  `encryption provider configuration file "testdata/invalid-configs/kms/invalid-content.yaml" is empty`,
		},
		{
			name:     "missing file",
			filepath: "testdata/invalid-configs/kms/file-that-does-not-exist.yaml",
			wantHash: "",
			wantErr:  `error reading encryption provider configuration file "testdata/invalid-configs/kms/file-that-does-not-exist.yaml": open testdata/invalid-configs/kms/file-that-does-not-exist.yaml: no such file or directory`,
		},
		{
			name:     "valid file",
			filepath: "testdata/valid-configs/secret-box-first.yaml",
			wantHash: "k8s:enc:unstable:1:c638c0327dbc3276dd1fcf3e67895d19ebca16b91ae0d19af24ef0759b8e0f66",
			wantErr:  ``,
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, err := GetEncryptionConfigHash(tt.filepath)
			if errString(err) != tt.wantErr {
				t.Errorf("GetEncryptionConfigHash() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.wantHash {
				t.Errorf("GetEncryptionConfigHash() got = %v, want %v", got, tt.wantHash)
			}
		})
	}
}
