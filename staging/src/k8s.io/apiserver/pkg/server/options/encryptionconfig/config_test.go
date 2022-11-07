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
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	apiserverconfig "k8s.io/apiserver/pkg/apis/config"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope"
	envelopekmsv2 "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

const (
	sampleText        = "abcdefghijklmnopqrstuvwxyz"
	sampleContextText = "0123456789"
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
	err error
}

func (t *testKMSv2EnvelopeService) Decrypt(ctx context.Context, uid string, req *envelopekmsv2.DecryptRequest) ([]byte, error) {
	if t.err != nil {
		return nil, t.err
	}
	return base64.StdEncoding.DecodeString(string(req.Ciphertext))
}

func (t *testKMSv2EnvelopeService) Encrypt(ctx context.Context, uid string, data []byte) (*envelopekmsv2.EncryptResponse, error) {
	if t.err != nil {
		return nil, t.err
	}
	return &envelopekmsv2.EncryptResponse{
		Ciphertext: []byte(base64.StdEncoding.EncodeToString(data)),
		KeyID:      "1",
	}, nil
}

func (t *testKMSv2EnvelopeService) Status(ctx context.Context) (*envelopekmsv2.StatusResponse, error) {
	if t.err != nil {
		return nil, t.err
	}
	return &envelopekmsv2.StatusResponse{Healthz: "ok", KeyID: "1", Version: "v2alpha1"}, nil
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
func newMockEnvelopeKMSv2Service(ctx context.Context, endpoint string, timeout time.Duration) (envelopekmsv2.Service, error) {
	return &testKMSv2EnvelopeService{nil}, nil
}

// The factory method to create mock envelope kmsv2 service which always returns error.
func newMockErrorEnvelopeKMSv2Service(endpoint string, timeout time.Duration) (envelopekmsv2.Service, error) {
	return &testKMSv2EnvelopeService{errors.New("test")}, nil
}

func TestLegacyConfig(t *testing.T) {
	legacyV1Config := "testdata/valid-configs/legacy.yaml"
	legacyConfigObject, err := loadConfig(legacyV1Config)
	cacheSize := int32(10)
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
						APIVersion: "v1",
						Name:       "testprovider",
						Endpoint:   "unix:///tmp/testprovider.sock",
						CacheSize:  &cacheSize,
						Timeout:    &metav1.Duration{Duration: 3 * time.Second},
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
	if d := cmp.Diff(expected, legacyConfigObject); d != "" {
		t.Fatalf("EncryptionConfig mismatch (-want +got):\n%s", d)
	}
}

func TestEncryptionProviderConfigCorrect(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2, true)()
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
	correctConfigWithIdentityFirst := "testdata/valid-configs/identity-first.yaml"
	identityFirstTransformerOverrides, _, err := LoadEncryptionConfig(correctConfigWithIdentityFirst, false, ctx.Done())
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithIdentityFirst)
	}

	correctConfigWithAesGcmFirst := "testdata/valid-configs/aes-gcm-first.yaml"
	aesGcmFirstTransformerOverrides, _, err := LoadEncryptionConfig(correctConfigWithAesGcmFirst, false, ctx.Done())
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithAesGcmFirst)
	}

	correctConfigWithAesCbcFirst := "testdata/valid-configs/aes-cbc-first.yaml"
	aesCbcFirstTransformerOverrides, _, err := LoadEncryptionConfig(correctConfigWithAesCbcFirst, false, ctx.Done())
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithAesCbcFirst)
	}

	correctConfigWithSecretboxFirst := "testdata/valid-configs/secret-box-first.yaml"
	secretboxFirstTransformerOverrides, _, err := LoadEncryptionConfig(correctConfigWithSecretboxFirst, false, ctx.Done())
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithSecretboxFirst)
	}

	correctConfigWithKMSFirst := "testdata/valid-configs/kms-first.yaml"
	kmsFirstTransformerOverrides, _, err := LoadEncryptionConfig(correctConfigWithKMSFirst, false, ctx.Done())
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithKMSFirst)
	}

	correctConfigWithKMSv2First := "testdata/valid-configs/kmsv2-first.yaml"
	kmsv2FirstTransformerOverrides, _, err := LoadEncryptionConfig(correctConfigWithKMSv2First, false, ctx.Done())
	if err != nil {
		t.Fatalf("error while parsing configuration file: %s.\nThe file was:\n%s", err, correctConfigWithKMSv2First)
	}

	// Pick the transformer for any of the returned resources.
	identityFirstTransformer := identityFirstTransformerOverrides[schema.ParseGroupResource("secrets")]
	aesGcmFirstTransformer := aesGcmFirstTransformerOverrides[schema.ParseGroupResource("secrets")]
	aesCbcFirstTransformer := aesCbcFirstTransformerOverrides[schema.ParseGroupResource("secrets")]
	secretboxFirstTransformer := secretboxFirstTransformerOverrides[schema.ParseGroupResource("secrets")]
	kmsFirstTransformer := kmsFirstTransformerOverrides[schema.ParseGroupResource("secrets")]
	kmsv2FirstTransformer := kmsv2FirstTransformerOverrides[schema.ParseGroupResource("secrets")]

	dataCtx := value.DefaultContext([]byte(sampleContextText))
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

func TestKMSPluginHealthz(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KMSv2, true)()

	testCases := []struct {
		desc    string
		config  string
		want    []healthChecker
		wantErr string
		kmsv2   bool
		kmsv1   bool
	}{
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
			config: "testdata/valid-configs/kms/multiple-providers-kmsv2.yaml",
			want: []healthChecker{
				&kmsv2PluginProbe{
					name: "foo",
					ttl:  3 * time.Second,
				},
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
			config, err := loadConfig(tt.config)
			if errStr := errString(err); errStr != tt.wantErr {
				t.Fatalf("unexpected error state got=%s want=%s", errStr, tt.wantErr)
			}
			if len(tt.wantErr) > 0 {
				return
			}

			_, got, kmsUsed, err := getTransformerOverridesAndKMSPluginProbes(config, testContext(t).Done())
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

	service, _ := newMockEnvelopeKMSv2Service(ctx, "unix:///tmp/testprovider.sock", 3*time.Second)
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
			_ = tt.probe.check(ctx)
			if tt.probe.ttl != tt.wantTTL {
				t.Fatalf("want ttl %v, got ttl %v", tt.wantTTL, tt.probe.ttl)
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

	ctx := context.Background()
	dataCtx := value.DefaultContext([]byte("authenticated_data"))

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
	_, _, err = p.TransformFromStorage(ctx, out, value.DefaultContext([]byte("incorrect_context")))
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
	transformers, _, err := LoadEncryptionConfig(encryptionConfigPath, false, ctx.Done())
	if err != nil {
		t.Fatal(err)
	}
	if len(transformers) != 1 {
		t.Fatalf("input config does not have exactly one resource: %s", encryptionConfigPath)
	}
	for _, transformer := range transformers {
		return transformer
	}
	panic("unreachable")
}

func TestIsKMSv2ProviderHealthyError(t *testing.T) {
	testCases := []struct {
		desc           string
		statusResponse *envelopekmsv2.StatusResponse
	}{
		{
			desc: "healthz status is not ok",
			statusResponse: &envelopekmsv2.StatusResponse{
				Healthz: "unhealthy",
			},
		},
		{
			desc: "version is not v2alpha1",
			statusResponse: &envelopekmsv2.StatusResponse{
				Version: "v1beta1",
			},
		},
		{
			desc: "missing keyID",
			statusResponse: &envelopekmsv2.StatusResponse{
				Healthz: "ok",
				Version: "v2alpha1",
			},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			if err := isKMSv2ProviderHealthy("testplugin", tt.statusResponse); err == nil {
				t.Fatalf("isKMSv2ProviderHealthy() should have returned an error")
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
