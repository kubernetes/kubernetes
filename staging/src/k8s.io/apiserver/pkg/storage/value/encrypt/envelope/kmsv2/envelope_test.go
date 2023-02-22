/*
Copyright 2022 The Kubernetes Authors.

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

// Package kmsv2 transforms values for storage at rest using a Envelope v2 provider
package kmsv2

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	kmstypes "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2/v2alpha1"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	kmsservice "k8s.io/kms/pkg/service"
	testingclock "k8s.io/utils/clock/testing"
)

const (
	testText        = "abcdefghijklmnopqrstuvwxyz"
	testContextText = "0123456789"
	testKeyHash     = "sha256:6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b"
	testKeyVersion  = "1"
	testCacheTTL    = 10 * time.Second
)

var (
	errCode = "empty"
)

// testEnvelopeService is a mock Envelope service which can be used to simulate remote Envelope services
// for testing of Envelope based encryption providers.
type testEnvelopeService struct {
	annotations map[string][]byte
	disabled    bool
	keyVersion  string
}

func (t *testEnvelopeService) Decrypt(ctx context.Context, uid string, req *kmsservice.DecryptRequest) ([]byte, error) {
	if t.disabled {
		return nil, fmt.Errorf("Envelope service was disabled")
	}
	if len(uid) == 0 {
		return nil, fmt.Errorf("uid is required")
	}
	if len(req.KeyID) == 0 {
		return nil, fmt.Errorf("keyID is required")
	}
	return base64.StdEncoding.DecodeString(string(req.Ciphertext))
}

func (t *testEnvelopeService) Encrypt(ctx context.Context, uid string, data []byte) (*kmsservice.EncryptResponse, error) {
	if t.disabled {
		return nil, fmt.Errorf("Envelope service was disabled")
	}
	if len(uid) == 0 {
		return nil, fmt.Errorf("uid is required")
	}
	annotations := make(map[string][]byte)
	if t.annotations != nil {
		for k, v := range t.annotations {
			annotations[k] = v
		}
	} else {
		annotations["local-kek.kms.kubernetes.io"] = []byte("encrypted-local-kek")
	}
	return &kmsservice.EncryptResponse{Ciphertext: []byte(base64.StdEncoding.EncodeToString(data)), KeyID: t.keyVersion, Annotations: annotations}, nil
}

func (t *testEnvelopeService) Status(ctx context.Context) (*kmsservice.StatusResponse, error) {
	if t.disabled {
		return nil, fmt.Errorf("Envelope service was disabled")
	}
	return &kmsservice.StatusResponse{KeyID: t.keyVersion}, nil
}

func (t *testEnvelopeService) SetDisabledStatus(status bool) {
	t.disabled = status
}

func (t *testEnvelopeService) SetAnnotations(annotations map[string][]byte) {
	t.annotations = annotations
}

func (t *testEnvelopeService) Rotate() {
	i, _ := strconv.Atoi(t.keyVersion)
	t.keyVersion = strconv.FormatInt(int64(i+1), 10)
}

func newTestEnvelopeService() *testEnvelopeService {
	return &testEnvelopeService{
		keyVersion: testKeyVersion,
	}
}

// Throw error if Envelope transformer tries to contact Envelope without hitting cache.
func TestEnvelopeCaching(t *testing.T) {
	testCases := []struct {
		desc                     string
		cacheTTL                 time.Duration
		simulateKMSPluginFailure bool
		expectedError            string
	}{
		{
			desc:                     "entry in cache should withstand plugin failure",
			cacheTTL:                 5 * time.Minute,
			simulateKMSPluginFailure: true,
		},
		{
			desc:                     "cache entry expired should not withstand plugin failure",
			cacheTTL:                 1 * time.Millisecond,
			simulateKMSPluginFailure: true,
			expectedError:            "failed to decrypt DEK, error: Envelope service was disabled",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			envelopeService := newTestEnvelopeService()
			fakeClock := testingclock.NewFakeClock(time.Now())
			envelopeTransformer := newEnvelopeTransformerWithClock(envelopeService, testProviderName,
				func(ctx context.Context) (string, error) {
					return "", nil
				},
				func(ctx context.Context) error {
					return nil
				},
				aestransformer.NewGCMTransformer, tt.cacheTTL, fakeClock)

			ctx := testContext(t)
			dataCtx := value.DefaultContext([]byte(testContextText))
			originalText := []byte(testText)

			transformedData, err := envelopeTransformer.TransformToStorage(ctx, originalText, dataCtx)
			if err != nil {
				t.Fatalf("envelopeTransformer: error while transforming data to storage: %s", err)
			}
			untransformedData, _, err := envelopeTransformer.TransformFromStorage(ctx, transformedData, dataCtx)
			if err != nil {
				t.Fatalf("could not decrypt Envelope transformer's encrypted data even once: %v", err)
			}
			if !bytes.Equal(untransformedData, originalText) {
				t.Fatalf("envelopeTransformer transformed data incorrectly. Expected: %v, got %v", originalText, untransformedData)
			}

			envelopeService.SetDisabledStatus(tt.simulateKMSPluginFailure)
			fakeClock.Step(2 * time.Minute)
			// Subsequent read for the same data should work fine due to caching.
			untransformedData, _, err = envelopeTransformer.TransformFromStorage(ctx, transformedData, dataCtx)
			if tt.expectedError != "" {
				if err == nil {
					t.Fatalf("expected error: %v, got nil", tt.expectedError)
				}
				if err.Error() != tt.expectedError {
					t.Fatalf("expected error: %v, got: %v", tt.expectedError, err)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if !bytes.Equal(untransformedData, originalText) {
					t.Fatalf("envelopeTransformer transformed data incorrectly. Expected: %v, got %v", originalText, untransformedData)
				}
			}
		})
	}
}

// Test keyIDGetter as part of envelopeTransformer, throws error if returned err or staleness is incorrect.
func TestEnvelopeTransformerKeyIDGetter(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		desc          string
		expectedStale bool
		testErr       error
		testKeyID     string
	}{
		{
			desc:          "keyIDGetter returns err",
			expectedStale: false,
			testErr:       fmt.Errorf("failed to perform status section of the healthz check for KMS Provider"),
			testKeyID:     "",
		},
		{
			desc:          "keyIDGetter returns same keyID",
			expectedStale: false,
			testErr:       nil,
			testKeyID:     testKeyVersion,
		},
		{
			desc:          "keyIDGetter returns different keyID",
			expectedStale: true,
			testErr:       nil,
			testKeyID:     "2",
		},
	}

	for _, tt := range testCases {
		tt := tt
		t.Run(tt.desc, func(t *testing.T) {
			t.Parallel()
			envelopeService := newTestEnvelopeService()
			envelopeTransformer := NewEnvelopeTransformer(envelopeService, testProviderName,
				func(ctx context.Context) (string, error) {
					return tt.testKeyID, tt.testErr
				},
				func(ctx context.Context) error {
					return nil
				},
				aestransformer.NewGCMTransformer)

			ctx := testContext(t)
			dataCtx := value.DefaultContext([]byte(testContextText))
			originalText := []byte(testText)

			transformedData, err := envelopeTransformer.TransformToStorage(ctx, originalText, dataCtx)
			if err != nil {
				t.Fatalf("envelopeTransformer: error while transforming data (%v) to storage: %s", originalText, err)
			}

			_, stale, err := envelopeTransformer.TransformFromStorage(ctx, transformedData, dataCtx)
			if tt.testErr != nil {
				if err == nil {
					t.Fatalf("envelopeTransformer: expected error: %v, got nil", tt.testErr)
				}
				if err.Error() != tt.testErr.Error() {
					t.Fatalf("envelopeTransformer: expected error: %v, got: %v", tt.testErr, err)
				}
			} else {
				if err != nil {
					t.Fatalf("envelopeTransformer: unexpected error: %v", err)
				}
				if stale != tt.expectedStale {
					t.Fatalf("envelopeTransformer TransformFromStorage determined keyID staleness incorrectly, expected: %v, got %v", tt.expectedStale, stale)
				}
			}
		})
	}
}

func TestTransformToStorageError(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name        string
		annotations map[string][]byte
	}{
		{
			name: "invalid annotation key",
			annotations: map[string][]byte{
				"http://foo.example.com": []byte("bar"),
			},
		},
		{
			name: "annotation value size too large",
			annotations: map[string][]byte{
				"simple": []byte(strings.Repeat("a", 32*1024)),
			},
		},
		{
			name: "annotations size too large",
			annotations: map[string][]byte{
				"simple":  []byte(strings.Repeat("a", 31*1024)),
				"simple2": []byte(strings.Repeat("a", 1024)),
			},
		},
	}

	for _, tt := range testCases {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			envelopeService := newTestEnvelopeService()
			envelopeService.SetAnnotations(tt.annotations)
			envelopeTransformer := NewEnvelopeTransformer(envelopeService, testProviderName,
				func(ctx context.Context) (string, error) {
					return "", nil
				},
				func(ctx context.Context) error {
					return nil
				},
				aestransformer.NewGCMTransformer)
			ctx := testContext(t)
			dataCtx := value.DefaultContext([]byte(testContextText))

			_, err := envelopeTransformer.TransformToStorage(ctx, []byte(testText), dataCtx)
			if err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !strings.Contains(err.Error(), "failed to validate annotations") {
				t.Fatalf("expected error to contain 'failed to validate annotations', got %v", err)
			}
		})
	}
}

func TestEncodeDecode(t *testing.T) {
	envelopeTransformer := &envelopeTransformer{}

	obj := &kmstypes.EncryptedObject{
		EncryptedData: []byte{0x01, 0x02, 0x03},
		KeyID:         "1",
		EncryptedDEK:  []byte{0x04, 0x05, 0x06},
	}

	data, err := envelopeTransformer.doEncode(obj)
	if err != nil {
		t.Fatalf("envelopeTransformer: error while encoding data: %s", err)
	}
	got, err := envelopeTransformer.doDecode(data)
	if err != nil {
		t.Fatalf("envelopeTransformer: error while decoding data: %s", err)
	}
	// reset internal field modified by marshaling obj
	obj.XXX_sizecache = 0
	if !reflect.DeepEqual(got, obj) {
		t.Fatalf("envelopeTransformer: decoded data does not match original data. Got: %v, want %v", got, obj)
	}
}

func TestValidateEncryptedObject(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		desc          string
		originalData  *kmstypes.EncryptedObject
		expectedError error
	}{
		{
			desc:          "encrypted object is nil",
			originalData:  nil,
			expectedError: fmt.Errorf("encrypted object is nil"),
		},
		{
			desc: "encrypted data is nil",
			originalData: &kmstypes.EncryptedObject{
				KeyID:        "1",
				EncryptedDEK: []byte{0x01, 0x02, 0x03},
			},
			expectedError: fmt.Errorf("encrypted data is empty"),
		},
		{
			desc: "encrypted data is []byte{}",
			originalData: &kmstypes.EncryptedObject{
				EncryptedDEK:  []byte{0x01, 0x02, 0x03},
				EncryptedData: []byte{},
			},
			expectedError: fmt.Errorf("encrypted data is empty"),
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			err := validateEncryptedObject(tt.originalData)
			if err == nil {
				t.Fatalf("envelopeTransformer: expected error while decoding data, got nil")
			}

			if err.Error() != tt.expectedError.Error() {
				t.Fatalf("doDecode() error: expected %v, got %v", tt.expectedError, err)
			}
		})
	}
}

func TestValidateAnnotations(t *testing.T) {
	t.Parallel()
	successCases := []map[string][]byte{
		{"a.com": []byte("bar")},
		{"k8s.io": []byte("bar")},
		{"dev.k8s.io": []byte("bar")},
		{"dev.k8s.io.": []byte("bar")},
		{"foo.example.com": []byte("bar")},
		{"this.is.a.really.long.fqdn": []byte("bar")},
		{"bbc.co.uk": []byte("bar")},
		{"10.0.0.1": []byte("bar")}, // DNS labels can start with numbers and there is no requirement for letters.
		{"hyphens-are-good.k8s.io": []byte("bar")},
		{strings.Repeat("a", 63) + ".k8s.io": []byte("bar")},
		{strings.Repeat("a", 63) + "." + strings.Repeat("b", 63) + "." + strings.Repeat("c", 63) + "." + strings.Repeat("d", 54) + ".k8s.io": []byte("bar")},
	}
	t.Run("success", func(t *testing.T) {
		for i := range successCases {
			t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
				t.Parallel()
				if err := validateAnnotations(successCases[i]); err != nil {
					t.Errorf("case[%d] expected success, got %#v", i, err)
				}
			})
		}
	})

	atleastTwoSegmentsErrorMsg := "should be a domain with at least two segments separated by dots"
	moreThan63CharsErrorMsg := "must be no more than 63 characters"
	moreThan253CharsErrorMsg := "must be no more than 253 characters"
	dns1123SubdomainErrorMsg := "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character"

	annotationsNameErrorCases := []struct {
		annotations map[string][]byte
		expect      string
	}{
		{map[string][]byte{".": []byte("bar")}, dns1123SubdomainErrorMsg},
		{map[string][]byte{"...": []byte("bar")}, dns1123SubdomainErrorMsg},
		{map[string][]byte{".io": []byte("bar")}, dns1123SubdomainErrorMsg},
		{map[string][]byte{"com": []byte("bar")}, atleastTwoSegmentsErrorMsg},
		{map[string][]byte{".com": []byte("bar")}, dns1123SubdomainErrorMsg},
		{map[string][]byte{"Dev.k8s.io": []byte("bar")}, dns1123SubdomainErrorMsg},
		{map[string][]byte{".foo.example.com": []byte("bar")}, dns1123SubdomainErrorMsg},
		{map[string][]byte{"*.example.com": []byte("bar")}, dns1123SubdomainErrorMsg},
		{map[string][]byte{"*.bar.com": []byte("bar")}, dns1123SubdomainErrorMsg},
		{map[string][]byte{"*.foo.bar.com": []byte("bar")}, dns1123SubdomainErrorMsg},
		{map[string][]byte{"underscores_are_bad.k8s.io": []byte("bar")}, dns1123SubdomainErrorMsg},
		{map[string][]byte{"foo@bar.example.com": []byte("bar")}, dns1123SubdomainErrorMsg},
		{map[string][]byte{"http://foo.example.com": []byte("bar")}, dns1123SubdomainErrorMsg},
		{map[string][]byte{strings.Repeat("a", 64) + ".k8s.io": []byte("bar")}, moreThan63CharsErrorMsg},
		{map[string][]byte{strings.Repeat("a", 63) + "." + strings.Repeat("b", 63) + "." + strings.Repeat("c", 63) + "." + strings.Repeat("d", 55) + ".k8s.io": []byte("bar")}, moreThan253CharsErrorMsg},
	}

	t.Run("name error", func(t *testing.T) {
		for i := range annotationsNameErrorCases {
			t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
				t.Parallel()
				err := validateAnnotations(annotationsNameErrorCases[i].annotations)
				if err == nil {
					t.Errorf("case[%d]: expected failure", i)
				} else {
					if !strings.Contains(err.Error(), annotationsNameErrorCases[i].expect) {
						t.Errorf("case[%d]: error details do not include %q: %q", i, annotationsNameErrorCases[i].expect, err)
					}
				}
			})
		}
	})

	maxSizeErrMsg := "which exceeds the max size of"
	annotationsSizeErrorCases := []struct {
		annotations map[string][]byte
		expect      string
	}{
		{map[string][]byte{"simple": []byte(strings.Repeat("a", 33*1024))}, maxSizeErrMsg},
		{map[string][]byte{"simple": []byte(strings.Repeat("a", 32*1024))}, maxSizeErrMsg},
		{map[string][]byte{"simple": []byte(strings.Repeat("a", 64*1024))}, maxSizeErrMsg},
		{map[string][]byte{"simple": []byte(strings.Repeat("a", 31*1024)), "simple2": []byte(strings.Repeat("a", 1024))}, maxSizeErrMsg},
		{map[string][]byte{strings.Repeat("a", 253): []byte(strings.Repeat("a", 32*1024))}, maxSizeErrMsg},
	}
	t.Run("size error", func(t *testing.T) {
		for i := range annotationsSizeErrorCases {
			t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
				t.Parallel()
				err := validateAnnotations(annotationsSizeErrorCases[i].annotations)
				if err == nil {
					t.Errorf("case[%d]: expected failure", i)
				} else {
					if !strings.Contains(err.Error(), annotationsSizeErrorCases[i].expect) {
						t.Errorf("case[%d]: error details do not include %q: %q", i, annotationsSizeErrorCases[i].expect, err)
					}
				}
			})
		}
	})
}

func TestValidateKeyID(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name              string
		keyID             string
		expectedError     string
		expectedErrorCode string
	}{
		{
			name:              "valid key ID",
			keyID:             "1234",
			expectedError:     "",
			expectedErrorCode: "ok",
		},
		{
			name:              "empty key ID",
			keyID:             "",
			expectedError:     "keyID is empty",
			expectedErrorCode: "empty",
		},
		{
			name:              "keyID size is greater than 1 kB",
			keyID:             strings.Repeat("a", 1024+1),
			expectedError:     "which exceeds the max size of",
			expectedErrorCode: "too_long",
		},
	}

	for _, tt := range testCases {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			errCode, err := ValidateKeyID(tt.keyID)
			if tt.expectedError != "" {
				if err == nil {
					t.Fatalf("expected error %q, got nil", tt.expectedError)
				}
				if !strings.Contains(err.Error(), tt.expectedError) {
					t.Fatalf("expected error %q, got %q", tt.expectedError, err)
				}
			} else {
				if err != nil {
					t.Fatalf("expected no error, got %q", err)
				}
			}
			if tt.expectedErrorCode != string(errCode) {
				t.Fatalf("expected %s errCode, got %s", tt.expectedErrorCode, string(errCode))
			}
		})
	}
}

func TestValidateEncryptedDEK(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name          string
		encryptedDEK  []byte
		expectedError string
	}{
		{
			name:          "encrypted DEK is nil",
			encryptedDEK:  nil,
			expectedError: "encrypted DEK is empty",
		},
		{
			name:          "encrypted DEK is empty",
			encryptedDEK:  []byte{},
			expectedError: "encrypted DEK is empty",
		},
		{
			name:          "encrypted DEK size is greater than 1 kB",
			encryptedDEK:  bytes.Repeat([]byte("a"), 1024+1),
			expectedError: "which exceeds the max size of",
		},
		{
			name:          "valid encrypted DEK",
			encryptedDEK:  []byte{0x01, 0x02, 0x03},
			expectedError: "",
		},
	}

	for _, tt := range testCases {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := validateEncryptedDEK(tt.encryptedDEK)
			if tt.expectedError != "" {
				if err == nil {
					t.Fatalf("expected error %q, got nil", tt.expectedError)
				}
				if !strings.Contains(err.Error(), tt.expectedError) {
					t.Fatalf("expected error %q, got %q", tt.expectedError, err)
				}
			} else {
				if err != nil {
					t.Fatalf("expected no error, got %q", err)
				}
			}
		})
	}
}

func TestEnvelopeMetrics(t *testing.T) {
	envelopeService := newTestEnvelopeService()
	envelopeTransformer := NewEnvelopeTransformer(envelopeService, testProviderName,
		func(ctx context.Context) (string, error) {
			return testKeyVersion, nil
		},
		// health probe check to ensure keyID freshness
		func(ctx context.Context) error {
			metrics.RecordInvalidKeyIDFromStatus(testProviderName, errCode)
			return nil
		},
		aestransformer.NewGCMTransformer)

	dataCtx := value.DefaultContext([]byte(testContextText))

	kmsv2Transformer := value.PrefixTransformer{Prefix: []byte("k8s:enc:kms:v2:"), Transformer: envelopeTransformer}

	testCases := []struct {
		desc                  string
		keyVersionFromEncrypt string
		prefix                value.Transformer
		metrics               []string
		want                  string
	}{
		{
			desc:                  "keyIDHash total",
			keyVersionFromEncrypt: testKeyVersion,
			prefix:                value.NewPrefixTransformers(nil, kmsv2Transformer),
			metrics: []string{
				"apiserver_envelope_encryption_key_id_hash_total",
			},
			want: fmt.Sprintf(`
				# HELP apiserver_envelope_encryption_key_id_hash_total [ALPHA] Number of times a keyID is used split by transformation type and provider.
				# TYPE apiserver_envelope_encryption_key_id_hash_total counter
				apiserver_envelope_encryption_key_id_hash_total{key_id_hash="%s",provider_name="%s",transformation_type="%s"} 1
        		apiserver_envelope_encryption_key_id_hash_total{key_id_hash="%s",provider_name="%s",transformation_type="%s"} 1
				`, testKeyHash, testProviderName, metrics.FromStorageLabel, testKeyHash, testProviderName, metrics.ToStorageLabel),
		},
		{
			// keyVersionFromEncrypt is returned from kms v2 envelope service
			// when it is different from the key ID returned from last status call
			// it will trigger health probe check immediately to ensure keyID freshness
			// during probe check above, it will call RecordInvalidKeyIDFromStatus
			desc:                  "invalid KeyID From Status Total",
			keyVersionFromEncrypt: "2",
			prefix:                value.NewPrefixTransformers(nil, kmsv2Transformer),
			metrics: []string{
				"apiserver_envelope_encryption_invalid_key_id_from_status_total",
			},
			want: fmt.Sprintf(`
			# HELP apiserver_envelope_encryption_invalid_key_id_from_status_total [ALPHA] Number of times an invalid keyID is returned by the Status RPC call split by error.
			# TYPE apiserver_envelope_encryption_invalid_key_id_from_status_total counter
			apiserver_envelope_encryption_invalid_key_id_from_status_total{error="%s",provider_name="%s"} 1
			`, errCode, testProviderName),
		},
	}

	metrics.DekCacheInterArrivals.Reset()
	metrics.KeyIDHashTotal.Reset()
	metrics.InvalidKeyIDFromStatusTotal.Reset()

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			defer metrics.DekCacheInterArrivals.Reset()
			defer metrics.KeyIDHashTotal.Reset()
			defer metrics.InvalidKeyIDFromStatusTotal.Reset()
			ctx := testContext(t)
			envelopeService.keyVersion = tt.keyVersionFromEncrypt
			transformedData, err := tt.prefix.TransformToStorage(ctx, []byte(testText), dataCtx)
			if err != nil {
				t.Fatal(err)
			}
			tt.prefix.TransformFromStorage(ctx, transformedData, dataCtx)

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), tt.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
