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
	"flag"
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/gogo/protobuf/proto"

	"k8s.io/apimachinery/pkg/util/uuid"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage/value"
	kmstypes "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2/v2"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	kmsservice "k8s.io/kms/pkg/service"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

const (
	testText        = "abcdefghijklmnopqrstuvwxyz"
	testContextText = "0123456789"
	testKeyHash     = "sha256:6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b"
	testKeyVersion  = "1"
)

// testEnvelopeService is a mock Envelope service which can be used to simulate remote Envelope services
// for testing of Envelope based encryption providers.
type testEnvelopeService struct {
	annotations  map[string][]byte
	disabled     bool
	keyVersion   string
	ciphertext   []byte
	decryptCalls int
}

func (t *testEnvelopeService) Decrypt(ctx context.Context, uid string, req *kmsservice.DecryptRequest) ([]byte, error) {
	t.decryptCalls++
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

	ciphertext := t.ciphertext
	if ciphertext == nil {
		ciphertext = []byte(base64.StdEncoding.EncodeToString(data))
	}

	return &kmsservice.EncryptResponse{Ciphertext: ciphertext, KeyID: t.keyVersion, Annotations: annotations}, nil
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

func (t *testEnvelopeService) SetCiphertext(ciphertext []byte) {
	t.ciphertext = ciphertext
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
		expectedDecryptCalls     int
	}{
		{
			desc:                     "entry in cache should withstand plugin failure",
			cacheTTL:                 5 * time.Minute,
			simulateKMSPluginFailure: true,
			expectedDecryptCalls:     0, // should not hit KMS plugin
		},
		{
			desc:                     "cache entry expired should not withstand plugin failure",
			cacheTTL:                 1 * time.Millisecond,
			simulateKMSPluginFailure: true,
			expectedError:            "failed to decrypt DEK, error: Envelope service was disabled",
			expectedDecryptCalls:     10, // should hit KMS plugin for each read after cache entry expired and fail
		},
		{
			desc:                     "cache entry expired should work after cache refresh",
			cacheTTL:                 1 * time.Millisecond,
			simulateKMSPluginFailure: false,
			expectedDecryptCalls:     1, // should hit KMS plugin just for the 1st read after cache entry expired
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			ctx := testContext(t)

			envelopeService := newTestEnvelopeService()
			fakeClock := testingclock.NewFakeClock(time.Now())

			state, err := testStateFunc(ctx, envelopeService, fakeClock)()
			if err != nil {
				t.Fatal(err)
			}

			transformer := newEnvelopeTransformerWithClock(envelopeService, testProviderName,
				func() (State, error) { return state, nil },
				tt.cacheTTL, fakeClock)

			dataCtx := value.DefaultContext(testContextText)
			originalText := []byte(testText)

			transformedData, err := transformer.TransformToStorage(ctx, originalText, dataCtx)
			if err != nil {
				t.Fatalf("envelopeTransformer: error while transforming data to storage: %s", err)
			}
			untransformedData, _, err := transformer.TransformFromStorage(ctx, transformedData, dataCtx)
			if err != nil {
				t.Fatalf("could not decrypt Envelope transformer's encrypted data even once: %v", err)
			}
			if !bytes.Equal(untransformedData, originalText) {
				t.Fatalf("envelopeTransformer transformed data incorrectly. Expected: %v, got %v", originalText, untransformedData)
			}

			fakeClock.Step(2 * time.Minute)
			state, err = testStateFunc(ctx, envelopeService, fakeClock)()
			if err != nil {
				t.Fatal(err)
			}
			envelopeService.SetDisabledStatus(tt.simulateKMSPluginFailure)

			for i := 0; i < 10; i++ {
				// Subsequent reads for the same data should work fine due to caching.
				untransformedData, _, err = transformer.TransformFromStorage(ctx, transformedData, dataCtx)
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
			}
			if envelopeService.decryptCalls != tt.expectedDecryptCalls {
				t.Fatalf("expected %d decrypt calls, got %d", tt.expectedDecryptCalls, envelopeService.decryptCalls)
			}
		})
	}
}

func testStateFunc(ctx context.Context, envelopeService kmsservice.Service, clock clock.Clock) func() (State, error) {
	return func() (State, error) {
		transformer, resp, cacheKey, errGen := GenerateTransformer(ctx, string(uuid.NewUUID()), envelopeService)
		if errGen != nil {
			return State{}, errGen
		}
		return State{
			Transformer:         transformer,
			EncryptedDEK:        resp.Ciphertext,
			KeyID:               resp.KeyID,
			Annotations:         resp.Annotations,
			UID:                 "panda",
			ExpirationTimestamp: clock.Now().Add(time.Hour),
			CacheKey:            cacheKey,
		}, nil
	}
}

// TestEnvelopeTransformerStaleness validates that staleness checks on read honor the data returned from the StateFunc.
func TestEnvelopeTransformerStaleness(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		desc          string
		expectedStale bool
		testErr       error
		testKeyID     string
	}{
		{
			desc:          "stateFunc returns err",
			expectedStale: false,
			testErr:       fmt.Errorf("failed to perform status section of the healthz check for KMS Provider"),
			testKeyID:     "",
		},
		{
			desc:          "stateFunc returns same keyID",
			expectedStale: false,
			testErr:       nil,
			testKeyID:     testKeyVersion,
		},
		{
			desc:          "stateFunc returns different keyID",
			expectedStale: true,
			testErr:       nil,
			testKeyID:     "2",
		},
	}

	for _, tt := range testCases {
		tt := tt
		t.Run(tt.desc, func(t *testing.T) {
			t.Parallel()

			ctx := testContext(t)

			envelopeService := newTestEnvelopeService()
			state, err := testStateFunc(ctx, envelopeService, clock.RealClock{})()
			if err != nil {
				t.Fatal(err)
			}
			var stateErr error

			transformer := NewEnvelopeTransformer(envelopeService, testProviderName,
				func() (State, error) { return state, stateErr },
			)

			dataCtx := value.DefaultContext(testContextText)
			originalText := []byte(testText)

			transformedData, err := transformer.TransformToStorage(ctx, originalText, dataCtx)
			if err != nil {
				t.Fatalf("envelopeTransformer: error while transforming data (%v) to storage: %s", originalText, err)
			}

			// inject test data before performing a read
			state.KeyID = tt.testKeyID
			stateErr = tt.testErr

			_, stale, err := transformer.TransformFromStorage(ctx, transformedData, dataCtx)
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

func TestEnvelopeTransformerStateFunc(t *testing.T) {
	t.Parallel()

	ctx := testContext(t)

	envelopeService := newTestEnvelopeService()
	state, err := testStateFunc(ctx, envelopeService, clock.RealClock{})()
	if err != nil {
		t.Fatal(err)
	}

	// start with a broken state
	stateErr := fmt.Errorf("some state error")

	transformer := NewEnvelopeTransformer(envelopeService, testProviderName,
		func() (State, error) { return state, stateErr },
	)

	dataCtx := value.DefaultContext(testContextText)
	originalText := []byte(testText)

	t.Run("nothing works when the state is broken", func(t *testing.T) {
		_, err := transformer.TransformToStorage(ctx, originalText, dataCtx)
		if err != stateErr {
			t.Fatalf("expected state error, got: %v", err)
		}
		data, err := proto.Marshal(&kmstypes.EncryptedObject{
			EncryptedData: []byte{1},
			KeyID:         "2",
			EncryptedDEK:  []byte{3},
			Annotations:   nil,
		})
		if err != nil {
			t.Fatal(err)
		}
		_, _, err = transformer.TransformFromStorage(ctx, data, dataCtx)
		if err != stateErr {
			t.Fatalf("expected state error, got: %v", err)
		}
	})

	// fix the state
	stateErr = nil

	var encryptedData []byte

	t.Run("everything works when the state is fixed", func(t *testing.T) {
		encryptedData, err = transformer.TransformToStorage(ctx, originalText, dataCtx)
		if err != nil {
			t.Fatal(err)
		}
		_, _, err = transformer.TransformFromStorage(ctx, encryptedData, dataCtx)
		if err != nil {
			t.Fatal(err)
		}
	})

	// break the plugin
	envelopeService.SetDisabledStatus(true)

	t.Run("everything works even when the plugin is down but the state is valid", func(t *testing.T) {
		data, err := transformer.TransformToStorage(ctx, originalText, dataCtx)
		if err != nil {
			t.Fatal(err)
		}
		_, _, err = transformer.TransformFromStorage(ctx, data, dataCtx)
		if err != nil {
			t.Fatal(err)
		}
	})

	// make the state invalid
	state.ExpirationTimestamp = time.Now().Add(-time.Hour)

	t.Run("writes fail when the plugin is down and the state is invalid", func(t *testing.T) {
		_, err := transformer.TransformToStorage(ctx, originalText, dataCtx)
		if !strings.Contains(errString(err), `EDEK with keyID "1" expired at`) {
			t.Fatalf("expected expiration error, got: %v", err)
		}
	})

	t.Run("reads succeed when the plugin is down and the state is invalid", func(t *testing.T) {
		_, _, err = transformer.TransformFromStorage(ctx, encryptedData, dataCtx)
		if err != nil {
			t.Fatal(err)
		}
	})

	t.Run("reads for a different DEK fail when the plugin is down and the state is invalid", func(t *testing.T) {
		obj := &kmstypes.EncryptedObject{}
		if err := proto.Unmarshal(encryptedData, obj); err != nil {
			t.Fatal(err)
		}
		obj.EncryptedDEK = append(obj.EncryptedDEK, 1) // skip StateFunc transformer
		data, err := proto.Marshal(obj)
		if err != nil {
			t.Fatal(err)
		}

		_, _, err = transformer.TransformFromStorage(ctx, data, dataCtx)
		if errString(err) != "failed to decrypt DEK, error: Envelope service was disabled" {
			t.Fatal(err)
		}
	})
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

			ctx := testContext(t)

			envelopeService := newTestEnvelopeService()
			envelopeService.SetAnnotations(tt.annotations)
			transformer := NewEnvelopeTransformer(envelopeService, testProviderName,
				testStateFunc(ctx, envelopeService, clock.RealClock{}),
			)
			dataCtx := value.DefaultContext(testContextText)

			_, err := transformer.TransformToStorage(ctx, []byte(testText), dataCtx)
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
	transformer := &envelopeTransformer{}

	obj := &kmstypes.EncryptedObject{
		EncryptedData: []byte{0x01, 0x02, 0x03},
		KeyID:         "1",
		EncryptedDEK:  []byte{0x04, 0x05, 0x06},
	}

	data, err := transformer.doEncode(obj)
	if err != nil {
		t.Fatalf("envelopeTransformer: error while encoding data: %s", err)
	}
	got, err := transformer.doDecode(data)
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
			i := i
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
			i := i
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
			i := i
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
	transformer := NewEnvelopeTransformer(envelopeService, testProviderName,
		testStateFunc(testContext(t), envelopeService, clock.RealClock{}),
	)

	dataCtx := value.DefaultContext(testContextText)

	kmsv2Transformer := value.PrefixTransformer{Prefix: []byte("k8s:enc:kms:v2:"), Transformer: transformer}

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
	}

	metrics.KeyIDHashTotal.Reset()
	metrics.InvalidKeyIDFromStatusTotal.Reset()

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			defer metrics.KeyIDHashTotal.Reset()
			defer metrics.InvalidKeyIDFromStatusTotal.Reset()
			ctx := testContext(t)
			envelopeService.keyVersion = tt.keyVersionFromEncrypt
			transformedData, err := tt.prefix.TransformToStorage(ctx, []byte(testText), dataCtx)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := tt.prefix.TransformFromStorage(ctx, transformedData, dataCtx); err != nil {
				t.Fatal(err)
			}

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), tt.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestEnvelopeLogging(t *testing.T) {
	klog.InitFlags(nil)
	flag.Set("v", "6")
	flag.Parse()

	testCases := []struct {
		desc     string
		ctx      context.Context
		wantLogs []string
	}{
		{
			desc: "no request info in context",
			ctx:  testContext(t),
			wantLogs: []string{
				`"encrypting content using envelope service" uid="UID"`,
				`"encrypting content using DEK" uid="UID" key="0123456789" group="" version="" resource="" subresource="" verb="" namespace="" name=""`,
				`"decrypting content using envelope service" uid="UID" key="0123456789" group="" version="" resource="" subresource="" verb="" namespace="" name=""`,
			},
		},
		{
			desc: "request info in context",
			ctx: genericapirequest.WithRequestInfo(testContext(t), &genericapirequest.RequestInfo{
				APIGroup:    "awesome.bears.com",
				APIVersion:  "v1",
				Resource:    "pandas",
				Subresource: "status",
				Namespace:   "kube-system",
				Name:        "panda",
				Verb:        "update",
			}),
			wantLogs: []string{
				`"encrypting content using envelope service" uid="UID"`,
				`"encrypting content using DEK" uid="UID" key="0123456789" group="awesome.bears.com" version="v1" resource="pandas" subresource="status" verb="update" namespace="kube-system" name="panda"`,
				`"decrypting content using envelope service" uid="UID" key="0123456789" group="awesome.bears.com" version="v1" resource="pandas" subresource="status" verb="update" namespace="kube-system" name="panda"`,
			},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.desc, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)
			klog.LogToStderr(false)
			defer klog.LogToStderr(true)

			envelopeService := newTestEnvelopeService()
			fakeClock := testingclock.NewFakeClock(time.Now())
			transformer := newEnvelopeTransformerWithClock(envelopeService, testProviderName,
				testStateFunc(tc.ctx, envelopeService, clock.RealClock{}),
				1*time.Second, fakeClock)

			dataCtx := value.DefaultContext([]byte(testContextText))
			originalText := []byte(testText)

			transformedData, err := transformer.TransformToStorage(tc.ctx, originalText, dataCtx)
			if err != nil {
				t.Fatalf("envelopeTransformer: error while transforming data to storage: %v", err)
			}

			// advance the clock to trigger cache to expire, so we make a decrypt call that will log
			fakeClock.Step(2 * time.Second)

			_, _, err = transformer.TransformFromStorage(tc.ctx, transformedData, dataCtx)
			if err != nil {
				t.Fatalf("could not decrypt Envelope transformer's encrypted data even once: %v", err)
			}

			klog.Flush()
			klog.SetOutput(&bytes.Buffer{}) // prevent further writes into buf
			capturedOutput := buf.String()

			// replace the uid with a constant to make the test output stable and assertable
			capturedOutput = regexp.MustCompile(`uid="[^"]+"`).ReplaceAllString(capturedOutput, `uid="UID"`)

			for _, wantLog := range tc.wantLogs {
				if !strings.Contains(capturedOutput, wantLog) {
					t.Errorf("expected log %q, got %q", wantLog, capturedOutput)
				}
			}
		})
	}
}

func TestCacheNotCorrupted(t *testing.T) {
	ctx := testContext(t)

	envelopeService := newTestEnvelopeService()
	envelopeService.SetAnnotations(map[string][]byte{
		"encrypted-dek.kms.kubernetes.io": []byte("encrypted-dek-0"),
	})

	fakeClock := testingclock.NewFakeClock(time.Now())

	state, err := testStateFunc(ctx, envelopeService, fakeClock)()
	if err != nil {
		t.Fatal(err)
	}

	transformer := newEnvelopeTransformerWithClock(envelopeService, testProviderName,
		func() (State, error) { return state, nil },
		1*time.Second, fakeClock)

	dataCtx := value.DefaultContext(testContextText)
	originalText := []byte(testText)

	transformedData1, err := transformer.TransformToStorage(ctx, originalText, dataCtx)
	if err != nil {
		t.Fatalf("envelopeTransformer: error while transforming data to storage: %s", err)
	}

	// this is to mimic a plugin that sets a static response for ciphertext
	// but uses the annotation field to send the actual encrypted DEK.
	envelopeService.SetCiphertext(state.EncryptedDEK)
	// for this plugin, it indicates a change in the remote key ID as the returned
	// encrypted DEK is different.
	envelopeService.SetAnnotations(map[string][]byte{
		"encrypted-dek.kms.kubernetes.io": []byte("encrypted-dek-1"),
	})

	state, err = testStateFunc(ctx, envelopeService, fakeClock)()
	if err != nil {
		t.Fatal(err)
	}

	transformer = newEnvelopeTransformerWithClock(envelopeService, testProviderName,
		func() (State, error) { return state, nil },
		1*time.Second, fakeClock)

	transformedData2, err := transformer.TransformToStorage(ctx, originalText, dataCtx)
	if err != nil {
		t.Fatalf("envelopeTransformer: error while transforming data to storage: %s", err)
	}

	if _, _, err := transformer.TransformFromStorage(ctx, transformedData1, dataCtx); err != nil {
		t.Fatal(err)
	}
	if _, _, err := transformer.TransformFromStorage(ctx, transformedData2, dataCtx); err != nil {
		t.Fatal(err)
	}
}

func TestGenerateCacheKey(t *testing.T) {
	encryptedDEK1 := []byte{1, 2, 3}
	keyID1 := "id1"
	annotations1 := map[string][]byte{"a": {4, 5}, "b": {6, 7}}

	encryptedDEK2 := []byte{4, 5, 6}
	keyID2 := "id2"
	annotations2 := map[string][]byte{"x": {9, 10}, "y": {11, 12}}

	// generate all possible combinations of the above
	testCases := []struct {
		encryptedDEK []byte
		keyID        string
		annotations  map[string][]byte
	}{
		{encryptedDEK1, keyID1, annotations1},
		{encryptedDEK1, keyID1, annotations2},
		{encryptedDEK1, keyID2, annotations1},
		{encryptedDEK1, keyID2, annotations2},
		{encryptedDEK2, keyID1, annotations1},
		{encryptedDEK2, keyID1, annotations2},
		{encryptedDEK2, keyID2, annotations1},
		{encryptedDEK2, keyID2, annotations2},
	}

	for _, tc := range testCases {
		tc := tc
		for _, tc2 := range testCases {
			tc2 := tc2
			t.Run(fmt.Sprintf("%+v-%+v", tc, tc2), func(t *testing.T) {
				key1, err1 := generateCacheKey(tc.encryptedDEK, tc.keyID, tc.annotations)
				key2, err2 := generateCacheKey(tc2.encryptedDEK, tc2.keyID, tc2.annotations)
				if err1 != nil || err2 != nil {
					t.Errorf("generateCacheKey() want err=nil, got err1=%q, err2=%q", errString(err1), errString(err2))
				}
				if bytes.Equal(key1, key2) != reflect.DeepEqual(tc, tc2) {
					t.Errorf("expected %v, got %v", reflect.DeepEqual(tc, tc2), bytes.Equal(key1, key2))
				}
			})
		}
	}
}

func errString(err error) string {
	if err == nil {
		return ""
	}

	return err.Error()
}
