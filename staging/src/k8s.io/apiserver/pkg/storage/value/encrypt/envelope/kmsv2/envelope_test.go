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

	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	kmstypes "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2/v2alpha1"
	kmsservice "k8s.io/kms/service"
)

const (
	testText              = "abcdefghijklmnopqrstuvwxyz"
	testContextText       = "0123456789"
	testEnvelopeCacheSize = 10
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
		keyVersion: "1",
	}
}

// Throw error if Envelope transformer tries to contact Envelope without hitting cache.
func TestEnvelopeCaching(t *testing.T) {
	testCases := []struct {
		desc                     string
		cacheSize                int
		simulateKMSPluginFailure bool
		expectedError            string
	}{
		{
			desc:                     "positive cache size should withstand plugin failure",
			cacheSize:                1000,
			simulateKMSPluginFailure: true,
		},
		{
			desc:                     "cache disabled size should not withstand plugin failure",
			cacheSize:                0,
			simulateKMSPluginFailure: true,
			expectedError:            "failed to decrypt DEK, error: Envelope service was disabled",
		},
		{
			desc:                     "cache disabled, no plugin failure should succeed",
			cacheSize:                0,
			simulateKMSPluginFailure: false,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			envelopeService := newTestEnvelopeService()
			envelopeTransformer := NewEnvelopeTransformer(envelopeService, tt.cacheSize, aestransformer.NewGCMTransformer)
			ctx := context.Background()
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

// Makes Envelope transformer hit cache limit, throws error if it misbehaves.
func TestEnvelopeCacheLimit(t *testing.T) {
	envelopeTransformer := NewEnvelopeTransformer(newTestEnvelopeService(), testEnvelopeCacheSize, aestransformer.NewGCMTransformer)
	ctx := context.Background()
	dataCtx := value.DefaultContext([]byte(testContextText))

	transformedOutputs := map[int][]byte{}

	// Overwrite lots of entries in the map
	for i := 0; i < 2*testEnvelopeCacheSize; i++ {
		numberText := []byte(strconv.Itoa(i))

		res, err := envelopeTransformer.TransformToStorage(ctx, numberText, dataCtx)
		transformedOutputs[i] = res
		if err != nil {
			t.Fatalf("envelopeTransformer: error while transforming data (%v) to storage: %s", numberText, err)
		}
	}

	// Try reading all the data now, ensuring cache misses don't cause a concern.
	for i := 0; i < 2*testEnvelopeCacheSize; i++ {
		numberText := []byte(strconv.Itoa(i))

		output, _, err := envelopeTransformer.TransformFromStorage(ctx, transformedOutputs[i], dataCtx)
		if err != nil {
			t.Fatalf("envelopeTransformer: error while transforming data (%v) from storage: %s", transformedOutputs[i], err)
		}

		if !bytes.Equal(numberText, output) {
			t.Fatalf("envelopeTransformer transformed data incorrectly using cache. Expected: %v, got %v", numberText, output)
		}
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
			envelopeTransformer := NewEnvelopeTransformer(envelopeService, 0, aestransformer.NewGCMTransformer)
			ctx := context.Background()
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
		name          string
		keyID         string
		expectedError string
	}{
		{
			name:          "valid key ID",
			keyID:         "1234",
			expectedError: "",
		},
		{
			name:          "empty key ID",
			keyID:         "",
			expectedError: "keyID is empty",
		},
		{
			name:          "keyID size is greater than 1 kB",
			keyID:         strings.Repeat("a", 1024+1),
			expectedError: "which exceeds the max size of",
		},
	}

	for _, tt := range testCases {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			err := validateKeyID(tt.keyID)
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
