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
	"testing"

	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	kmstypes "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2/v2alpha1"
)

const (
	testText              = "abcdefghijklmnopqrstuvwxyz"
	testContextText       = "0123456789"
	testEnvelopeCacheSize = 10
)

// testEnvelopeService is a mock Envelope service which can be used to simulate remote Envelope services
// for testing of Envelope based encryption providers.
type testEnvelopeService struct {
	disabled   bool
	keyVersion string
}

func (t *testEnvelopeService) Decrypt(ctx context.Context, uid string, req *DecryptRequest) ([]byte, error) {
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

func (t *testEnvelopeService) Encrypt(ctx context.Context, uid string, data []byte) (*EncryptResponse, error) {
	if t.disabled {
		return nil, fmt.Errorf("Envelope service was disabled")
	}
	if len(uid) == 0 {
		return nil, fmt.Errorf("uid is required")
	}
	return &EncryptResponse{Ciphertext: []byte(base64.StdEncoding.EncodeToString(data)), KeyID: t.keyVersion, Annotations: map[string][]byte{"kms.kubernetes.io/local-kek": []byte("encrypted-local-kek")}}, nil
}

func (t *testEnvelopeService) Status(ctx context.Context) (*StatusResponse, error) {
	if t.disabled {
		return nil, fmt.Errorf("Envelope service was disabled")
	}
	return &StatusResponse{KeyID: t.keyVersion}, nil
}

func (t *testEnvelopeService) SetDisabledStatus(status bool) {
	t.disabled = status
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
	}{
		{
			desc:                     "positive cache size should withstand plugin failure",
			cacheSize:                1000,
			simulateKMSPluginFailure: true,
		},
		{
			desc:      "cache disabled size should not withstand plugin failure",
			cacheSize: 0,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			envelopeService := newTestEnvelopeService()
			envelopeTransformer, err := NewEnvelopeTransformer(envelopeService, tt.cacheSize, aestransformer.NewGCMTransformer)
			if err != nil {
				t.Fatalf("failed to initialize envelope transformer: %v", err)
			}
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
			// Subsequent read for the same data should work fine due to caching.
			untransformedData, _, err = envelopeTransformer.TransformFromStorage(ctx, transformedData, dataCtx)
			if err != nil {
				t.Fatalf("could not decrypt Envelope transformer's encrypted data using just cache: %v", err)
			}
			if !bytes.Equal(untransformedData, originalText) {
				t.Fatalf("envelopeTransformer transformed data incorrectly using cache. Got: %v, want %v", untransformedData, originalText)
			}
		})
	}
}

// Makes Envelope transformer hit cache limit, throws error if it misbehaves.
func TestEnvelopeCacheLimit(t *testing.T) {
	envelopeTransformer, err := NewEnvelopeTransformer(newTestEnvelopeService(), testEnvelopeCacheSize, aestransformer.NewGCMTransformer)
	if err != nil {
		t.Fatalf("failed to initialize envelope transformer: %v", err)
	}
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

func TestDecodeError(t *testing.T) {
	et := &envelopeTransformer{}

	testCases := []struct {
		desc          string
		originalData  func() []byte
		expectedError error
	}{
		{
			desc: "encrypted data is nil",
			originalData: func() []byte {
				data, _ := et.doEncode(&kmstypes.EncryptedObject{})
				return data
			},
			expectedError: fmt.Errorf("encrypted data is nil after unmarshal"),
		},
		{
			desc: "keyID is nil",
			originalData: func() []byte {
				data, _ := et.doEncode(&kmstypes.EncryptedObject{
					EncryptedData: []byte{0x01, 0x02, 0x03},
				})
				return data
			},
			expectedError: fmt.Errorf("keyID is empty after unmarshal"),
		},
		{
			desc: "encrypted dek is nil",
			originalData: func() []byte {
				data, _ := et.doEncode(&kmstypes.EncryptedObject{
					EncryptedData: []byte{0x01, 0x02, 0x03},
					KeyID:         "1",
				})
				return data
			},
			expectedError: fmt.Errorf("encrypted dek is nil after unmarshal"),
		},
	}

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			_, err := et.doDecode(tt.originalData())
			if err == nil {
				t.Fatalf("envelopeTransformer: expected error while decoding data, got nil")
			}

			if err.Error() != tt.expectedError.Error() {
				t.Fatalf("doDecode() error: expected %v, got %v", tt.expectedError, err)
			}
		})
	}
}
