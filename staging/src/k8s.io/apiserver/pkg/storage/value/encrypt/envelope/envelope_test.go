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

package envelope

import (
	"bytes"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"strconv"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
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

func (t *testEnvelopeService) Decrypt(data []byte) ([]byte, error) {
	if t.disabled {
		return nil, fmt.Errorf("Envelope service was disabled")
	}
	dataChunks := strings.SplitN(string(data), ":", 2)
	if len(dataChunks) != 2 {
		return nil, fmt.Errorf("invalid data encountered for decryption: %s. Missing key version", data)
	}
	return base64.StdEncoding.DecodeString(dataChunks[1])
}

func (t *testEnvelopeService) Encrypt(data []byte) ([]byte, error) {
	if t.disabled {
		return nil, fmt.Errorf("Envelope service was disabled")
	}
	return []byte(t.keyVersion + ":" + base64.StdEncoding.EncodeToString(data)), nil
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
			expectedError:            "Envelope service was disabled",
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
			cbcTransformer := func(block cipher.Block) (value.Transformer, error) {
				return aestransformer.NewCBCTransformer(block), nil
			}
			envelopeTransformer := NewEnvelopeTransformer(envelopeService, tt.cacheSize, cbcTransformer)
			ctx := context.Background()
			dataCtx := value.DefaultContext(testContextText)
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
	cbcTransformer := func(block cipher.Block) (value.Transformer, error) {
		return aestransformer.NewCBCTransformer(block), nil
	}
	envelopeTransformer := NewEnvelopeTransformer(newTestEnvelopeService(), testEnvelopeCacheSize, cbcTransformer)
	ctx := context.Background()
	dataCtx := value.DefaultContext(testContextText)

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

func BenchmarkEnvelopeCBCRead(b *testing.B) {
	cbcTransformer := func(block cipher.Block) (value.Transformer, error) {
		return aestransformer.NewCBCTransformer(block), nil
	}
	envelopeTransformer := NewEnvelopeTransformer(newTestEnvelopeService(), testEnvelopeCacheSize, cbcTransformer)
	benchmarkRead(b, envelopeTransformer, 1024)
}

func BenchmarkAESCBCRead(b *testing.B) {
	block, err := aes.NewCipher(bytes.Repeat([]byte("a"), 32))
	if err != nil {
		b.Fatal(err)
	}

	aesCBCTransformer := aestransformer.NewCBCTransformer(block)
	benchmarkRead(b, aesCBCTransformer, 1024)
}

func BenchmarkEnvelopeGCMRead(b *testing.B) {
	envelopeTransformer := NewEnvelopeTransformer(newTestEnvelopeService(), testEnvelopeCacheSize, aestransformer.NewGCMTransformer)
	benchmarkRead(b, envelopeTransformer, 1024)
}

func BenchmarkAESGCMRead(b *testing.B) {
	block, err := aes.NewCipher(bytes.Repeat([]byte("a"), 32))
	if err != nil {
		b.Fatal(err)
	}

	aesGCMTransformer, err := aestransformer.NewGCMTransformer(block)
	if err != nil {
		b.Fatal(err)
	}

	benchmarkRead(b, aesGCMTransformer, 1024)
}

func benchmarkRead(b *testing.B, transformer value.Transformer, valueLength int) {
	ctx := context.Background()
	dataCtx := value.DefaultContext(testContextText)
	v := bytes.Repeat([]byte("0123456789abcdef"), valueLength/16)

	out, err := transformer.TransformToStorage(ctx, v, dataCtx)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		from, stale, err := transformer.TransformFromStorage(ctx, out, dataCtx)
		if err != nil {
			b.Fatal(err)
		}
		if stale {
			b.Fatalf("unexpected data: %t %q", stale, from)
		}
	}
	b.StopTimer()
}

// remove after 1.13
func TestBackwardsCompatibility(t *testing.T) {
	envelopeService := newTestEnvelopeService()
	cbcTransformer := func(block cipher.Block) (value.Transformer, error) {
		return aestransformer.NewCBCTransformer(block), nil
	}
	envelopeTransformerInst := NewEnvelopeTransformer(envelopeService, testEnvelopeCacheSize, cbcTransformer)
	ctx := context.Background()
	dataCtx := value.DefaultContext(testContextText)
	originalText := []byte(testText)

	transformedData, err := oldTransformToStorage(ctx, envelopeTransformerInst.(*envelopeTransformer), originalText, dataCtx)
	if err != nil {
		t.Fatalf("envelopeTransformer: error while transforming data to storage: %s", err)
	}
	untransformedData, _, err := envelopeTransformerInst.TransformFromStorage(ctx, transformedData, dataCtx)
	if err != nil {
		t.Fatalf("could not decrypt Envelope transformer's encrypted data even once: %v", err)
	}
	if !bytes.Equal(untransformedData, originalText) {
		t.Fatalf("envelopeTransformer transformed data incorrectly. Expected: %v, got %v", originalText, untransformedData)
	}

	envelopeService.SetDisabledStatus(true)
	// Subsequent read for the same data should work fine due to caching.
	untransformedData, _, err = envelopeTransformerInst.TransformFromStorage(ctx, transformedData, dataCtx)
	if err != nil {
		t.Fatalf("could not decrypt Envelope transformer's encrypted data using just cache: %v", err)
	}
	if !bytes.Equal(untransformedData, originalText) {
		t.Fatalf("envelopeTransformer transformed data incorrectly using cache. Expected: %v, got %v", originalText, untransformedData)
	}
}

// remove after 1.13
func oldTransformToStorage(ctx context.Context, t *envelopeTransformer, data []byte, dataCtx value.Context) ([]byte, error) {
	newKey, err := generateKey(32)
	if err != nil {
		return nil, err
	}

	encKey, err := t.envelopeService.Encrypt(newKey)
	if err != nil {
		return nil, err
	}

	transformer, err := t.addTransformer(encKey, newKey)
	if err != nil {
		return nil, err
	}

	// Append the length of the encrypted DEK as the first 2 bytes.
	encKeyLen := make([]byte, 2)
	encKeyBytes := []byte(encKey)
	binary.BigEndian.PutUint16(encKeyLen, uint16(len(encKeyBytes)))

	prefix := append(encKeyLen, encKeyBytes...)

	prefixedData := make([]byte, len(prefix), len(data)+len(prefix))
	copy(prefixedData, prefix)
	result, err := transformer.TransformToStorage(ctx, data, dataCtx)
	if err != nil {
		return nil, err
	}
	prefixedData = append(prefixedData, result...)
	return prefixedData, nil
}
