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
	"crypto/aes"
	"encoding/base64"
	"fmt"
	"strconv"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
)

const (
	testContextText = "0123456789"
)

// testEnvelopeService is a mock Envelope service which can be used to simulate remote Envelope services
// for testing of Envelope based encryption providers.
type testEnvelopeService struct {
	keyVersion string
}

func (t *testEnvelopeService) Decrypt(data []byte) ([]byte, error) {
	dataChunks := strings.SplitN(string(data), ":", 2)
	if len(dataChunks) != 2 {
		return nil, fmt.Errorf("invalid data encountered for decryption: %s. Missing key version", data)
	}
	return base64.StdEncoding.DecodeString(dataChunks[1])
}

func (t *testEnvelopeService) Encrypt(data []byte) ([]byte, error) {
	return []byte(t.keyVersion + ":" + base64.StdEncoding.EncodeToString(data)), nil
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

func BenchmarkEnvelopeCBCRead(b *testing.B) {
	envelopeTransformer, err := NewEnvelopeTransformer(newTestEnvelopeService(), aestransformer.NewCBCTransformer)
	if err != nil {
		b.Fatalf("failed to initialize envelope transformer: %v", err)
	}
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
	envelopeTransformer, err := NewEnvelopeTransformer(newTestEnvelopeService(), aestransformer.NewGCMTransformer)
	if err != nil {
		b.Fatalf("failed to initialize envelope transformer: %v", err)
	}
	benchmarkRead(b, envelopeTransformer, 1024)
}

func BenchmarkAESGCMRead(b *testing.B) {
	block, err := aes.NewCipher(bytes.Repeat([]byte("a"), 32))
	if err != nil {
		b.Fatal(err)
	}

	aesGCMTransformer := aestransformer.NewGCMTransformer(block)
	benchmarkRead(b, aesGCMTransformer, 1024)
}

func benchmarkRead(b *testing.B, transformer value.Transformer, valueLength int) {
	context := value.DefaultContext([]byte(testContextText))
	v := bytes.Repeat([]byte("0123456789abcdef"), valueLength/16)

	out, err := transformer.TransformToStorage(v, context)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		from, stale, err := transformer.TransformFromStorage(out, context)
		if err != nil {
			b.Fatal(err)
		}
		if stale {
			b.Fatalf("unexpected data: %t %q", stale, from)
		}
	}
	b.StopTimer()
}
