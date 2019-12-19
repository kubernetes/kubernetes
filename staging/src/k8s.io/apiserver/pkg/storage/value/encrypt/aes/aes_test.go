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

package aes

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/storage/value"
)

func TestGCMDataStable(t *testing.T) {
	block, err := aes.NewCipher([]byte("0123456789abcdef"))
	if err != nil {
		t.Fatal(err)
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		t.Fatal(err)
	}
	// IMPORTANT: If you must fix this test, then all previously encrypted data from previously compiled versions is broken unless you hardcode the nonce size to 12
	if aead.NonceSize() != 12 {
		t.Fatalf("The underlying Golang crypto size has changed, old version of AES on disk will not be readable unless the AES implementation is changed to hardcode nonce size.")
	}
}

func TestGCMKeyRotation(t *testing.T) {
	testErr := fmt.Errorf("test error")
	block1, err := aes.NewCipher([]byte("abcdefghijklmnop"))
	if err != nil {
		t.Fatal(err)
	}
	block2, err := aes.NewCipher([]byte("0123456789abcdef"))
	if err != nil {
		t.Fatal(err)
	}

	context := value.DefaultContext([]byte("authenticated_data"))

	p := value.NewPrefixTransformers(testErr,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewGCMTransformer(block1)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewGCMTransformer(block2)},
	)
	out, err := p.TransformToStorage([]byte("firstvalue"), context)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(out, []byte("first:")) {
		t.Fatalf("unexpected prefix: %q", out)
	}
	from, stale, err := p.TransformFromStorage(out, context)
	if err != nil {
		t.Fatal(err)
	}
	if stale || !bytes.Equal([]byte("firstvalue"), from) {
		t.Fatalf("unexpected data: %t %q", stale, from)
	}

	// verify changing the context fails storage
	_, _, err = p.TransformFromStorage(out, value.DefaultContext([]byte("incorrect_context")))
	if err == nil {
		t.Fatalf("expected unauthenticated data")
	}

	// reverse the order, use the second key
	p = value.NewPrefixTransformers(testErr,
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewGCMTransformer(block2)},
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewGCMTransformer(block1)},
	)
	from, stale, err = p.TransformFromStorage(out, context)
	if err != nil {
		t.Fatal(err)
	}
	if !stale || !bytes.Equal([]byte("firstvalue"), from) {
		t.Fatalf("unexpected data: %t %q", stale, from)
	}
}

func TestCBCKeyRotation(t *testing.T) {
	testErr := fmt.Errorf("test error")
	block1, err := aes.NewCipher([]byte("abcdefghijklmnop"))
	if err != nil {
		t.Fatal(err)
	}
	block2, err := aes.NewCipher([]byte("0123456789abcdef"))
	if err != nil {
		t.Fatal(err)
	}

	context := value.DefaultContext([]byte("authenticated_data"))

	p := value.NewPrefixTransformers(testErr,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewCBCTransformer(block1)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewCBCTransformer(block2)},
	)
	out, err := p.TransformToStorage([]byte("firstvalue"), context)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(out, []byte("first:")) {
		t.Fatalf("unexpected prefix: %q", out)
	}
	from, stale, err := p.TransformFromStorage(out, context)
	if err != nil {
		t.Fatal(err)
	}
	if stale || !bytes.Equal([]byte("firstvalue"), from) {
		t.Fatalf("unexpected data: %t %q", stale, from)
	}

	// verify changing the context fails storage
	_, _, err = p.TransformFromStorage(out, value.DefaultContext([]byte("incorrect_context")))
	if err != nil {
		t.Fatalf("CBC mode does not support authentication: %v", err)
	}

	// reverse the order, use the second key
	p = value.NewPrefixTransformers(testErr,
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewCBCTransformer(block2)},
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewCBCTransformer(block1)},
	)
	from, stale, err = p.TransformFromStorage(out, context)
	if err != nil {
		t.Fatal(err)
	}
	if !stale || !bytes.Equal([]byte("firstvalue"), from) {
		t.Fatalf("unexpected data: %t %q", stale, from)
	}
}

func BenchmarkGCMRead(b *testing.B) {
	tests := []struct {
		keyLength   int
		valueLength int
		expectStale bool
	}{
		{keyLength: 16, valueLength: 1024, expectStale: false},
		{keyLength: 32, valueLength: 1024, expectStale: false},
		{keyLength: 32, valueLength: 16384, expectStale: false},
		{keyLength: 32, valueLength: 16384, expectStale: true},
	}
	for _, t := range tests {
		name := fmt.Sprintf("%vKeyLength/%vValueLength/%vExpectStale", t.keyLength, t.valueLength, t.expectStale)
		b.Run(name, func(b *testing.B) {
			benchmarkGCMRead(b, t.keyLength, t.valueLength, t.expectStale)
		})
	}
}

func BenchmarkGCMWrite(b *testing.B) {
	tests := []struct {
		keyLength   int
		valueLength int
	}{
		{keyLength: 16, valueLength: 1024},
		{keyLength: 32, valueLength: 1024},
		{keyLength: 32, valueLength: 16384},
	}
	for _, t := range tests {
		name := fmt.Sprintf("%vKeyLength/%vValueLength", t.keyLength, t.valueLength)
		b.Run(name, func(b *testing.B) {
			benchmarkGCMWrite(b, t.keyLength, t.valueLength)
		})
	}
}

func benchmarkGCMRead(b *testing.B, keyLength int, valueLength int, expectStale bool) {
	block1, err := aes.NewCipher(bytes.Repeat([]byte("a"), keyLength))
	if err != nil {
		b.Fatal(err)
	}
	block2, err := aes.NewCipher(bytes.Repeat([]byte("b"), keyLength))
	if err != nil {
		b.Fatal(err)
	}
	p := value.NewPrefixTransformers(nil,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewGCMTransformer(block1)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewGCMTransformer(block2)},
	)

	context := value.DefaultContext([]byte("authenticated_data"))
	v := bytes.Repeat([]byte("0123456789abcdef"), valueLength/16)

	out, err := p.TransformToStorage(v, context)
	if err != nil {
		b.Fatal(err)
	}
	// reverse the key order if expecting stale
	if expectStale {
		p = value.NewPrefixTransformers(nil,
			value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewGCMTransformer(block2)},
			value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewGCMTransformer(block1)},
		)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		from, stale, err := p.TransformFromStorage(out, context)
		if err != nil {
			b.Fatal(err)
		}
		if expectStale != stale {
			b.Fatalf("unexpected data: %q, expect stale %t but got %t", from, expectStale, stale)
		}
	}
	b.StopTimer()
}

func benchmarkGCMWrite(b *testing.B, keyLength int, valueLength int) {
	block1, err := aes.NewCipher(bytes.Repeat([]byte("a"), keyLength))
	if err != nil {
		b.Fatal(err)
	}
	block2, err := aes.NewCipher(bytes.Repeat([]byte("b"), keyLength))
	if err != nil {
		b.Fatal(err)
	}
	p := value.NewPrefixTransformers(nil,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewGCMTransformer(block1)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewGCMTransformer(block2)},
	)

	context := value.DefaultContext([]byte("authenticated_data"))
	v := bytes.Repeat([]byte("0123456789abcdef"), valueLength/16)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := p.TransformToStorage(v, context)
		if err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

func BenchmarkCBCRead(b *testing.B) {
	tests := []struct {
		keyLength   int
		valueLength int
		expectStale bool
	}{
		{keyLength: 32, valueLength: 1024, expectStale: false},
		{keyLength: 32, valueLength: 16384, expectStale: false},
		{keyLength: 32, valueLength: 16384, expectStale: true},
	}
	for _, t := range tests {
		name := fmt.Sprintf("%vKeyLength/%vValueLength/%vExpectStale", t.keyLength, t.valueLength, t.expectStale)
		b.Run(name, func(b *testing.B) {
			benchmarkCBCRead(b, t.keyLength, t.valueLength, t.expectStale)
		})
	}
}

func BenchmarkCBCWrite(b *testing.B) {
	tests := []struct {
		keyLength   int
		valueLength int
	}{
		{keyLength: 32, valueLength: 1024},
		{keyLength: 32, valueLength: 16384},
	}
	for _, t := range tests {
		name := fmt.Sprintf("%vKeyLength/%vValueLength", t.keyLength, t.valueLength)
		b.Run(name, func(b *testing.B) {
			benchmarkCBCWrite(b, t.keyLength, t.valueLength)
		})
	}
}

func benchmarkCBCRead(b *testing.B, keyLength int, valueLength int, expectStale bool) {
	block1, err := aes.NewCipher(bytes.Repeat([]byte("a"), keyLength))
	if err != nil {
		b.Fatal(err)
	}
	block2, err := aes.NewCipher(bytes.Repeat([]byte("b"), keyLength))
	if err != nil {
		b.Fatal(err)
	}
	p := value.NewPrefixTransformers(nil,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewCBCTransformer(block1)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewCBCTransformer(block2)},
	)

	context := value.DefaultContext([]byte("authenticated_data"))
	v := bytes.Repeat([]byte("0123456789abcdef"), valueLength/16)

	out, err := p.TransformToStorage(v, context)
	if err != nil {
		b.Fatal(err)
	}
	// reverse the key order if expecting stale
	if expectStale {
		p = value.NewPrefixTransformers(nil,
			value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewCBCTransformer(block2)},
			value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewCBCTransformer(block1)},
		)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		from, stale, err := p.TransformFromStorage(out, context)
		if err != nil {
			b.Fatal(err)
		}
		if expectStale != stale {
			b.Fatalf("unexpected data: %q, expect stale %t but got %t", from, expectStale, stale)
		}
	}
	b.StopTimer()
}

func benchmarkCBCWrite(b *testing.B, keyLength int, valueLength int) {
	block1, err := aes.NewCipher(bytes.Repeat([]byte("a"), keyLength))
	if err != nil {
		b.Fatal(err)
	}
	block2, err := aes.NewCipher(bytes.Repeat([]byte("b"), keyLength))
	if err != nil {
		b.Fatal(err)
	}
	p := value.NewPrefixTransformers(nil,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewCBCTransformer(block1)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewCBCTransformer(block2)},
	)

	context := value.DefaultContext([]byte("authenticated_data"))
	v := bytes.Repeat([]byte("0123456789abcdef"), valueLength/16)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := p.TransformToStorage(v, context)
		if err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

func TestRoundTrip(t *testing.T) {
	lengths := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 128, 1024}

	aes16block, err := aes.NewCipher([]byte(bytes.Repeat([]byte("a"), 16)))
	if err != nil {
		t.Fatal(err)
	}
	aes24block, err := aes.NewCipher([]byte(bytes.Repeat([]byte("b"), 24)))
	if err != nil {
		t.Fatal(err)
	}
	aes32block, err := aes.NewCipher([]byte(bytes.Repeat([]byte("c"), 32)))
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		context value.Context
		t       value.Transformer
	}{
		{name: "GCM 16 byte key", t: NewGCMTransformer(aes16block)},
		{name: "GCM 24 byte key", t: NewGCMTransformer(aes24block)},
		{name: "GCM 32 byte key", t: NewGCMTransformer(aes32block)},
		{name: "CBC 32 byte key", t: NewCBCTransformer(aes32block)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			context := tt.context
			if context == nil {
				context = value.DefaultContext("")
			}
			for _, l := range lengths {
				data := make([]byte, l)
				if _, err := io.ReadFull(rand.Reader, data); err != nil {
					t.Fatalf("unable to read sufficient random bytes: %v", err)
				}
				original := append([]byte{}, data...)

				ciphertext, err := tt.t.TransformToStorage(data, context)
				if err != nil {
					t.Errorf("TransformToStorage error = %v", err)
					continue
				}

				result, stale, err := tt.t.TransformFromStorage(ciphertext, context)
				if err != nil {
					t.Errorf("TransformFromStorage error = %v", err)
					continue
				}
				if stale {
					t.Errorf("unexpected stale output")
					continue
				}

				switch {
				case l == 0:
					if len(result) != 0 {
						t.Errorf("Round trip failed len=%d\noriginal:\n%s\nresult:\n%s", l, hex.Dump(original), hex.Dump(result))
					}
				case !reflect.DeepEqual(original, result):
					t.Errorf("Round trip failed len=%d\noriginal:\n%s\nresult:\n%s", l, hex.Dump(original), hex.Dump(result))
				}
			}
		})
	}
}
