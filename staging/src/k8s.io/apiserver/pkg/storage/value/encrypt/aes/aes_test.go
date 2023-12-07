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
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"reflect"
	"sync"
	"sync/atomic"
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
		t.Errorf("The underlying Golang crypto size has changed, old version of AES on disk will not be readable unless the AES implementation is changed to hardcode nonce size.")
	}

	transformerCounterNonce, _, err := NewGCMTransformerWithUniqueKeyUnsafe()
	if err != nil {
		t.Fatal(err)
	}
	if nonceSize := transformerCounterNonce.(*gcm).aead.NonceSize(); nonceSize != 12 {
		t.Errorf("counter nonce: backwards incompatible change to nonce size detected: %d", nonceSize)
	}

	transformerRandomNonce, err := NewGCMTransformer(block)
	if err != nil {
		t.Fatal(err)
	}
	if nonceSize := transformerRandomNonce.(*gcm).aead.NonceSize(); nonceSize != 12 {
		t.Errorf("random nonce: backwards incompatible change to nonce size detected: %d", nonceSize)
	}
}

func TestGCMUnsafeNonceOverflow(t *testing.T) {
	var msgFatal string
	var count int

	nonceGen := &nonceGenerator{
		fatal: func(msg string) {
			msgFatal = msg
			count++
		},
	}

	block, err := aes.NewCipher([]byte("abcdefghijklmnop"))
	if err != nil {
		t.Fatal(err)
	}
	transformer, err := newGCMTransformerWithUniqueKeyUnsafe(block, nonceGen)
	if err != nil {
		t.Fatal(err)
	}

	assertNonce(t, &nonceGen.nonce, 0)

	runEncrypt(t, transformer)

	assertNonce(t, &nonceGen.nonce, 1)

	runEncrypt(t, transformer)

	assertNonce(t, &nonceGen.nonce, 2)

	nonceGen.nonce.Store(math.MaxUint64 - 1) // pretend lots of encryptions occurred

	runEncrypt(t, transformer)

	assertNonce(t, &nonceGen.nonce, math.MaxUint64)

	if count != 0 {
		t.Errorf("fatal should not have been called yet")
	}

	runEncrypt(t, transformer)

	assertNonce(t, &nonceGen.nonce, 0)

	if count != 1 {
		t.Errorf("fatal should have been once, got %d", count)
	}

	if msgFatal != "aes-gcm detected nonce overflow - cryptographic wear out has occurred" {
		t.Errorf("unexpected message: %s", msgFatal)
	}
}

func assertNonce(t *testing.T, nonce *atomic.Uint64, want uint64) {
	t.Helper()

	if got := nonce.Load(); want != got {
		t.Errorf("nonce should equal %d, got %d", want, got)
	}
}

func runEncrypt(t *testing.T, transformer value.Transformer) {
	t.Helper()

	ctx := context.Background()
	dataCtx := value.DefaultContext("authenticated_data")

	_, err := transformer.TransformToStorage(ctx, []byte("firstvalue"), dataCtx)
	if err != nil {
		t.Fatal(err)
	}
}

// TestGCMUnsafeCompatibility asserts that encryptions performed via
// NewGCMTransformerWithUniqueKeyUnsafe can be decrypted via NewGCMTransformer.
func TestGCMUnsafeCompatibility(t *testing.T) {
	transformerEncrypt, key, err := NewGCMTransformerWithUniqueKeyUnsafe()
	if err != nil {
		t.Fatal(err)
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		t.Fatal(err)
	}

	transformerDecrypt := newGCMTransformer(t, block, nil)

	ctx := context.Background()
	dataCtx := value.DefaultContext("authenticated_data")

	plaintext := []byte("firstvalue")

	ciphertext, err := transformerEncrypt.TransformToStorage(ctx, plaintext, dataCtx)
	if err != nil {
		t.Fatal(err)
	}

	if bytes.Equal(plaintext, ciphertext) {
		t.Errorf("plaintext %q matches ciphertext %q", string(plaintext), string(ciphertext))
	}

	plaintextAgain, _, err := transformerDecrypt.TransformFromStorage(ctx, ciphertext, dataCtx)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(plaintext, plaintextAgain) {
		t.Errorf("expected original plaintext %q, got %q", string(plaintext), string(plaintextAgain))
	}
}

func TestGCMLegacyDataCompatibility(t *testing.T) {
	block, err := aes.NewCipher([]byte("snorlax_awesomes"))
	if err != nil {
		t.Fatal(err)
	}

	transformerDecrypt := newGCMTransformer(t, block, nil)

	// recorded output from NewGCMTransformer at commit 3b1fc60d8010dd8b53e97ba80e4710dbb430beee
	const legacyCiphertext = "\x9f'\xc8\xfc\xea\x8aX\xc4g\xd8\xe47\xdb\xf2\xd8YU\xf9\xb4\xbd\x91/N\xf9g\u05c8\xa0\xcb\ay}\xac\n?\n\bE`\\\xa8Z\xc8V+J\xe1"

	ctx := context.Background()
	dataCtx := value.DefaultContext("bamboo")

	plaintext := []byte("pandas are the best")

	plaintextAgain, _, err := transformerDecrypt.TransformFromStorage(ctx, []byte(legacyCiphertext), dataCtx)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(plaintext, plaintextAgain) {
		t.Errorf("expected original plaintext %q, got %q", string(plaintext), string(plaintextAgain))
	}
}

func TestExtendedNonceGCMLegacyDataCompatibility(t *testing.T) {
	// recorded output from NewKDFExtendedNonceGCMTransformerWithUniqueSeed from https://github.com/kubernetes/kubernetes/pull/118828
	const (
		legacyKey        = "]@2:\x82\x0f\xf9Uag^;\x95\xe8\x18g\xc5\xfd\xd5a\xd3Z\x88\xa2Ћ\b\xaa\x9dO\xcf\\"
		legacyCiphertext = "$Bu\x9e3\x94_\xba\xd7\t\xdbWz\x0f\x03\x7fا\t\xfcv\x97\x9b\x89B \x9d\xeb\xce˝W\xef\xe3\xd6\xffj\x1e\xf6\xee\x9aP\x03\xb9\x83;0C\xce\xc1\xe4{5\x17[\x15\x11\a\xa8\xd2Ak\x0e)k\xbff\xb5\xd1\x02\xfc\xefߚx\xf2\x93\xd2q"
	)

	transformerDecrypt := newHKDFExtendedNonceGCMTransformerTest(t, nil, []byte(legacyKey))

	ctx := context.Background()
	dataCtx := value.DefaultContext("bamboo")

	plaintext := []byte("pandas are the best")

	plaintextAgain, _, err := transformerDecrypt.TransformFromStorage(ctx, []byte(legacyCiphertext), dataCtx)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(plaintext, plaintextAgain) {
		t.Errorf("expected original plaintext %q, got %q", string(plaintext), string(plaintextAgain))
	}
}

func TestGCMUnsafeNonceGen(t *testing.T) {
	block, err := aes.NewCipher([]byte("abcdefghijklmnop"))
	if err != nil {
		t.Fatal(err)
	}
	transformer := newGCMTransformerWithUniqueKeyUnsafeTest(t, block, nil)

	ctx := context.Background()
	dataCtx := value.DefaultContext("authenticated_data")

	const count = 1_000

	counters := make([]uint64, count)

	// run a bunch of go routines to make sure we are go routine safe
	// on both the nonce generation and the actual encryption/decryption
	var wg sync.WaitGroup
	for i := 0; i < count; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()

			plaintext := bytes.Repeat([]byte{byte(i % 8)}, count)

			out, err := transformer.TransformToStorage(ctx, plaintext, dataCtx)
			if err != nil {
				t.Error(err)
				return
			}

			nonce := out[:12]
			randomN := nonce[:4]

			if bytes.Equal(randomN, make([]byte, len(randomN))) {
				t.Error("got all zeros for random four byte nonce")
			}

			counter := nonce[4:]
			counters[binary.LittleEndian.Uint64(counter)-1]++ // subtract one because the counter starts at 1, not 0

			plaintextAgain, _, err := transformer.TransformFromStorage(ctx, out, dataCtx)
			if err != nil {
				t.Error(err)
				return
			}

			if !bytes.Equal(plaintext, plaintextAgain) {
				t.Errorf("expected original plaintext %q, got %q", string(plaintext), string(plaintextAgain))
			}
		}()
	}
	wg.Wait()

	want := make([]uint64, count)
	for i := range want {
		want[i] = 1
	}

	if !reflect.DeepEqual(want, counters) {
		t.Error("unexpected counter state")
	}
}

func TestGCMNonce(t *testing.T) {
	t.Run("gcm", func(t *testing.T) {
		testGCMNonce(t, newGCMTransformer, 0, func(_ int, nonce []byte) {
			if bytes.Equal(nonce, make([]byte, len(nonce))) {
				t.Error("got all zeros for nonce")
			}
		})
	})

	t.Run("gcm unsafe", func(t *testing.T) {
		testGCMNonce(t, newGCMTransformerWithUniqueKeyUnsafeTest, 0, func(i int, nonce []byte) {
			counter := binary.LittleEndian.Uint64(nonce)
			if uint64(i+1) != counter { // add one because the counter starts at 1, not 0
				t.Errorf("counter nonce is invalid: want %d, got %d", i+1, counter)
			}
		})
	})

	t.Run("gcm extended nonce", func(t *testing.T) {
		testGCMNonce(t, newHKDFExtendedNonceGCMTransformerTest, infoSizeExtendedNonceGCM, func(_ int, nonce []byte) {
			if bytes.Equal(nonce, make([]byte, len(nonce))) {
				t.Error("got all zeros for nonce")
			}
		})
	})
}

func testGCMNonce(t *testing.T, f transformerFunc, infoLen int, check func(int, []byte)) {
	key := []byte("abcdefghijklmnopabcdefghijklmnop")
	block, err := aes.NewCipher(key)
	if err != nil {
		t.Fatal(err)
	}
	transformer := f(t, block, key)

	ctx := context.Background()
	dataCtx := value.DefaultContext("authenticated_data")

	const count = 1_000

	for i := 0; i < count; i++ {
		i := i

		out, err := transformer.TransformToStorage(ctx, bytes.Repeat([]byte{byte(i % 8)}, count), dataCtx)
		if err != nil {
			t.Fatal(err)
		}

		info := out[:infoLen]
		nonce := out[infoLen : 12+infoLen]
		randomN := nonce[:4]

		if bytes.Equal(randomN, make([]byte, len(randomN))) {
			t.Error("got all zeros for first four bytes")
		}

		if infoLen != 0 {
			if bytes.Equal(info, make([]byte, infoLen)) {
				t.Error("got all zeros for info")
			}
		}

		check(i, nonce[4:])
	}
}

func TestGCMKeyRotation(t *testing.T) {
	t.Run("gcm", func(t *testing.T) {
		testGCMKeyRotation(t, newGCMTransformer)
	})

	t.Run("gcm unsafe", func(t *testing.T) {
		testGCMKeyRotation(t, newGCMTransformerWithUniqueKeyUnsafeTest)
	})

	t.Run("gcm extended", func(t *testing.T) {
		testGCMKeyRotation(t, newHKDFExtendedNonceGCMTransformerTest)
	})
}

func testGCMKeyRotation(t *testing.T, f transformerFunc) {
	key1 := []byte("abcdefghijklmnopabcdefghijklmnop")
	key2 := []byte("0123456789abcdef0123456789abcdef")

	testErr := fmt.Errorf("test error")
	block1, err := aes.NewCipher(key1)
	if err != nil {
		t.Fatal(err)
	}
	block2, err := aes.NewCipher(key2)
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	dataCtx := value.DefaultContext("authenticated_data")

	p := value.NewPrefixTransformers(testErr,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: f(t, block1, key1)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: f(t, block2, key2)},
	)
	out, err := p.TransformToStorage(ctx, []byte("firstvalue"), dataCtx)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(out, []byte("first:")) {
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
	if err == nil {
		t.Fatalf("expected unauthenticated data")
	}

	// reverse the order, use the second key
	p = value.NewPrefixTransformers(testErr,
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: f(t, block2, key2)},
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: f(t, block1, key1)},
	)
	from, stale, err = p.TransformFromStorage(ctx, out, dataCtx)
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

	ctx := context.Background()
	dataCtx := value.DefaultContext("authenticated_data")

	p := value.NewPrefixTransformers(testErr,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewCBCTransformer(block1)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewCBCTransformer(block2)},
	)
	out, err := p.TransformToStorage(ctx, []byte("firstvalue"), dataCtx)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(out, []byte("first:")) {
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
	p = value.NewPrefixTransformers(testErr,
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewCBCTransformer(block2)},
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewCBCTransformer(block1)},
	)
	from, stale, err = p.TransformFromStorage(ctx, out, dataCtx)
	if err != nil {
		t.Fatal(err)
	}
	if !stale || !bytes.Equal([]byte("firstvalue"), from) {
		t.Fatalf("unexpected data: %t %q", stale, from)
	}
}

var gcmBenchmarks = []namedTransformerFunc{
	{name: "gcm-random-nonce", f: newGCMTransformer},
	{name: "gcm-counter-nonce", f: newGCMTransformerWithUniqueKeyUnsafeTest},
	{name: "gcm-extended-nonce", f: newHKDFExtendedNonceGCMTransformerTest},
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
			for _, n := range gcmBenchmarks {
				n := n
				if t.keyLength == 16 && n.name == "gcm-extended-nonce" {
					continue // gcm-extended-nonce requires 32 byte keys
				}
				b.Run(n.name, func(b *testing.B) {
					b.ReportAllocs()
					benchmarkGCMRead(b, n.f, t.keyLength, t.valueLength, t.expectStale)
				})
			}
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
			for _, n := range gcmBenchmarks {
				n := n
				if t.keyLength == 16 && n.name == "gcm-extended-nonce" {
					continue // gcm-extended-nonce requires 32 byte keys
				}
				b.Run(n.name, func(b *testing.B) {
					b.ReportAllocs()
					benchmarkGCMWrite(b, n.f, t.keyLength, t.valueLength)
				})
			}
		})
	}
}

func benchmarkGCMRead(b *testing.B, f transformerFunc, keyLength int, valueLength int, expectStale bool) {
	key1 := bytes.Repeat([]byte("a"), keyLength)
	key2 := bytes.Repeat([]byte("b"), keyLength)

	block1, err := aes.NewCipher(key1)
	if err != nil {
		b.Fatal(err)
	}
	block2, err := aes.NewCipher(key2)
	if err != nil {
		b.Fatal(err)
	}
	p := value.NewPrefixTransformers(nil,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: f(b, block1, key1)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: f(b, block2, key2)},
	)

	ctx := context.Background()
	dataCtx := value.DefaultContext("authenticated_data")
	v := bytes.Repeat([]byte("0123456789abcdef"), valueLength/16)

	out, err := p.TransformToStorage(ctx, v, dataCtx)
	if err != nil {
		b.Fatal(err)
	}
	// reverse the key order if expecting stale
	if expectStale {
		p = value.NewPrefixTransformers(nil,
			value.PrefixTransformer{Prefix: []byte("second:"), Transformer: f(b, block2, key2)},
			value.PrefixTransformer{Prefix: []byte("first:"), Transformer: f(b, block1, key1)},
		)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		from, stale, err := p.TransformFromStorage(ctx, out, dataCtx)
		if err != nil {
			b.Fatal(err)
		}
		if expectStale != stale {
			b.Fatalf("unexpected data: %q, expect stale %t but got %t", from, expectStale, stale)
		}
	}
	b.StopTimer()
}

func benchmarkGCMWrite(b *testing.B, f transformerFunc, keyLength int, valueLength int) {
	key1 := bytes.Repeat([]byte("a"), keyLength)
	key2 := bytes.Repeat([]byte("b"), keyLength)

	block1, err := aes.NewCipher(key1)
	if err != nil {
		b.Fatal(err)
	}
	block2, err := aes.NewCipher(key2)
	if err != nil {
		b.Fatal(err)
	}
	p := value.NewPrefixTransformers(nil,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: f(b, block1, key1)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: f(b, block2, key2)},
	)

	ctx := context.Background()
	dataCtx := value.DefaultContext("authenticated_data")
	v := bytes.Repeat([]byte("0123456789abcdef"), valueLength/16)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := p.TransformToStorage(ctx, v, dataCtx)
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

	ctx := context.Background()
	dataCtx := value.DefaultContext("authenticated_data")
	v := bytes.Repeat([]byte("0123456789abcdef"), valueLength/16)

	out, err := p.TransformToStorage(ctx, v, dataCtx)
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
		from, stale, err := p.TransformFromStorage(ctx, out, dataCtx)
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

	ctx := context.Background()
	dataCtx := value.DefaultContext("authenticated_data")
	v := bytes.Repeat([]byte("0123456789abcdef"), valueLength/16)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := p.TransformToStorage(ctx, v, dataCtx)
		if err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

func TestRoundTrip(t *testing.T) {
	lengths := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 128, 1024}

	aes16block, err := aes.NewCipher(bytes.Repeat([]byte("a"), 16))
	if err != nil {
		t.Fatal(err)
	}
	aes24block, err := aes.NewCipher(bytes.Repeat([]byte("b"), 24))
	if err != nil {
		t.Fatal(err)
	}
	key32 := bytes.Repeat([]byte("c"), 32)
	aes32block, err := aes.NewCipher(key32)
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	tests := []struct {
		name string
		t    value.Transformer
	}{
		{name: "GCM 16 byte key", t: newGCMTransformer(t, aes16block, nil)},
		{name: "GCM 24 byte key", t: newGCMTransformer(t, aes24block, nil)},
		{name: "GCM 32 byte key", t: newGCMTransformer(t, aes32block, nil)},
		{name: "GCM 16 byte unsafe key", t: newGCMTransformerWithUniqueKeyUnsafeTest(t, aes16block, nil)},
		{name: "GCM 24 byte unsafe key", t: newGCMTransformerWithUniqueKeyUnsafeTest(t, aes24block, nil)},
		{name: "GCM 32 byte unsafe key", t: newGCMTransformerWithUniqueKeyUnsafeTest(t, aes32block, nil)},
		{name: "GCM 32 byte seed", t: newHKDFExtendedNonceGCMTransformerTest(t, nil, key32)},
		{name: "CBC 32 byte key", t: NewCBCTransformer(aes32block)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dataCtx := value.DefaultContext("/foo/bar")
			for _, l := range lengths {
				data := make([]byte, l)
				if _, err := io.ReadFull(rand.Reader, data); err != nil {
					t.Fatalf("unable to read sufficient random bytes: %v", err)
				}
				original := append([]byte{}, data...)

				ciphertext, err := tt.t.TransformToStorage(ctx, data, dataCtx)
				if err != nil {
					t.Errorf("TransformToStorage error = %v", err)
					continue
				}

				result, stale, err := tt.t.TransformFromStorage(ctx, ciphertext, dataCtx)
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

type namedTransformerFunc struct {
	name string
	f    transformerFunc
}

type transformerFunc func(t testing.TB, block cipher.Block, key []byte) value.Transformer

func newGCMTransformer(t testing.TB, block cipher.Block, _ []byte) value.Transformer {
	t.Helper()

	transformer, err := NewGCMTransformer(block)
	if err != nil {
		t.Fatal(err)
	}

	return transformer
}

func newGCMTransformerWithUniqueKeyUnsafeTest(t testing.TB, block cipher.Block, _ []byte) value.Transformer {
	t.Helper()

	nonceGen := &nonceGenerator{fatal: die}
	transformer, err := newGCMTransformerWithUniqueKeyUnsafe(block, nonceGen)
	if err != nil {
		t.Fatal(err)
	}

	return transformer
}

func newHKDFExtendedNonceGCMTransformerTest(t testing.TB, _ cipher.Block, key []byte) value.Transformer {
	t.Helper()

	transformer, err := NewHKDFExtendedNonceGCMTransformer(key)
	if err != nil {
		t.Fatal(err)
	}

	return transformer
}
