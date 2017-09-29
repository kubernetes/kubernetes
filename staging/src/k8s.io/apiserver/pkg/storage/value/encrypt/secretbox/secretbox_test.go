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

package secretbox

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/storage/value"
)

var (
	key1 = [32]byte{0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01}
	key2 = [32]byte{0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02}
)

func TestSecretboxKeyRotation(t *testing.T) {
	testErr := fmt.Errorf("test error")
	context := value.DefaultContext([]byte("authenticated_data"))

	p := value.NewPrefixTransformers(testErr,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewSecretboxTransformer(key1)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewSecretboxTransformer(key2)},
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

	// verify changing the context does not fails storage
	// Secretbox is not currently an authenticating store
	from, stale, err = p.TransformFromStorage(out, value.DefaultContext([]byte("incorrect_context")))
	if err != nil {
		t.Fatalf("secretbox is not authenticated")
	}

	// reverse the order, use the second key
	p = value.NewPrefixTransformers(testErr,
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewSecretboxTransformer(key2)},
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewSecretboxTransformer(key1)},
	)
	from, stale, err = p.TransformFromStorage(out, context)
	if err != nil {
		t.Fatal(err)
	}
	if !stale || !bytes.Equal([]byte("firstvalue"), from) {
		t.Fatalf("unexpected data: %t %q", stale, from)
	}
}

func BenchmarkSecretboxRead_32_1024(b *testing.B)        { benchmarkSecretboxRead(b, 32, 1024, false) }
func BenchmarkSecretboxRead_32_16384(b *testing.B)       { benchmarkSecretboxRead(b, 32, 16384, false) }
func BenchmarkSecretboxRead_32_16384_Stale(b *testing.B) { benchmarkSecretboxRead(b, 32, 16384, true) }

func BenchmarkSecretboxWrite_32_1024(b *testing.B)  { benchmarkSecretboxWrite(b, 32, 1024) }
func BenchmarkSecretboxWrite_32_16384(b *testing.B) { benchmarkSecretboxWrite(b, 32, 16384) }

func benchmarkSecretboxRead(b *testing.B, keyLength int, valueLength int, expectStale bool) {
	p := value.NewPrefixTransformers(nil,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewSecretboxTransformer(key1)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewSecretboxTransformer(key2)},
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
			value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewSecretboxTransformer(key2)},
			value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewSecretboxTransformer(key1)},
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

func benchmarkSecretboxWrite(b *testing.B, keyLength int, valueLength int) {
	p := value.NewPrefixTransformers(nil,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewSecretboxTransformer(key1)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewSecretboxTransformer(key2)},
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

	tests := []struct {
		name    string
		context value.Context
		t       value.Transformer
	}{
		{name: "Secretbox 32 byte key", t: NewSecretboxTransformer(key1)},
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
