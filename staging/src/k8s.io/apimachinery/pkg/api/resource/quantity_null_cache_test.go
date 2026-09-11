/*
Copyright The Kubernetes Authors.

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

package resource

import (
	"bytes"
	"encoding/json"
	"testing"
)

// TestQuantityNullDecodeClearsCachedSerialization verifies that decoding null invalidates
// the cached string, which both null branches used to leave in place.
func TestQuantityNullDecodeClearsCachedSerialization(t *testing.T) {
	decoders := []struct {
		name   string
		decode func(*Quantity) error
	}{
		{"json", func(q *Quantity) error { return json.Unmarshal([]byte("null"), q) }},
		{"cbor", func(q *Quantity) error { return q.UnmarshalCBOR([]byte{0xf6}) }},
	}
	for _, value := range []string{"1", "1Ki", "1.5Gi", "1e6", "0"} {
		for _, decoder := range decoders {
			for _, dec := range []bool{false, true} {
				name := decoder.name + "/" + value
				if dec {
					name += "/dec"
				}
				t.Run(name, func(t *testing.T) {
					q := MustParse(value)
					if dec {
						q.ToDec()
					}
					format := q.Format
					_ = q.String() // populate the cache the decoder has to invalidate

					if err := decoder.decode(&q); err != nil {
						t.Fatal(err)
					}
					if !q.IsZero() || q.Sign() != 0 || q.Value() != 0 {
						t.Errorf("null decode left %v, want zero", q.Value())
					}
					if q.Format != format {
						t.Errorf("format = %q, want %q", q.Format, format)
					}
					// MarshalJSON copies q.s when it is set; the others go through
					// String(), which only recomputes an empty cache, never a stale one.
					if encoded, err := json.Marshal(q); err != nil {
						t.Fatal(err)
					} else if string(encoded) != `"0"` {
						t.Errorf("json = %s, want \"0\"", encoded)
					}
					// The codec's own output for "0", so the encoding form is not pinned.
					wantCBOR, err := MustParse("0").MarshalCBOR()
					if err != nil {
						t.Fatal(err)
					}
					if encoded, err := q.MarshalCBOR(); err != nil {
						t.Fatal(err)
					} else if !bytes.Equal(encoded, wantCBOR) {
						t.Errorf("cbor = %x, want %x", encoded, wantCBOR)
					}
					if got := q.ToUnstructured(); got != "0" {
						t.Errorf("unstructured = %#v, want \"0\"", got)
					}
					if got := q.String(); got != "0" {
						t.Errorf("String = %q, want \"0\"", got)
					}
					if err := decoder.decode(&q); err != nil {
						t.Fatal(err)
					}
					if got := q.String(); got != "0" {
						t.Errorf("second null decode = %q, want \"0\"", got)
					}
				})
			}
		}
	}
}
