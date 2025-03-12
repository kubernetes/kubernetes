/*
Copyright 2025 The Kubernetes Authors.

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

package runtime

import (
	"bytes"
	"io"
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestVarint(t *testing.T) {
	varintBuffer := make([]byte, maxUint64VarIntLength)
	offset := encodeVarintGenerated(varintBuffer, len(varintBuffer), math.MaxUint64)
	used := len(varintBuffer) - offset
	if used != maxUint64VarIntLength {
		t.Fatalf("expected encodeVarintGenerated to use %d bytes to encode MaxUint64, got %d", maxUint64VarIntLength, used)
	}
}

func TestNestedMarshalToWriter(t *testing.T) {
	testcases := []struct {
		name string
		raw  []byte
	}{
		{
			name: "zero-length",
			raw:  []byte{},
		},
		{
			name: "simple",
			raw:  []byte{0x00, 0x01, 0x02, 0x03},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			u := &Unknown{
				ContentType:     "ct",
				ContentEncoding: "ce",
				TypeMeta: TypeMeta{
					APIVersion: "v1",
					Kind:       "k",
				},
			}

			// Marshal normally with Raw inlined
			u.Raw = tc.raw
			marshalData, err := u.Marshal()
			if err != nil {
				t.Fatal(err)
			}
			u.Raw = nil

			// Marshal with NestedMarshalTo
			nestedMarshalData := make([]byte, len(marshalData))
			n, err := u.NestedMarshalTo(nestedMarshalData, copyMarshaler(tc.raw), uint64(len(tc.raw)))
			if err != nil {
				t.Fatal(err)
			}
			if n != len(marshalData) {
				t.Errorf("NestedMarshalTo returned %d, expected %d", n, len(marshalData))
			}
			if e, a := marshalData, nestedMarshalData; !bytes.Equal(e, a) {
				t.Errorf("NestedMarshalTo and Marshal differ:\n%s", cmp.Diff(e, a))
			}

			// Streaming marshal with MarshalToWriter
			buf := bytes.NewBuffer(nil)
			n, err = u.MarshalToWriter(buf, len(tc.raw), func(w io.Writer) (int, error) {
				return w.Write(tc.raw)
			})
			if err != nil {
				t.Fatal(err)
			}
			if n != len(marshalData) {
				t.Errorf("MarshalToWriter returned %d, expected %d", n, len(marshalData))
			}
			if e, a := marshalData, buf.Bytes(); !bytes.Equal(e, a) {
				t.Errorf("MarshalToWriter and Marshal differ:\n%s", cmp.Diff(e, a))
			}
		})
	}
}

type copyMarshaler []byte

func (c copyMarshaler) MarshalTo(dest []byte) (int, error) {
	n := copy(dest, []byte(c))
	return n, nil
}
