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
	"errors"
	"io"
	"math"
	"reflect"
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

func (c copyMarshaler) Size() int {
	return len(c)
}

func (c copyMarshaler) MarshalTo(dest []byte) (int, error) {
	n := copy(dest, []byte(c))
	return n, nil
}

func TestUnmarshalRawZeroCopy(t *testing.T) {
	testCases := []struct {
		name          string
		obj           *Unknown
		extraWireData []byte
		expectShare   bool
	}{
		{
			name: "decode an Unknown obj with zero-copy Raw slice",
			obj: &Unknown{
				TypeMeta:        TypeMeta{APIVersion: "group/version", Kind: "Carp"},
				Raw:             []byte("hello world"),
				ContentEncoding: "encoding",
				ContentType:     ContentTypeProtobuf,
			},
			expectShare: true,
		},
		{
			name: "decode an Unknown obj with empty Raw slice (0x12, 0x00)",
			obj: &Unknown{
				TypeMeta:    TypeMeta{APIVersion: "group/version", Kind: "Carp"},
				Raw:         []byte{},
				ContentType: ContentTypeProtobuf,
			},
			expectShare: false,
		},
		{
			name: "decode an Unknown obj with nil Raw slice",
			obj: &Unknown{
				TypeMeta:    TypeMeta{APIVersion: "group/version", Kind: "Carp"},
				Raw:         nil,
				ContentType: ContentTypeProtobuf,
			},
			expectShare: false,
		},
		{
			name: "skip unknown fields across wire types 0, 1, 2, 3/4, and 5",
			obj: &Unknown{
				TypeMeta: TypeMeta{APIVersion: "group/version", Kind: "Carp"},
				Raw:      []byte("payload"),
			},
			extraWireData: []byte{
				0x28, 0x01, // field 5, wire type 0 (varint)
				0x31, 1, 2, 3, 4, 5, 6, 7, 8, // field 6, wire type 1 (64-bit)
				0x3a, 0x03, 'a', 'b', 'c', // field 7, wire type 2 (bytes)
				0x43, 0x48, 0x01, 0x44, // field 8, wire type 3/4 (group containing field 9 varint)
				0x55, 1, 2, 3, 4, // field 10, wire type 5 (32-bit)
			},
			expectShare: true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			data, err := tc.obj.Marshal()
			if err != nil {
				t.Fatal(err)
			}
			data = append(data, tc.extraWireData...)

			var want Unknown
			if err := want.Unmarshal(data); err != nil {
				t.Fatal(err)
			}

			var decoded Unknown
			if err := decoded.UnmarshalRawZeroCopy(data); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(&decoded, &want) {
				t.Fatalf("UnmarshalRawZeroCopy() = %#v, want %#v", &decoded, &want)
			}

			if tc.expectShare {
				if cap(decoded.Raw) != len(decoded.Raw) {
					t.Fatalf("cap(decoded.Raw) = %d, want %d (3-index slice cap == len)", cap(decoded.Raw), len(decoded.Raw))
				}
				rawIndex := bytes.Index(data, tc.obj.Raw)
				if rawIndex < 0 {
					t.Fatal("expected Raw bytes to be present in marshaled data")
				}
				data[rawIndex] ^= 0xff
				if reflect.DeepEqual(decoded.Raw, tc.obj.Raw) {
					t.Fatal("expected decoded.Raw to share the backing array with input data")
				}
			} else if cap(decoded.Raw) != 0 {
				t.Fatalf("cap(decoded.Raw) = %d, want 0 so empty/nil Raw never aliases input data", cap(decoded.Raw))
			}
		})
	}

	errorCases := []struct {
		name string
		data []byte
	}{
		{
			name: "varint overflow",
			data: []byte{0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01},
		},
		{
			name: "negative length in Raw field",
			data: []byte{0x12, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01},
		},
		{
			name: "truncated Raw field",
			data: []byte{0x12, 0x05, 'a', 'b'},
		},
		{
			name: "unexpected end of group in unknown field",
			data: []byte{0x43, 0x44, 0x44},
		},
	}
	for _, tc := range errorCases {
		t.Run(tc.name, func(t *testing.T) {
			var want Unknown
			wantErr := want.Unmarshal(tc.data)

			var got Unknown
			gotErr := got.UnmarshalRawZeroCopy(tc.data)
			if !errors.Is(gotErr, wantErr) && (gotErr == nil || wantErr == nil || gotErr.Error() != wantErr.Error()) {
				t.Fatalf("UnmarshalRawZeroCopy() error = %v, want %v", gotErr, wantErr)
			}
		})
	}
}
