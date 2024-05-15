/*
Copyright 2024 The Kubernetes Authors.

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

package modes_test

import (
	"encoding/hex"
	"fmt"
	"math"
	"testing"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/internal/modes"

	"github.com/google/go-cmp/cmp"
)

// TestAppendixA roundtrips the examples of encoded CBOR data items in RFC 8949 Appendix A. For
// completeness, all examples from the appendix are included, even those those that are rejected by
// this decoder or are re-encoded to a different sequence of CBOR bytes (with explanation).
func TestAppendixA(t *testing.T) {
	hex := func(h string) []byte {
		b, err := hex.DecodeString(h)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}

	eq := conversion.EqualitiesOrDie(
		// NaN float64 values are always inequal and have multiple representations. RFC 8949
		// Section 4.2.2 recommends protocols not supporting NaN payloads or signaling NaNs
		// choose a single representation for all NaN values. For the purposes of this test,
		// all NaN representations are equivalent.
		func(a float64, b float64) bool {
			if math.IsNaN(a) && math.IsNaN(b) {
				return true
			}
			return math.Float64bits(a) == math.Float64bits(b)
		},
	)

	const (
		reasonArrayFixedLength  = "indefinite-length arrays are re-encoded with fixed length"
		reasonByteString        = "strings are encoded as the byte string major type"
		reasonMapFixedLength    = "indefinite-length maps are re-encoded with fixed length"
		reasonMapSorted         = "map entries are sorted"
		reasonStringFixedLength = "indefinite-length strings are re-encoded with fixed length"
		reasonTagIgnored        = "unrecognized tag numbers are ignored"
		reasonTimeToInterface   = "times decode to interface{} as RFC3339 timestamps for JSON interoperability"
	)

	for _, tc := range []struct {
		example []byte // example data item
		decoded interface{}
		reject  string   // reason the decoder rejects the example
		encoded []byte   // re-encoded object (only if different from example encoding)
		reasons []string // reasons for re-encode difference

		// TODO: The cases with nonempty fixme are known to be not working and fixing them
		// is an alpha criteria. They're present and skipped for visibility.
		fixme string
	}{
		{
			example: hex("00"),
			decoded: int64(0),
		},
		{
			example: hex("01"),
			decoded: int64(1),
		},
		{
			example: hex("0a"),
			decoded: int64(10),
		},
		{
			example: hex("17"),
			decoded: int64(23),
		},
		{
			example: hex("1818"),
			decoded: int64(24),
		},
		{
			example: hex("1819"),
			decoded: int64(25),
		},
		{
			example: hex("1864"),
			decoded: int64(100),
		},
		{
			example: hex("1903e8"),
			decoded: int64(1000),
		},
		{
			example: hex("1a000f4240"),
			decoded: int64(1000000),
		},
		{
			example: hex("1b000000e8d4a51000"),
			decoded: int64(1000000000000),
		},
		{
			example: hex("1bffffffffffffffff"),
			reject:  "2^64-1 overflows int64 and falling back to float64 (as with JSON) loses distinction between float and integer",
		},
		{
			example: hex("c249010000000000000000"),
			reject:  "decoding tagged positive bigint value to interface{} can't reproduce this value without losing distinction between float and integer",
		},
		{
			example: hex("3bffffffffffffffff"),
			reject:  "-2^64-1 overflows int64 and falling back to float64 (as with JSON) loses distinction between float and integer",
		},
		{
			example: hex("c349010000000000000000"),
			reject:  "-18446744073709551617 overflows int64 and falling back to float64 (as with JSON) loses distinction between float and integer",
		},
		{
			example: hex("20"),
			decoded: int64(-1),
		},
		{
			example: hex("29"),
			decoded: int64(-10),
		},
		{
			example: hex("3863"),
			decoded: int64(-100),
		},
		{
			example: hex("3903e7"),
			decoded: int64(-1000),
		},
		{
			example: hex("f90000"),
			decoded: 0.0,
		},
		{
			example: hex("f98000"),
			decoded: math.Copysign(0, -1),
		},
		{
			example: hex("f93c00"),
			decoded: 1.0,
		},
		{
			example: hex("fb3ff199999999999a"),
			decoded: 1.1,
		},
		{
			example: hex("f93e00"),
			decoded: 1.5,
		},
		{
			example: hex("f97bff"),
			decoded: 65504.0,
		},
		{
			example: hex("fa47c35000"),
			decoded: 100000.0,
		},
		{
			example: hex("fa7f7fffff"),
			decoded: 3.4028234663852886e+38,
		},
		{
			example: hex("fb7e37e43c8800759c"),
			decoded: 1.0e+300,
		},
		{
			example: hex("f90001"),
			decoded: 5.960464477539063e-8,
		},
		{
			example: hex("f90400"),
			decoded: 0.00006103515625,
		},
		{
			example: hex("f9c400"),
			decoded: -4.0,
		},
		{
			example: hex("fbc010666666666666"),
			decoded: -4.1,
		},
		{
			example: hex("f97c00"),
			reject:  "floating-point NaN and infinities are not accepted",
		},
		{
			example: hex("f97e00"),
			reject:  "floating-point NaN and infinities are not accepted",
		},
		{
			example: hex("f9fc00"),
			reject:  "floating-point NaN and infinities are not accepted",
		},
		{
			example: hex("fa7f800000"),
			reject:  "floating-point NaN and infinities are not accepted",
		},
		{
			example: hex("fa7fc00000"),
			reject:  "floating-point NaN and infinities are not accepted",
		},
		{
			example: hex("faff800000"),
			reject:  "floating-point NaN and infinities are not accepted",
		},
		{
			example: hex("fb7ff0000000000000"),
			reject:  "floating-point NaN and infinities are not accepted",
		},
		{
			example: hex("fb7ff8000000000000"),
			reject:  "floating-point NaN and infinities are not accepted",
		},
		{
			example: hex("fbfff0000000000000"),
			reject:  "floating-point NaN and infinities are not accepted",
		},
		{
			example: hex("f4"),
			decoded: false,
		},
		{
			example: hex("f5"),
			decoded: true,
		},
		{
			example: hex("f6"),
			decoded: nil,
		},
		{
			example: hex("f7"),
			reject:  "only simple values false, true, and null have a clear analog",
			fixme:   "the undefined simple value should not successfully decode as nil",
		},
		{
			example: hex("f0"),
			reject:  "only simple values false, true, and null have a clear analog",
			fixme:   "simple values other than false, true, and null should be rejected",
		},
		{
			example: hex("f8ff"),
			reject:  "only simple values false, true, and null have a clear analog",
			fixme:   "simple values other than false, true, and null should be rejected",
		},
		{
			example: hex("c074323031332d30332d32315432303a30343a30305a"),
			decoded: "2013-03-21T20:04:00Z",
			encoded: hex("54323031332d30332d32315432303a30343a30305a"),
			reasons: []string{
				reasonByteString,
				reasonTimeToInterface,
			},
		},
		{
			example: hex("c11a514b67b0"),
			decoded: "2013-03-21T20:04:00Z",
			encoded: hex("54323031332d30332d32315432303a30343a30305a"),
			reasons: []string{
				reasonByteString,
				reasonTimeToInterface,
			},
		},
		{
			example: hex("c1fb41d452d9ec200000"),
			decoded: "2013-03-21T20:04:00.5Z",
			encoded: hex("56323031332d30332d32315432303a30343a30302e355a"),
			reasons: []string{
				reasonByteString,
				reasonTimeToInterface,
			},
		},
		{
			example: hex("d74401020304"),
			decoded: "\x01\x02\x03\x04",
			encoded: hex("4401020304"),
			reasons: []string{
				reasonTagIgnored,
			},
		},
		{
			example: hex("d818456449455446"),
			decoded: "dIETF",
			encoded: hex("456449455446"),
			reasons: []string{
				reasonTagIgnored,
			},
		},
		{
			example: hex("d82076687474703a2f2f7777772e6578616d706c652e636f6d"),
			decoded: "http://www.example.com",
			encoded: hex("56687474703a2f2f7777772e6578616d706c652e636f6d"),
			reasons: []string{
				reasonByteString,
				reasonTagIgnored,
			},
		},
		{
			example: hex("40"),
			decoded: "",
		},
		{
			example: hex("4401020304"),
			decoded: "\x01\x02\x03\x04",
		},
		{
			example: hex("60"),
			decoded: "",
			encoded: hex("40"),
			reasons: []string{
				reasonByteString,
			},
		},
		{
			example: hex("6161"),
			decoded: "a",
			encoded: hex("4161"),
			reasons: []string{
				reasonByteString,
			},
		},
		{
			example: hex("6449455446"),
			decoded: "IETF",
			encoded: hex("4449455446"),
			reasons: []string{
				reasonByteString,
			},
		},
		{
			example: hex("62225c"),
			decoded: "\"\\",
			encoded: hex("42225c"),
			reasons: []string{
				reasonByteString,
			},
		},
		{
			example: hex("62c3bc"),
			decoded: "ü",
			encoded: hex("42c3bc"),
			reasons: []string{
				reasonByteString,
			},
		},
		{
			example: hex("63e6b0b4"),
			decoded: "水",
			encoded: hex("43e6b0b4"),
			reasons: []string{
				reasonByteString,
			},
		},
		{
			example: hex("64f0908591"),
			decoded: "𐅑",
			encoded: hex("44f0908591"),
			reasons: []string{
				reasonByteString,
			},
		},
		{
			example: hex("80"),
			decoded: []interface{}{},
		},
		{
			example: hex("83010203"),
			decoded: []interface{}{int64(1), int64(2), int64(3)},
		},
		{
			example: hex("8301820203820405"),
			decoded: []interface{}{int64(1), []interface{}{int64(2), int64(3)}, []interface{}{int64(4), int64(5)}},
		},
		{
			example: hex("98190102030405060708090a0b0c0d0e0f101112131415161718181819"),
			decoded: []interface{}{int64(1), int64(2), int64(3), int64(4), int64(5), int64(6), int64(7), int64(8), int64(9), int64(10), int64(11), int64(12), int64(13), int64(14), int64(15), int64(16), int64(17), int64(18), int64(19), int64(20), int64(21), int64(22), int64(23), int64(24), int64(25)},
		},
		{
			example: hex("a0"),
			decoded: map[string]interface{}{},
		},
		{
			example: hex("a201020304"),
			reject:  "integer map keys don't correspond with field names or unstructured keys",
		},
		{
			example: hex("a26161016162820203"),
			decoded: map[string]interface{}{
				"a": int64(1),
				"b": []interface{}{int64(2), int64(3)},
			},
			encoded: hex("a24161014162820203"),
			reasons: []string{
				reasonByteString,
			},
		},
		{
			example: hex("826161a161626163"),
			decoded: []interface{}{
				"a",
				map[string]interface{}{"b": "c"},
			},
			encoded: hex("824161a141624163"),
			reasons: []string{
				reasonByteString,
			},
		},
		{
			example: hex("a56161614161626142616361436164614461656145"),
			decoded: map[string]interface{}{
				"a": "A",
				"b": "B",
				"c": "C",
				"d": "D",
				"e": "E",
			},
			encoded: hex("a54161414141624142416341434164414441654145"),
			reasons: []string{
				reasonByteString,
			},
		},
		{
			example: hex("5f42010243030405ff"),
			decoded: "\x01\x02\x03\x04\x05",
			encoded: hex("450102030405"),
			reasons: []string{
				reasonStringFixedLength,
			},
		},
		{
			example: hex("7f657374726561646d696e67ff"),
			decoded: "streaming",
			encoded: hex("4973747265616d696e67"),
			reasons: []string{
				reasonByteString,
				reasonStringFixedLength,
			},
		},
		{
			example: hex("9fff"),
			decoded: []interface{}{},
			encoded: hex("80"),
			reasons: []string{
				reasonArrayFixedLength,
			},
		},
		{
			example: hex("9f018202039f0405ffff"),
			decoded: []interface{}{
				int64(1),
				[]interface{}{int64(2), int64(3)},
				[]interface{}{int64(4), int64(5)},
			},
			encoded: hex("8301820203820405"),
			reasons: []string{
				reasonArrayFixedLength,
			},
		},
		{
			example: hex("9f01820203820405ff"),
			decoded: []interface{}{
				int64(1),
				[]interface{}{int64(2), int64(3)},
				[]interface{}{int64(4), int64(5)},
			},
			encoded: hex("8301820203820405"),
			reasons: []string{
				reasonArrayFixedLength,
			},
		},
		{
			example: hex("83018202039f0405ff"),
			decoded: []interface{}{
				int64(1),
				[]interface{}{int64(2), int64(3)},
				[]interface{}{int64(4), int64(5)},
			},
			encoded: hex("8301820203820405"),
			reasons: []string{
				reasonArrayFixedLength,
			},
		},
		{
			example: hex("83019f0203ff820405"),
			decoded: []interface{}{
				int64(1),
				[]interface{}{int64(2), int64(3)},
				[]interface{}{int64(4), int64(5)},
			},
			encoded: hex("8301820203820405"),
			reasons: []string{
				reasonArrayFixedLength,
			},
		},
		{
			example: hex("9f0102030405060708090a0b0c0d0e0f101112131415161718181819ff"),
			decoded: []interface{}{
				int64(1), int64(2), int64(3), int64(4), int64(5),
				int64(6), int64(7), int64(8), int64(9), int64(10),
				int64(11), int64(12), int64(13), int64(14), int64(15),
				int64(16), int64(17), int64(18), int64(19), int64(20),
				int64(21), int64(22), int64(23), int64(24), int64(25),
			},
			encoded: hex("98190102030405060708090a0b0c0d0e0f101112131415161718181819"),
			reasons: []string{
				reasonArrayFixedLength,
			},
		},
		{
			example: hex("bf61610161629f0203ffff"),
			decoded: map[string]interface{}{
				"a": int64(1),
				"b": []interface{}{int64(2), int64(3)},
			},
			encoded: hex("a24161014162820203"),
			reasons: []string{
				reasonArrayFixedLength,
				reasonByteString,
				reasonMapFixedLength,
			},
		},
		{
			example: hex("826161bf61626163ff"),
			decoded: []interface{}{"a", map[string]interface{}{"b": "c"}},
			encoded: hex("824161a141624163"),
			reasons: []string{
				reasonByteString,
				reasonMapFixedLength,
			},
		},
		{
			example: hex("bf6346756ef563416d7421ff"),
			decoded: map[string]interface{}{
				"Amt": int64(-2),
				"Fun": true,
			},
			encoded: hex("a243416d74214346756ef5"),
			reasons: []string{
				reasonByteString,
				reasonMapFixedLength,
				reasonMapSorted,
			},
		},
	} {
		t.Run(fmt.Sprintf("%x", tc.example), func(t *testing.T) {
			if tc.fixme != "" {
				t.Skip(tc.fixme) // TODO: Remove once all cases are fixed.
			}

			var decoded interface{}
			err := modes.Decode.Unmarshal(tc.example, &decoded)
			if err != nil {
				if tc.reject != "" {
					t.Logf("expected decode error (%s) occurred: %v", tc.reject, err)
					return
				}
				t.Fatalf("unexpected decode error: %v", err)
			} else if tc.reject != "" {
				t.Fatalf("expected decode error (%v) did not occur", tc.reject)
			}

			if !eq.DeepEqual(tc.decoded, decoded) {
				t.Fatal(cmp.Diff(tc.decoded, decoded))
			}

			actual, err := modes.Encode.Marshal(decoded)
			if err != nil {
				t.Fatal(err)
			}

			expected := tc.example
			if tc.encoded != nil {
				expected = tc.encoded
				if len(tc.reasons) == 0 {
					t.Fatal("invalid test case: missing reasons for difference between the example encoding and the actual encoding")
				}
				diff := cmp.Diff(tc.example, tc.encoded)
				if diff == "" {
					t.Fatal("invalid test case: no difference between the example encoding and the expected re-encoding")
				}
				t.Logf("expecting the following differences from the example encoding on re-encode:\n%s", diff)
				t.Logf("reasons for encoding differences:")
				for _, reason := range tc.reasons {
					t.Logf("- %s", reason)
				}

			}

			if diff := cmp.Diff(expected, actual); diff != "" {
				t.Errorf("re-encoded object differs from expected:\n%s", diff)
			}
		})
	}
}
