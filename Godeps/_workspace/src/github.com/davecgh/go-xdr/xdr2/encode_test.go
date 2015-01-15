/*
 * Copyright (c) 2012-2014 Dave Collins <dave@davec.name>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

package xdr_test

import (
	"fmt"
	"math"
	"reflect"
	"testing"
	"time"

	. "github.com/davecgh/go-xdr/xdr2"
)

// testExpectedMRet is a convenience method to test an expected number of bytes
// written and error for a marshal.
func testExpectedMRet(t *testing.T, name string, n, wantN int, err, wantErr error) bool {
	// First ensure the number of bytes written is the expected value.  The
	// bytes read should be accurate even when an error occurs.
	if n != wantN {
		t.Errorf("%s: unexpected num bytes written - got: %v want: %v\n",
			name, n, wantN)
		return false
	}

	// Next check for the expected error.
	if reflect.TypeOf(err) != reflect.TypeOf(wantErr) {
		t.Errorf("%s: failed to detect error - got: %v <%[2]T> want: %T",
			name, err, wantErr)
		return false
	}
	if rerr, ok := err.(*MarshalError); ok {
		if werr, ok := wantErr.(*MarshalError); ok {
			if rerr.ErrorCode != werr.ErrorCode {
				t.Errorf("%s: failed to detect error code - "+
					"got: %v want: %v", name,
					rerr.ErrorCode, werr.ErrorCode)
				return false
			}
		}
	}

	return true
}

// TestMarshal ensures the Marshal function works properly with all types.
func TestMarshal(t *testing.T) {
	// Variables for various unsupported Marshal types.
	var nilInterface interface{}
	var testChan chan int
	var testFunc func()
	var testComplex64 complex64
	var testComplex128 complex128

	// testInterface is used to test Marshal with values nested in an
	// interface.
	testInterface := interface{}(17)

	// structMarshalTestIn is input data for the big struct test of all
	// supported types.
	structMarshalTestIn := allTypesTest{
		127,                                // A
		255,                                // B
		32767,                              // C
		65535,                              // D
		2147483647,                         // E
		4294967295,                         // F
		9223372036854775807,                // G
		18446744073709551615,               // H
		true,                               // I
		3.14,                               // J
		3.141592653589793,                  // K
		"xdr",                              // L
		[]byte{1, 2, 3, 4},                 // M
		[3]byte{1, 2, 3},                   // N
		[]int16{512, 1024, 2048},           // O
		[2]subTest{{"one", 1}, {"two", 2}}, // P
		&subTest{"bar", 3},                 // Q
		map[string]uint32{"map1": 1},       // R
		time.Unix(1396581888, 0).UTC(),     // S
	}

	// structMarshalTestWant is the expected output after marshalling
	// structMarshalTestIn.
	structMarshalTestWant := []byte{
		0x00, 0x00, 0x00, 0x7F, // A
		0x00, 0x00, 0x00, 0xFF, // B
		0x00, 0x00, 0x7F, 0xFF, // C
		0x00, 0x00, 0xFF, 0xFF, // D
		0x7F, 0xFF, 0xFF, 0xFF, // E
		0xFF, 0xFF, 0xFF, 0xFF, // F
		0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // G
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // H
		0x00, 0x00, 0x00, 0x01, // I
		0x40, 0x48, 0xF5, 0xC3, // J
		0x40, 0x09, 0x21, 0xfb, 0x54, 0x44, 0x2d, 0x18, // K
		0x00, 0x00, 0x00, 0x03, 0x78, 0x64, 0x72, 0x00, // L
		0x00, 0x00, 0x00, 0x04, 0x01, 0x02, 0x03, 0x04, // M
		0x01, 0x02, 0x03, 0x00, // N
		0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00,
		0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00, // O
		0x00, 0x00, 0x00, 0x03, 0x6F, 0x6E, 0x65, 0x00, // P[0].A
		0x00, 0x00, 0x00, 0x01, // P[0].B
		0x00, 0x00, 0x00, 0x03, 0x74, 0x77, 0x6F, 0x00, // P[1].A
		0x00, 0x00, 0x00, 0x02, // P[1].B
		0x00, 0x00, 0x00, 0x03, 0x62, 0x61, 0x72, 0x00, // Q.A
		0x00, 0x00, 0x00, 0x03, // Q.B
		0x00, 0x00, 0x00, 0x01, // R length
		0x00, 0x00, 0x00, 0x04, 0x6D, 0x61, 0x70, 0x31, // R key map1
		0x00, 0x00, 0x00, 0x01, // R value map1
		0x00, 0x00, 0x00, 0x14, 0x32, 0x30, 0x31, 0x34,
		0x2d, 0x30, 0x34, 0x2d, 0x30, 0x34, 0x54, 0x30,
		0x33, 0x3a, 0x32, 0x34, 0x3a, 0x34, 0x38, 0x5a, // S
	}

	tests := []struct {
		in        interface{} // input value
		wantBytes []byte      // expected bytes
		wantN     int         // expected/max number of bytes written
		err       error       // expected error
	}{
		// interface
		{testInterface, []byte{0x00, 0x00, 0x00, 0x11}, 4, nil},
		{&testInterface, []byte{0x00, 0x00, 0x00, 0x11}, 4, nil},

		// int8 - XDR Integer
		{int8(0), []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{int8(64), []byte{0x00, 0x00, 0x00, 0x40}, 4, nil},
		{int8(127), []byte{0x00, 0x00, 0x00, 0x7F}, 4, nil},
		{int8(-1), []byte{0xFF, 0xFF, 0xFF, 0xFF}, 4, nil},
		{int8(-128), []byte{0xFF, 0xFF, 0xFF, 0x80}, 4, nil},
		// Expected Failure -- Short write
		{int8(127), []byte{0x00, 0x00, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},

		// uint8 - XDR Unsigned Integer
		{uint8(0), []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{uint8(64), []byte{0x00, 0x00, 0x00, 0x40}, 4, nil},
		{uint8(255), []byte{0x00, 0x00, 0x00, 0xFF}, 4, nil},
		// Expected Failure -- Short write
		{uint8(255), []byte{0x00, 0x00}, 2, &MarshalError{ErrorCode: ErrIO}},

		// int16 - XDR Integer
		{int16(0), []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{int16(1024), []byte{0x00, 0x00, 0x04, 0x00}, 4, nil},
		{int16(32767), []byte{0x00, 0x00, 0x7F, 0xFF}, 4, nil},
		{int16(-1), []byte{0xFF, 0xFF, 0xFF, 0xFF}, 4, nil},
		{int16(-32768), []byte{0xFF, 0xFF, 0x80, 0x00}, 4, nil},
		// Expected Failure -- Short write
		{int16(-32768), []byte{0xFF}, 1, &MarshalError{ErrorCode: ErrIO}},

		// uint16 - XDR Unsigned Integer
		{uint16(0), []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{uint16(1024), []byte{0x00, 0x00, 0x04, 0x00}, 4, nil},
		{uint16(65535), []byte{0x00, 0x00, 0xFF, 0xFF}, 4, nil},
		// Expected Failure -- Short write
		{uint16(65535), []byte{0x00, 0x00}, 2, &MarshalError{ErrorCode: ErrIO}},

		// int32 - XDR Integer
		{int32(0), []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{int32(262144), []byte{0x00, 0x04, 0x00, 0x00}, 4, nil},
		{int32(2147483647), []byte{0x7F, 0xFF, 0xFF, 0xFF}, 4, nil},
		{int32(-1), []byte{0xFF, 0xFF, 0xFF, 0xFF}, 4, nil},
		{int32(-2147483648), []byte{0x80, 0x00, 0x00, 0x00}, 4, nil},
		// Expected Failure -- Short write
		{int32(2147483647), []byte{0x7F, 0xFF, 0xFF}, 3, &MarshalError{ErrorCode: ErrIO}},

		// uint32 - XDR Unsigned Integer
		{uint32(0), []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{uint32(262144), []byte{0x00, 0x04, 0x00, 0x00}, 4, nil},
		{uint32(4294967295), []byte{0xFF, 0xFF, 0xFF, 0xFF}, 4, nil},
		// Expected Failure -- Short write
		{uint32(262144), []byte{0x00, 0x04, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},

		// int64 - XDR Hyper Integer
		{int64(0), []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{int64(1 << 34), []byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{int64(1 << 42), []byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{int64(9223372036854775807), []byte{0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, 8, nil},
		{int64(-1), []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, 8, nil},
		{int64(-9223372036854775808), []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		// Expected Failure -- Short write
		{int64(-9223372036854775808), []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 7, &MarshalError{ErrorCode: ErrIO}},

		// uint64 - XDR Unsigned Hyper Integer
		{uint64(0), []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{uint64(1 << 34), []byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{uint64(1 << 42), []byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{uint64(18446744073709551615), []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, 8, nil},
		{uint64(9223372036854775808), []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		// Expected Failure -- Short write
		{uint64(9223372036854775808), []byte{0x80}, 1, &MarshalError{ErrorCode: ErrIO}},

		// bool - XDR Integer
		{false, []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{true, []byte{0x00, 0x00, 0x00, 0x01}, 4, nil},
		// Expected Failure -- Short write
		{true, []byte{0x00, 0x00, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},

		// float32 - XDR Floating-Point
		{float32(0), []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{float32(3.14), []byte{0x40, 0x48, 0xF5, 0xC3}, 4, nil},
		{float32(1234567.0), []byte{0x49, 0x96, 0xB4, 0x38}, 4, nil},
		{float32(math.Inf(-1)), []byte{0xFF, 0x80, 0x00, 0x00}, 4, nil},
		{float32(math.Inf(0)), []byte{0x7F, 0x80, 0x00, 0x00}, 4, nil},
		// Expected Failure -- Short write
		{float32(3.14), []byte{0x40, 0x48, 0xF5}, 3, &MarshalError{ErrorCode: ErrIO}},

		// float64 - XDR Double-precision Floating-Point
		{float64(0), []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{float64(3.141592653589793), []byte{0x40, 0x09, 0x21, 0xfb, 0x54, 0x44, 0x2d, 0x18}, 8, nil},
		{float64(math.Inf(-1)), []byte{0xFF, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{float64(math.Inf(0)), []byte{0x7F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		// Expected Failure -- Short write
		{float64(3.141592653589793), []byte{0x40, 0x09, 0x21, 0xfb, 0x54, 0x44, 0x2d}, 7, &MarshalError{ErrorCode: ErrIO}},

		// string - XDR String
		{"", []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{"xdr", []byte{0x00, 0x00, 0x00, 0x03, 0x78, 0x64, 0x72, 0x00}, 8, nil},
		{"τ=2π", []byte{0x00, 0x00, 0x00, 0x06, 0xCF, 0x84, 0x3D, 0x32, 0xCF, 0x80, 0x00, 0x00}, 12, nil},
		// Expected Failures -- Short write in length and payload
		{"xdr", []byte{0x00, 0x00, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},
		{"xdr", []byte{0x00, 0x00, 0x00, 0x03, 0x78}, 5, &MarshalError{ErrorCode: ErrIO}},

		// []byte - XDR Variable Opaque
		{[]byte{0x01}, []byte{0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00}, 8, nil},
		{[]byte{0x01, 0x02, 0x03}, []byte{0x00, 0x00, 0x00, 0x03, 0x01, 0x02, 0x03, 0x00}, 8, nil},
		// Expected Failures -- Short write in length and payload
		{[]byte{0x01}, []byte{0x00, 0x00, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},
		{[]byte{0x01}, []byte{0x00, 0x00, 0x00, 0x01, 0x01}, 5, &MarshalError{ErrorCode: ErrIO}},

		// [#]byte - XDR Fixed Opaque
		{[1]byte{0x01}, []byte{0x01, 0x00, 0x00, 0x00}, 4, nil}, // No & here to test unaddressable arrays
		{&[2]byte{0x01, 0x02}, []byte{0x01, 0x02, 0x00, 0x00}, 4, nil},
		{&[3]byte{0x01, 0x02, 0x03}, []byte{0x01, 0x02, 0x03, 0x00}, 4, nil},
		{&[4]byte{0x01, 0x02, 0x03, 0x04}, []byte{0x01, 0x02, 0x03, 0x04}, 4, nil},
		{&[5]byte{0x01, 0x02, 0x03, 0x04, 0x05}, []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00}, 8, nil},
		// Expected Failure -- Short write
		{[1]byte{0x01}, []byte{0x01, 0x00, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},

		// []<type> - XDR Variable-Length Array
		{&[]int16{512, 1024, 2048},
			[]byte{0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00},
			16, nil},
		{[]bool{true, false}, []byte{0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00}, 12, nil},
		// Expected Failures -- Short write in number of elements and
		// payload
		{[]bool{true, false}, []byte{0x00, 0x00, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},
		{[]bool{true, false}, []byte{0x00, 0x00, 0x00, 0x02, 0x00}, 5, &MarshalError{ErrorCode: ErrIO}},

		// [#]<type> - XDR Fixed-Length Array
		{&[2]uint32{512, 1024}, []byte{0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00}, 8, nil},
		// Expected Failures -- Short write in number of elements and
		// payload
		{[2]uint32{512, 1024}, []byte{0x00, 0x00, 0x02}, 3, &MarshalError{ErrorCode: ErrIO}},
		{[2]uint32{512, 1024}, []byte{0x00, 0x00, 0x02, 0x00, 0x00}, 5, &MarshalError{ErrorCode: ErrIO}},

		// map[string]uint32
		{map[string]uint32{"map1": 1},
			[]byte{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x6D, 0x61, 0x70, 0x31, 0x00, 0x00, 0x00, 0x01},
			16, nil},
		// Expected Failures -- Short write in number of elements, key,
		// and payload
		{map[string]uint32{"map1": 1}, []byte{0x00, 0x00, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},
		{map[string]uint32{"map1": 1}, []byte{0x00, 0x00, 0x00, 0x01, 0x00}, 5, &MarshalError{ErrorCode: ErrIO}},
		{map[string]uint32{"map1": 1}, []byte{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04}, 8, &MarshalError{ErrorCode: ErrIO}},
		{map[string]uint32{"map1": 1},
			[]byte{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x6D, 0x61, 0x70, 0x31},
			12, &MarshalError{ErrorCode: ErrIO}},

		// time.Time - XDR String per RFC3339
		{time.Unix(1396581888, 0).UTC(),
			[]byte{
				0x00, 0x00, 0x00, 0x14, 0x32, 0x30, 0x31, 0x34,
				0x2d, 0x30, 0x34, 0x2d, 0x30, 0x34, 0x54, 0x30,
				0x33, 0x3a, 0x32, 0x34, 0x3a, 0x34, 0x38, 0x5a,
			}, 24, nil},
		// Expected Failure -- Short write
		{time.Unix(1396581888, 0).UTC(), []byte{0x00, 0x00, 0x00, 0x14, 0x32, 0x30, 0x31, 0x34}, 8, &MarshalError{ErrorCode: ErrIO}},

		// struct - XDR Structure -- test struct contains all supported types
		{&structMarshalTestIn, structMarshalTestWant, len(structMarshalTestWant), nil},
		{opaqueStruct{[]uint8{1}, [1]uint8{2}},
			[]byte{
				0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
				0x00, 0x00, 0x00, 0x02,
			}, 12, nil},
		// Expected Failures -- Short write in variable length,
		// variable payload, and fixed payload.
		{structMarshalTestIn, structMarshalTestWant[:3], 3, &MarshalError{ErrorCode: ErrIO}},
		{opaqueStruct{[]uint8{1}, [1]uint8{2}}, []byte{0x00, 0x00, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},
		{opaqueStruct{[]uint8{1}, [1]uint8{2}}, []byte{0x00, 0x00, 0x00, 0x01}, 4, &MarshalError{ErrorCode: ErrIO}},
		{opaqueStruct{[]uint8{1}, [1]uint8{2}},
			[]byte{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01},
			8, &MarshalError{ErrorCode: ErrIO}},

		// Expected errors
		{nilInterface, []byte{}, 0, &MarshalError{ErrorCode: ErrNilInterface}},
		{&nilInterface, []byte{}, 0, &MarshalError{ErrorCode: ErrNilInterface}},
		{(*interface{})(nil), []byte{}, 0, &MarshalError{ErrorCode: ErrBadArguments}},
		{testChan, []byte{}, 0, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{&testChan, []byte{}, 0, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{testFunc, []byte{}, 0, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{&testFunc, []byte{}, 0, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{testComplex64, []byte{}, 0, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{&testComplex64, []byte{}, 0, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{testComplex128, []byte{}, 0, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{&testComplex128, []byte{}, 0, &MarshalError{ErrorCode: ErrUnsupportedType}},
	}

	for i, test := range tests {
		data := newFixedWriter(test.wantN)
		n, err := Marshal(data, test.in)

		// First ensure the number of bytes written is the expected
		// value and the error is the expected one.
		testName := fmt.Sprintf("Marshal #%d", i)
		testExpectedMRet(t, testName, n, test.wantN, err, test.err)

		rv := data.Bytes()
		if len(rv) != len(test.wantBytes) {
			t.Errorf("%s: unexpected len - got: %v want: %v\n",
				testName, len(rv), len(test.wantBytes))
			continue
		}
		if !reflect.DeepEqual(rv, test.wantBytes) {
			t.Errorf("%s: unexpected result - got: %v want: %v\n",
				testName, rv, test.wantBytes)
			continue
		}
	}
}

// encodeFunc is used to identify which public function of the Encoder object
// a test applies to.
type encodeFunc int

const (
	fEncodeBool encodeFunc = iota
	fEncodeDouble
	fEncodeEnum
	fEncodeFixedOpaque
	fEncodeFloat
	fEncodeHyper
	fEncodeInt
	fEncodeOpaque
	fEncodeString
	fEncodeUhyper
	fEncodeUint
)

// Map of encodeFunc values to names for pretty printing.
var encodeFuncStrings = map[encodeFunc]string{
	fEncodeBool:        "EncodeBool",
	fEncodeDouble:      "EncodeDouble",
	fEncodeEnum:        "EncodeEnum",
	fEncodeFixedOpaque: "EncodeFixedOpaque",
	fEncodeFloat:       "EncodeFloat",
	fEncodeHyper:       "EncodeHyper",
	fEncodeInt:         "EncodeInt",
	fEncodeOpaque:      "EncodeOpaque",
	fEncodeString:      "EncodeString",
	fEncodeUhyper:      "EncodeUhyper",
	fEncodeUint:        "EncodeUint",
}

// String implements the fmt.Stringer interface and returns the encode function
// as a human-readable string.
func (f encodeFunc) String() string {
	if s := encodeFuncStrings[f]; s != "" {
		return s
	}
	return fmt.Sprintf("Unknown encodeFunc (%d)", f)
}

// TestEncoder ensures an Encoder works as intended.
func TestEncoder(t *testing.T) {
	tests := []struct {
		f         encodeFunc  // function to use to encode
		in        interface{} // input value
		wantBytes []byte      // expected bytes
		wantN     int         // expected number of bytes written
		err       error       // expected error
	}{
		// Bool
		{fEncodeBool, false, []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{fEncodeBool, true, []byte{0x00, 0x00, 0x00, 0x01}, 4, nil},
		// Expected Failure -- Short write
		{fEncodeBool, true, []byte{0x00, 0x00, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},

		// Double
		{fEncodeDouble, float64(0), []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{fEncodeDouble, float64(3.141592653589793), []byte{0x40, 0x09, 0x21, 0xfb, 0x54, 0x44, 0x2d, 0x18}, 8, nil},
		{fEncodeDouble, float64(math.Inf(-1)), []byte{0xFF, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{fEncodeDouble, float64(math.Inf(0)), []byte{0x7F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		// Expected Failure -- Short write
		{fEncodeDouble, float64(3.141592653589793), []byte{0x40, 0x09, 0x21, 0xfb, 0x54, 0x44, 0x2d}, 7, &MarshalError{ErrorCode: ErrIO}},

		// Enum
		{fEncodeEnum, int32(0), []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{fEncodeEnum, int32(1), []byte{0x00, 0x00, 0x00, 0x01}, 4, nil},
		// Expected Failures -- Invalid enum values
		{fEncodeEnum, int32(2), []byte{}, 0, &MarshalError{ErrorCode: ErrBadEnumValue}},
		{fEncodeEnum, int32(1234), []byte{}, 0, &MarshalError{ErrorCode: ErrBadEnumValue}},

		// FixedOpaque
		{fEncodeFixedOpaque, []byte{0x01}, []byte{0x01, 0x00, 0x00, 0x00}, 4, nil},
		{fEncodeFixedOpaque, []byte{0x01, 0x02}, []byte{0x01, 0x02, 0x00, 0x00}, 4, nil},
		{fEncodeFixedOpaque, []byte{0x01, 0x02, 0x03}, []byte{0x01, 0x02, 0x03, 0x00}, 4, nil},
		{fEncodeFixedOpaque, []byte{0x01, 0x02, 0x03, 0x04}, []byte{0x01, 0x02, 0x03, 0x04}, 4, nil},
		{fEncodeFixedOpaque, []byte{0x01, 0x02, 0x03, 0x04, 0x05}, []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00}, 8, nil},
		// Expected Failure -- Short write
		{fEncodeFixedOpaque, []byte{0x01}, []byte{0x01, 0x00, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},

		// Float
		{fEncodeFloat, float32(0), []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{fEncodeFloat, float32(3.14), []byte{0x40, 0x48, 0xF5, 0xC3}, 4, nil},
		{fEncodeFloat, float32(1234567.0), []byte{0x49, 0x96, 0xB4, 0x38}, 4, nil},
		{fEncodeFloat, float32(math.Inf(-1)), []byte{0xFF, 0x80, 0x00, 0x00}, 4, nil},
		{fEncodeFloat, float32(math.Inf(0)), []byte{0x7F, 0x80, 0x00, 0x00}, 4, nil},
		// Expected Failure -- Short write
		{fEncodeFloat, float32(3.14), []byte{0x40, 0x48, 0xF5}, 3, &MarshalError{ErrorCode: ErrIO}},

		// Hyper
		{fEncodeHyper, int64(0), []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{fEncodeHyper, int64(1 << 34), []byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{fEncodeHyper, int64(1 << 42), []byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{fEncodeHyper, int64(9223372036854775807), []byte{0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, 8, nil},
		{fEncodeHyper, int64(-1), []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, 8, nil},
		{fEncodeHyper, int64(-9223372036854775808), []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		// Expected Failure -- Short write
		{fEncodeHyper, int64(-9223372036854775808), []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 7, &MarshalError{ErrorCode: ErrIO}},

		// Int
		{fEncodeInt, int32(0), []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{fEncodeInt, int32(262144), []byte{0x00, 0x04, 0x00, 0x00}, 4, nil},
		{fEncodeInt, int32(2147483647), []byte{0x7F, 0xFF, 0xFF, 0xFF}, 4, nil},
		{fEncodeInt, int32(-1), []byte{0xFF, 0xFF, 0xFF, 0xFF}, 4, nil},
		{fEncodeInt, int32(-2147483648), []byte{0x80, 0x00, 0x00, 0x00}, 4, nil},
		// Expected Failure -- Short write
		{fEncodeInt, int32(2147483647), []byte{0x7F, 0xFF, 0xFF}, 3, &MarshalError{ErrorCode: ErrIO}},

		// Opaque
		{fEncodeOpaque, []byte{0x01}, []byte{0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00}, 8, nil},
		{fEncodeOpaque, []byte{0x01, 0x02, 0x03}, []byte{0x00, 0x00, 0x00, 0x03, 0x01, 0x02, 0x03, 0x00}, 8, nil},
		// Expected Failures -- Short write in length and payload
		{fEncodeOpaque, []byte{0x01}, []byte{0x00, 0x00, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},
		{fEncodeOpaque, []byte{0x01}, []byte{0x00, 0x00, 0x00, 0x01, 0x01}, 5, &MarshalError{ErrorCode: ErrIO}},

		// String
		{fEncodeString, "", []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{fEncodeString, "xdr", []byte{0x00, 0x00, 0x00, 0x03, 0x78, 0x64, 0x72, 0x00}, 8, nil},
		{fEncodeString, "τ=2π", []byte{0x00, 0x00, 0x00, 0x06, 0xCF, 0x84, 0x3D, 0x32, 0xCF, 0x80, 0x00, 0x00}, 12, nil},
		// Expected Failures -- Short write in length and payload
		{fEncodeString, "xdr", []byte{0x00, 0x00, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},
		{fEncodeString, "xdr", []byte{0x00, 0x00, 0x00, 0x03, 0x78}, 5, &MarshalError{ErrorCode: ErrIO}},

		// Uhyper
		{fEncodeUhyper, uint64(0), []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{fEncodeUhyper, uint64(1 << 34), []byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{fEncodeUhyper, uint64(1 << 42), []byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		{fEncodeUhyper, uint64(18446744073709551615), []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, 8, nil},
		{fEncodeUhyper, uint64(9223372036854775808), []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 8, nil},
		// Expected Failure -- Short write
		{fEncodeUhyper, uint64(9223372036854775808), []byte{0x80}, 1, &MarshalError{ErrorCode: ErrIO}},

		// Uint
		{fEncodeUint, uint32(0), []byte{0x00, 0x00, 0x00, 0x00}, 4, nil},
		{fEncodeUint, uint32(262144), []byte{0x00, 0x04, 0x00, 0x00}, 4, nil},
		{fEncodeUint, uint32(4294967295), []byte{0xFF, 0xFF, 0xFF, 0xFF}, 4, nil},
		// Expected Failure -- Short write
		{fEncodeUint, uint32(262144), []byte{0x00, 0x04, 0x00}, 3, &MarshalError{ErrorCode: ErrIO}},
	}

	validEnums := make(map[int32]bool)
	validEnums[0] = true
	validEnums[1] = true

	var err error
	var n int

	for i, test := range tests {
		err = nil
		data := newFixedWriter(test.wantN)
		enc := NewEncoder(data)
		switch test.f {
		case fEncodeBool:
			in := test.in.(bool)
			n, err = enc.EncodeBool(in)
		case fEncodeDouble:
			in := test.in.(float64)
			n, err = enc.EncodeDouble(in)
		case fEncodeEnum:
			in := test.in.(int32)
			n, err = enc.EncodeEnum(in, validEnums)
		case fEncodeFixedOpaque:
			in := test.in.([]byte)
			n, err = enc.EncodeFixedOpaque(in)
		case fEncodeFloat:
			in := test.in.(float32)
			n, err = enc.EncodeFloat(in)
		case fEncodeHyper:
			in := test.in.(int64)
			n, err = enc.EncodeHyper(in)
		case fEncodeInt:
			in := test.in.(int32)
			n, err = enc.EncodeInt(in)
		case fEncodeOpaque:
			in := test.in.([]byte)
			n, err = enc.EncodeOpaque(in)
		case fEncodeString:
			in := test.in.(string)
			n, err = enc.EncodeString(in)
		case fEncodeUhyper:
			in := test.in.(uint64)
			n, err = enc.EncodeUhyper(in)
		case fEncodeUint:
			in := test.in.(uint32)
			n, err = enc.EncodeUint(in)
		default:
			t.Errorf("%v #%d unrecognized function", test.f, i)
			continue
		}

		// First ensure the number of bytes written is the expected
		// value and the error is the expected one.
		testName := fmt.Sprintf("%v #%d", test.f, i)
		testExpectedMRet(t, testName, n, test.wantN, err, test.err)

		// Finally, ensure the written bytes are what is expected.
		rv := data.Bytes()
		if len(rv) != len(test.wantBytes) {
			t.Errorf("%s: unexpected len - got: %v want: %v\n",
				testName, len(rv), len(test.wantBytes))
			continue
		}
		if !reflect.DeepEqual(rv, test.wantBytes) {
			t.Errorf("%s: unexpected result - got: %v want: %v\n",
				testName, rv, test.wantBytes)
			continue
		}
	}
}

// TestMarshalCorners ensures the Marshal function properly handles various
// cases not already covered by the other tests.
func TestMarshalCorners(t *testing.T) {
	// Ensure encode of an invalid reflect value returns the expected
	// error.
	testName := "Encode invalid reflect value"
	expectedN := 0
	expectedErr := error(&MarshalError{ErrorCode: ErrUnsupportedType})
	expectedVal := []byte{}
	data := newFixedWriter(expectedN)
	n, err := TstEncode(data)(reflect.Value{})
	testExpectedMRet(t, testName, n, expectedN, err, expectedErr)
	if !reflect.DeepEqual(data.Bytes(), expectedVal) {
		t.Errorf("%s: unexpected result - got: %x want: %x\n",
			testName, data.Bytes(), expectedVal)
	}

	// Ensure marshal of a struct with both exported and unexported fields
	// skips the unexported fields but still marshals to the exported
	// fields.
	type unexportedStruct struct {
		unexported int
		Exported   int
	}
	testName = "Marshal struct with exported and unexported fields"
	tstruct := unexportedStruct{0, 1}
	expectedN = 4
	expectedErr = nil
	expectedVal = []byte{0x00, 0x00, 0x00, 0x01}
	data = newFixedWriter(expectedN)
	n, err = Marshal(data, tstruct)
	testExpectedMRet(t, testName, n, expectedN, err, expectedErr)
	if !reflect.DeepEqual(data.Bytes(), expectedVal) {
		t.Errorf("%s: unexpected result - got: %x want: %x\n",
			testName, data.Bytes(), expectedVal)
	}

}
