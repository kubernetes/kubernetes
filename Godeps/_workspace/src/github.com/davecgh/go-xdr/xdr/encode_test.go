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

package xdr

import (
	"fmt"
	"math"
	"reflect"
	"testing"
	"time"
)

// TestMarshal executes all of the tests described by marshalTests.
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
		subTest{"bar", 3},                  // Q
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
		in   interface{}
		want []byte
		err  error
	}{
		// interface
		{testInterface, []byte{0x00, 0x00, 0x00, 0x11}, nil},
		{&testInterface, []byte{0x00, 0x00, 0x00, 0x11}, nil},

		// int8 - XDR Integer
		{int8(0), []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{int8(64), []byte{0x00, 0x00, 0x00, 0x40}, nil},
		{int8(127), []byte{0x00, 0x00, 0x00, 0x7F}, nil},
		{int8(-1), []byte{0xFF, 0xFF, 0xFF, 0xFF}, nil},
		{int8(-128), []byte{0xFF, 0xFF, 0xFF, 0x80}, nil},

		// uint8 - XDR Unsigned Integer
		{uint8(0), []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{uint8(64), []byte{0x00, 0x00, 0x00, 0x40}, nil},
		{uint8(255), []byte{0x00, 0x00, 0x00, 0xFF}, nil},

		// int16 - XDR Integer
		{int16(0), []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{int16(1024), []byte{0x00, 0x00, 0x04, 0x00}, nil},
		{int16(32767), []byte{0x00, 0x00, 0x7F, 0xFF}, nil},
		{int16(-1), []byte{0xFF, 0xFF, 0xFF, 0xFF}, nil},
		{int16(-32768), []byte{0xFF, 0xFF, 0x80, 0x00}, nil},

		// uint16 - XDR Unsigned Integer
		{uint16(0), []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{uint16(1024), []byte{0x00, 0x00, 0x04, 0x00}, nil},
		{uint16(65535), []byte{0x00, 0x00, 0xFF, 0xFF}, nil},

		// int32 - XDR Integer
		{int32(0), []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{int32(262144), []byte{0x00, 0x04, 0x00, 0x00}, nil},
		{int32(2147483647), []byte{0x7F, 0xFF, 0xFF, 0xFF}, nil},
		{int32(-1), []byte{0xFF, 0xFF, 0xFF, 0xFF}, nil},
		{int32(-2147483648), []byte{0x80, 0x00, 0x00, 0x00}, nil},

		// uint32 - XDR Unsigned Integer
		{uint32(0), []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{uint32(262144), []byte{0x00, 0x04, 0x00, 0x00}, nil},
		{uint32(4294967295), []byte{0xFF, 0xFF, 0xFF, 0xFF}, nil},

		// int64 - XDR Hyper Integer
		{int64(0), []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},
		{int64(1 << 34), []byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, nil},
		{int64(1 << 42), []byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},
		{int64(9223372036854775807), []byte{0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, nil},
		{int64(-1), []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, nil},
		{int64(-9223372036854775808), []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},

		// uint64 - XDR Unsigned Hyper Integer
		{uint64(0), []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},
		{uint64(1 << 34), []byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, nil},
		{uint64(1 << 42), []byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},
		{uint64(18446744073709551615), []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, nil},
		{uint64(9223372036854775808), []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},

		// bool - XDR Integer
		{false, []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{true, []byte{0x00, 0x00, 0x00, 0x01}, nil},

		// float32 - XDR Floating-Point
		{float32(0), []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{float32(3.14), []byte{0x40, 0x48, 0xF5, 0xC3}, nil},
		{float32(1234567.0), []byte{0x49, 0x96, 0xB4, 0x38}, nil},
		{float32(math.Inf(-1)), []byte{0xFF, 0x80, 0x00, 0x00}, nil},
		{float32(math.Inf(0)), []byte{0x7F, 0x80, 0x00, 0x00}, nil},

		// float64 - XDR Double-precision Floating-Point
		{float64(0), []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},
		{float64(3.141592653589793), []byte{0x40, 0x09, 0x21, 0xfb, 0x54, 0x44, 0x2d, 0x18}, nil},
		{float64(math.Inf(-1)), []byte{0xFF, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},
		{float64(math.Inf(0)), []byte{0x7F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},

		// string - XDR String
		{"", []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{"xdr", []byte{0x00, 0x00, 0x00, 0x03, 0x78, 0x64, 0x72, 0x00}, nil},
		{"τ=2π", []byte{0x00, 0x00, 0x00, 0x06, 0xCF, 0x84, 0x3D, 0x32, 0xCF, 0x80, 0x00, 0x00}, nil},

		// []byte - XDR Variable Opaque
		{[]byte{0x01}, []byte{0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00}, nil},
		{[]byte{0x01, 0x02, 0x03}, []byte{0x00, 0x00, 0x00, 0x03, 0x01, 0x02, 0x03, 0x00}, nil},

		// [#]byte - XDR Fixed Opaque
		{[1]byte{0x01}, []byte{0x01, 0x00, 0x00, 0x00}, nil}, // No & here to test unaddressable arrays
		{&[2]byte{0x01, 0x02}, []byte{0x01, 0x02, 0x00, 0x00}, nil},
		{&[3]byte{0x01, 0x02, 0x03}, []byte{0x01, 0x02, 0x03, 0x00}, nil},
		{&[4]byte{0x01, 0x02, 0x03, 0x04}, []byte{0x01, 0x02, 0x03, 0x04}, nil},
		{&[5]byte{0x01, 0x02, 0x03, 0x04, 0x05}, []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00}, nil},

		// []<type> - XDR Variable-Length Array
		{&[]int16{512, 1024, 2048},
			[]byte{0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00},
			nil},
		{[]bool{true, false}, []byte{0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00}, nil},

		// [#]<type> - XDR Fixed-Length Array
		{&[2]uint32{512, 1024}, []byte{0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00}, nil},

		// map[string]uint32
		{map[string]uint32{"map1": 1},
			[]byte{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x6D, 0x61, 0x70, 0x31, 0x00, 0x00, 0x00, 0x01},
			nil},

		// time.Time - XDR String per RFC3339
		{time.Unix(1396581888, 0).UTC(),
			[]byte{
				0x00, 0x00, 0x00, 0x14, 0x32, 0x30, 0x31, 0x34,
				0x2d, 0x30, 0x34, 0x2d, 0x30, 0x34, 0x54, 0x30,
				0x33, 0x3a, 0x32, 0x34, 0x3a, 0x34, 0x38, 0x5a,
			}, nil},

		// struct - XDR Structure -- test struct contains all supported types
		{&structMarshalTestIn, structMarshalTestWant, nil},

		// Expected errors
		{nilInterface, nil, &MarshalError{ErrorCode: ErrNilInterface}},
		{&nilInterface, nil, &MarshalError{ErrorCode: ErrNilInterface}},
		{testChan, nil, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{&testChan, nil, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{testFunc, nil, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{&testFunc, nil, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{testComplex64, nil, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{&testComplex64, nil, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{testComplex128, nil, &MarshalError{ErrorCode: ErrUnsupportedType}},
		{&testComplex128, nil, &MarshalError{ErrorCode: ErrUnsupportedType}},
	}

	for i, test := range tests {
		rv, err := Marshal(test.in)
		if reflect.TypeOf(err) != reflect.TypeOf(test.err) {
			t.Errorf("Marshal #%d failed to detect error - got: %v <%T> want: %T",
				i, err, err, test.err)
			continue
		}
		if rerr, ok := err.(*MarshalError); ok {
			if terr, ok := test.err.(*MarshalError); ok {
				if rerr.ErrorCode != terr.ErrorCode {
					t.Errorf("Marshal #%d failed to detect error code - got: %v want: %v",
						i, rerr.ErrorCode, terr.ErrorCode)
					continue
				}
				// Got expected error.  Move on to the next test.
				continue
			}
		}

		if len(rv) != len(test.want) {
			t.Errorf("Marshal #%d len got: %v want: %v\n", i, len(rv), len(test.want))
			continue
		}
		if !reflect.DeepEqual(rv, test.want) {
			t.Errorf("Marshal #%d got: %v want: %v\n", i, rv, test.want)
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

func (f encodeFunc) String() string {
	if s := encodeFuncStrings[f]; s != "" {
		return s
	}
	return fmt.Sprintf("Unknown encodeFunc (%d)", f)
}

// TestEncoder executes all of the tests described by encodeTests.
func TestEncoder(t *testing.T) {
	tests := []struct {
		f    encodeFunc
		in   interface{}
		want []byte
		err  error
	}{
		// Bool
		{fEncodeBool, false, []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeBool, true, []byte{0x00, 0x00, 0x00, 0x01}, nil},

		// Double
		{fEncodeDouble, float64(0), []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeDouble, float64(3.141592653589793), []byte{0x40, 0x09, 0x21, 0xfb, 0x54, 0x44, 0x2d, 0x18}, nil},
		{fEncodeDouble, float64(math.Inf(-1)), []byte{0xFF, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeDouble, float64(math.Inf(0)), []byte{0x7F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},

		// Enum
		{fEncodeEnum, int32(0), []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeEnum, int32(1), []byte{0x00, 0x00, 0x00, 0x01}, nil},
		{fEncodeEnum, int32(2), nil, &MarshalError{ErrorCode: ErrBadEnumValue}},
		{fEncodeEnum, int32(1234), nil, &MarshalError{ErrorCode: ErrBadEnumValue}},

		// FixedOpaque
		{fEncodeFixedOpaque, []byte{0x01}, []byte{0x01, 0x00, 0x00, 0x00}, nil},
		{fEncodeFixedOpaque, []byte{0x01, 0x02}, []byte{0x01, 0x02, 0x00, 0x00}, nil},
		{fEncodeFixedOpaque, []byte{0x01, 0x02, 0x03}, []byte{0x01, 0x02, 0x03, 0x00}, nil},
		{fEncodeFixedOpaque, []byte{0x01, 0x02, 0x03, 0x04}, []byte{0x01, 0x02, 0x03, 0x04}, nil},
		{fEncodeFixedOpaque, []byte{0x01, 0x02, 0x03, 0x04, 0x05}, []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00}, nil},

		// Float
		{fEncodeFloat, float32(0), []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeFloat, float32(3.14), []byte{0x40, 0x48, 0xF5, 0xC3}, nil},
		{fEncodeFloat, float32(1234567.0), []byte{0x49, 0x96, 0xB4, 0x38}, nil},
		{fEncodeFloat, float32(math.Inf(-1)), []byte{0xFF, 0x80, 0x00, 0x00}, nil},
		{fEncodeFloat, float32(math.Inf(0)), []byte{0x7F, 0x80, 0x00, 0x00}, nil},

		// Hyper
		{fEncodeHyper, int64(0), []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeHyper, int64(1 << 34), []byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeHyper, int64(1 << 42), []byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeHyper, int64(9223372036854775807), []byte{0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, nil},
		{fEncodeHyper, int64(-1), []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, nil},
		{fEncodeHyper, int64(-9223372036854775808), []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},

		// Int
		{fEncodeInt, int32(0), []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeInt, int32(262144), []byte{0x00, 0x04, 0x00, 0x00}, nil},
		{fEncodeInt, int32(2147483647), []byte{0x7F, 0xFF, 0xFF, 0xFF}, nil},
		{fEncodeInt, int32(-1), []byte{0xFF, 0xFF, 0xFF, 0xFF}, nil},
		{fEncodeInt, int32(-2147483648), []byte{0x80, 0x00, 0x00, 0x00}, nil},

		// Opaque
		{fEncodeOpaque, []byte{0x01}, []byte{0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00}, nil},
		{fEncodeOpaque, []byte{0x01, 0x02, 0x03}, []byte{0x00, 0x00, 0x00, 0x03, 0x01, 0x02, 0x03, 0x00}, nil},

		// String
		{fEncodeString, "", []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeString, "xdr", []byte{0x00, 0x00, 0x00, 0x03, 0x78, 0x64, 0x72, 0x00}, nil},
		{fEncodeString, "τ=2π", []byte{0x00, 0x00, 0x00, 0x06, 0xCF, 0x84, 0x3D, 0x32, 0xCF, 0x80, 0x00, 0x00}, nil},

		// Uhyper
		{fEncodeUhyper, uint64(0), []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeUhyper, uint64(1 << 34), []byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeUhyper, uint64(1 << 42), []byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeUhyper, uint64(18446744073709551615), []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, nil},
		{fEncodeUhyper, uint64(9223372036854775808), []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, nil},

		// Uint
		{fEncodeUint, uint32(0), []byte{0x00, 0x00, 0x00, 0x00}, nil},
		{fEncodeUint, uint32(262144), []byte{0x00, 0x04, 0x00, 0x00}, nil},
		{fEncodeUint, uint32(4294967295), []byte{0xFF, 0xFF, 0xFF, 0xFF}, nil},
	}

	validEnums := make(map[int32]bool)
	validEnums[0] = true
	validEnums[1] = true

	var err error
	enc := NewEncoder()

	for i, test := range tests {
		err = nil
		enc.Reset()
		switch test.f {
		case fEncodeBool:
			in := test.in.(bool)
			err = enc.EncodeBool(in)
		case fEncodeDouble:
			in := test.in.(float64)
			err = enc.EncodeDouble(in)
		case fEncodeEnum:
			in := test.in.(int32)
			err = enc.EncodeEnum(in, validEnums)
		case fEncodeFixedOpaque:
			in := test.in.([]byte)
			err = enc.EncodeFixedOpaque(in)
		case fEncodeFloat:
			in := test.in.(float32)
			err = enc.EncodeFloat(in)
		case fEncodeHyper:
			in := test.in.(int64)
			err = enc.EncodeHyper(in)
		case fEncodeInt:
			in := test.in.(int32)
			err = enc.EncodeInt(in)
		case fEncodeOpaque:
			in := test.in.([]byte)
			err = enc.EncodeOpaque(in)
		case fEncodeString:
			in := test.in.(string)
			err = enc.EncodeString(in)
		case fEncodeUhyper:
			in := test.in.(uint64)
			err = enc.EncodeUhyper(in)
		case fEncodeUint:
			in := test.in.(uint32)
			err = enc.EncodeUint(in)
		default:
			t.Errorf("%v #%d unrecognized function", test.f, i)
			continue
		}
		if reflect.TypeOf(err) != reflect.TypeOf(test.err) {
			t.Errorf("%v #%d failed to detect error - got: %v <%T> want: %T",
				test.f, i, err, err, test.err)
			continue
		}
		if rerr, ok := err.(*MarshalError); ok {
			if terr, ok := test.err.(*MarshalError); ok {
				if rerr.ErrorCode != terr.ErrorCode {
					t.Errorf("%v #%d failed to detect error code - got: %v want: %v",
						test.f, i, rerr.ErrorCode, terr.ErrorCode)
					continue
				}
				// Got expected error.  Move on to the next test.
				continue
			}
		}

		rv := enc.Data()
		if len(rv) != len(test.want) {
			t.Errorf("%v #%d len got: %v want: %v\n", test.f, i, len(rv), len(test.want))
			continue
		}
		if !reflect.DeepEqual(rv, test.want) {
			t.Errorf("%v #%d got: %v want: %v\n", test.f, i, rv, test.want)
			continue
		}
	}
}
