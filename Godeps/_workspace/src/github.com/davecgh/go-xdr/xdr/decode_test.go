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

// subTest is used to allow testing of the Unmarshal function into struct fields
// which are structs themselves.
type subTest struct {
	A string
	B uint8
}

// allTypesTest is used to allow testing of the Unmarshal function into struct
// fields of all supported types.
type allTypesTest struct {
	A int8
	B uint8
	C int16
	D uint16
	E int32
	F uint32
	G int64
	H uint64
	I bool
	J float32
	K float64
	L string
	M []byte
	N [3]byte
	O []int16
	P [2]subTest
	Q subTest
	R map[string]uint32
	S time.Time
}

// TestUnmarshal executes all of the tests described by unmarshalTests.
func TestUnmarshal(t *testing.T) {
	// structTestIn is input data for the big struct test of all supported
	// types.
	structTestIn := []byte{
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
		0x00, 0x00, 0x00, 0x02, // R length
		0x00, 0x00, 0x00, 0x04, 0x6D, 0x61, 0x70, 0x31, // R key map1
		0x00, 0x00, 0x00, 0x01, // R value map1
		0x00, 0x00, 0x00, 0x04, 0x6D, 0x61, 0x70, 0x32, // R key map2
		0x00, 0x00, 0x00, 0x02, // R value map2
		0x00, 0x00, 0x00, 0x14, 0x32, 0x30, 0x31, 0x34,
		0x2d, 0x30, 0x34, 0x2d, 0x30, 0x34, 0x54, 0x30,
		0x33, 0x3a, 0x32, 0x34, 0x3a, 0x34, 0x38, 0x5a, // S
	}

	// structTestWant is the expected output after unmarshalling
	// structTestIn.
	structTestWant := allTypesTest{
		127,                                     // A
		255,                                     // B
		32767,                                   // C
		65535,                                   // D
		2147483647,                              // E
		4294967295,                              // F
		9223372036854775807,                     // G
		18446744073709551615,                    // H
		true,                                    // I
		3.14,                                    // J
		3.141592653589793,                       // K
		"xdr",                                   // L
		[]byte{1, 2, 3, 4},                      // M
		[3]byte{1, 2, 3},                        // N
		[]int16{512, 1024, 2048},                // O
		[2]subTest{{"one", 1}, {"two", 2}},      // P
		subTest{"bar", 3},                       // Q
		map[string]uint32{"map1": 1, "map2": 2}, // R
		time.Unix(1396581888, 0).UTC(),          // S
	}

	tests := []struct {
		in   []byte
		want interface{}
		err  error
	}{
		// int8 - XDR Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, int8(0), nil},
		{[]byte{0x00, 0x00, 0x00, 0x40}, int8(64), nil},
		{[]byte{0x00, 0x00, 0x00, 0x7F}, int8(127), nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, int8(-1), nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0x80}, int8(-128), nil},
		// Expected Failures -- 128, -129 overflow int8
		{[]byte{0x00, 0x00, 0x00, 0x80}, int8(0), &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0xFF, 0xFF, 0xFF, 0x7F}, int8(0), &UnmarshalError{ErrorCode: ErrOverflow}},

		// uint8 - XDR Unsigned Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, uint8(0), nil},
		{[]byte{0x00, 0x00, 0x00, 0x40}, uint8(64), nil},
		{[]byte{0x00, 0x00, 0x00, 0xFF}, uint8(255), nil},
		// Expected Failures -- 256, -1 overflow uint8
		{[]byte{0x00, 0x00, 0x01, 0x00}, uint8(0), &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, uint8(0), &UnmarshalError{ErrorCode: ErrOverflow}},

		// int16 - XDR Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, int16(0), nil},
		{[]byte{0x00, 0x00, 0x04, 0x00}, int16(1024), nil},
		{[]byte{0x00, 0x00, 0x7F, 0xFF}, int16(32767), nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, int16(-1), nil},
		{[]byte{0xFF, 0xFF, 0x80, 0x00}, int16(-32768), nil},
		// Expected Failures -- 32768, -32769 overflow int16
		{[]byte{0x00, 0x00, 0x80, 0x00}, int16(0), &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0xFF, 0xFF, 0x7F, 0xFF}, int16(0), &UnmarshalError{ErrorCode: ErrOverflow}},

		// uint16 - XDR Unsigned Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, uint16(0), nil},
		{[]byte{0x00, 0x00, 0x04, 0x00}, uint16(1024), nil},
		{[]byte{0x00, 0x00, 0xFF, 0xFF}, uint16(65535), nil},
		// Expected Failures -- 65536, -1 overflow uint16
		{[]byte{0x00, 0x01, 0x00, 0x00}, uint16(0), &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, uint16(0), &UnmarshalError{ErrorCode: ErrOverflow}},

		// int32 - XDR Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, int32(0), nil},
		{[]byte{0x00, 0x04, 0x00, 0x00}, int32(262144), nil},
		{[]byte{0x7F, 0xFF, 0xFF, 0xFF}, int32(2147483647), nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, int32(-1), nil},
		{[]byte{0x80, 0x00, 0x00, 0x00}, int32(-2147483648), nil},

		// uint32 - XDR Unsigned Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, uint32(0), nil},
		{[]byte{0x00, 0x04, 0x00, 0x00}, uint32(262144), nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, uint32(4294967295), nil},

		// int64 - XDR Hyper Integer
		{[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, int64(0), nil},
		{[]byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, int64(1 << 34), nil},
		{[]byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, int64(1 << 42), nil},
		{[]byte{0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, int64(9223372036854775807), nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, int64(-1), nil},
		{[]byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, int64(-9223372036854775808), nil},

		// uint64 - XDR Unsigned Hyper Integer
		{[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, uint64(0), nil},
		{[]byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, uint64(1 << 34), nil},
		{[]byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, uint64(1 << 42), nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, uint64(18446744073709551615), nil},
		{[]byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, uint64(9223372036854775808), nil},

		// bool - XDR Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, false, nil},
		{[]byte{0x00, 0x00, 0x00, 0x01}, true, nil},
		// Expected Failures -- only 0 or 1 is a valid bool
		{[]byte{0x01, 0x00, 0x00, 0x00}, true, &UnmarshalError{ErrorCode: ErrBadEnumValue}},
		{[]byte{0x00, 0x00, 0x40, 0x00}, true, &UnmarshalError{ErrorCode: ErrBadEnumValue}},

		// float32 - XDR Floating-Point
		{[]byte{0x00, 0x00, 0x00, 0x00}, float32(0), nil},
		{[]byte{0x40, 0x48, 0xF5, 0xC3}, float32(3.14), nil},
		{[]byte{0x49, 0x96, 0xB4, 0x38}, float32(1234567.0), nil},
		{[]byte{0xFF, 0x80, 0x00, 0x00}, float32(math.Inf(-1)), nil},
		{[]byte{0x7F, 0x80, 0x00, 0x00}, float32(math.Inf(0)), nil},

		// float64 - XDR Double-precision Floating-Point
		{[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, float64(0), nil},
		{[]byte{0x40, 0x09, 0x21, 0xfb, 0x54, 0x44, 0x2d, 0x18}, float64(3.141592653589793), nil},
		{[]byte{0xFF, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, float64(math.Inf(-1)), nil},
		{[]byte{0x7F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, float64(math.Inf(0)), nil},

		// string - XDR String
		{[]byte{0x00, 0x00, 0x00, 0x00}, "", nil},
		{[]byte{0x00, 0x00, 0x00, 0x03, 0x78, 0x64, 0x72, 0x00}, "xdr", nil},
		{[]byte{0x00, 0x00, 0x00, 0x06, 0xCF, 0x84, 0x3D, 0x32, 0xCF, 0x80, 0x00, 0x00}, "τ=2π", nil},
		// Expected Failure -- String len 255 larger than available bytes
		{[]byte{0x00, 0x00, 0x00, 0xFF}, "", &UnmarshalError{ErrorCode: ErrUnexpectedEnd}},

		// []byte - XDR Variable Opaque
		{[]byte{0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00}, []byte{0x01}, nil},
		{[]byte{0x00, 0x00, 0x00, 0x03, 0x01, 0x02, 0x03, 0x00}, []byte{0x01, 0x02, 0x03}, nil},
		// Expected Failure -- 255 bytes of opaque data larger than available bytes
		{[]byte{0x00, 0x00, 0x00, 0xFF}, []byte{}, &UnmarshalError{ErrorCode: ErrUnexpectedEnd}},

		// [#]byte - XDR Fixed Opaque
		{[]byte{0x01, 0x00, 0x00, 0x00}, [1]byte{0x01}, nil},
		{[]byte{0x01, 0x02, 0x00, 0x00}, [2]byte{0x01, 0x02}, nil},
		{[]byte{0x01, 0x02, 0x03, 0x00}, [3]byte{0x01, 0x02, 0x03}, nil},
		{[]byte{0x01, 0x02, 0x03, 0x04}, [4]byte{0x01, 0x02, 0x03, 0x04}, nil},
		{[]byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00}, [5]byte{0x01, 0x02, 0x03, 0x04, 0x05}, nil},
		// Expected Failure -- fixed opaque data not padded
		{[]byte{0x01}, [1]byte{}, &UnmarshalError{ErrorCode: ErrUnexpectedEnd}},

		// []<type> - XDR Variable-Length Array
		{[]byte{0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00},
			[]int16{512, 1024, 2048}, nil},
		{[]byte{0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00}, []bool{true, false}, nil},
		// Expected Failure -- 2 entries in array - not enough bytes
		{[]byte{0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01}, []bool{}, &UnmarshalError{ErrorCode: ErrUnexpectedEnd}},

		// [#]<type> - XDR Fixed-Length Array
		{[]byte{0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00}, [2]uint32{512, 1024}, nil},
		{[]byte{0x00, 0x00, 0x02, 0x00}, [2]uint32{}, &UnmarshalError{ErrorCode: ErrUnexpectedEnd}},

		// map[string]uint32
		{[]byte{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x6D, 0x61, 0x70, 0x31, 0x00, 0x00, 0x00, 0x01},
			map[string]uint32{"map1": 1}, nil},
		// Expected Failure -- 1 map element - not enough bytes
		{[]byte{0x00, 0x00, 0x00, 0x01}, map[string]uint32{}, &UnmarshalError{ErrorCode: ErrUnexpectedEnd}},

		// time.Time - XDR String per RFC3339
		{[]byte{
			0x00, 0x00, 0x00, 0x14, 0x32, 0x30, 0x31, 0x34,
			0x2d, 0x30, 0x34, 0x2d, 0x30, 0x34, 0x54, 0x30,
			0x33, 0x3a, 0x32, 0x34, 0x3a, 0x34, 0x38, 0x5a,
		}, time.Unix(1396581888, 0).UTC(), nil},

		// struct - XDR Structure -- test struct contains all supported types
		{structTestIn, structTestWant, nil},
	}

	for i, test := range tests {
		// Create a new pointer to the appropriate type
		want := reflect.New(reflect.TypeOf(test.want)).Interface()
		rest, err := Unmarshal(test.in, want)
		if reflect.TypeOf(err) != reflect.TypeOf(test.err) {
			t.Errorf("Unmarshal #%d failed to detect error - got: %v <%T> want: %T",
				i, err, err, test.err)
			continue
		}
		if rerr, ok := err.(*UnmarshalError); ok {
			if terr, ok := test.err.(*UnmarshalError); ok {
				if rerr.ErrorCode != terr.ErrorCode {
					t.Errorf("Unmarshal #%d failed to detect error code - got: %v want: %v",
						i, rerr.ErrorCode, terr.ErrorCode)
					continue
				}
				// Got expected error.  Move on to the next test.
				continue
			}
		}

		if len(rest) != 0 {
			t.Errorf("Unmarshal #%d len got: %v want: %v\n", i, len(rest), 0)
			continue
		}
		wantElem := reflect.Indirect(reflect.ValueOf(want)).Interface()
		if !reflect.DeepEqual(wantElem, test.want) {
			t.Errorf("Unmarshal #%d got: %v want: %v\n", i, wantElem, test.want)
			continue
		}
	}
}

// decodeFunc is used to identify which public function of the Decoder object
// a test applies to.
type decodeFunc int

const (
	fDecodeBool decodeFunc = iota
	fDecodeDouble
	fDecodeEnum
	fDecodeFixedOpaque
	fDecodeFloat
	fDecodeHyper
	fDecodeInt
	fDecodeOpaque
	fDecodeString
	fDecodeUhyper
	fDecodeUint
)

// Map of decodeFunc values to names for pretty printing.
var decodeFuncStrings = map[decodeFunc]string{
	fDecodeBool:        "DecodeBool",
	fDecodeDouble:      "DecodeDouble",
	fDecodeEnum:        "DecodeEnum",
	fDecodeFixedOpaque: "DecodeFixedOpaque",
	fDecodeFloat:       "DecodeFloat",
	fDecodeHyper:       "DecodeHyper",
	fDecodeInt:         "DecodeInt",
	fDecodeOpaque:      "DecodeOpaque",
	fDecodeString:      "DecodeString",
	fDecodeUhyper:      "DecodeUhyper",
	fDecodeUint:        "DecodeUint",
}

func (f decodeFunc) String() string {
	if s := decodeFuncStrings[f]; s != "" {
		return s
	}
	return fmt.Sprintf("Unknown decodeFunc (%d)", f)
}

// TestDecoder executes all of the tests described by decodeTests.
func TestDecoder(t *testing.T) {
	tests := []struct {
		f    decodeFunc
		in   []byte
		want interface{}
		err  error
	}{
		// Bool
		{fDecodeBool, []byte{0x00, 0x00, 0x00, 0x00}, false, nil},
		{fDecodeBool, []byte{0x00, 0x00, 0x00, 0x01}, true, nil},
		// Expected Failures -- only 0 or 1 is a valid bool
		{fDecodeBool, []byte{0x01, 0x00, 0x00, 0x00}, true, &UnmarshalError{ErrorCode: ErrBadEnumValue}},
		{fDecodeBool, []byte{0x00, 0x00, 0x40, 0x00}, true, &UnmarshalError{ErrorCode: ErrBadEnumValue}},

		// Double
		{fDecodeDouble, []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, float64(0), nil},
		{fDecodeDouble, []byte{0x40, 0x09, 0x21, 0xfb, 0x54, 0x44, 0x2d, 0x18}, float64(3.141592653589793), nil},
		{fDecodeDouble, []byte{0xFF, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, float64(math.Inf(-1)), nil},
		{fDecodeDouble, []byte{0x7F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, float64(math.Inf(0)), nil},

		// Enum
		{fDecodeEnum, []byte{0x00, 0x00, 0x00, 0x00}, int32(0), nil},
		{fDecodeEnum, []byte{0x00, 0x00, 0x00, 0x01}, int32(1), nil},
		{fDecodeEnum, []byte{0x00, 0x00, 0x00, 0x02}, nil, &UnmarshalError{ErrorCode: ErrBadEnumValue}},
		{fDecodeEnum, []byte{0x12, 0x34, 0x56, 0x78}, nil, &UnmarshalError{ErrorCode: ErrBadEnumValue}},

		// FixedOpaque
		{fDecodeFixedOpaque, []byte{0x01, 0x00, 0x00, 0x00}, []byte{0x01}, nil},
		{fDecodeFixedOpaque, []byte{0x01, 0x02, 0x00, 0x00}, []byte{0x01, 0x02}, nil},
		{fDecodeFixedOpaque, []byte{0x01, 0x02, 0x03, 0x00}, []byte{0x01, 0x02, 0x03}, nil},
		{fDecodeFixedOpaque, []byte{0x01, 0x02, 0x03, 0x04}, []byte{0x01, 0x02, 0x03, 0x04}, nil},
		{fDecodeFixedOpaque, []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00}, []byte{0x01, 0x02, 0x03, 0x04, 0x05}, nil},
		// Expected Failure -- fixed opaque data not padded
		{fDecodeFixedOpaque, []byte{0x01}, []byte{0x00}, &UnmarshalError{ErrorCode: ErrUnexpectedEnd}},

		// Float
		{fDecodeFloat, []byte{0x00, 0x00, 0x00, 0x00}, float32(0), nil},
		{fDecodeFloat, []byte{0x40, 0x48, 0xF5, 0xC3}, float32(3.14), nil},
		{fDecodeFloat, []byte{0x49, 0x96, 0xB4, 0x38}, float32(1234567.0), nil},
		{fDecodeFloat, []byte{0xFF, 0x80, 0x00, 0x00}, float32(math.Inf(-1)), nil},
		{fDecodeFloat, []byte{0x7F, 0x80, 0x00, 0x00}, float32(math.Inf(0)), nil},

		// Hyper
		{fDecodeHyper, []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, int64(0), nil},
		{fDecodeHyper, []byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, int64(1 << 34), nil},
		{fDecodeHyper, []byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, int64(1 << 42), nil},
		{fDecodeHyper, []byte{0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, int64(9223372036854775807), nil},
		{fDecodeHyper, []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, int64(-1), nil},
		{fDecodeHyper, []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, int64(-9223372036854775808), nil},

		// Int
		{fDecodeInt, []byte{0x00, 0x00, 0x00, 0x00}, int32(0), nil},
		{fDecodeInt, []byte{0x00, 0x04, 0x00, 0x00}, int32(262144), nil},
		{fDecodeInt, []byte{0x7F, 0xFF, 0xFF, 0xFF}, int32(2147483647), nil},
		{fDecodeInt, []byte{0xFF, 0xFF, 0xFF, 0xFF}, int32(-1), nil},
		{fDecodeInt, []byte{0x80, 0x00, 0x00, 0x00}, int32(-2147483648), nil},

		// Opaque
		{fDecodeOpaque, []byte{0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00}, []byte{0x01}, nil},
		{fDecodeOpaque, []byte{0x00, 0x00, 0x00, 0x03, 0x01, 0x02, 0x03, 0x00}, []byte{0x01, 0x02, 0x03}, nil},
		// Expected Failure -- 255 bytes of opaque data larger than available bytes
		{fDecodeOpaque, []byte{0x00, 0x00, 0x00, 0xFF}, []byte{}, &UnmarshalError{ErrorCode: ErrUnexpectedEnd}},

		// String
		{fDecodeString, []byte{0x00, 0x00, 0x00, 0x00}, "", nil},
		{fDecodeString, []byte{0x00, 0x00, 0x00, 0x03, 0x78, 0x64, 0x72, 0x00}, "xdr", nil},
		{fDecodeString, []byte{0x00, 0x00, 0x00, 0x06, 0xCF, 0x84, 0x3D, 0x32, 0xCF, 0x80, 0x00, 0x00}, "τ=2π", nil},
		// Expected Failure -- String len 255 larger than available bytes
		{fDecodeString, []byte{0x00, 0x00, 0x00, 0xFF}, "", &UnmarshalError{ErrorCode: ErrUnexpectedEnd}},

		// Uhyper
		{fDecodeUhyper, []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, uint64(0), nil},
		{fDecodeUhyper, []byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, uint64(1 << 34), nil},
		{fDecodeUhyper, []byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, uint64(1 << 42), nil},
		{fDecodeUhyper, []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, uint64(18446744073709551615), nil},
		{fDecodeUhyper, []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, uint64(9223372036854775808), nil},

		// Uint
		{fDecodeUint, []byte{0x00, 0x00, 0x00, 0x00}, uint32(0), nil},
		{fDecodeUint, []byte{0x00, 0x04, 0x00, 0x00}, uint32(262144), nil},
		{fDecodeUint, []byte{0xFF, 0xFF, 0xFF, 0xFF}, uint32(4294967295), nil},
	}

	validEnums := make(map[int32]bool)
	validEnums[0] = true
	validEnums[1] = true

	var rv interface{}
	var err error

	for i, test := range tests {
		err = nil
		dec := NewDecoder(test.in)
		switch test.f {
		case fDecodeBool:
			rv, err = dec.DecodeBool()
		case fDecodeDouble:
			rv, err = dec.DecodeDouble()
		case fDecodeEnum:
			rv, err = dec.DecodeEnum(validEnums)
		case fDecodeFixedOpaque:
			want := test.want.([]byte)
			rv, err = dec.DecodeFixedOpaque(int32(len(want)))
		case fDecodeFloat:
			rv, err = dec.DecodeFloat()
		case fDecodeHyper:
			rv, err = dec.DecodeHyper()
		case fDecodeInt:
			rv, err = dec.DecodeInt()
		case fDecodeOpaque:
			rv, err = dec.DecodeOpaque()
		case fDecodeString:
			rv, err = dec.DecodeString()
		case fDecodeUhyper:
			rv, err = dec.DecodeUhyper()
		case fDecodeUint:
			rv, err = dec.DecodeUint()
		default:
			t.Errorf("%v #%d unrecognized function", test.f, i)
			continue
		}
		if reflect.TypeOf(err) != reflect.TypeOf(test.err) {
			t.Errorf("%v #%d failed to detect error - got: %v <%T> want: %T",
				test.f, i, err, err, test.err)
			continue
		}
		if rerr, ok := err.(*UnmarshalError); ok {
			if terr, ok := test.err.(*UnmarshalError); ok {
				if rerr.ErrorCode != terr.ErrorCode {
					t.Errorf("%v #%d failed to detect error code - got: %v want: %v",
						test.f, i, rerr.ErrorCode, terr.ErrorCode)
					continue
				}
				// Got expected error.  Move on to the next test.
				continue
			}
		}

		if !reflect.DeepEqual(rv, test.want) {
			t.Errorf("%v #%d got: %v want: %v\n", test.f, i, rv, test.want)
			continue
		}
	}
}
