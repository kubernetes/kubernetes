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
	"bytes"
	"fmt"
	"math"
	"reflect"
	"testing"
	"time"

	. "github.com/davecgh/go-xdr/xdr2"
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
	Q *subTest
	R map[string]uint32
	S time.Time
}

// opaqueStruct is used to test handling of uint8 slices and arrays.
type opaqueStruct struct {
	Slice []uint8  `xdropaque:"false"`
	Array [1]uint8 `xdropaque:"false"`
}

// testExpectedURet is a convenience method to test an expected number of bytes
// read and error for an unmarshal.
func testExpectedURet(t *testing.T, name string, n, wantN int, err, wantErr error) bool {
	// First ensure the number of bytes read is the expected value.  The
	// byes read should be accurate even when an error occurs.
	if n != wantN {
		t.Errorf("%s: unexpected num bytes read - got: %v want: %v\n",
			name, n, wantN)
		return false
	}

	// Next check for the expected error.
	if reflect.TypeOf(err) != reflect.TypeOf(wantErr) {
		t.Errorf("%s: failed to detect error - got: %v <%[2]T> want: %T",
			name, err, wantErr)
		return false
	}
	if rerr, ok := err.(*UnmarshalError); ok {
		if werr, ok := wantErr.(*UnmarshalError); ok {
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

// TestUnmarshal ensures the Unmarshal function works properly with all types.
func TestUnmarshal(t *testing.T) {
	// Variables for various unsupported Unmarshal types.
	var nilInterface interface{}
	var testChan chan int
	var testFunc func()
	var testComplex64 complex64
	var testComplex128 complex128

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
		&subTest{"bar", 3},                      // Q
		map[string]uint32{"map1": 1, "map2": 2}, // R
		time.Unix(1396581888, 0).UTC(),          // S
	}

	tests := []struct {
		in      []byte      // input bytes
		wantVal interface{} // expected value
		wantN   int         // expected number of bytes read
		err     error       // expected error
	}{
		// int8 - XDR Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, int8(0), 4, nil},
		{[]byte{0x00, 0x00, 0x00, 0x40}, int8(64), 4, nil},
		{[]byte{0x00, 0x00, 0x00, 0x7F}, int8(127), 4, nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, int8(-1), 4, nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0x80}, int8(-128), 4, nil},
		// Expected Failures -- 128, -129 overflow int8 and not enough
		// bytes
		{[]byte{0x00, 0x00, 0x00, 0x80}, int8(0), 4, &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0xFF, 0xFF, 0xFF, 0x7F}, int8(0), 4, &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0x00, 0x00, 0x00}, int8(0), 3, &UnmarshalError{ErrorCode: ErrIO}},

		// uint8 - XDR Unsigned Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, uint8(0), 4, nil},
		{[]byte{0x00, 0x00, 0x00, 0x40}, uint8(64), 4, nil},
		{[]byte{0x00, 0x00, 0x00, 0xFF}, uint8(255), 4, nil},
		// Expected Failures -- 256, -1 overflow uint8 and not enough
		// bytes
		{[]byte{0x00, 0x00, 0x01, 0x00}, uint8(0), 4, &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, uint8(0), 4, &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0x00, 0x00, 0x00}, uint8(0), 3, &UnmarshalError{ErrorCode: ErrIO}},

		// int16 - XDR Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, int16(0), 4, nil},
		{[]byte{0x00, 0x00, 0x04, 0x00}, int16(1024), 4, nil},
		{[]byte{0x00, 0x00, 0x7F, 0xFF}, int16(32767), 4, nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, int16(-1), 4, nil},
		{[]byte{0xFF, 0xFF, 0x80, 0x00}, int16(-32768), 4, nil},
		// Expected Failures -- 32768, -32769 overflow int16 and not
		// enough bytes
		{[]byte{0x00, 0x00, 0x80, 0x00}, int16(0), 4, &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0xFF, 0xFF, 0x7F, 0xFF}, int16(0), 4, &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0x00, 0x00, 0x00}, uint16(0), 3, &UnmarshalError{ErrorCode: ErrIO}},

		// uint16 - XDR Unsigned Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, uint16(0), 4, nil},
		{[]byte{0x00, 0x00, 0x04, 0x00}, uint16(1024), 4, nil},
		{[]byte{0x00, 0x00, 0xFF, 0xFF}, uint16(65535), 4, nil},
		// Expected Failures -- 65536, -1 overflow uint16 and not enough
		// bytes
		{[]byte{0x00, 0x01, 0x00, 0x00}, uint16(0), 4, &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, uint16(0), 4, &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0x00, 0x00, 0x00}, uint16(0), 3, &UnmarshalError{ErrorCode: ErrIO}},

		// int32 - XDR Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, int32(0), 4, nil},
		{[]byte{0x00, 0x04, 0x00, 0x00}, int32(262144), 4, nil},
		{[]byte{0x7F, 0xFF, 0xFF, 0xFF}, int32(2147483647), 4, nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, int32(-1), 4, nil},
		{[]byte{0x80, 0x00, 0x00, 0x00}, int32(-2147483648), 4, nil},
		// Expected Failure -- not enough bytes
		{[]byte{0x00, 0x00, 0x00}, int32(0), 3, &UnmarshalError{ErrorCode: ErrIO}},

		// uint32 - XDR Unsigned Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, uint32(0), 4, nil},
		{[]byte{0x00, 0x04, 0x00, 0x00}, uint32(262144), 4, nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, uint32(4294967295), 4, nil},
		// Expected Failure -- not enough bytes
		{[]byte{0x00, 0x00, 0x00}, uint32(0), 3, &UnmarshalError{ErrorCode: ErrIO}},

		// int64 - XDR Hyper Integer
		{[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, int64(0), 8, nil},
		{[]byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, int64(1 << 34), 8, nil},
		{[]byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, int64(1 << 42), 8, nil},
		{[]byte{0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, int64(9223372036854775807), 8, nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, int64(-1), 8, nil},
		{[]byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, int64(-9223372036854775808), 8, nil},
		// Expected Failures -- not enough bytes
		{[]byte{0x7f, 0xff, 0xff}, int64(0), 3, &UnmarshalError{ErrorCode: ErrIO}},
		{[]byte{0x7f, 0x00, 0xff, 0x00}, int64(0), 4, &UnmarshalError{ErrorCode: ErrIO}},

		// uint64 - XDR Unsigned Hyper Integer
		{[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, uint64(0), 8, nil},
		{[]byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, uint64(1 << 34), 8, nil},
		{[]byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, uint64(1 << 42), 8, nil},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, uint64(18446744073709551615), 8, nil},
		{[]byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, uint64(9223372036854775808), 8, nil},
		// Expected Failures -- not enough bytes
		{[]byte{0xff, 0xff, 0xff}, uint64(0), 3, &UnmarshalError{ErrorCode: ErrIO}},
		{[]byte{0xff, 0x00, 0xff, 0x00}, uint64(0), 4, &UnmarshalError{ErrorCode: ErrIO}},

		// bool - XDR Integer
		{[]byte{0x00, 0x00, 0x00, 0x00}, false, 4, nil},
		{[]byte{0x00, 0x00, 0x00, 0x01}, true, 4, nil},
		// Expected Failures -- only 0 or 1 is a valid bool
		{[]byte{0x01, 0x00, 0x00, 0x00}, true, 4, &UnmarshalError{ErrorCode: ErrBadEnumValue}},
		{[]byte{0x00, 0x00, 0x40, 0x00}, true, 4, &UnmarshalError{ErrorCode: ErrBadEnumValue}},

		// float32 - XDR Floating-Point
		{[]byte{0x00, 0x00, 0x00, 0x00}, float32(0), 4, nil},
		{[]byte{0x40, 0x48, 0xF5, 0xC3}, float32(3.14), 4, nil},
		{[]byte{0x49, 0x96, 0xB4, 0x38}, float32(1234567.0), 4, nil},
		{[]byte{0xFF, 0x80, 0x00, 0x00}, float32(math.Inf(-1)), 4, nil},
		{[]byte{0x7F, 0x80, 0x00, 0x00}, float32(math.Inf(0)), 4, nil},
		// Expected Failures -- not enough bytes
		{[]byte{0xff, 0xff}, float32(0), 2, &UnmarshalError{ErrorCode: ErrIO}},
		{[]byte{0xff, 0x00, 0xff}, float32(0), 3, &UnmarshalError{ErrorCode: ErrIO}},

		// float64 - XDR Double-precision Floating-Point
		{[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, float64(0), 8, nil},
		{[]byte{0x40, 0x09, 0x21, 0xfb, 0x54, 0x44, 0x2d, 0x18}, float64(3.141592653589793), 8, nil},
		{[]byte{0xFF, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, float64(math.Inf(-1)), 8, nil},
		{[]byte{0x7F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, float64(math.Inf(0)), 8, nil},
		// Expected Failures -- not enough bytes
		{[]byte{0xff, 0xff, 0xff}, float64(0), 3, &UnmarshalError{ErrorCode: ErrIO}},
		{[]byte{0xff, 0x00, 0xff, 0x00}, float64(0), 4, &UnmarshalError{ErrorCode: ErrIO}},

		// string - XDR String
		{[]byte{0x00, 0x00, 0x00, 0x00}, "", 4, nil},
		{[]byte{0x00, 0x00, 0x00, 0x03, 0x78, 0x64, 0x72, 0x00}, "xdr", 8, nil},
		{[]byte{0x00, 0x00, 0x00, 0x06, 0xCF, 0x84, 0x3D, 0x32, 0xCF, 0x80, 0x00, 0x00}, "τ=2π", 12, nil},
		// Expected Failures -- not enough bytes for length, length
		// larger than allowed, and len larger than available bytes.
		{[]byte{0x00, 0x00, 0xFF}, "", 3, &UnmarshalError{ErrorCode: ErrIO}},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, "", 4, &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0x00, 0x00, 0x00, 0xFF}, "", 4, &UnmarshalError{ErrorCode: ErrIO}},

		// []byte - XDR Variable Opaque
		{[]byte{0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00}, []byte{0x01}, 8, nil},
		{[]byte{0x00, 0x00, 0x00, 0x03, 0x01, 0x02, 0x03, 0x00}, []byte{0x01, 0x02, 0x03}, 8, nil},
		// Expected Failures -- not enough bytes for length, length
		// larger than allowed, and data larger than available bytes.
		{[]byte{0x00, 0x00, 0xFF}, []byte{}, 3, &UnmarshalError{ErrorCode: ErrIO}},
		{[]byte{0xFF, 0xFF, 0xFF, 0xFF}, []byte{}, 4, &UnmarshalError{ErrorCode: ErrOverflow}},
		{[]byte{0x00, 0x00, 0x00, 0xFF}, []byte{}, 4, &UnmarshalError{ErrorCode: ErrIO}},

		// [#]byte - XDR Fixed Opaque
		{[]byte{0x01, 0x00, 0x00, 0x00}, [1]byte{0x01}, 4, nil},
		{[]byte{0x01, 0x02, 0x00, 0x00}, [2]byte{0x01, 0x02}, 4, nil},
		{[]byte{0x01, 0x02, 0x03, 0x00}, [3]byte{0x01, 0x02, 0x03}, 4, nil},
		{[]byte{0x01, 0x02, 0x03, 0x04}, [4]byte{0x01, 0x02, 0x03, 0x04}, 4, nil},
		{[]byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00}, [5]byte{0x01, 0x02, 0x03, 0x04, 0x05}, 8, nil},
		// Expected Failure -- fixed opaque data not padded
		{[]byte{0x01}, [1]byte{}, 1, &UnmarshalError{ErrorCode: ErrIO}},

		// []<type> - XDR Variable-Length Array
		{[]byte{0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00},
			[]int16{512, 1024, 2048}, 16, nil},
		{[]byte{0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00}, []bool{true, false}, 12, nil},
		// Expected Failure -- 2 entries in array - not enough bytes
		{[]byte{0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01}, []bool{}, 8, &UnmarshalError{ErrorCode: ErrIO}},

		// [#]<type> - XDR Fixed-Length Array
		{[]byte{0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00}, [2]uint32{512, 1024}, 8, nil},
		// Expected Failure -- 2 entries in array - not enough bytes
		{[]byte{0x00, 0x00, 0x00, 0x02}, [2]uint32{}, 4, &UnmarshalError{ErrorCode: ErrIO}},

		// map[string]uint32
		{[]byte{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x6D, 0x61, 0x70, 0x31, 0x00, 0x00, 0x00, 0x01},
			map[string]uint32{"map1": 1}, 16, nil},
		// Expected Failures -- not enough bytes in length, 1 map
		// element no extra bytes, 1 map element not enough bytes for
		// key, 1 map element not enough bytes for value.
		{[]byte{0x00, 0x00, 0x00}, map[string]uint32{}, 3, &UnmarshalError{ErrorCode: ErrIO}},
		{[]byte{0x00, 0x00, 0x00, 0x01}, map[string]uint32{}, 4, &UnmarshalError{ErrorCode: ErrIO}},
		{[]byte{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00}, map[string]uint32{}, 7, &UnmarshalError{ErrorCode: ErrIO}},
		{[]byte{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x6D, 0x61, 0x70, 0x31},
			map[string]uint32{}, 12, &UnmarshalError{ErrorCode: ErrIO}},

		// time.Time - XDR String per RFC3339
		{[]byte{
			0x00, 0x00, 0x00, 0x14, 0x32, 0x30, 0x31, 0x34,
			0x2d, 0x30, 0x34, 0x2d, 0x30, 0x34, 0x54, 0x30,
			0x33, 0x3a, 0x32, 0x34, 0x3a, 0x34, 0x38, 0x5a,
		}, time.Unix(1396581888, 0).UTC(), 24, nil},
		// Expected Failures -- not enough bytes, improperly formatted
		// time
		{[]byte{0x00, 0x00, 0x00}, time.Time{}, 3, &UnmarshalError{ErrorCode: ErrIO}},
		{[]byte{0x00, 0x00, 0x00, 0x00}, time.Time{}, 4, &UnmarshalError{ErrorCode: ErrParseTime}},

		// struct - XDR Structure -- test struct contains all supported types
		{structTestIn, structTestWant, len(structTestIn), nil},
		{[]byte{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02},
			opaqueStruct{[]uint8{1}, [1]uint8{2}}, 12, nil},
		// Expected Failures -- normal struct not enough bytes, non
		// opaque data not enough bytes for slice, non opaque data not
		// enough bytes for slice.
		{[]byte{0x00, 0x00}, allTypesTest{}, 2, &UnmarshalError{ErrorCode: ErrIO}},
		{[]byte{0x00, 0x00, 0x00}, opaqueStruct{}, 3, &UnmarshalError{ErrorCode: ErrIO}},
		{[]byte{0x00, 0x00, 0x00, 0x00, 0x00}, opaqueStruct{}, 5, &UnmarshalError{ErrorCode: ErrIO}},

		// Expected errors
		{nil, nilInterface, 0, &UnmarshalError{ErrorCode: ErrNilInterface}},
		{nil, &nilInterface, 0, &UnmarshalError{ErrorCode: ErrNilInterface}},
		{nil, testChan, 0, &UnmarshalError{ErrorCode: ErrUnsupportedType}},
		{nil, &testChan, 0, &UnmarshalError{ErrorCode: ErrUnsupportedType}},
		{nil, testFunc, 0, &UnmarshalError{ErrorCode: ErrUnsupportedType}},
		{nil, &testFunc, 0, &UnmarshalError{ErrorCode: ErrUnsupportedType}},
		{nil, testComplex64, 0, &UnmarshalError{ErrorCode: ErrUnsupportedType}},
		{nil, &testComplex64, 0, &UnmarshalError{ErrorCode: ErrUnsupportedType}},
		{nil, testComplex128, 0, &UnmarshalError{ErrorCode: ErrUnsupportedType}},
		{nil, &testComplex128, 0, &UnmarshalError{ErrorCode: ErrUnsupportedType}},
	}

	for i, test := range tests {
		// Attempt to unmarshal to a non-pointer version of each
		// positive test type to ensure the appropriate error is
		// returned.
		if test.err == nil && test.wantVal != nil {
			testName := fmt.Sprintf("Unmarshal #%d (non-pointer)", i)
			wantErr := &UnmarshalError{ErrorCode: ErrBadArguments}

			wvt := reflect.TypeOf(test.wantVal)
			want := reflect.New(wvt).Elem().Interface()
			n, err := Unmarshal(bytes.NewReader(test.in), want)
			if !testExpectedURet(t, testName, n, 0, err, wantErr) {
				continue
			}
		}

		testName := fmt.Sprintf("Unmarshal #%d", i)
		// Create a new pointer to the appropriate type.
		var want interface{}
		if test.wantVal != nil {
			wvt := reflect.TypeOf(test.wantVal)
			want = reflect.New(wvt).Interface()
		}
		n, err := Unmarshal(bytes.NewReader(test.in), want)

		// First ensure the number of bytes read is the expected value
		// and the error is the expected one.
		if !testExpectedURet(t, testName, n, test.wantN, err, test.err) {
			continue
		}
		if test.err != nil {
			continue
		}

		// Finally, ensure the read value is the expected one.
		wantElem := reflect.Indirect(reflect.ValueOf(want)).Interface()
		if !reflect.DeepEqual(wantElem, test.wantVal) {
			t.Errorf("%s: unexpected result - got: %v want: %v\n",
				testName, wantElem, test.wantVal)
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

// String implements the fmt.Stringer interface and returns the encode function
// as a human-readable string.
func (f decodeFunc) String() string {
	if s := decodeFuncStrings[f]; s != "" {
		return s
	}
	return fmt.Sprintf("Unknown decodeFunc (%d)", f)
}

// TestDecoder ensures a Decoder works as intended.
func TestDecoder(t *testing.T) {
	tests := []struct {
		f       decodeFunc  // function to use to decode
		in      []byte      // input bytes
		wantVal interface{} // expected value
		wantN   int         // expected number of bytes read
		err     error       // expected error
	}{
		// Bool
		{fDecodeBool, []byte{0x00, 0x00, 0x00, 0x00}, false, 4, nil},
		{fDecodeBool, []byte{0x00, 0x00, 0x00, 0x01}, true, 4, nil},
		// Expected Failures -- only 0 or 1 is a valid bool
		{fDecodeBool, []byte{0x01, 0x00, 0x00, 0x00}, true, 4, &UnmarshalError{ErrorCode: ErrBadEnumValue}},
		{fDecodeBool, []byte{0x00, 0x00, 0x40, 0x00}, true, 4, &UnmarshalError{ErrorCode: ErrBadEnumValue}},

		// Double
		{fDecodeDouble, []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, float64(0), 8, nil},
		{fDecodeDouble, []byte{0x40, 0x09, 0x21, 0xfb, 0x54, 0x44, 0x2d, 0x18}, float64(3.141592653589793), 8, nil},
		{fDecodeDouble, []byte{0xFF, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, float64(math.Inf(-1)), 8, nil},
		{fDecodeDouble, []byte{0x7F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, float64(math.Inf(0)), 8, nil},

		// Enum
		{fDecodeEnum, []byte{0x00, 0x00, 0x00, 0x00}, int32(0), 4, nil},
		{fDecodeEnum, []byte{0x00, 0x00, 0x00, 0x01}, int32(1), 4, nil},
		{fDecodeEnum, []byte{0x00, 0x00, 0x00, 0x02}, nil, 4, &UnmarshalError{ErrorCode: ErrBadEnumValue}},
		{fDecodeEnum, []byte{0x12, 0x34, 0x56, 0x78}, nil, 4, &UnmarshalError{ErrorCode: ErrBadEnumValue}},
		{fDecodeEnum, []byte{0x00}, nil, 1, &UnmarshalError{ErrorCode: ErrIO}},

		// FixedOpaque
		{fDecodeFixedOpaque, []byte{0x01, 0x00, 0x00, 0x00}, []byte{0x01}, 4, nil},
		{fDecodeFixedOpaque, []byte{0x01, 0x02, 0x00, 0x00}, []byte{0x01, 0x02}, 4, nil},
		{fDecodeFixedOpaque, []byte{0x01, 0x02, 0x03, 0x00}, []byte{0x01, 0x02, 0x03}, 4, nil},
		{fDecodeFixedOpaque, []byte{0x01, 0x02, 0x03, 0x04}, []byte{0x01, 0x02, 0x03, 0x04}, 4, nil},
		{fDecodeFixedOpaque, []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00}, []byte{0x01, 0x02, 0x03, 0x04, 0x05}, 8, nil},
		// Expected Failure -- fixed opaque data not padded
		{fDecodeFixedOpaque, []byte{0x01}, []byte{0x00}, 1, &UnmarshalError{ErrorCode: ErrIO}},

		// Float
		{fDecodeFloat, []byte{0x00, 0x00, 0x00, 0x00}, float32(0), 4, nil},
		{fDecodeFloat, []byte{0x40, 0x48, 0xF5, 0xC3}, float32(3.14), 4, nil},
		{fDecodeFloat, []byte{0x49, 0x96, 0xB4, 0x38}, float32(1234567.0), 4, nil},
		{fDecodeFloat, []byte{0xFF, 0x80, 0x00, 0x00}, float32(math.Inf(-1)), 4, nil},
		{fDecodeFloat, []byte{0x7F, 0x80, 0x00, 0x00}, float32(math.Inf(0)), 4, nil},

		// Hyper
		{fDecodeHyper, []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, int64(0), 8, nil},
		{fDecodeHyper, []byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, int64(1 << 34), 8, nil},
		{fDecodeHyper, []byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, int64(1 << 42), 8, nil},
		{fDecodeHyper, []byte{0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, int64(9223372036854775807), 8, nil},
		{fDecodeHyper, []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, int64(-1), 8, nil},
		{fDecodeHyper, []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, int64(-9223372036854775808), 8, nil},

		// Int
		{fDecodeInt, []byte{0x00, 0x00, 0x00, 0x00}, int32(0), 4, nil},
		{fDecodeInt, []byte{0x00, 0x04, 0x00, 0x00}, int32(262144), 4, nil},
		{fDecodeInt, []byte{0x7F, 0xFF, 0xFF, 0xFF}, int32(2147483647), 4, nil},
		{fDecodeInt, []byte{0xFF, 0xFF, 0xFF, 0xFF}, int32(-1), 4, nil},
		{fDecodeInt, []byte{0x80, 0x00, 0x00, 0x00}, int32(-2147483648), 4, nil},

		// Opaque
		{fDecodeOpaque, []byte{0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00}, []byte{0x01}, 8, nil},
		{fDecodeOpaque, []byte{0x00, 0x00, 0x00, 0x03, 0x01, 0x02, 0x03, 0x00}, []byte{0x01, 0x02, 0x03}, 8, nil},
		// Expected Failures -- not enough bytes for length, length
		// larger than allowed, and data larger than available bytes.
		{fDecodeOpaque, []byte{0x00, 0x00, 0xFF}, []byte{}, 3, &UnmarshalError{ErrorCode: ErrIO}},
		{fDecodeOpaque, []byte{0xFF, 0xFF, 0xFF, 0xFF}, []byte{}, 4, &UnmarshalError{ErrorCode: ErrOverflow}},
		{fDecodeOpaque, []byte{0x7F, 0xFF, 0xFF, 0xFD}, []byte{}, 4, &UnmarshalError{ErrorCode: ErrOverflow}},
		{fDecodeOpaque, []byte{0x00, 0x00, 0x00, 0xFF}, []byte{}, 4, &UnmarshalError{ErrorCode: ErrIO}},

		// String
		{fDecodeString, []byte{0x00, 0x00, 0x00, 0x00}, "", 4, nil},
		{fDecodeString, []byte{0x00, 0x00, 0x00, 0x03, 0x78, 0x64, 0x72, 0x00}, "xdr", 8, nil},
		{fDecodeString, []byte{0x00, 0x00, 0x00, 0x06, 0xCF, 0x84, 0x3D, 0x32, 0xCF, 0x80, 0x00, 0x00}, "τ=2π", 12, nil},
		// Expected Failures -- not enough bytes for length, length
		// larger than allowed, and len larger than available bytes.
		{fDecodeString, []byte{0x00, 0x00, 0xFF}, "", 3, &UnmarshalError{ErrorCode: ErrIO}},
		{fDecodeString, []byte{0xFF, 0xFF, 0xFF, 0xFF}, "", 4, &UnmarshalError{ErrorCode: ErrOverflow}},
		{fDecodeString, []byte{0x00, 0x00, 0x00, 0xFF}, "", 4, &UnmarshalError{ErrorCode: ErrIO}},

		// Uhyper
		{fDecodeUhyper, []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, uint64(0), 8, nil},
		{fDecodeUhyper, []byte{0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00}, uint64(1 << 34), 8, nil},
		{fDecodeUhyper, []byte{0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00}, uint64(1 << 42), 8, nil},
		{fDecodeUhyper, []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, uint64(18446744073709551615), 8, nil},
		{fDecodeUhyper, []byte{0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, uint64(9223372036854775808), 8, nil},

		// Uint
		{fDecodeUint, []byte{0x00, 0x00, 0x00, 0x00}, uint32(0), 4, nil},
		{fDecodeUint, []byte{0x00, 0x04, 0x00, 0x00}, uint32(262144), 4, nil},
		{fDecodeUint, []byte{0xFF, 0xFF, 0xFF, 0xFF}, uint32(4294967295), 4, nil},
	}

	validEnums := make(map[int32]bool)
	validEnums[0] = true
	validEnums[1] = true

	var rv interface{}
	var n int
	var err error

	for i, test := range tests {
		err = nil
		dec := NewDecoder(bytes.NewReader(test.in))
		switch test.f {
		case fDecodeBool:
			rv, n, err = dec.DecodeBool()
		case fDecodeDouble:
			rv, n, err = dec.DecodeDouble()
		case fDecodeEnum:
			rv, n, err = dec.DecodeEnum(validEnums)
		case fDecodeFixedOpaque:
			want := test.wantVal.([]byte)
			rv, n, err = dec.DecodeFixedOpaque(int32(len(want)))
		case fDecodeFloat:
			rv, n, err = dec.DecodeFloat()
		case fDecodeHyper:
			rv, n, err = dec.DecodeHyper()
		case fDecodeInt:
			rv, n, err = dec.DecodeInt()
		case fDecodeOpaque:
			rv, n, err = dec.DecodeOpaque()
		case fDecodeString:
			rv, n, err = dec.DecodeString()
		case fDecodeUhyper:
			rv, n, err = dec.DecodeUhyper()
		case fDecodeUint:
			rv, n, err = dec.DecodeUint()
		default:
			t.Errorf("%v #%d unrecognized function", test.f, i)
			continue
		}

		// First ensure the number of bytes read is the expected value
		// and the error is the expected one.
		testName := fmt.Sprintf("%v #%d", test.f, i)
		if !testExpectedURet(t, testName, n, test.wantN, err, test.err) {
			continue
		}
		if test.err != nil {
			continue
		}

		// Finally, ensure the read value is the expected one.
		if !reflect.DeepEqual(rv, test.wantVal) {
			t.Errorf("%s: unexpected result - got: %v want: %v\n",
				testName, rv, test.wantVal)
			continue
		}
	}
}

// TestUnmarshalCorners ensures the Unmarshal function properly handles various
// cases not already covered by the other tests.
func TestUnmarshalCorners(t *testing.T) {
	buf := []byte{
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x02,
	}

	// Ensure unmarshal to unsettable pointer returns the expected error.
	testName := "Unmarshal to unsettable pointer"
	var i32p *int32
	expectedN := 0
	expectedErr := error(&UnmarshalError{ErrorCode: ErrNotSettable})
	n, err := Unmarshal(bytes.NewReader(buf), i32p)
	testExpectedURet(t, testName, n, expectedN, err, expectedErr)

	// Ensure decode of unsettable pointer returns the expected error.
	testName = "Decode to unsettable pointer"
	expectedN = 0
	expectedErr = &UnmarshalError{ErrorCode: ErrNotSettable}
	n, err = TstDecode(bytes.NewReader(buf))(reflect.ValueOf(i32p))
	testExpectedURet(t, testName, n, expectedN, err, expectedErr)

	// Ensure unmarshal to indirected unsettable pointer returns the
	// expected error.
	testName = "Unmarshal to indirected unsettable pointer"
	ii32p := interface{}(i32p)
	expectedN = 0
	expectedErr = &UnmarshalError{ErrorCode: ErrNotSettable}
	n, err = Unmarshal(bytes.NewReader(buf), &ii32p)
	testExpectedURet(t, testName, n, expectedN, err, expectedErr)

	// Ensure unmarshal to embedded unsettable interface value returns the
	// expected error.
	testName = "Unmarshal to embedded unsettable interface value"
	var i32 int32
	ii32 := interface{}(i32)
	expectedN = 0
	expectedErr = &UnmarshalError{ErrorCode: ErrNotSettable}
	n, err = Unmarshal(bytes.NewReader(buf), &ii32)
	testExpectedURet(t, testName, n, expectedN, err, expectedErr)

	// Ensure unmarshal to embedded interface value works properly.
	testName = "Unmarshal to embedded interface value"
	ii32vp := interface{}(&i32)
	expectedN = 4
	expectedErr = nil
	ii32vpr := int32(1)
	expectedVal := interface{}(&ii32vpr)
	n, err = Unmarshal(bytes.NewReader(buf), &ii32vp)
	if testExpectedURet(t, testName, n, expectedN, err, expectedErr) {
		if !reflect.DeepEqual(ii32vp, expectedVal) {
			t.Errorf("%s: unexpected result - got: %v want: %v\n",
				testName, ii32vp, expectedVal)
		}
	}

	// Ensure decode of an invalid reflect value returns the expected
	// error.
	testName = "Decode invalid reflect value"
	expectedN = 0
	expectedErr = error(&UnmarshalError{ErrorCode: ErrUnsupportedType})
	n, err = TstDecode(bytes.NewReader(buf))(reflect.Value{})
	testExpectedURet(t, testName, n, expectedN, err, expectedErr)

	// Ensure unmarshal to a slice with a cap and 0 length adjusts the
	// length properly.
	testName = "Unmarshal to capped slice"
	cappedSlice := make([]bool, 0, 1)
	expectedN = 8
	expectedErr = nil
	expectedVal = []bool{true}
	n, err = Unmarshal(bytes.NewReader(buf), &cappedSlice)
	if testExpectedURet(t, testName, n, expectedN, err, expectedErr) {
		if !reflect.DeepEqual(cappedSlice, expectedVal) {
			t.Errorf("%s: unexpected result - got: %v want: %v\n",
				testName, cappedSlice, expectedVal)
		}
	}

	// Ensure unmarshal to struct with both exported and unexported fields
	// skips the unexported fields but still unmarshals to the exported
	// fields.
	type unexportedStruct struct {
		unexported int
		Exported   int
	}
	testName = "Unmarshal to struct with exported and unexported fields"
	var tstruct unexportedStruct
	expectedN = 4
	expectedErr = nil
	expectedVal = unexportedStruct{0, 1}
	n, err = Unmarshal(bytes.NewReader(buf), &tstruct)
	if testExpectedURet(t, testName, n, expectedN, err, expectedErr) {
		if !reflect.DeepEqual(tstruct, expectedVal) {
			t.Errorf("%s: unexpected result - got: %v want: %v\n",
				testName, tstruct, expectedVal)
		}
	}

	// Ensure decode to struct with unsettable fields return expected error.
	type unsettableStruct struct {
		Exported int
	}
	testName = "Decode to struct with unsettable fields"
	var ustruct unsettableStruct
	expectedN = 0
	expectedErr = error(&UnmarshalError{ErrorCode: ErrNotSettable})
	n, err = TstDecode(bytes.NewReader(buf))(reflect.ValueOf(ustruct))
	testExpectedURet(t, testName, n, expectedN, err, expectedErr)

	// Ensure decode to struct with unsettable pointer fields return
	// expected error.
	type unsettablePointerStruct struct {
		Exported *int
	}
	testName = "Decode to struct with unsettable pointer fields"
	var upstruct unsettablePointerStruct
	expectedN = 0
	expectedErr = error(&UnmarshalError{ErrorCode: ErrNotSettable})
	n, err = TstDecode(bytes.NewReader(buf))(reflect.ValueOf(upstruct))
	testExpectedURet(t, testName, n, expectedN, err, expectedErr)
}
