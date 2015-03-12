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
	"io"
	"math"
	"reflect"
	"time"
)

var errIOEncode = "%s while encoding %d bytes"

/*
Marshal writes the XDR encoding of v to writer w and returns the number of bytes
written.  It traverses v recursively and automatically indirects pointers
through arbitrary depth to encode the actual value pointed to.

Marshal uses reflection to determine the type of the concrete value contained by
v and performs a mapping of Go types to the underlying XDR types as follows:

	Go Type -> XDR Type
	--------------------
	int8, int16, int32, int -> XDR Integer
	uint8, uint16, uint32, uint -> XDR Unsigned Integer
	int64 -> XDR Hyper Integer
	uint64 -> XDR Unsigned Hyper Integer
	bool -> XDR Boolean
	float32 -> XDR Floating-Point
	float64 -> XDR Double-Precision Floating-Point
	string -> XDR String
	byte -> XDR Integer
	[]byte -> XDR Variable-Length Opaque Data
	[#]byte -> XDR Fixed-Length Opaque Data
	[]<type> -> XDR Variable-Length Array
	[#]<type> -> XDR Fixed-Length Array
	struct -> XDR Structure
	map -> XDR Variable-Length Array of two-element XDR Structures
	time.Time -> XDR String encoded with RFC3339 nanosecond precision

Notes and Limitations:

	* Automatic marshalling of variable and fixed-length arrays of uint8s
	  requires a special struct tag `xdropaque:"false"` since byte slices and
	  byte arrays are assumed to be opaque data and byte is a Go alias for uint8
	  thus indistinguishable under reflection
	* Channel, complex, and function types cannot be encoded
	* Interfaces without a concrete value cannot be encoded
	* Cyclic data structures are not supported and will result in infinite loops
	* Strings are marshalled with UTF-8 character encoding which differs from
	  the XDR specification of ASCII, however UTF-8 is backwards compatible with
	  ASCII so this should rarely cause issues

If any issues are encountered during the marshalling process, a MarshalError is
returned with a human readable description as well as an ErrorCode value for
further inspection from sophisticated callers.  Some potential issues are
unsupported Go types, attempting to encode more opaque data than can be
represented by a single opaque XDR entry, and exceeding max slice limitations.
*/
func Marshal(w io.Writer, v interface{}) (int, error) {
	if v == nil {
		msg := "can't marshal nil interface"
		err := marshalError("Marshal", ErrNilInterface, msg, nil, nil)
		return 0, err
	}

	vv := reflect.ValueOf(v)
	vve := vv
	for vve.Kind() == reflect.Ptr {
		if vve.IsNil() {
			msg := fmt.Sprintf("can't marshal nil pointer '%v'",
				vv.Type().String())
			err := marshalError("Marshal", ErrBadArguments, msg,
				nil, nil)
			return 0, err
		}
		vve = vve.Elem()
	}

	enc := Encoder{w: w}
	return enc.encode(vve)
}

// An Encoder wraps an io.Writer that will receive the XDR encoded byte stream.
// See NewEncoder.
type Encoder struct {
	w io.Writer
}

// EncodeInt writes the XDR encoded representation of the passed 32-bit signed
// integer to the encapsulated writer and returns the number of bytes written.
//
// A MarshalError with an error code of ErrIO is returned if writing the data
// fails.
//
// Reference:
// 	RFC Section 4.1 - Integer
// 	32-bit big-endian signed integer in range [-2147483648, 2147483647]
func (enc *Encoder) EncodeInt(v int32) (int, error) {
	var b [4]byte
	b[0] = byte(v >> 24)
	b[1] = byte(v >> 16)
	b[2] = byte(v >> 8)
	b[3] = byte(v)

	n, err := enc.w.Write(b[:])
	if err != nil {
		msg := fmt.Sprintf(errIOEncode, err.Error(), 4)
		err := marshalError("EncodeInt", ErrIO, msg, b[:n], err)
		return n, err
	}

	return n, nil
}

// EncodeInt writes the XDR encoded representation of the passed 32-bit
// unsigned integer to the encapsulated writer and returns the number of bytes
// written.
//
// A MarshalError with an error code of ErrIO is returned if writing the data
// fails.
//
// Reference:
// 	RFC Section 4.2 - Unsigned Integer
// 	32-bit big-endian unsigned integer in range [0, 4294967295]
func (enc *Encoder) EncodeUint(v uint32) (int, error) {
	var b [4]byte
	b[0] = byte(v >> 24)
	b[1] = byte(v >> 16)
	b[2] = byte(v >> 8)
	b[3] = byte(v)

	n, err := enc.w.Write(b[:])
	if err != nil {
		msg := fmt.Sprintf(errIOEncode, err.Error(), 4)
		err := marshalError("EncodeUint", ErrIO, msg, b[:n], err)
		return n, err
	}

	return n, nil
}

// EncodeEnum treats the passed 32-bit signed integer as an enumeration value
// and, if it is in the list of passed valid enumeration values, writes the XDR
// encoded representation of it to the encapsulated writer.  It returns the
// number of bytes written.
//
// A MarshalError is returned if the enumeration value is not one of the
// provided valid values or if writing the data fails.

//
// Reference:
// 	RFC Section 4.3 - Enumeration
// 	Represented as an XDR encoded signed integer
func (enc *Encoder) EncodeEnum(v int32, validEnums map[int32]bool) (int, error) {
	if !validEnums[v] {
		err := marshalError("EncodeEnum", ErrBadEnumValue,
			"invalid enum", v, nil)
		return 0, err
	}
	return enc.EncodeInt(v)
}

// EncodeInt writes the XDR encoded representation of the passed boolean to the
// encapsulated writer and returns the number of bytes written.
//
// A MarshalError with an error code of ErrIO is returned if writing the data
// fails.
//
// Reference:
// 	RFC Section 4.4 - Boolean
// 	Represented as an XDR encoded enumeration where 0 is false and 1 is true
func (enc *Encoder) EncodeBool(v bool) (int, error) {
	i := int32(0)
	if v == true {
		i = 1
	}
	return enc.EncodeInt(i)
}

// EncodeHyper writes the XDR encoded representation of the passed 64-bit
// signed integer to the encapsulated writer and returns the number of bytes
// written.
//
// A MarshalError with an error code of ErrIO is returned if writing the data
// fails.
//
// Reference:
// 	RFC Section 4.5 - Hyper Integer
// 	64-bit big-endian signed integer in range [-9223372036854775808, 9223372036854775807]
func (enc *Encoder) EncodeHyper(v int64) (int, error) {
	var b [8]byte
	b[0] = byte(v >> 56)
	b[1] = byte(v >> 48)
	b[2] = byte(v >> 40)
	b[3] = byte(v >> 32)
	b[4] = byte(v >> 24)
	b[5] = byte(v >> 16)
	b[6] = byte(v >> 8)
	b[7] = byte(v)

	n, err := enc.w.Write(b[:])
	if err != nil {
		msg := fmt.Sprintf(errIOEncode, err.Error(), 8)
		err := marshalError("EncodeHyper", ErrIO, msg, b[:n], err)
		return n, err
	}

	return n, nil
}

// EncodeUhyper writes the XDR encoded representation of the passed 64-bit
// unsigned integer to the encapsulated writer and returns the number of bytes
// written.
//
// A MarshalError with an error code of ErrIO is returned if writing the data
// fails.
//
// Reference:
// 	RFC Section 4.5 - Unsigned Hyper Integer
// 	64-bit big-endian unsigned integer in range [0, 18446744073709551615]
func (enc *Encoder) EncodeUhyper(v uint64) (int, error) {
	var b [8]byte
	b[0] = byte(v >> 56)
	b[1] = byte(v >> 48)
	b[2] = byte(v >> 40)
	b[3] = byte(v >> 32)
	b[4] = byte(v >> 24)
	b[5] = byte(v >> 16)
	b[6] = byte(v >> 8)
	b[7] = byte(v)

	n, err := enc.w.Write(b[:])
	if err != nil {
		msg := fmt.Sprintf(errIOEncode, err.Error(), 8)
		err := marshalError("EncodeUhyper", ErrIO, msg, b[:n], err)
		return n, err
	}

	return n, nil
}

// EncodeFloat writes the XDR encoded representation of the passed 32-bit
// (single-precision) floating point to the encapsulated writer and returns the
// number of bytes written.
//
// A MarshalError with an error code of ErrIO is returned if writing the data
// fails.
//
// Reference:
// 	RFC Section 4.6 - Floating Point
// 	32-bit single-precision IEEE 754 floating point
func (enc *Encoder) EncodeFloat(v float32) (int, error) {
	ui := math.Float32bits(v)
	return enc.EncodeUint(ui)
}

// EncodeDouble writes the XDR encoded representation of the passed 64-bit
// (double-precision) floating point to the encapsulated writer and returns the
// number of bytes written.
//
// A MarshalError with an error code of ErrIO is returned if writing the data
// fails.
//
// Reference:
// 	RFC Section 4.7 -  Double-Precision Floating Point
// 	64-bit double-precision IEEE 754 floating point
func (enc *Encoder) EncodeDouble(v float64) (int, error) {
	ui := math.Float64bits(v)
	return enc.EncodeUhyper(ui)
}

// RFC Section 4.8 -  Quadruple-Precision Floating Point
// 128-bit quadruple-precision floating point
// Not Implemented

// EncodeFixedOpaque treats the passed byte slice as opaque data of a fixed
// size and writes the XDR encoded representation of it  to the encapsulated
// writer.  It returns the number of bytes written.
//
// A MarshalError with an error code of ErrIO is returned if writing the data
// fails.
//
// Reference:
// 	RFC Section 4.9 - Fixed-Length Opaque Data
// 	Fixed-length uninterpreted data zero-padded to a multiple of four
func (enc *Encoder) EncodeFixedOpaque(v []byte) (int, error) {
	l := len(v)
	pad := (4 - (l % 4)) % 4

	// Write the actual bytes.
	n, err := enc.w.Write(v)
	if err != nil {
		msg := fmt.Sprintf(errIOEncode, err.Error(), len(v))
		err := marshalError("EncodeFixedOpaque", ErrIO, msg, v[:n], err)
		return n, err
	}

	// Write any padding if needed.
	if pad > 0 {
		b := make([]byte, pad)
		n2, err := enc.w.Write(b)
		n += n2
		if err != nil {
			written := make([]byte, l+n2)
			copy(written, v)
			copy(written[l:], b[:n2])
			msg := fmt.Sprintf(errIOEncode, err.Error(), l+pad)
			err := marshalError("EncodeFixedOpaque", ErrIO, msg,
				written, err)
			return n, err
		}
	}

	return n, nil
}

// EncodeOpaque treats the passed byte slice as opaque data of a variable
// size and writes the XDR encoded representation of it to the encapsulated
// writer.  It returns the number of bytes written.
//
// A MarshalError with an error code of ErrIO is returned if writing the data
// fails.
//
// Reference:
// 	RFC Section 4.10 - Variable-Length Opaque Data
// 	Unsigned integer length followed by fixed opaque data of that length
func (enc *Encoder) EncodeOpaque(v []byte) (int, error) {
	// Length of opaque data.
	n, err := enc.EncodeUint(uint32(len(v)))
	if err != nil {
		return n, err
	}

	n2, err := enc.EncodeFixedOpaque(v)
	n += n2
	return n, err
}

// EncodeString writes the XDR encoded representation of the passed string
// to the encapsulated writer and returns the number of bytes written.
// Character encoding is assumed to be UTF-8 and therefore ASCII compatible.  If
// the underlying character encoding is not compatible with this assumption, the
// data can instead be written as variable-length opaque data (EncodeOpaque) and
// manually converted as needed.
//
// A MarshalError with an error code of ErrIO is returned if writing the data
// fails.
//
// Reference:
// 	RFC Section 4.11 - String
// 	Unsigned integer length followed by bytes zero-padded to a multiple of four
func (enc *Encoder) EncodeString(v string) (int, error) {
	// Length of string.
	n, err := enc.EncodeUint(uint32(len(v)))
	if err != nil {
		return n, err
	}

	n2, err := enc.EncodeFixedOpaque([]byte(v))
	n += n2
	return n, err
}

// encodeFixedArray writes the XDR encoded representation of each element
// in the passed array represented by the reflection value to the encapsulated
// writer and returns the number of bytes written.  The ignoreOpaque flag
// controls whether or not uint8 (byte) elements should be encoded individually
// or as a fixed sequence of opaque data.
//
// A MarshalError is returned if any issues are encountered while encoding
// the array elements.
//
// Reference:
// 	RFC Section 4.12 - Fixed-Length Array
// 	Individually XDR encoded array elements
func (enc *Encoder) encodeFixedArray(v reflect.Value, ignoreOpaque bool) (int, error) {
	// Treat [#]byte (byte is alias for uint8) as opaque data unless ignored.
	if !ignoreOpaque && v.Type().Elem().Kind() == reflect.Uint8 {
		// Create a slice of the underlying array for better efficiency
		// when possible.  Can't create a slice of an unaddressable
		// value.
		if v.CanAddr() {
			return enc.EncodeFixedOpaque(v.Slice(0, v.Len()).Bytes())
		}

		// When the underlying array isn't addressable fall back to
		// copying the array into a new slice.  This is rather ugly, but
		// the inability to create a constant slice from an
		// unaddressable array is a limitation of Go.
		slice := make([]byte, v.Len(), v.Len())
		reflect.Copy(reflect.ValueOf(slice), v)
		return enc.EncodeFixedOpaque(slice)
	}

	// Encode each array element.
	var n int
	for i := 0; i < v.Len(); i++ {
		n2, err := enc.encode(v.Index(i))
		n += n2
		if err != nil {
			return n, err
		}
	}

	return n, nil
}

// encodeArray writes an XDR encoded integer representing the number of
// elements in the passed slice represented by the reflection value followed by
// the XDR encoded representation of each element in slice to the encapsulated
// writer and returns the number of bytes written.  The ignoreOpaque flag
// controls whether or not uint8 (byte) elements should be encoded individually
// or as a variable sequence of opaque data.
//
// A MarshalError is returned if any issues are encountered while encoding
// the array elements.
//
// Reference:
// 	RFC Section 4.13 - Variable-Length Array
// 	Unsigned integer length followed by individually XDR encoded array elements
func (enc *Encoder) encodeArray(v reflect.Value, ignoreOpaque bool) (int, error) {
	numItems := uint32(v.Len())
	n, err := enc.EncodeUint(numItems)
	if err != nil {
		return n, err
	}

	n2, err := enc.encodeFixedArray(v, ignoreOpaque)
	n += n2
	return n, err
}

// encodeStruct writes an XDR encoded representation of each value in the
// exported fields of the struct represented by the passed reflection value to
// the encapsulated writer and returns the number of bytes written.  Pointers
// are automatically indirected through arbitrary depth to encode the actual
// value pointed to.
//
// A MarshalError is returned if any issues are encountered while encoding
// the elements.
//
// Reference:
// 	RFC Section 4.14 - Structure
// 	XDR encoded elements in the order of their declaration in the struct
func (enc *Encoder) encodeStruct(v reflect.Value) (int, error) {
	var n int
	vt := v.Type()
	for i := 0; i < v.NumField(); i++ {
		// Skip unexported fields and indirect through pointers.
		vtf := vt.Field(i)
		if vtf.PkgPath != "" {
			continue
		}
		vf := v.Field(i)
		vf = enc.indirect(vf)

		// Handle non-opaque data to []uint8 and [#]uint8 based on struct tag.
		tag := vtf.Tag.Get("xdropaque")
		if tag == "false" {
			switch vf.Kind() {
			case reflect.Slice:
				n2, err := enc.encodeArray(vf, true)
				n += n2
				if err != nil {
					return n, err
				}
				continue

			case reflect.Array:
				n2, err := enc.encodeFixedArray(vf, true)
				n += n2
				if err != nil {
					return n, err
				}
				continue
			}
		}

		// Encode each struct field.
		n2, err := enc.encode(vf)
		n += n2
		if err != nil {
			return n, err
		}
	}

	return n, nil
}

// RFC Section 4.15 - Discriminated Union
// RFC Section 4.16 - Void
// RFC Section 4.17 - Constant
// RFC Section 4.18 - Typedef
// RFC Section 4.19 - Optional data
// RFC Sections 4.15 though 4.19 only apply to the data specification language
// which is not implemented by this package.  In the case of discriminated
// unions, struct tags are used to perform a similar function.

// encodeMap treats the map represented by the passed reflection value as a
// variable-length array of 2-element structures whose fields are of the same
// type as the map keys and elements and writes its XDR encoded representation
// to the encapsulated writer.  It returns the number of bytes written.
//
// A MarshalError is returned if any issues are encountered while encoding
// the elements.
func (enc *Encoder) encodeMap(v reflect.Value) (int, error) {
	// Number of elements.
	n, err := enc.EncodeUint(uint32(v.Len()))
	if err != nil {
		return n, err
	}

	// Encode each key and value according to their type.
	for _, key := range v.MapKeys() {
		n2, err := enc.encode(key)
		n += n2
		if err != nil {
			return n, err
		}

		n2, err = enc.encode(v.MapIndex(key))
		n += n2
		if err != nil {
			return n, err
		}
	}

	return n, nil
}

// encodeInterface examines the interface represented by the passed reflection
// value to detect whether it is an interface that can be encoded if it is,
// extracts the underlying value to pass back into the encode function for
// encoding according to its type.
//
// A MarshalError is returned if any issues are encountered while encoding
// the interface.
func (enc *Encoder) encodeInterface(v reflect.Value) (int, error) {
	if v.IsNil() || !v.CanInterface() {
		msg := fmt.Sprintf("can't encode nil interface")
		err := marshalError("encodeInterface", ErrNilInterface, msg,
			nil, nil)
		return 0, err
	}

	// Extract underlying value from the interface and indirect through pointers.
	ve := reflect.ValueOf(v.Interface())
	ve = enc.indirect(ve)
	return enc.encode(ve)
}

// encode is the main workhorse for marshalling via reflection.  It uses
// the passed reflection value to choose the XDR primitives to encode into
// the encapsulated writer and returns the number of bytes written.  It is a
// recursive function, so cyclic data structures are not supported and will
// result in an infinite loop.
func (enc *Encoder) encode(v reflect.Value) (int, error) {
	if !v.IsValid() {
		msg := fmt.Sprintf("type '%s' is not valid", v.Kind().String())
		err := marshalError("encode", ErrUnsupportedType, msg, nil, nil)
		return 0, err
	}

	// Indirect through pointers to get at the concrete value.
	ve := enc.indirect(v)

	// Handle time.Time values by encoding them as an RFC3339 formatted
	// string with nanosecond precision.  Check the type string before
	// doing a full blown conversion to interface and type assertion since
	// checking a string is much quicker.
	if ve.Type().String() == "time.Time" && ve.CanInterface() {
		viface := ve.Interface()
		if tv, ok := viface.(time.Time); ok {
			return enc.EncodeString(tv.Format(time.RFC3339Nano))
		}
	}

	// Handle native Go types.
	switch ve.Kind() {
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int:
		return enc.EncodeInt(int32(ve.Int()))

	case reflect.Int64:
		return enc.EncodeHyper(ve.Int())

	case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint:
		return enc.EncodeUint(uint32(ve.Uint()))

	case reflect.Uint64:
		return enc.EncodeUhyper(ve.Uint())

	case reflect.Bool:
		return enc.EncodeBool(ve.Bool())

	case reflect.Float32:
		return enc.EncodeFloat(float32(ve.Float()))

	case reflect.Float64:
		return enc.EncodeDouble(ve.Float())

	case reflect.String:
		return enc.EncodeString(ve.String())

	case reflect.Array:
		return enc.encodeFixedArray(ve, false)

	case reflect.Slice:
		return enc.encodeArray(ve, false)

	case reflect.Struct:
		return enc.encodeStruct(ve)

	case reflect.Map:
		return enc.encodeMap(ve)

	case reflect.Interface:
		return enc.encodeInterface(ve)
	}

	// The only unhandled types left are unsupported.  At the time of this
	// writing the only remaining unsupported types that exist are
	// reflect.Uintptr and reflect.UnsafePointer.
	msg := fmt.Sprintf("unsupported Go type '%s'", ve.Kind().String())
	err := marshalError("encode", ErrUnsupportedType, msg, nil, nil)
	return 0, err
}

// indirect dereferences pointers until it reaches a non-pointer.  This allows
// transparent encoding through arbitrary levels of indirection.
func (enc *Encoder) indirect(v reflect.Value) reflect.Value {
	rv := v
	for rv.Kind() == reflect.Ptr {
		rv = rv.Elem()
	}
	return rv
}

// NewEncoder returns an object that can be used to manually choose fields to
// XDR encode to the passed writer w.  Typically, Marshal should be used instead
// of manually creating an Encoder. An Encoder, along with several of its
// methods to encode XDR primitives, is exposed so it is possible to perform
// manual encoding of data without relying on reflection should it be necessary
// in complex scenarios where automatic reflection-based encoding won't work.
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{w: w}
}
