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
	"time"
)

var errMaxXdr = "data exceeds max xdr size limit"

/*
Marshal returns the XDR encoding of v.  It traverses v recursively and
automatically indirects pointers through arbitrary depth to encode the actual
value pointed to.

Marshal uses reflection to determine the type of the concrete value contained
by v and performs a mapping of Go types to the underlying XDR types as follows:

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
func Marshal(v interface{}) (rv []byte, err error) {
	if v == nil {
		msg := "can't marshal nil interface"
		err = marshalError("Marshal", ErrNilInterface, msg, nil)
		return nil, err
	}

	vv := reflect.ValueOf(v)
	vve := vv
	for vve.Kind() == reflect.Ptr {
		if vve.IsNil() {
			msg := fmt.Sprintf("can't marshal nil pointer '%v'",
				vv.Type().String())
			err = marshalError("Marshal", ErrBadArguments, msg, nil)
			return nil, err
		}
		vve = vve.Elem()
	}

	enc := Encoder{}
	err = enc.encode(vve)
	return enc.data, err
}

// An Encoder contains information about the state of an encode operation
// from an interface value into an XDR-encoded byte slice.  See NewEncoder.
type Encoder struct {
	data []byte
	off  int
}

// EncodeInt appends the XDR encoded representation of the passed 32-bit signed
// integer to the Encoder's data.
//
// A MarshalError is returned if appending the data would overflow the
// internal data slice.
//
// Reference:
// 	RFC Section 4.1 - Integer
// 	32-bit big-endian signed integer in range [-2147483648, 2147483647]
func (enc *Encoder) EncodeInt(v int32) (err error) {
	b := make([]byte, 4, 4)
	b[0] = byte(v >> 24)
	b[1] = byte(v >> 16)
	b[2] = byte(v >> 8)
	b[3] = byte(v)

	if len(enc.data) > maxInt-4 {
		err = marshalError("EncodeInt", ErrOverflow, errMaxSlice, b)
		return
	}

	enc.data = append(enc.data, b...)
	return
}

// EncodeInt appends the XDR encoded representation of the passed 32-bit
// unsigned integer to the Encoder's data.
//
// A MarshalError is returned if appending the data would overflow the
// internal data slice.
//
// Reference:
// 	RFC Section 4.2 - Unsigned Integer
// 	32-bit big-endian unsigned integer in range [0, 4294967295]
func (enc *Encoder) EncodeUint(v uint32) (err error) {
	b := make([]byte, 4, 4)
	b[0] = byte(v >> 24)
	b[1] = byte(v >> 16)
	b[2] = byte(v >> 8)
	b[3] = byte(v)

	if len(enc.data) > maxInt-4 {
		err = marshalError("EncodeUint", ErrOverflow, errMaxSlice, b)
		return
	}

	enc.data = append(enc.data, b...)
	return
}

// EncodeEnum treats the passed 32-bit signed integer as an enumeration value
// and, if it is in the list of passed valid enumeration values, appends the XDR
// encoded representation of it to the Encoder's data.
//
// A MarshalError is returned if the enumeration value is not one of the provided
// valid values or appending the data would overflow the internal data slice.
//
// Reference:
// 	RFC Section 4.3 - Enumeration
// 	Represented as an XDR encoded signed integer
func (enc *Encoder) EncodeEnum(v int32, validEnums map[int32]bool) (err error) {
	if !validEnums[v] {
		err = marshalError("EncodeEnum", ErrBadEnumValue, "invalid enum", v)
		return
	}
	return enc.EncodeInt(v)
}

// EncodeInt appends the XDR encoded representation of the passed boolean
// to the Encoder's data.
//
// A MarshalError is returned if appending the data would overflow the
// internal data slice.
//
// Reference:
// 	RFC Section 4.4 - Boolean
// 	Represented as an XDR encoded enumeration where 0 is false and 1 is true
func (enc *Encoder) EncodeBool(v bool) (err error) {
	i := int32(0)
	if v == true {
		i = 1
	}
	return enc.EncodeInt(i)
}

// EncodeHyper appends the XDR encoded representation of the passed 64-bit
// signed integer to the Encoder's data.
//
// A MarshalError is returned if appending the data would overflow the
// internal data slice.
//
// Reference:
// 	RFC Section 4.5 - Hyper Integer
// 	64-bit big-endian signed integer in range [-9223372036854775808, 9223372036854775807]
func (enc *Encoder) EncodeHyper(v int64) (err error) {
	b := make([]byte, 8, 8)
	b[0] = byte(v >> 56)
	b[1] = byte(v >> 48)
	b[2] = byte(v >> 40)
	b[3] = byte(v >> 32)
	b[4] = byte(v >> 24)
	b[5] = byte(v >> 16)
	b[6] = byte(v >> 8)
	b[7] = byte(v)

	if len(enc.data) > maxInt-8 {
		err = marshalError("EncodeHyper", ErrOverflow, errMaxSlice, b)
		return
	}

	enc.data = append(enc.data, b...)
	return
}

// EncodeUhyper appends the XDR encoded representation of the passed 64-bit
// unsigned integer to the Encoder's data.
//
// A MarshalError is returned if appending the data would overflow the
// internal data slice.
//
// Reference:
// 	RFC Section 4.5 - Unsigned Hyper Integer
// 	64-bit big-endian unsigned integer in range [0, 18446744073709551615]
func (enc *Encoder) EncodeUhyper(v uint64) (err error) {
	b := make([]byte, 8, 8)
	b[0] = byte(v >> 56)
	b[1] = byte(v >> 48)
	b[2] = byte(v >> 40)
	b[3] = byte(v >> 32)
	b[4] = byte(v >> 24)
	b[5] = byte(v >> 16)
	b[6] = byte(v >> 8)
	b[7] = byte(v)

	if len(enc.data) > maxInt-8 {
		err = marshalError("EncodeUhyper", ErrOverflow, errMaxSlice, b)
		return
	}

	enc.data = append(enc.data, b...)
	return
}

// EncodeFloat appends the XDR encoded representation of the passed 32-bit
// (single-precision) floating point to the Encoder's data.
//
// A MarshalError is returned if appending the data would overflow the
// internal data slice.
//
// Reference:
// 	RFC Section 4.6 - Floating Point
// 	32-bit single-precision IEEE 754 floating point
func (enc *Encoder) EncodeFloat(v float32) (err error) {
	ui := math.Float32bits(v)
	return enc.EncodeUint(ui)
}

// EncodeDouble appends the XDR encoded representation of the passed 64-bit
// (double-precision) floating point to the Encoder's data.
//
// A MarshalError is returned if appending the data would overflow the
// internal data slice.
//
// Reference:
// 	RFC Section 4.7 -  Double-Precision Floating Point
// 	64-bit double-precision IEEE 754 floating point
func (enc *Encoder) EncodeDouble(v float64) (err error) {
	ui := math.Float64bits(v)
	return enc.EncodeUhyper(ui)
}

// RFC Section 4.8 -  Quadruple-Precision Floating Point
// 128-bit quadruple-precision floating point
// Not Implemented

// EncodeFixedOpaque treats the passed byte slice as opaque data of a fixed
// size and appends the XDR encoded representation of it to the Encoder's data.
//
// A MarshalError is returned if appending the data would overflow the
// internal data slice.
//
// Reference:
// 	RFC Section 4.9 - Fixed-Length Opaque Data
// 	Fixed-length uninterpreted data zero-padded to a multiple of four
func (enc *Encoder) EncodeFixedOpaque(v []byte) (err error) {
	l := len(v)
	pad := (4 - (l % 4)) % 4
	size := l + pad

	if len(enc.data) > maxInt-size {
		err = marshalError("EncodeFixedOpaque", ErrOverflow, errMaxSlice, size)
		return
	}

	enc.data = append(enc.data, v...)
	if pad > 0 {
		b := make([]byte, pad, pad)
		for i := 0; i < pad; i++ {
			b[i] = 0
		}
		enc.data = append(enc.data, b...)
	}
	return
}

// EncodeOpaque treats the passed byte slice as opaque data of a variable
// size and appends the XDR encoded representation of it to the Encoder's data.
//
// A MarshalError is returned if appending the data would overflow the
// internal data slice or the data in the byte slice is larger than XDR
// supports.
//
// Reference:
// 	RFC Section 4.10 - Variable-Length Opaque Data
// 	Unsigned integer length followed by fixed opaque data of that length
func (enc *Encoder) EncodeOpaque(v []byte) (err error) {
	dataLen := len(v)
	if uint(dataLen) > math.MaxUint32 {
		err = marshalError("EncodeOpaque", ErrOverflow, errMaxXdr, dataLen)
		return
	}
	err = enc.EncodeUint(uint32(dataLen))
	if err != nil {
		return
	}
	err = enc.EncodeFixedOpaque(v)
	return
}

// EncodeString appends the XDR encoded representation of the passed string
// to the Encoder's data.  Character encoding is assumed to be UTF-8 and
// therefore ASCII compatible.  If the underlying character encoding
// is not compatible with this assumption, the data can instead be written as
// variable-length opaque data (EncodeOpaque) and manually converted as needed.
//
// A MarshalError is returned if appending the data would overflow the
// internal data slice or the length of the string is larger than XDR supports.
//
// Reference:
// 	RFC Section 4.11 - String
// 	Unsigned integer length followed by bytes zero-padded to a multiple of four
func (enc *Encoder) EncodeString(v string) (err error) {
	dataLen := len(v)
	if uint(dataLen) > math.MaxUint32 {
		err = marshalError("EncodeString", ErrOverflow, errMaxXdr, dataLen)
		return
	}
	err = enc.EncodeUint(uint32(dataLen))
	if err != nil {
		return
	}
	err = enc.EncodeFixedOpaque([]byte(v))
	return
}

// encodeFixedArray appends the XDR encoded representation of each element
// in the passed array represented by the reflection value to the Encoder's
// data.  The ignoreOpaque flag controls whether or not uint8 (byte) elements
// should be encoded individually or as a fixed sequence of opaque data.
//
// A MarshalError is returned if any issues are encountered while encoding
// the array elements.
//
// Reference:
// 	RFC Section 4.12 - Fixed-Length Array
// 	Individually XDR encoded array elements
func (enc *Encoder) encodeFixedArray(v reflect.Value, ignoreOpaque bool) (err error) {
	// Treat [#]byte (byte is alias for uint8) as opaque data unless ignored.
	if !ignoreOpaque && v.Type().Elem().Kind() == reflect.Uint8 {
		// Create a slice of the underlying array for better efficiency when
		// possible.  Can't create a slice of an unaddressable value.
		if v.CanAddr() {
			err = enc.EncodeFixedOpaque(v.Slice(0, v.Len()).Bytes())
			return
		}

		// When the underlying array isn't addressable fall back to copying the
		// array into a new slice.  This is rather ugly, but the inability to
		// create a constant slice from an unaddressable array seems to be a
		// limitation of Go.
		slice := make([]byte, v.Len(), v.Len())
		reflect.Copy(reflect.ValueOf(slice), v)
		err = enc.EncodeFixedOpaque(slice)
		return
	}

	// Encode each array element.
	for i := 0; i < v.Len(); i++ {
		err = enc.encode(v.Index(i))
		if err != nil {
			return err
		}
	}
	return
}

// encodeArray appends an XDR encoded integer representing the number of
// elements in the passed slice represented by the reflection value followed by
// the XDR encoded representation of each element in slice to the Encoder's
// data.  The ignoreOpaque flag controls whether or not uint8 (byte) elements
// should be encoded individually or as a variable sequence of opaque data.
//
// A MarshalError is returned if any issues are encountered while encoding
// the array elements.
//
// Reference:
// 	RFC Section 4.13 - Variable-Length Array
// 	Unsigned integer length followed by individually XDR encoded array elements
func (enc *Encoder) encodeArray(v reflect.Value, ignoreOpaque bool) (err error) {
	numItems := uint32(v.Len())
	err = enc.encode(reflect.ValueOf(numItems))
	if err != nil {
		return err
	}
	err = enc.encodeFixedArray(v, ignoreOpaque)
	return
}

// encodeStruct appends an XDR encoded representation of each value in the
// exported fields of the struct represented by the passed reflection value to
// the Encoder's data.  Pointers are automatically indirected through arbitrary
// depth to encode the actual value pointed to.
//
// A MarshalError is returned if any issues are encountered while encoding
// the elements.
//
// Reference:
// 	RFC Section 4.14 - Structure
// 	XDR encoded elements in the order of their declaration in the struct
func (enc *Encoder) encodeStruct(v reflect.Value) (err error) {
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
				err = enc.encodeArray(vf, true)
				if err != nil {
					return
				}
				continue

			case reflect.Array:
				err = enc.encodeFixedArray(vf, true)
				if err != nil {
					return
				}
				continue
			}
		}

		// Encode each struct field.
		err = enc.encode(vf)
		if err != nil {
			return err
		}
	}
	return
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
// type as the map keys and elements and appends its XDR encoded representation
// to the Encoder's data.
//
// A MarshalError is returned if any issues are encountered while encoding
// the elements.
func (enc *Encoder) encodeMap(v reflect.Value) (err error) {
	dataLen := v.Len()
	if uint(dataLen) > math.MaxUint32 {
		err = marshalError("encodeMap", ErrOverflow, errMaxXdr, dataLen)
		return
	}
	err = enc.EncodeUint(uint32(dataLen))
	if err != nil {
		return
	}

	// Encode each key and value according to their type.
	for _, key := range v.MapKeys() {
		err = enc.encode(key)
		if err != nil {
			return
		}

		err = enc.encode(v.MapIndex(key))
		if err != nil {
			return
		}
	}
	return
}

// encodeInterface examines the interface represented by the passed reflection
// value to detect whether it is an interface that can be encoded if it is,
// extracts the underlying value to pass back into the encode function for
// encoding according to its type.
//
// A MarshalError is returned if any issues are encountered while encoding
// the interface.
func (enc *Encoder) encodeInterface(v reflect.Value) (err error) {
	if v.IsNil() || !v.CanInterface() {
		msg := fmt.Sprintf("can't encode nil interface")
		err = marshalError("encodeInterface", ErrNilInterface, msg, nil)
		return err
	}

	// Extract underlying value from the interface and indirect through pointers.
	ve := reflect.ValueOf(v.Interface())
	ve = enc.indirect(ve)
	return enc.encode(ve)
}

// encode is the main workhorse for marshalling via reflection.  It uses
// the passed reflection value to choose the XDR primitives to encode into
// the data field of the Encoder.  It is a recursive function, so cyclic data
// structures are not supported and will result in an infinite loop.
func (enc *Encoder) encode(v reflect.Value) (err error) {
	if !v.IsValid() {
		msg := fmt.Sprintf("type '%s' is not valid", v.Kind().String())
		err = marshalError("encode", ErrUnsupportedType, msg, nil)
		return
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
			err = enc.EncodeString(tv.Format(time.RFC3339Nano))
			return
		}
	}

	// Handle native Go types.
	switch ve.Kind() {
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int:
		err = enc.EncodeInt(int32(ve.Int()))

	case reflect.Int64:
		err = enc.EncodeHyper(ve.Int())

	case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint:
		err = enc.EncodeUint(uint32(ve.Uint()))

	case reflect.Uint64:
		err = enc.EncodeUhyper(ve.Uint())

	case reflect.Bool:
		err = enc.EncodeBool(ve.Bool())

	case reflect.Float32:
		err = enc.EncodeFloat(float32(ve.Float()))

	case reflect.Float64:
		err = enc.EncodeDouble(ve.Float())

	case reflect.String:
		err = enc.EncodeString(ve.String())

	case reflect.Array:
		err = enc.encodeFixedArray(ve, false)

	case reflect.Slice:
		err = enc.encodeArray(ve, false)

	case reflect.Struct:
		err = enc.encodeStruct(ve)

	case reflect.Map:
		err = enc.encodeMap(ve)

	case reflect.Interface:
		err = enc.encodeInterface(ve)

	// reflect.Uintptr, reflect.UnsafePointer
	default:
		msg := fmt.Sprintf("unsupported Go type '%s'", ve.Kind().String())
		err = marshalError("encode", ErrUnsupportedType, msg, nil)
	}

	return
}

// indirect dereferences pointers until it reaches a non-pointer.  This allows
// transparent encoding through arbitrary levels of indirection.
func (enc *Encoder) indirect(v reflect.Value) (rv reflect.Value) {
	rv = v
	for rv.Kind() == reflect.Ptr {
		rv = rv.Elem()
	}
	return
}

// Data returns the XDR encoded data stored in the Encoder.
func (enc *Encoder) Data() (rv []byte) {
	return enc.data[:]
}

// Reset clears the internal XDR encoded data so the Encoder may be reused.
func (enc *Encoder) Reset() {
	enc.data = enc.data[0:0]
}

// NewEncoder returns an object that can be used to manually build an XDR
// encoded byte slice.  Typically, Marshal should be used instead of manually
// creating an Encoder. An Encoder, along with several of its methods to encode
// XDR primitives, is exposed so it is possible to perform manual encoding of
// data without relying on reflection should it be necessary in complex
// scenarios where automatic reflection-based decoding won't work.
func NewEncoder() *Encoder {
	return &Encoder{}
}
