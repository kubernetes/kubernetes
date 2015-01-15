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

const maxInt = int(^uint(0) >> 1)

var errMaxSlice = "data exceeds max slice limit"
var errInsufficientBytes = "insufficient bytes to decode %d bytes"
var errInsufficientPad = "insufficient pad bytes to decode %d bytes"

/*
Unmarshal parses XDR-encoded data into the value pointed to by v.  An
addressable pointer must be provided since Unmarshal needs to both store the
result of the decode as well as obtain target type information.  Unmarhsal
traverses v recursively and automatically indirects pointers through arbitrary
depth, allocating them as necessary, to decode the data into the underlying value
pointed to.

Unmarshal uses reflection to determine the type of the concrete value contained
by v and performs a mapping of underlying XDR types to Go types as follows:

	Go Type <- XDR Type
	--------------------
	int8, int16, int32, int <- XDR Integer
	uint8, uint16, uint32, uint <- XDR Unsigned Integer
	int64 <- XDR Hyper Integer
	uint64 <- XDR Unsigned Hyper Integer
	bool <- XDR Boolean
	float32 <- XDR Floating-Point
	float64 <- XDR Double-Precision Floating-Point
	string <- XDR String
	byte <- XDR Integer
	[]byte <- XDR Variable-Length Opaque Data
	[#]byte <- XDR Fixed-Length Opaque Data
	[]<type> <- XDR Variable-Length Array
	[#]<type> <- XDR Fixed-Length Array
	struct <- XDR Structure
	map <- XDR Variable-Length Array of two-element XDR Structures
	time.Time <- XDR String encoded with RFC3339 nanosecond precision

Notes and Limitations:

	* Automatic unmarshalling of variable and fixed-length arrays of uint8s
	  requires a special struct tag `xdropaque:"false"` since byte slices and
	  byte arrays are assumed to be opaque data and byte is a Go alias for uint8
	  thus indistinguishable under reflection
	* Cyclic data structures are not supported and will result in infinite loops

If any issues are encountered during the unmarshalling process, an
UnmarshalError is returned with a human readable description as well as
an ErrorCode value for further inspection from sophisticated callers.  Some
potential issues are unsupported Go types, attempting to decode a value which is
too large to fit into a specified Go type, and exceeding max slice limitations.
*/
func Unmarshal(data []byte, v interface{}) (rest []byte, err error) {
	if v == nil {
		msg := "can't unmarshal to nil interface"
		err = unmarshalError("Unmarshal", ErrNilInterface, msg, nil)
		return data, err
	}

	vv := reflect.ValueOf(v)
	if vv.Kind() != reflect.Ptr {
		msg := fmt.Sprintf("can't unmarshal to non-pointer '%v' - use & operator",
			vv.Type().String())
		err = unmarshalError("Unmarshal", ErrBadArguments, msg, nil)
		return data, err
	}
	if vv.IsNil() && !vv.CanSet() {
		msg := fmt.Sprintf("can't unmarshal to unsettable '%v' - use & operator",
			vv.Type().String())
		err = unmarshalError("Unmarshal", ErrNotSettable, msg, nil)
		return data, err
	}

	d := Decoder{data: data, off: 0}
	err = d.decode(vv)
	return d.data, err
}

// A Decoder contains information about the state of a decode operation
// from an XDR-encoded byte slice into interface values and provides several
// exposed methods to manually decode various XDR primitives without relying
// on reflection.  The NewDecoder function can be used to get a new Decoder
// directly.
//
// Typically, Unmarshal should be used instead of manual decoding.  A
// Decoder is exposed so it is possible to perform manual decoding should it be
// necessary in complex scenarios where automatic reflection-based decoding
// won't work.
type Decoder struct {
	data []byte
	off  int
}

// DecodeInt treats the next 4 bytes as an XDR encoded integer and returns the
// result as an int32.
//
// An UnmarshalError is returned if there are insufficient bytes remaining.
//
// Reference:
// 	RFC Section 4.1 - Integer
// 	32-bit big-endian signed integer in range [-2147483648, 2147483647]
func (d *Decoder) DecodeInt() (rv int32, err error) {
	data := d.data
	if len(data) < 4 {
		msg := fmt.Sprintf(errInsufficientBytes, 4)
		err = unmarshalError("DecodeInt", ErrUnexpectedEnd, msg, data)
		return
	}
	rv = int32(data[3]) | int32(data[2])<<8 |
		int32(data[1])<<16 | int32(data[0])<<24

	d.data = data[4:]
	return
}

// DecodeUint treats the next 4 bytes as an XDR encoded unsigned integer and
// returns the result as a uint32.
//
// An UnmarshalError is returned if there are insufficient bytes remaining.
//
// Reference:
// 	RFC Section 4.2 - Unsigned Integer
// 	32-bit big-endian unsigned integer in range [0, 4294967295]
func (d *Decoder) DecodeUint() (rv uint32, err error) {
	data := d.data
	if len(data) < 4 {
		msg := fmt.Sprintf(errInsufficientBytes, 4)
		err = unmarshalError("DecodeUint", ErrUnexpectedEnd, msg, data)
		return
	}
	rv = uint32(data[3]) | uint32(data[2])<<8 |
		uint32(data[1])<<16 | uint32(data[0])<<24

	d.data = data[4:]
	return
}

// DecodeEnum treats the next 4 bytes as an XDR encoded enumeration value and
// returns the result as an int32 after verifying that the value is in the
// provided map of valid values.
//
// An UnmarshalError is returned if there are insufficient bytes remaining or
// the parsed enumeration value is not one of the provided valid values.
//
// Reference:
// 	RFC Section 4.3 - Enumeration
// 	Represented as an XDR encoded signed integer
func (d *Decoder) DecodeEnum(validEnums map[int32]bool) (rv int32, err error) {
	val, err := d.DecodeInt()
	if err != nil {
		return
	}
	if !validEnums[val] {
		err = unmarshalError("DecodeEnum", ErrBadEnumValue, "invalid enum", val)
		return
	}
	return val, nil
}

// DecodeBool treats the next 4 bytes as an XDR encoded boolean value and
// returns the result as a bool.
//
// An UnmarshalError is returned if there are insufficient bytes remaining or
// the parsed value is not a 0 or 1.
//
// Reference:
// 	RFC Section 4.4 - Boolean
// 	Represented as an XDR encoded enumeration where 0 is false and 1 is true
func (d *Decoder) DecodeBool() (rv bool, err error) {
	val, err := d.DecodeInt()
	if err != nil {
		return
	}
	switch val {
	case 0:
		return false, nil
	case 1:
		return true, nil
	}
	err = unmarshalError("DecodeBool", ErrBadEnumValue, "bool not 0 or 1", val)
	return
}

// DecodeHyper treats the next 8 bytes as an XDR encoded hyper value and
// returns the result as an int64.
//
// An UnmarshalError is returned if there are insufficient bytes remaining.
//
// Reference:
// 	RFC Section 4.5 - Hyper Integer
// 	64-bit big-endian signed integer in range [-9223372036854775808, 9223372036854775807]
func (d *Decoder) DecodeHyper() (rv int64, err error) {
	data := d.data
	if len(data) < 8 {
		msg := fmt.Sprintf(errInsufficientBytes, 8)
		err = unmarshalError("DecodeHyper", ErrUnexpectedEnd, msg, data)
		return
	}
	rv = int64(data[7]) | int64(data[6])<<8 |
		int64(data[5])<<16 | int64(data[4])<<24 |
		int64(data[3])<<32 | int64(data[2])<<40 |
		int64(data[1])<<48 | int64(data[0])<<56

	d.data = data[8:]
	return
}

// DecodeUhyper treats the next 8  bytes as an XDR encoded unsigned hyper value
// and returns the result as a uint64.
//
// An UnmarshalError is returned if there are insufficient bytes remaining.
//
// Reference:
// 	RFC Section 4.5 - Unsigned Hyper Integer
// 	64-bit big-endian unsigned integer in range [0, 18446744073709551615]
func (d *Decoder) DecodeUhyper() (rv uint64, err error) {
	data := d.data
	if len(data) < 8 {
		msg := fmt.Sprintf(errInsufficientBytes, 8)
		err = unmarshalError("DecodeUhyper", ErrUnexpectedEnd, msg, data)
		return
	}
	rv = uint64(data[7]) | uint64(data[6])<<8 |
		uint64(data[5])<<16 | uint64(data[4])<<24 |
		uint64(data[3])<<32 | uint64(data[2])<<40 |
		uint64(data[1])<<48 | uint64(data[0])<<56

	d.data = data[8:]
	return
}

// DecodeFloat treats the next 4 bytes as an XDR encoded floating point and
// returns the result as a float32.
//
// An UnmarshalError is returned if there are insufficient bytes remaining.
//
// Reference:
// 	RFC Section 4.6 - Floating Point
// 	32-bit single-precision IEEE 754 floating point
func (d *Decoder) DecodeFloat() (rv float32, err error) {
	data := d.data
	if len(data) < 4 {
		msg := fmt.Sprintf(errInsufficientBytes, 4)
		err = unmarshalError("DecodeFloat", ErrUnexpectedEnd, msg, data)
		return
	}
	val := uint32(data[3]) | uint32(data[2])<<8 |
		uint32(data[1])<<16 | uint32(data[0])<<24

	d.data = data[4:]
	return math.Float32frombits(val), nil
}

// DecodeDouble treats the next 8 bytes as an XDR encoded double-precision
// floating point and returns the result as a float64.
//
// An UnmarshalError is returned if there are insufficient bytes remaining.
//
// Reference:
// 	RFC Section 4.7 -  Double-Precision Floating Point
// 	64-bit double-precision IEEE 754 floating point
func (d *Decoder) DecodeDouble() (rv float64, err error) {
	data := d.data
	if len(data) < 8 {
		msg := fmt.Sprintf(errInsufficientBytes, 8)
		err = unmarshalError("DecodeDouble", ErrUnexpectedEnd, msg, data)
		return
	}
	val := uint64(data[7]) | uint64(data[6])<<8 |
		uint64(data[5])<<16 | uint64(data[4])<<24 |
		uint64(data[3])<<32 | uint64(data[2])<<40 |
		uint64(data[1])<<48 | uint64(data[0])<<56

	d.data = data[8:]
	return math.Float64frombits(val), nil
}

// RFC Section 4.8 -  Quadruple-Precision Floating Point
// 128-bit quadruple-precision floating point
// Not Implemented

// DecodeFixedOpaque treats the next 'size' bytes as XDR encoded opaque data and
// returns the result as a byte slice.
//
// An UnmarshalError is returned if there are insufficient bytes remaining to
// satisfy the passed size, including the necessary padding to make it a
// multiple of 4.
//
// Reference:
// 	RFC Section 4.9 - Fixed-Length Opaque Data
// 	Fixed-length uninterpreted data zero-padded to a multiple of four
func (d *Decoder) DecodeFixedOpaque(size int32) (rv []byte, err error) {
	if size == 0 {
		return
	}
	data := d.data
	if int32(len(data)) < size {
		msg := fmt.Sprintf(errInsufficientBytes, size)
		err = unmarshalError("DecodeFixedOpaque", ErrUnexpectedEnd, msg, data)
		return
	}
	pad := (4 - (size % 4)) % 4
	if int32(len(data[size:])) < pad {
		msg := fmt.Sprintf(errInsufficientPad, size+pad)
		err = unmarshalError("DecodeFixedOpaque", ErrUnexpectedEnd, msg, data)
		return
	}
	rv = data[0:size]

	d.data = data[size+pad:]
	return
}

// DecodeOpaque treats the next bytes as variable length XDR encoded opaque
// data and returns the result as a byte slice.
//
// An UnmarshalError is returned if there are insufficient bytes remaining or
// the opaque data is larger than the max length of a Go slice.
//
// Reference:
// 	RFC Section 4.10 - Variable-Length Opaque Data
// 	Unsigned integer length followed by fixed opaque data of that length
func (d *Decoder) DecodeOpaque() (rv []byte, err error) {
	dataLen, err := d.DecodeUint()
	if err != nil {
		return
	}
	if uint(dataLen) > uint(maxInt) {
		err = unmarshalError("DecodeOpaque", ErrOverflow, errMaxSlice, dataLen)
		return
	}
	return d.DecodeFixedOpaque(int32(dataLen))
}

// DecodeString treats the next bytes as a variable length XDR encoded string
// and returns the result as a string.  Character encoding is assumed to be
// UTF-8 and therefore ASCII compatible.  If the underlying character encoding
// is not compatibile with this assumption, the data can instead be read as
// variable-length opaque data (DecodeOpaque) and manually converted as needed.
//
// An UnmarshalError is returned if there are insufficient bytes remaining or
// the string data is larger than the max length of a Go slice.
//
// Reference:
// 	RFC Section 4.11 - String
// 	Unsigned integer length followed by bytes zero-padded to a multiple of four
func (d *Decoder) DecodeString() (rv string, err error) {
	dataLen, err := d.DecodeUint()
	if err != nil {
		return
	}
	if uint(dataLen) > uint(maxInt) {
		err = unmarshalError("DecodeString", ErrOverflow, errMaxSlice, dataLen)
		return
	}
	opaque, err := d.DecodeFixedOpaque(int32(dataLen))
	if err != nil {
		return
	}
	rv = string(opaque)
	return
}

// decodeFixedArray treats the next bytes as a series of XDR encoded elements
// of the same type as the array represented by the reflection value and decodes
// each element into the passed array.  The ignoreOpaque flag controls whether
// or not uint8 (byte) elements should be decoded individually or as a fixed
// sequence of opaque data.
//
// An UnmarshalError is returned if any issues are encountered while decoding
// the array elements.
//
// Reference:
// 	RFC Section 4.12 - Fixed-Length Array
// 	Individually XDR encoded array elements
func (d *Decoder) decodeFixedArray(v reflect.Value, ignoreOpaque bool) (err error) {
	// Treat [#]byte (byte is alias for uint8) as opaque data unless ignored.
	if !ignoreOpaque && v.Type().Elem().Kind() == reflect.Uint8 {
		data, err := d.DecodeFixedOpaque(int32(v.Len()))
		if err != nil {
			return err
		}
		reflect.Copy(v, reflect.ValueOf(data))
		return nil
	}

	// Decode each array element.
	for i := 0; i < v.Len(); i++ {
		err = d.decode(v.Index(i))
		if err != nil {
			return
		}
	}
	return
}

// decodeArray treats the next bytes as a variable length series of XDR encoded
// elements of the same type as the array represented by the reflection value.
// The number of elements is obtained by first decoding the unsigned integer
// element count.  Then each element is decoded into the passed array. The
// ignoreOpaque flag controls whether or not uint8 (byte) elements should be
// decoded individually or as a variable sequence of opaque data.
//
// An UnmarshalError is returned if any issues are encountered while decoding
// the array elements.
//
// Reference:
// 	RFC Section 4.13 - Variable-Length Array
// 	Unsigned integer length followed by individually XDR encoded array elements
func (d *Decoder) decodeArray(v reflect.Value, ignoreOpaque bool) (err error) {
	dataLen, err := d.DecodeUint()
	if err != nil {
		return
	}
	if uint(dataLen) > uint(maxInt) {
		err = unmarshalError("decodeArray", ErrOverflow, errMaxSlice, dataLen)
		return
	}

	// Allocate storage for the slice elements (the underlying array) if
	// existing slice does not enough capacity.
	sliceLen := int(dataLen)
	if v.Cap() < sliceLen {
		v.Set(reflect.MakeSlice(v.Type(), sliceLen, sliceLen))
	}
	if v.Len() < sliceLen {
		v.SetLen(sliceLen)
	}

	// Treat []byte (byte is alias for uint8) as opaque data unless ignored.
	if !ignoreOpaque && v.Type().Elem().Kind() == reflect.Uint8 {
		data, err := d.DecodeFixedOpaque(int32(sliceLen))
		if err != nil {
			return err
		}
		v.SetBytes(data)
		return nil
	}

	// Decode each slice element.
	for i := 0; i < sliceLen; i++ {
		err = d.decode(v.Index(i))
		if err != nil {
			return err
		}
	}
	return
}

// decodeStruct treats the next bytes as a series of XDR encoded elements
// of the same type as the exported fields of the struct represented by the
// passed reflection value.  Pointers are automatically indirected and
// allocated as necessary.
//
// An UnmarshalError is returned if any issues are encountered while decoding
// the elements.
//
// Reference:
// 	RFC Section 4.14 - Structure
// 	XDR encoded elements in the order of their declaration in the struct
func (d *Decoder) decodeStruct(v reflect.Value) (err error) {
	vt := v.Type()
	for i := 0; i < v.NumField(); i++ {
		// Skip unexported fields.
		vtf := vt.Field(i)
		if vtf.PkgPath != "" {
			continue
		}

		// Indirect through pointers allocating them as needed and ensure
		// the field is settable.
		vf := v.Field(i)
		vf, err = d.indirect(vf)
		if err != nil {
			return err
		}
		if !vf.CanSet() {
			msg := fmt.Sprintf("can't decode to unsettable '%v'",
				vf.Type().String())
			err = unmarshalError("decodeStruct", ErrNotSettable, msg, nil)
			return err
		}

		// Handle non-opaque data to []uint8 and [#]uint8 based on struct tag.
		tag := vtf.Tag.Get("xdropaque")
		if tag == "false" {
			switch vf.Kind() {
			case reflect.Slice:
				err = d.decodeArray(vf, true)
				if err != nil {
					return
				}
				continue

			case reflect.Array:
				err = d.decodeFixedArray(vf, true)
				if err != nil {
					return
				}
				continue
			}
		}

		// Decode each struct field.
		err = d.decode(vf)
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

// decodeMap treats the next bytes as an XDR encoded variable array of 2-element
// structures whose fields are of the same type as the map keys and elements
// represented by the passed reflection value.  Pointers are automatically
// indirected and allocated as necessary.
//
// An UnmarshalError is returned if any issues are encountered while decoding
// the elements.
func (d *Decoder) decodeMap(v reflect.Value) (err error) {
	dataLen, err := d.DecodeUint()
	if err != nil {
		return
	}

	// Allocate storage for the underlying map if needed.
	vt := v.Type()
	if v.IsNil() {
		v.Set(reflect.MakeMap(vt))
	}

	// Decode each key and value according to their type.
	keyType := vt.Key()
	elemType := vt.Elem()
	for i := uint32(0); i < dataLen; i++ {
		key := reflect.New(keyType).Elem()
		err = d.decode(key)
		if err != nil {
			return
		}

		val := reflect.New(elemType).Elem()
		err = d.decode(val)
		if err != nil {
			return
		}
		v.SetMapIndex(key, val)
	}
	return
}

// decodeInterface examines the interface represented by the passed reflection
// value to detect whether it is an interface that can be decoded into and
// if it is, extracts the underlying value to pass back into the decode function
// for decoding according to its type.
//
// An UnmarshalError is returned if any issues are encountered while decoding
// the interface.
func (d *Decoder) decodeInterface(v reflect.Value) (err error) {
	if v.IsNil() || !v.CanInterface() {
		msg := fmt.Sprintf("can't decode to nil interface")
		err = unmarshalError("decodeInterface", ErrNilInterface, msg, nil)
		return err
	}

	// Extract underlying value from the interface and indirect through pointers
	// allocating them as needed.
	ve := reflect.ValueOf(v.Interface())
	ve, err = d.indirect(ve)
	if err != nil {
		return err
	}
	if !ve.CanSet() {
		msg := fmt.Sprintf("can't decode to unsettable '%v'", ve.Type().String())
		err = unmarshalError("decodeInterface", ErrNotSettable, msg, nil)
		return err
	}
	return d.decode(ve)
}

// decode is the main workhorse for unmarshalling via reflection.  It uses
// the passed reflection value to choose the XDR primitives to decode from
// the XDR encoded data stored in the Decoder.  It is a recursive function,
// so cyclic data structures are not supported and will result in an infinite
// loop.
func (d *Decoder) decode(v reflect.Value) (err error) {
	if !v.IsValid() {
		msg := fmt.Sprintf("type '%s' is not valid", v.Kind().String())
		err = unmarshalError("decode", ErrUnsupportedType, msg, nil)
		return
	}

	// Indirect through pointers allocating them as needed.
	ve, err := d.indirect(v)
	if err != nil {
		return err
	}

	// Handle time.Time values by decoding them as an RFC3339 formatted
	// string with nanosecond precision.  Check the type string rather
	// than doing a full blown conversion to interface and type assertion
	// since checking a string is much quicker.
	if ve.Type().String() == "time.Time" {
		// Read the value as a string and parse it.
		timeString, err := d.DecodeString()
		if err != nil {
			return err
		}
		ttv, err := time.Parse(time.RFC3339, timeString)
		if err != nil {
			return err
		}
		ve.Set(reflect.ValueOf(ttv))
		return nil
	}

	// Handle native Go types.
	switch ve.Kind() {
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int:
		i, err := d.DecodeInt()
		if err != nil {
			return err
		}
		if ve.OverflowInt(int64(i)) {
			msg := fmt.Sprintf("signed integer too large to fit '%s'",
				ve.Kind().String())
			err = unmarshalError("decode", ErrOverflow, msg, i)
			return err
		}
		ve.SetInt(int64(i))

	case reflect.Int64:
		i, err := d.DecodeHyper()
		if err != nil {
			return err
		}
		ve.SetInt(i)

	case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint:
		ui, err := d.DecodeUint()
		if err != nil {
			return err
		}
		if ve.OverflowUint(uint64(ui)) {
			msg := fmt.Sprintf("unsigned integer too large to fit '%s'",
				ve.Kind().String())
			err = unmarshalError("decode", ErrOverflow, msg, ui)
			return err
		}
		ve.SetUint(uint64(ui))

	case reflect.Uint64:
		ui, err := d.DecodeUhyper()
		if err != nil {
			return err
		}
		ve.SetUint(ui)

	case reflect.Bool:
		b, err := d.DecodeBool()
		if err != nil {
			return err
		}
		ve.SetBool(b)

	case reflect.Float32:
		f, err := d.DecodeFloat()
		if err != nil {
			return err
		}
		ve.SetFloat(float64(f))

	case reflect.Float64:
		f, err := d.DecodeDouble()
		if err != nil {
			return err
		}
		ve.SetFloat(f)

	case reflect.String:
		s, err := d.DecodeString()
		if err != nil {
			return err
		}
		ve.SetString(s)

	case reflect.Array:
		err := d.decodeFixedArray(ve, false)
		if err != nil {
			return err
		}

	case reflect.Slice:
		err := d.decodeArray(ve, false)
		if err != nil {
			return err
		}

	case reflect.Struct:
		err := d.decodeStruct(ve)
		if err != nil {
			return err
		}

	case reflect.Map:
		err := d.decodeMap(ve)
		if err != nil {
			return err
		}

	case reflect.Interface:
		err := d.decodeInterface(ve)
		if err != nil {
			return err
		}

	// reflect.Uintptr, reflect.UnsafePointer
	default:
		msg := fmt.Sprintf("unsupported Go type '%s'", ve.Kind().String())
		err = unmarshalError("decode", ErrUnsupportedType, msg, nil)
	}

	return
}

// indirect dereferences pointers allocating them as needed until it reaches
// a non-pointer.  This allows transparent decoding through arbitrary levels
// of indirection.
func (d *Decoder) indirect(v reflect.Value) (rv reflect.Value, err error) {
	rv = v
	for rv.Kind() == reflect.Ptr {
		// Allocate pointer if needed.
		isNil := rv.IsNil()
		if isNil && !rv.CanSet() {
			msg := fmt.Sprintf("unable to allocate pointer for '%v'", rv.Type().String())
			err = unmarshalError("indirect", ErrNotSettable, msg, nil)
			return
		}
		if isNil {
			rv.Set(reflect.New(rv.Type().Elem()))
		}
		rv = rv.Elem()
	}
	return
}

// NewDecoder returns a Decoder that can be used to manually decode XDR data
// from a provided byte slice.  Typically, Unmarshal should be used instead of
// manually creating a Decoder.
func NewDecoder(bytes []byte) *Decoder {
	return &Decoder{data: bytes}
}
