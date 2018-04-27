// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package tls implements functionality for dealing with TLS-encoded data,
// as defined in RFC 5246.  This includes parsing and generation of TLS-encoded
// data, together with utility functions for dealing with the DigitallySigned
// TLS type.
package tls

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

// This file holds utility functions for TLS encoding/decoding data
// as per RFC 5246 section 4.

// A structuralError suggests that the TLS data is valid, but the Go type
// which is receiving it doesn't match.
type structuralError struct {
	field string
	msg   string
}

func (e structuralError) Error() string {
	var prefix string
	if e.field != "" {
		prefix = e.field + ": "
	}
	return "tls: structure error: " + prefix + e.msg
}

// A syntaxError suggests that the TLS data is invalid.
type syntaxError struct {
	field string
	msg   string
}

func (e syntaxError) Error() string {
	var prefix string
	if e.field != "" {
		prefix = e.field + ": "
	}
	return "tls: syntax error: " + prefix + e.msg
}

// Uint24 is an unsigned 3-byte integer.
type Uint24 uint32

// Enum is an unsigned integer.
type Enum uint64

var (
	uint8Type  = reflect.TypeOf(uint8(0))
	uint16Type = reflect.TypeOf(uint16(0))
	uint24Type = reflect.TypeOf(Uint24(0))
	uint32Type = reflect.TypeOf(uint32(0))
	uint64Type = reflect.TypeOf(uint64(0))
	enumType   = reflect.TypeOf(Enum(0))
)

// Unmarshal parses the TLS-encoded data in b and uses the reflect package to
// fill in an arbitrary value pointed at by val.  Because Unmarshal uses the
// reflect package, the structs being written to must use exported fields
// (upper case names).
//
// The mappings between TLS types and Go types is as follows; some fields
// must have tags (to indicate their encoded size).
//
//	TLS		Go		Required Tags
//	opaque		byte / uint8
//	uint8		byte / uint8
//	uint16		uint16
//	uint24		tls.Uint24
//	uint32		uint32
//	uint64		uint64
//	enum		tls.Enum	size:S or maxval:N
//	Type<N,M>	[]Type		minlen:N,maxlen:M
//	opaque[N]	[N]byte / [N]uint8
//	uint8[N]	[N]byte / [N]uint8
//	struct { }	struct { }
//	select(T) {
//	 case e1: Type	*T		selector:Field,val:e1
//	}
//
// TLS variants (RFC 5246 s4.6.1) are only supported when the value of the
// associated enumeration type is available earlier in the same enclosing
// struct, and each possible variant is marked with a selector tag (to
// indicate which field selects the variants) and a val tag (to indicate
// what value of the selector picks this particular field).
//
// For example, a TLS structure:
//
//   enum { e1(1), e2(2) } EnumType;
//   struct {
//      EnumType sel;
//      select(sel) {
//         case e1: uint16
//         case e2: uint32
//      } data;
//   } VariantItem;
//
// would have a corresponding Go type:
//
//   type VariantItem struct {
//      Sel    tls.Enum  `tls:"maxval:2"`
//      Data16 *uint16   `tls:"selector:Sel,val:1"`
//      Data32 *uint32   `tls:"selector:Sel,val:2"`
//    }
//
// TLS fixed-length vectors of types other than opaque or uint8 are not supported.
//
// For TLS variable-length vectors that are themselves used in other vectors,
// create a single-field structure to represent the inner type. For example, for:
//
//   opaque InnerType<1..65535>;
//   struct {
//     InnerType inners<1,65535>;
//   } Something;
//
// convert to:
//
//   type InnerType struct {
//      Val    []byte       `tls:"minlen:1,maxlen:65535"`
//   }
//   type Something struct {
//      Inners []InnerType  `tls:"minlen:1,maxlen:65535"`
//   }
//
// If the encoded value does not fit in the Go type, Unmarshal returns a parse error.
func Unmarshal(b []byte, val interface{}) ([]byte, error) {
	return UnmarshalWithParams(b, val, "")
}

// UnmarshalWithParams allows field parameters to be specified for the
// top-level element. The form of the params is the same as the field tags.
func UnmarshalWithParams(b []byte, val interface{}, params string) ([]byte, error) {
	info, err := fieldTagToFieldInfo(params, "")
	if err != nil {
		return nil, err
	}
	// The passed in interface{} is a pointer (to allow the value to be written
	// to); extract the pointed-to object as a reflect.Value, so parseField
	// can do various introspection things.
	v := reflect.ValueOf(val).Elem()
	offset, err := parseField(v, b, 0, info)
	if err != nil {
		return nil, err
	}
	return b[offset:], nil
}

// Return the number of bytes needed to encode values up to (and including) x.
func byteCount(x uint64) uint {
	switch {
	case x < 0x100:
		return 1
	case x < 0x10000:
		return 2
	case x < 0x1000000:
		return 3
	case x < 0x100000000:
		return 4
	case x < 0x10000000000:
		return 5
	case x < 0x1000000000000:
		return 6
	case x < 0x100000000000000:
		return 7
	default:
		return 8
	}
}

type fieldInfo struct {
	count    uint // Number of bytes
	countSet bool
	minlen   uint64 // Only relevant for slices
	maxlen   uint64 // Only relevant for slices
	selector string // Only relevant for select sub-values
	val      uint64 // Only relevant for select sub-values
	name     string // Used for better error messages
}

func (i *fieldInfo) fieldName() string {
	if i == nil {
		return ""
	}
	return i.name
}

// Given a tag string, return a fieldInfo describing the field.
func fieldTagToFieldInfo(str string, name string) (*fieldInfo, error) {
	var info *fieldInfo
	// Iterate over clauses in the tag, ignoring any that don't parse properly.
	for _, part := range strings.Split(str, ",") {
		switch {
		case strings.HasPrefix(part, "maxval:"):
			if v, err := strconv.ParseUint(part[7:], 10, 64); err == nil {
				info = &fieldInfo{count: byteCount(v), countSet: true}
			}
		case strings.HasPrefix(part, "size:"):
			if sz, err := strconv.ParseUint(part[5:], 10, 32); err == nil {
				info = &fieldInfo{count: uint(sz), countSet: true}
			}
		case strings.HasPrefix(part, "maxlen:"):
			v, err := strconv.ParseUint(part[7:], 10, 64)
			if err != nil {
				continue
			}
			if info == nil {
				info = &fieldInfo{}
			}
			info.count = byteCount(v)
			info.countSet = true
			info.maxlen = v
		case strings.HasPrefix(part, "minlen:"):
			v, err := strconv.ParseUint(part[7:], 10, 64)
			if err != nil {
				continue
			}
			if info == nil {
				info = &fieldInfo{}
			}
			info.minlen = v
		case strings.HasPrefix(part, "selector:"):
			if info == nil {
				info = &fieldInfo{}
			}
			info.selector = part[9:]
		case strings.HasPrefix(part, "val:"):
			v, err := strconv.ParseUint(part[4:], 10, 64)
			if err != nil {
				continue
			}
			if info == nil {
				info = &fieldInfo{}
			}
			info.val = v
		}
	}
	if info != nil {
		info.name = name
		if info.selector == "" {
			if info.count < 1 {
				return nil, structuralError{name, "field of unknown size in " + str}
			} else if info.count > 8 {
				return nil, structuralError{name, "specified size too large in " + str}
			} else if info.minlen > info.maxlen {
				return nil, structuralError{name, "specified length range inverted in " + str}
			} else if info.val > 0 {
				return nil, structuralError{name, "specified selector value but not field in " + str}
			}
		}
	} else if name != "" {
		info = &fieldInfo{name: name}
	}
	return info, nil
}

// Check that a value fits into a field described by a fieldInfo structure.
func (i fieldInfo) check(val uint64, fldName string) error {
	if val >= (1 << (8 * i.count)) {
		return structuralError{fldName, fmt.Sprintf("value %d too large for size", val)}
	}
	if i.maxlen != 0 {
		if val < i.minlen {
			return structuralError{fldName, fmt.Sprintf("value %d too small for minimum %d", val, i.minlen)}
		}
		if val > i.maxlen {
			return structuralError{fldName, fmt.Sprintf("value %d too large for maximum %d", val, i.maxlen)}
		}
	}
	return nil
}

// readVarUint reads an big-endian unsigned integer of the given size in
// bytes.
func readVarUint(data []byte, info *fieldInfo) (uint64, error) {
	if info == nil || !info.countSet {
		return 0, structuralError{info.fieldName(), "no field size information available"}
	}
	if len(data) < int(info.count) {
		return 0, syntaxError{info.fieldName(), "truncated variable-length integer"}
	}
	var result uint64
	for i := uint(0); i < info.count; i++ {
		result = (result << 8) | uint64(data[i])
	}
	if err := info.check(result, info.name); err != nil {
		return 0, err
	}
	return result, nil
}

// parseField is the main parsing function. Given a byte slice and an offset
// (in bytes) into the data, it will try to parse a suitable ASN.1 value out
// and store it in the given Value.
func parseField(v reflect.Value, data []byte, initOffset int, info *fieldInfo) (int, error) {
	offset := initOffset
	rest := data[offset:]

	fieldType := v.Type()
	// First look for known fixed types.
	switch fieldType {
	case uint8Type:
		if len(rest) < 1 {
			return offset, syntaxError{info.fieldName(), "truncated uint8"}
		}
		v.SetUint(uint64(rest[0]))
		offset++
		return offset, nil
	case uint16Type:
		if len(rest) < 2 {
			return offset, syntaxError{info.fieldName(), "truncated uint16"}
		}
		v.SetUint(uint64(binary.BigEndian.Uint16(rest)))
		offset += 2
		return offset, nil
	case uint24Type:
		if len(rest) < 3 {
			return offset, syntaxError{info.fieldName(), "truncated uint24"}
		}
		v.SetUint(uint64(data[0])<<16 | uint64(data[1])<<8 | uint64(data[2]))
		offset += 3
		return offset, nil
	case uint32Type:
		if len(rest) < 4 {
			return offset, syntaxError{info.fieldName(), "truncated uint32"}
		}
		v.SetUint(uint64(binary.BigEndian.Uint32(rest)))
		offset += 4
		return offset, nil
	case uint64Type:
		if len(rest) < 8 {
			return offset, syntaxError{info.fieldName(), "truncated uint64"}
		}
		v.SetUint(uint64(binary.BigEndian.Uint64(rest)))
		offset += 8
		return offset, nil
	}

	// Now deal with user-defined types.
	switch v.Kind() {
	case enumType.Kind():
		// Assume that anything of the same kind as Enum is an Enum, so that
		// users can alias types of their own to Enum.
		val, err := readVarUint(rest, info)
		if err != nil {
			return offset, err
		}
		v.SetUint(val)
		offset += int(info.count)
		return offset, nil
	case reflect.Struct:
		structType := fieldType
		// TLS includes a select(Enum) {..} construct, where the value of an enum
		// indicates which variant field is present (like a C union). We require
		// that the enum value be an earlier field in the same structure (the selector),
		// and that each of the possible variant destination fields be pointers.
		// So the Go mapping looks like:
		//     type variantType struct {
		//         Which  tls.Enum  `tls:"size:1"`                // this is the selector
		//         Val1   *type1    `tls:"selector:Which,val:1"`  // this is a destination
		//         Val2   *type2    `tls:"selector:Which,val:1"`  // this is a destination
		//     }

		// To deal with this, we track any enum-like fields and their values...
		enums := make(map[string]uint64)
		// .. and we track which selector names we've seen (in the destination field tags),
		// and whether a destination for that selector has been chosen.
		selectorSeen := make(map[string]bool)
		for i := 0; i < structType.NumField(); i++ {
			// Find information about this field.
			tag := structType.Field(i).Tag.Get("tls")
			fieldInfo, err := fieldTagToFieldInfo(tag, structType.Field(i).Name)
			if err != nil {
				return offset, err
			}

			destination := v.Field(i)
			if fieldInfo.selector != "" {
				// This is a possible select(Enum) destination, so first check that the referenced
				// selector field has already been seen earlier in the struct.
				choice, ok := enums[fieldInfo.selector]
				if !ok {
					return offset, structuralError{fieldInfo.name, "selector not seen: " + fieldInfo.selector}
				}
				if structType.Field(i).Type.Kind() != reflect.Ptr {
					return offset, structuralError{fieldInfo.name, "choice field not a pointer type"}
				}
				// Is this the first mention of the selector field name?  If so, remember it.
				seen, ok := selectorSeen[fieldInfo.selector]
				if !ok {
					selectorSeen[fieldInfo.selector] = false
				}
				if choice != fieldInfo.val {
					// This destination field was not the chosen one, so make it nil (we checked
					// it was a pointer above).
					v.Field(i).Set(reflect.Zero(structType.Field(i).Type))
					continue
				}
				if seen {
					// We already saw a different destination field receive the value for this
					// selector value, which indicates a badly annotated structure.
					return offset, structuralError{fieldInfo.name, "duplicate selector value for " + fieldInfo.selector}
				}
				selectorSeen[fieldInfo.selector] = true
				// Make an object of the pointed-to type and parse into that.
				v.Field(i).Set(reflect.New(structType.Field(i).Type.Elem()))
				destination = v.Field(i).Elem()
			}
			offset, err = parseField(destination, data, offset, fieldInfo)
			if err != nil {
				return offset, err
			}

			// Remember any possible tls.Enum values encountered in case they are selectors.
			if structType.Field(i).Type.Kind() == enumType.Kind() {
				enums[structType.Field(i).Name] = v.Field(i).Uint()
			}

		}

		// Now we have seen all fields in the structure, check that all select(Enum) {..} selector
		// fields found a destination to put their data in.
		for selector, seen := range selectorSeen {
			if !seen {
				return offset, syntaxError{info.fieldName(), selector + ": unhandled value for selector"}
			}
		}
		return offset, nil
	case reflect.Array:
		datalen := v.Len()

		if datalen > len(rest) {
			return offset, syntaxError{info.fieldName(), "truncated array"}
		}
		inner := rest[:datalen]
		offset += datalen
		if fieldType.Elem().Kind() != reflect.Uint8 {
			// Only byte/uint8 arrays are supported
			return offset, structuralError{info.fieldName(), "unsupported array type: " + v.Type().String()}
		}
		reflect.Copy(v, reflect.ValueOf(inner))
		return offset, nil

	case reflect.Slice:
		sliceType := fieldType
		// Slices represent variable-length vectors, which are prefixed by a length field.
		// The fieldInfo indicates the size of that length field.
		varlen, err := readVarUint(rest, info)
		if err != nil {
			return offset, err
		}
		datalen := int(varlen)
		offset += int(info.count)
		rest = rest[info.count:]

		if datalen > len(rest) {
			return offset, syntaxError{info.fieldName(), "truncated slice"}
		}
		inner := rest[:datalen]
		offset += datalen
		if fieldType.Elem().Kind() == reflect.Uint8 {
			// Fast version for []byte
			v.Set(reflect.MakeSlice(sliceType, datalen, datalen))
			reflect.Copy(v, reflect.ValueOf(inner))
			return offset, nil
		}

		v.Set(reflect.MakeSlice(sliceType, 0, datalen))
		single := reflect.New(sliceType.Elem())
		for innerOffset := 0; innerOffset < len(inner); {
			var err error
			innerOffset, err = parseField(single.Elem(), inner, innerOffset, nil)
			if err != nil {
				return offset, err
			}
			v.Set(reflect.Append(v, single.Elem()))
		}
		return offset, nil

	default:
		return offset, structuralError{info.fieldName(), fmt.Sprintf("unsupported type: %s of kind %s", fieldType, v.Kind())}
	}
}

// Marshal returns the TLS encoding of val.
func Marshal(val interface{}) ([]byte, error) {
	return MarshalWithParams(val, "")
}

// MarshalWithParams returns the TLS encoding of val, and allows field
// parameters to be specified for the top-level element.  The form
// of the params is the same as the field tags.
func MarshalWithParams(val interface{}, params string) ([]byte, error) {
	info, err := fieldTagToFieldInfo(params, "")
	if err != nil {
		return nil, err
	}
	var out bytes.Buffer
	v := reflect.ValueOf(val)
	if err := marshalField(&out, v, info); err != nil {
		return nil, err
	}
	return out.Bytes(), err
}

func marshalField(out *bytes.Buffer, v reflect.Value, info *fieldInfo) error {
	var prefix string
	if info != nil && len(info.name) > 0 {
		prefix = info.name + ": "
	}
	fieldType := v.Type()
	// First look for known fixed types.
	switch fieldType {
	case uint8Type:
		out.WriteByte(byte(v.Uint()))
		return nil
	case uint16Type:
		scratch := make([]byte, 2)
		binary.BigEndian.PutUint16(scratch, uint16(v.Uint()))
		out.Write(scratch)
		return nil
	case uint24Type:
		i := v.Uint()
		if i > 0xffffff {
			return structuralError{info.fieldName(), fmt.Sprintf("uint24 overflow %d", i)}
		}
		scratch := make([]byte, 4)
		binary.BigEndian.PutUint32(scratch, uint32(i))
		out.Write(scratch[1:])
		return nil
	case uint32Type:
		scratch := make([]byte, 4)
		binary.BigEndian.PutUint32(scratch, uint32(v.Uint()))
		out.Write(scratch)
		return nil
	case uint64Type:
		scratch := make([]byte, 8)
		binary.BigEndian.PutUint64(scratch, uint64(v.Uint()))
		out.Write(scratch)
		return nil
	}

	// Now deal with user-defined types.
	switch v.Kind() {
	case enumType.Kind():
		i := v.Uint()
		if info == nil {
			return structuralError{info.fieldName(), "enum field tag missing"}
		}
		if err := info.check(i, prefix); err != nil {
			return err
		}
		scratch := make([]byte, 8)
		binary.BigEndian.PutUint64(scratch, uint64(i))
		out.Write(scratch[(8 - info.count):])
		return nil
	case reflect.Struct:
		structType := fieldType
		enums := make(map[string]uint64) // Values of any Enum fields
		// The comment parseField() describes the mapping of the TLS select(Enum) {..} construct;
		// here we have selector and source (rather than destination) fields.

		// Track which selector names we've seen (in the source field tags), and whether a source
		// value for that selector has been processed.
		selectorSeen := make(map[string]bool)
		for i := 0; i < structType.NumField(); i++ {
			// Find information about this field.
			tag := structType.Field(i).Tag.Get("tls")
			fieldInfo, err := fieldTagToFieldInfo(tag, structType.Field(i).Name)
			if err != nil {
				return err
			}

			source := v.Field(i)
			if fieldInfo.selector != "" {
				// This field is a possible source for a select(Enum) {..}.  First check
				// the selector field name has been seen.
				choice, ok := enums[fieldInfo.selector]
				if !ok {
					return structuralError{fieldInfo.name, "selector not seen: " + fieldInfo.selector}
				}
				if structType.Field(i).Type.Kind() != reflect.Ptr {
					return structuralError{fieldInfo.name, "choice field not a pointer type"}
				}
				// Is this the first mention of the selector field name? If so, remember it.
				seen, ok := selectorSeen[fieldInfo.selector]
				if !ok {
					selectorSeen[fieldInfo.selector] = false
				}
				if choice != fieldInfo.val {
					// This source was not chosen; police that it should be nil.
					if v.Field(i).Pointer() != uintptr(0) {
						return structuralError{fieldInfo.name, "unchosen field is non-nil"}
					}
					continue
				}
				if seen {
					// We already saw a different source field generate the value for this
					// selector value, which indicates a badly annotated structure.
					return structuralError{fieldInfo.name, "duplicate selector value for " + fieldInfo.selector}
				}
				selectorSeen[fieldInfo.selector] = true
				if v.Field(i).Pointer() == uintptr(0) {
					return structuralError{fieldInfo.name, "chosen field is nil"}
				}
				// Marshal from the pointed-to source object.
				source = v.Field(i).Elem()
			}

			var fieldData bytes.Buffer
			if err := marshalField(&fieldData, source, fieldInfo); err != nil {
				return err
			}
			out.Write(fieldData.Bytes())

			// Remember any tls.Enum values encountered in case they are selectors.
			if structType.Field(i).Type.Kind() == enumType.Kind() {
				enums[structType.Field(i).Name] = v.Field(i).Uint()
			}
		}
		// Now we have seen all fields in the structure, check that all select(Enum) {..} selector
		// fields found a source field get get their data from.
		for selector, seen := range selectorSeen {
			if !seen {
				return syntaxError{info.fieldName(), selector + ": unhandled value for selector"}
			}
		}
		return nil

	case reflect.Array:
		datalen := v.Len()
		arrayType := fieldType
		if arrayType.Elem().Kind() != reflect.Uint8 {
			// Only byte/uint8 arrays are supported
			return structuralError{info.fieldName(), "unsupported array type"}
		}
		bytes := make([]byte, datalen)
		for i := 0; i < datalen; i++ {
			bytes[i] = uint8(v.Index(i).Uint())
		}
		_, err := out.Write(bytes)
		return err

	case reflect.Slice:
		if info == nil {
			return structuralError{info.fieldName(), "slice field tag missing"}
		}

		sliceType := fieldType
		if sliceType.Elem().Kind() == reflect.Uint8 {
			// Fast version for []byte: first write the length as info.count bytes.
			datalen := v.Len()
			scratch := make([]byte, 8)
			binary.BigEndian.PutUint64(scratch, uint64(datalen))
			out.Write(scratch[(8 - info.count):])

			if err := info.check(uint64(datalen), prefix); err != nil {
				return err
			}
			// Then just write the data.
			bytes := make([]byte, datalen)
			for i := 0; i < datalen; i++ {
				bytes[i] = uint8(v.Index(i).Uint())
			}
			_, err := out.Write(bytes)
			return err
		}
		// General version: use a separate Buffer to write the slice entries into.
		var innerBuf bytes.Buffer
		for i := 0; i < v.Len(); i++ {
			if err := marshalField(&innerBuf, v.Index(i), nil); err != nil {
				return err
			}
		}

		// Now insert (and check) the size.
		size := uint64(innerBuf.Len())
		if err := info.check(size, prefix); err != nil {
			return err
		}
		scratch := make([]byte, 8)
		binary.BigEndian.PutUint64(scratch, size)
		out.Write(scratch[(8 - info.count):])

		// Then copy the data.
		_, err := out.Write(innerBuf.Bytes())
		return err

	default:
		return structuralError{info.fieldName(), fmt.Sprintf("unsupported type: %s of kind %s", fieldType, v.Kind())}
	}
}
