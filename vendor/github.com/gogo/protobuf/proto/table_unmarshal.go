// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2016 The Go Authors.  All rights reserved.
// https://github.com/golang/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package proto

import (
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"unicode/utf8"
)

// Unmarshal is the entry point from the generated .pb.go files.
// This function is not intended to be used by non-generated code.
// This function is not subject to any compatibility guarantee.
// msg contains a pointer to a protocol buffer struct.
// b is the data to be unmarshaled into the protocol buffer.
// a is a pointer to a place to store cached unmarshal information.
func (a *InternalMessageInfo) Unmarshal(msg Message, b []byte) error {
	// Load the unmarshal information for this message type.
	// The atomic load ensures memory consistency.
	u := atomicLoadUnmarshalInfo(&a.unmarshal)
	if u == nil {
		// Slow path: find unmarshal info for msg, update a with it.
		u = getUnmarshalInfo(reflect.TypeOf(msg).Elem())
		atomicStoreUnmarshalInfo(&a.unmarshal, u)
	}
	// Then do the unmarshaling.
	err := u.unmarshal(toPointer(&msg), b)
	return err
}

type unmarshalInfo struct {
	typ reflect.Type // type of the protobuf struct

	// 0 = only typ field is initialized
	// 1 = completely initialized
	initialized     int32
	lock            sync.Mutex                    // prevents double initialization
	dense           []unmarshalFieldInfo          // fields indexed by tag #
	sparse          map[uint64]unmarshalFieldInfo // fields indexed by tag #
	reqFields       []string                      // names of required fields
	reqMask         uint64                        // 1<<len(reqFields)-1
	unrecognized    field                         // offset of []byte to put unrecognized data (or invalidField if we should throw it away)
	extensions      field                         // offset of extensions field (of type proto.XXX_InternalExtensions), or invalidField if it does not exist
	oldExtensions   field                         // offset of old-form extensions field (of type map[int]Extension)
	extensionRanges []ExtensionRange              // if non-nil, implies extensions field is valid
	isMessageSet    bool                          // if true, implies extensions field is valid

	bytesExtensions field // offset of XXX_extensions with type []byte
}

// An unmarshaler takes a stream of bytes and a pointer to a field of a message.
// It decodes the field, stores it at f, and returns the unused bytes.
// w is the wire encoding.
// b is the data after the tag and wire encoding have been read.
type unmarshaler func(b []byte, f pointer, w int) ([]byte, error)

type unmarshalFieldInfo struct {
	// location of the field in the proto message structure.
	field field

	// function to unmarshal the data for the field.
	unmarshal unmarshaler

	// if a required field, contains a single set bit at this field's index in the required field list.
	reqMask uint64

	name string // name of the field, for error reporting
}

var (
	unmarshalInfoMap  = map[reflect.Type]*unmarshalInfo{}
	unmarshalInfoLock sync.Mutex
)

// getUnmarshalInfo returns the data structure which can be
// subsequently used to unmarshal a message of the given type.
// t is the type of the message (note: not pointer to message).
func getUnmarshalInfo(t reflect.Type) *unmarshalInfo {
	// It would be correct to return a new unmarshalInfo
	// unconditionally. We would end up allocating one
	// per occurrence of that type as a message or submessage.
	// We use a cache here just to reduce memory usage.
	unmarshalInfoLock.Lock()
	defer unmarshalInfoLock.Unlock()
	u := unmarshalInfoMap[t]
	if u == nil {
		u = &unmarshalInfo{typ: t}
		// Note: we just set the type here. The rest of the fields
		// will be initialized on first use.
		unmarshalInfoMap[t] = u
	}
	return u
}

// unmarshal does the main work of unmarshaling a message.
// u provides type information used to unmarshal the message.
// m is a pointer to a protocol buffer message.
// b is a byte stream to unmarshal into m.
// This is top routine used when recursively unmarshaling submessages.
func (u *unmarshalInfo) unmarshal(m pointer, b []byte) error {
	if atomic.LoadInt32(&u.initialized) == 0 {
		u.computeUnmarshalInfo()
	}
	if u.isMessageSet {
		return unmarshalMessageSet(b, m.offset(u.extensions).toExtensions())
	}
	var reqMask uint64 // bitmask of required fields we've seen.
	var errLater error
	for len(b) > 0 {
		// Read tag and wire type.
		// Special case 1 and 2 byte varints.
		var x uint64
		if b[0] < 128 {
			x = uint64(b[0])
			b = b[1:]
		} else if len(b) >= 2 && b[1] < 128 {
			x = uint64(b[0]&0x7f) + uint64(b[1])<<7
			b = b[2:]
		} else {
			var n int
			x, n = decodeVarint(b)
			if n == 0 {
				return io.ErrUnexpectedEOF
			}
			b = b[n:]
		}
		tag := x >> 3
		wire := int(x) & 7

		// Dispatch on the tag to one of the unmarshal* functions below.
		var f unmarshalFieldInfo
		if tag < uint64(len(u.dense)) {
			f = u.dense[tag]
		} else {
			f = u.sparse[tag]
		}
		if fn := f.unmarshal; fn != nil {
			var err error
			b, err = fn(b, m.offset(f.field), wire)
			if err == nil {
				reqMask |= f.reqMask
				continue
			}
			if r, ok := err.(*RequiredNotSetError); ok {
				// Remember this error, but keep parsing. We need to produce
				// a full parse even if a required field is missing.
				if errLater == nil {
					errLater = r
				}
				reqMask |= f.reqMask
				continue
			}
			if err != errInternalBadWireType {
				if err == errInvalidUTF8 {
					if errLater == nil {
						fullName := revProtoTypes[reflect.PtrTo(u.typ)] + "." + f.name
						errLater = &invalidUTF8Error{fullName}
					}
					continue
				}
				return err
			}
			// Fragments with bad wire type are treated as unknown fields.
		}

		// Unknown tag.
		if !u.unrecognized.IsValid() {
			// Don't keep unrecognized data; just skip it.
			var err error
			b, err = skipField(b, wire)
			if err != nil {
				return err
			}
			continue
		}
		// Keep unrecognized data around.
		// maybe in extensions, maybe in the unrecognized field.
		z := m.offset(u.unrecognized).toBytes()
		var emap map[int32]Extension
		var e Extension
		for _, r := range u.extensionRanges {
			if uint64(r.Start) <= tag && tag <= uint64(r.End) {
				if u.extensions.IsValid() {
					mp := m.offset(u.extensions).toExtensions()
					emap = mp.extensionsWrite()
					e = emap[int32(tag)]
					z = &e.enc
					break
				}
				if u.oldExtensions.IsValid() {
					p := m.offset(u.oldExtensions).toOldExtensions()
					emap = *p
					if emap == nil {
						emap = map[int32]Extension{}
						*p = emap
					}
					e = emap[int32(tag)]
					z = &e.enc
					break
				}
				if u.bytesExtensions.IsValid() {
					z = m.offset(u.bytesExtensions).toBytes()
					break
				}
				panic("no extensions field available")
			}
		}
		// Use wire type to skip data.
		var err error
		b0 := b
		b, err = skipField(b, wire)
		if err != nil {
			return err
		}
		*z = encodeVarint(*z, tag<<3|uint64(wire))
		*z = append(*z, b0[:len(b0)-len(b)]...)

		if emap != nil {
			emap[int32(tag)] = e
		}
	}
	if reqMask != u.reqMask && errLater == nil {
		// A required field of this message is missing.
		for _, n := range u.reqFields {
			if reqMask&1 == 0 {
				errLater = &RequiredNotSetError{n}
			}
			reqMask >>= 1
		}
	}
	return errLater
}

// computeUnmarshalInfo fills in u with information for use
// in unmarshaling protocol buffers of type u.typ.
func (u *unmarshalInfo) computeUnmarshalInfo() {
	u.lock.Lock()
	defer u.lock.Unlock()
	if u.initialized != 0 {
		return
	}
	t := u.typ
	n := t.NumField()

	// Set up the "not found" value for the unrecognized byte buffer.
	// This is the default for proto3.
	u.unrecognized = invalidField
	u.extensions = invalidField
	u.oldExtensions = invalidField
	u.bytesExtensions = invalidField

	// List of the generated type and offset for each oneof field.
	type oneofField struct {
		ityp  reflect.Type // interface type of oneof field
		field field        // offset in containing message
	}
	var oneofFields []oneofField

	for i := 0; i < n; i++ {
		f := t.Field(i)
		if f.Name == "XXX_unrecognized" {
			// The byte slice used to hold unrecognized input is special.
			if f.Type != reflect.TypeOf(([]byte)(nil)) {
				panic("bad type for XXX_unrecognized field: " + f.Type.Name())
			}
			u.unrecognized = toField(&f)
			continue
		}
		if f.Name == "XXX_InternalExtensions" {
			// Ditto here.
			if f.Type != reflect.TypeOf(XXX_InternalExtensions{}) {
				panic("bad type for XXX_InternalExtensions field: " + f.Type.Name())
			}
			u.extensions = toField(&f)
			if f.Tag.Get("protobuf_messageset") == "1" {
				u.isMessageSet = true
			}
			continue
		}
		if f.Name == "XXX_extensions" {
			// An older form of the extensions field.
			if f.Type == reflect.TypeOf((map[int32]Extension)(nil)) {
				u.oldExtensions = toField(&f)
				continue
			} else if f.Type == reflect.TypeOf(([]byte)(nil)) {
				u.bytesExtensions = toField(&f)
				continue
			}
			panic("bad type for XXX_extensions field: " + f.Type.Name())
		}
		if f.Name == "XXX_NoUnkeyedLiteral" || f.Name == "XXX_sizecache" {
			continue
		}

		oneof := f.Tag.Get("protobuf_oneof")
		if oneof != "" {
			oneofFields = append(oneofFields, oneofField{f.Type, toField(&f)})
			// The rest of oneof processing happens below.
			continue
		}

		tags := f.Tag.Get("protobuf")
		tagArray := strings.Split(tags, ",")
		if len(tagArray) < 2 {
			panic("protobuf tag not enough fields in " + t.Name() + "." + f.Name + ": " + tags)
		}
		tag, err := strconv.Atoi(tagArray[1])
		if err != nil {
			panic("protobuf tag field not an integer: " + tagArray[1])
		}

		name := ""
		for _, tag := range tagArray[3:] {
			if strings.HasPrefix(tag, "name=") {
				name = tag[5:]
			}
		}

		// Extract unmarshaling function from the field (its type and tags).
		unmarshal := fieldUnmarshaler(&f)

		// Required field?
		var reqMask uint64
		if tagArray[2] == "req" {
			bit := len(u.reqFields)
			u.reqFields = append(u.reqFields, name)
			reqMask = uint64(1) << uint(bit)
			// TODO: if we have more than 64 required fields, we end up
			// not verifying that all required fields are present.
			// Fix this, perhaps using a count of required fields?
		}

		// Store the info in the correct slot in the message.
		u.setTag(tag, toField(&f), unmarshal, reqMask, name)
	}

	// Find any types associated with oneof fields.
	// TODO: XXX_OneofFuncs returns more info than we need.  Get rid of some of it?
	fn := reflect.Zero(reflect.PtrTo(t)).MethodByName("XXX_OneofFuncs")
	// gogo: len(oneofFields) > 0 is needed for embedded oneof messages, without a marshaler and unmarshaler
	if fn.IsValid() && len(oneofFields) > 0 {
		res := fn.Call(nil)[3] // last return value from XXX_OneofFuncs: []interface{}
		for i := res.Len() - 1; i >= 0; i-- {
			v := res.Index(i)                             // interface{}
			tptr := reflect.ValueOf(v.Interface()).Type() // *Msg_X
			typ := tptr.Elem()                            // Msg_X

			f := typ.Field(0) // oneof implementers have one field
			baseUnmarshal := fieldUnmarshaler(&f)
			tags := strings.Split(f.Tag.Get("protobuf"), ",")
			fieldNum, err := strconv.Atoi(tags[1])
			if err != nil {
				panic("protobuf tag field not an integer: " + tags[1])
			}
			var name string
			for _, tag := range tags {
				if strings.HasPrefix(tag, "name=") {
					name = strings.TrimPrefix(tag, "name=")
					break
				}
			}

			// Find the oneof field that this struct implements.
			// Might take O(n^2) to process all of the oneofs, but who cares.
			for _, of := range oneofFields {
				if tptr.Implements(of.ityp) {
					// We have found the corresponding interface for this struct.
					// That lets us know where this struct should be stored
					// when we encounter it during unmarshaling.
					unmarshal := makeUnmarshalOneof(typ, of.ityp, baseUnmarshal)
					u.setTag(fieldNum, of.field, unmarshal, 0, name)
				}
			}
		}
	}

	// Get extension ranges, if any.
	fn = reflect.Zero(reflect.PtrTo(t)).MethodByName("ExtensionRangeArray")
	if fn.IsValid() {
		if !u.extensions.IsValid() && !u.oldExtensions.IsValid() && !u.bytesExtensions.IsValid() {
			panic("a message with extensions, but no extensions field in " + t.Name())
		}
		u.extensionRanges = fn.Call(nil)[0].Interface().([]ExtensionRange)
	}

	// Explicitly disallow tag 0. This will ensure we flag an error
	// when decoding a buffer of all zeros. Without this code, we
	// would decode and skip an all-zero buffer of even length.
	// [0 0] is [tag=0/wiretype=varint varint-encoded-0].
	u.setTag(0, zeroField, func(b []byte, f pointer, w int) ([]byte, error) {
		return nil, fmt.Errorf("proto: %s: illegal tag 0 (wire type %d)", t, w)
	}, 0, "")

	// Set mask for required field check.
	u.reqMask = uint64(1)<<uint(len(u.reqFields)) - 1

	atomic.StoreInt32(&u.initialized, 1)
}

// setTag stores the unmarshal information for the given tag.
// tag = tag # for field
// field/unmarshal = unmarshal info for that field.
// reqMask = if required, bitmask for field position in required field list. 0 otherwise.
// name = short name of the field.
func (u *unmarshalInfo) setTag(tag int, field field, unmarshal unmarshaler, reqMask uint64, name string) {
	i := unmarshalFieldInfo{field: field, unmarshal: unmarshal, reqMask: reqMask, name: name}
	n := u.typ.NumField()
	if tag >= 0 && (tag < 16 || tag < 2*n) { // TODO: what are the right numbers here?
		for len(u.dense) <= tag {
			u.dense = append(u.dense, unmarshalFieldInfo{})
		}
		u.dense[tag] = i
		return
	}
	if u.sparse == nil {
		u.sparse = map[uint64]unmarshalFieldInfo{}
	}
	u.sparse[uint64(tag)] = i
}

// fieldUnmarshaler returns an unmarshaler for the given field.
func fieldUnmarshaler(f *reflect.StructField) unmarshaler {
	if f.Type.Kind() == reflect.Map {
		return makeUnmarshalMap(f)
	}
	return typeUnmarshaler(f.Type, f.Tag.Get("protobuf"))
}

// typeUnmarshaler returns an unmarshaler for the given field type / field tag pair.
func typeUnmarshaler(t reflect.Type, tags string) unmarshaler {
	tagArray := strings.Split(tags, ",")
	encoding := tagArray[0]
	name := "unknown"
	ctype := false
	isTime := false
	isDuration := false
	isWktPointer := false
	proto3 := false
	validateUTF8 := true
	for _, tag := range tagArray[3:] {
		if strings.HasPrefix(tag, "name=") {
			name = tag[5:]
		}
		if tag == "proto3" {
			proto3 = true
		}
		if strings.HasPrefix(tag, "customtype=") {
			ctype = true
		}
		if tag == "stdtime" {
			isTime = true
		}
		if tag == "stdduration" {
			isDuration = true
		}
		if tag == "wktptr" {
			isWktPointer = true
		}
	}
	validateUTF8 = validateUTF8 && proto3

	// Figure out packaging (pointer, slice, or both)
	slice := false
	pointer := false
	if t.Kind() == reflect.Slice && t.Elem().Kind() != reflect.Uint8 {
		slice = true
		t = t.Elem()
	}
	if t.Kind() == reflect.Ptr {
		pointer = true
		t = t.Elem()
	}

	if ctype {
		if reflect.PtrTo(t).Implements(customType) {
			if slice {
				return makeUnmarshalCustomSlice(getUnmarshalInfo(t), name)
			}
			if pointer {
				return makeUnmarshalCustomPtr(getUnmarshalInfo(t), name)
			}
			return makeUnmarshalCustom(getUnmarshalInfo(t), name)
		} else {
			panic(fmt.Sprintf("custom type: type: %v, does not implement the proto.custom interface", t))
		}
	}

	if isTime {
		if pointer {
			if slice {
				return makeUnmarshalTimePtrSlice(getUnmarshalInfo(t), name)
			}
			return makeUnmarshalTimePtr(getUnmarshalInfo(t), name)
		}
		if slice {
			return makeUnmarshalTimeSlice(getUnmarshalInfo(t), name)
		}
		return makeUnmarshalTime(getUnmarshalInfo(t), name)
	}

	if isDuration {
		if pointer {
			if slice {
				return makeUnmarshalDurationPtrSlice(getUnmarshalInfo(t), name)
			}
			return makeUnmarshalDurationPtr(getUnmarshalInfo(t), name)
		}
		if slice {
			return makeUnmarshalDurationSlice(getUnmarshalInfo(t), name)
		}
		return makeUnmarshalDuration(getUnmarshalInfo(t), name)
	}

	if isWktPointer {
		switch t.Kind() {
		case reflect.Float64:
			if pointer {
				if slice {
					return makeStdDoubleValuePtrSliceUnmarshaler(getUnmarshalInfo(t), name)
				}
				return makeStdDoubleValuePtrUnmarshaler(getUnmarshalInfo(t), name)
			}
			if slice {
				return makeStdDoubleValueSliceUnmarshaler(getUnmarshalInfo(t), name)
			}
			return makeStdDoubleValueUnmarshaler(getUnmarshalInfo(t), name)
		case reflect.Float32:
			if pointer {
				if slice {
					return makeStdFloatValuePtrSliceUnmarshaler(getUnmarshalInfo(t), name)
				}
				return makeStdFloatValuePtrUnmarshaler(getUnmarshalInfo(t), name)
			}
			if slice {
				return makeStdFloatValueSliceUnmarshaler(getUnmarshalInfo(t), name)
			}
			return makeStdFloatValueUnmarshaler(getUnmarshalInfo(t), name)
		case reflect.Int64:
			if pointer {
				if slice {
					return makeStdInt64ValuePtrSliceUnmarshaler(getUnmarshalInfo(t), name)
				}
				return makeStdInt64ValuePtrUnmarshaler(getUnmarshalInfo(t), name)
			}
			if slice {
				return makeStdInt64ValueSliceUnmarshaler(getUnmarshalInfo(t), name)
			}
			return makeStdInt64ValueUnmarshaler(getUnmarshalInfo(t), name)
		case reflect.Uint64:
			if pointer {
				if slice {
					return makeStdUInt64ValuePtrSliceUnmarshaler(getUnmarshalInfo(t), name)
				}
				return makeStdUInt64ValuePtrUnmarshaler(getUnmarshalInfo(t), name)
			}
			if slice {
				return makeStdUInt64ValueSliceUnmarshaler(getUnmarshalInfo(t), name)
			}
			return makeStdUInt64ValueUnmarshaler(getUnmarshalInfo(t), name)
		case reflect.Int32:
			if pointer {
				if slice {
					return makeStdInt32ValuePtrSliceUnmarshaler(getUnmarshalInfo(t), name)
				}
				return makeStdInt32ValuePtrUnmarshaler(getUnmarshalInfo(t), name)
			}
			if slice {
				return makeStdInt32ValueSliceUnmarshaler(getUnmarshalInfo(t), name)
			}
			return makeStdInt32ValueUnmarshaler(getUnmarshalInfo(t), name)
		case reflect.Uint32:
			if pointer {
				if slice {
					return makeStdUInt32ValuePtrSliceUnmarshaler(getUnmarshalInfo(t), name)
				}
				return makeStdUInt32ValuePtrUnmarshaler(getUnmarshalInfo(t), name)
			}
			if slice {
				return makeStdUInt32ValueSliceUnmarshaler(getUnmarshalInfo(t), name)
			}
			return makeStdUInt32ValueUnmarshaler(getUnmarshalInfo(t), name)
		case reflect.Bool:
			if pointer {
				if slice {
					return makeStdBoolValuePtrSliceUnmarshaler(getUnmarshalInfo(t), name)
				}
				return makeStdBoolValuePtrUnmarshaler(getUnmarshalInfo(t), name)
			}
			if slice {
				return makeStdBoolValueSliceUnmarshaler(getUnmarshalInfo(t), name)
			}
			return makeStdBoolValueUnmarshaler(getUnmarshalInfo(t), name)
		case reflect.String:
			if pointer {
				if slice {
					return makeStdStringValuePtrSliceUnmarshaler(getUnmarshalInfo(t), name)
				}
				return makeStdStringValuePtrUnmarshaler(getUnmarshalInfo(t), name)
			}
			if slice {
				return makeStdStringValueSliceUnmarshaler(getUnmarshalInfo(t), name)
			}
			return makeStdStringValueUnmarshaler(getUnmarshalInfo(t), name)
		case uint8SliceType:
			if pointer {
				if slice {
					return makeStdBytesValuePtrSliceUnmarshaler(getUnmarshalInfo(t), name)
				}
				return makeStdBytesValuePtrUnmarshaler(getUnmarshalInfo(t), name)
			}
			if slice {
				return makeStdBytesValueSliceUnmarshaler(getUnmarshalInfo(t), name)
			}
			return makeStdBytesValueUnmarshaler(getUnmarshalInfo(t), name)
		default:
			panic(fmt.Sprintf("unknown wktpointer type %#v", t))
		}
	}

	// We'll never have both pointer and slice for basic types.
	if pointer && slice && t.Kind() != reflect.Struct {
		panic("both pointer and slice for basic type in " + t.Name())
	}

	switch t.Kind() {
	case reflect.Bool:
		if pointer {
			return unmarshalBoolPtr
		}
		if slice {
			return unmarshalBoolSlice
		}
		return unmarshalBoolValue
	case reflect.Int32:
		switch encoding {
		case "fixed32":
			if pointer {
				return unmarshalFixedS32Ptr
			}
			if slice {
				return unmarshalFixedS32Slice
			}
			return unmarshalFixedS32Value
		case "varint":
			// this could be int32 or enum
			if pointer {
				return unmarshalInt32Ptr
			}
			if slice {
				return unmarshalInt32Slice
			}
			return unmarshalInt32Value
		case "zigzag32":
			if pointer {
				return unmarshalSint32Ptr
			}
			if slice {
				return unmarshalSint32Slice
			}
			return unmarshalSint32Value
		}
	case reflect.Int64:
		switch encoding {
		case "fixed64":
			if pointer {
				return unmarshalFixedS64Ptr
			}
			if slice {
				return unmarshalFixedS64Slice
			}
			return unmarshalFixedS64Value
		case "varint":
			if pointer {
				return unmarshalInt64Ptr
			}
			if slice {
				return unmarshalInt64Slice
			}
			return unmarshalInt64Value
		case "zigzag64":
			if pointer {
				return unmarshalSint64Ptr
			}
			if slice {
				return unmarshalSint64Slice
			}
			return unmarshalSint64Value
		}
	case reflect.Uint32:
		switch encoding {
		case "fixed32":
			if pointer {
				return unmarshalFixed32Ptr
			}
			if slice {
				return unmarshalFixed32Slice
			}
			return unmarshalFixed32Value
		case "varint":
			if pointer {
				return unmarshalUint32Ptr
			}
			if slice {
				return unmarshalUint32Slice
			}
			return unmarshalUint32Value
		}
	case reflect.Uint64:
		switch encoding {
		case "fixed64":
			if pointer {
				return unmarshalFixed64Ptr
			}
			if slice {
				return unmarshalFixed64Slice
			}
			return unmarshalFixed64Value
		case "varint":
			if pointer {
				return unmarshalUint64Ptr
			}
			if slice {
				return unmarshalUint64Slice
			}
			return unmarshalUint64Value
		}
	case reflect.Float32:
		if pointer {
			return unmarshalFloat32Ptr
		}
		if slice {
			return unmarshalFloat32Slice
		}
		return unmarshalFloat32Value
	case reflect.Float64:
		if pointer {
			return unmarshalFloat64Ptr
		}
		if slice {
			return unmarshalFloat64Slice
		}
		return unmarshalFloat64Value
	case reflect.Map:
		panic("map type in typeUnmarshaler in " + t.Name())
	case reflect.Slice:
		if pointer {
			panic("bad pointer in slice case in " + t.Name())
		}
		if slice {
			return unmarshalBytesSlice
		}
		return unmarshalBytesValue
	case reflect.String:
		if validateUTF8 {
			if pointer {
				return unmarshalUTF8StringPtr
			}
			if slice {
				return unmarshalUTF8StringSlice
			}
			return unmarshalUTF8StringValue
		}
		if pointer {
			return unmarshalStringPtr
		}
		if slice {
			return unmarshalStringSlice
		}
		return unmarshalStringValue
	case reflect.Struct:
		// message or group field
		if !pointer {
			switch encoding {
			case "bytes":
				if slice {
					return makeUnmarshalMessageSlice(getUnmarshalInfo(t), name)
				}
				return makeUnmarshalMessage(getUnmarshalInfo(t), name)
			}
		}
		switch encoding {
		case "bytes":
			if slice {
				return makeUnmarshalMessageSlicePtr(getUnmarshalInfo(t), name)
			}
			return makeUnmarshalMessagePtr(getUnmarshalInfo(t), name)
		case "group":
			if slice {
				return makeUnmarshalGroupSlicePtr(getUnmarshalInfo(t), name)
			}
			return makeUnmarshalGroupPtr(getUnmarshalInfo(t), name)
		}
	}
	panic(fmt.Sprintf("unmarshaler not found type:%s encoding:%s", t, encoding))
}

// Below are all the unmarshalers for individual fields of various types.

func unmarshalInt64Value(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := int64(x)
	*f.toInt64() = v
	return b, nil
}

func unmarshalInt64Ptr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := int64(x)
	*f.toInt64Ptr() = &v
	return b, nil
}

func unmarshalInt64Slice(b []byte, f pointer, w int) ([]byte, error) {
	if w == WireBytes { // packed
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		res := b[x:]
		b = b[:x]
		for len(b) > 0 {
			x, n = decodeVarint(b)
			if n == 0 {
				return nil, io.ErrUnexpectedEOF
			}
			b = b[n:]
			v := int64(x)
			s := f.toInt64Slice()
			*s = append(*s, v)
		}
		return res, nil
	}
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := int64(x)
	s := f.toInt64Slice()
	*s = append(*s, v)
	return b, nil
}

func unmarshalSint64Value(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := int64(x>>1) ^ int64(x)<<63>>63
	*f.toInt64() = v
	return b, nil
}

func unmarshalSint64Ptr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := int64(x>>1) ^ int64(x)<<63>>63
	*f.toInt64Ptr() = &v
	return b, nil
}

func unmarshalSint64Slice(b []byte, f pointer, w int) ([]byte, error) {
	if w == WireBytes { // packed
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		res := b[x:]
		b = b[:x]
		for len(b) > 0 {
			x, n = decodeVarint(b)
			if n == 0 {
				return nil, io.ErrUnexpectedEOF
			}
			b = b[n:]
			v := int64(x>>1) ^ int64(x)<<63>>63
			s := f.toInt64Slice()
			*s = append(*s, v)
		}
		return res, nil
	}
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := int64(x>>1) ^ int64(x)<<63>>63
	s := f.toInt64Slice()
	*s = append(*s, v)
	return b, nil
}

func unmarshalUint64Value(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := uint64(x)
	*f.toUint64() = v
	return b, nil
}

func unmarshalUint64Ptr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := uint64(x)
	*f.toUint64Ptr() = &v
	return b, nil
}

func unmarshalUint64Slice(b []byte, f pointer, w int) ([]byte, error) {
	if w == WireBytes { // packed
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		res := b[x:]
		b = b[:x]
		for len(b) > 0 {
			x, n = decodeVarint(b)
			if n == 0 {
				return nil, io.ErrUnexpectedEOF
			}
			b = b[n:]
			v := uint64(x)
			s := f.toUint64Slice()
			*s = append(*s, v)
		}
		return res, nil
	}
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := uint64(x)
	s := f.toUint64Slice()
	*s = append(*s, v)
	return b, nil
}

func unmarshalInt32Value(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := int32(x)
	*f.toInt32() = v
	return b, nil
}

func unmarshalInt32Ptr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := int32(x)
	f.setInt32Ptr(v)
	return b, nil
}

func unmarshalInt32Slice(b []byte, f pointer, w int) ([]byte, error) {
	if w == WireBytes { // packed
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		res := b[x:]
		b = b[:x]
		for len(b) > 0 {
			x, n = decodeVarint(b)
			if n == 0 {
				return nil, io.ErrUnexpectedEOF
			}
			b = b[n:]
			v := int32(x)
			f.appendInt32Slice(v)
		}
		return res, nil
	}
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := int32(x)
	f.appendInt32Slice(v)
	return b, nil
}

func unmarshalSint32Value(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := int32(x>>1) ^ int32(x)<<31>>31
	*f.toInt32() = v
	return b, nil
}

func unmarshalSint32Ptr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := int32(x>>1) ^ int32(x)<<31>>31
	f.setInt32Ptr(v)
	return b, nil
}

func unmarshalSint32Slice(b []byte, f pointer, w int) ([]byte, error) {
	if w == WireBytes { // packed
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		res := b[x:]
		b = b[:x]
		for len(b) > 0 {
			x, n = decodeVarint(b)
			if n == 0 {
				return nil, io.ErrUnexpectedEOF
			}
			b = b[n:]
			v := int32(x>>1) ^ int32(x)<<31>>31
			f.appendInt32Slice(v)
		}
		return res, nil
	}
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := int32(x>>1) ^ int32(x)<<31>>31
	f.appendInt32Slice(v)
	return b, nil
}

func unmarshalUint32Value(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := uint32(x)
	*f.toUint32() = v
	return b, nil
}

func unmarshalUint32Ptr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := uint32(x)
	*f.toUint32Ptr() = &v
	return b, nil
}

func unmarshalUint32Slice(b []byte, f pointer, w int) ([]byte, error) {
	if w == WireBytes { // packed
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		res := b[x:]
		b = b[:x]
		for len(b) > 0 {
			x, n = decodeVarint(b)
			if n == 0 {
				return nil, io.ErrUnexpectedEOF
			}
			b = b[n:]
			v := uint32(x)
			s := f.toUint32Slice()
			*s = append(*s, v)
		}
		return res, nil
	}
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	v := uint32(x)
	s := f.toUint32Slice()
	*s = append(*s, v)
	return b, nil
}

func unmarshalFixed64Value(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireFixed64 {
		return b, errInternalBadWireType
	}
	if len(b) < 8 {
		return nil, io.ErrUnexpectedEOF
	}
	v := uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
	*f.toUint64() = v
	return b[8:], nil
}

func unmarshalFixed64Ptr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireFixed64 {
		return b, errInternalBadWireType
	}
	if len(b) < 8 {
		return nil, io.ErrUnexpectedEOF
	}
	v := uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
	*f.toUint64Ptr() = &v
	return b[8:], nil
}

func unmarshalFixed64Slice(b []byte, f pointer, w int) ([]byte, error) {
	if w == WireBytes { // packed
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		res := b[x:]
		b = b[:x]
		for len(b) > 0 {
			if len(b) < 8 {
				return nil, io.ErrUnexpectedEOF
			}
			v := uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
			s := f.toUint64Slice()
			*s = append(*s, v)
			b = b[8:]
		}
		return res, nil
	}
	if w != WireFixed64 {
		return b, errInternalBadWireType
	}
	if len(b) < 8 {
		return nil, io.ErrUnexpectedEOF
	}
	v := uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
	s := f.toUint64Slice()
	*s = append(*s, v)
	return b[8:], nil
}

func unmarshalFixedS64Value(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireFixed64 {
		return b, errInternalBadWireType
	}
	if len(b) < 8 {
		return nil, io.ErrUnexpectedEOF
	}
	v := int64(b[0]) | int64(b[1])<<8 | int64(b[2])<<16 | int64(b[3])<<24 | int64(b[4])<<32 | int64(b[5])<<40 | int64(b[6])<<48 | int64(b[7])<<56
	*f.toInt64() = v
	return b[8:], nil
}

func unmarshalFixedS64Ptr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireFixed64 {
		return b, errInternalBadWireType
	}
	if len(b) < 8 {
		return nil, io.ErrUnexpectedEOF
	}
	v := int64(b[0]) | int64(b[1])<<8 | int64(b[2])<<16 | int64(b[3])<<24 | int64(b[4])<<32 | int64(b[5])<<40 | int64(b[6])<<48 | int64(b[7])<<56
	*f.toInt64Ptr() = &v
	return b[8:], nil
}

func unmarshalFixedS64Slice(b []byte, f pointer, w int) ([]byte, error) {
	if w == WireBytes { // packed
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		res := b[x:]
		b = b[:x]
		for len(b) > 0 {
			if len(b) < 8 {
				return nil, io.ErrUnexpectedEOF
			}
			v := int64(b[0]) | int64(b[1])<<8 | int64(b[2])<<16 | int64(b[3])<<24 | int64(b[4])<<32 | int64(b[5])<<40 | int64(b[6])<<48 | int64(b[7])<<56
			s := f.toInt64Slice()
			*s = append(*s, v)
			b = b[8:]
		}
		return res, nil
	}
	if w != WireFixed64 {
		return b, errInternalBadWireType
	}
	if len(b) < 8 {
		return nil, io.ErrUnexpectedEOF
	}
	v := int64(b[0]) | int64(b[1])<<8 | int64(b[2])<<16 | int64(b[3])<<24 | int64(b[4])<<32 | int64(b[5])<<40 | int64(b[6])<<48 | int64(b[7])<<56
	s := f.toInt64Slice()
	*s = append(*s, v)
	return b[8:], nil
}

func unmarshalFixed32Value(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireFixed32 {
		return b, errInternalBadWireType
	}
	if len(b) < 4 {
		return nil, io.ErrUnexpectedEOF
	}
	v := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	*f.toUint32() = v
	return b[4:], nil
}

func unmarshalFixed32Ptr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireFixed32 {
		return b, errInternalBadWireType
	}
	if len(b) < 4 {
		return nil, io.ErrUnexpectedEOF
	}
	v := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	*f.toUint32Ptr() = &v
	return b[4:], nil
}

func unmarshalFixed32Slice(b []byte, f pointer, w int) ([]byte, error) {
	if w == WireBytes { // packed
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		res := b[x:]
		b = b[:x]
		for len(b) > 0 {
			if len(b) < 4 {
				return nil, io.ErrUnexpectedEOF
			}
			v := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
			s := f.toUint32Slice()
			*s = append(*s, v)
			b = b[4:]
		}
		return res, nil
	}
	if w != WireFixed32 {
		return b, errInternalBadWireType
	}
	if len(b) < 4 {
		return nil, io.ErrUnexpectedEOF
	}
	v := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	s := f.toUint32Slice()
	*s = append(*s, v)
	return b[4:], nil
}

func unmarshalFixedS32Value(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireFixed32 {
		return b, errInternalBadWireType
	}
	if len(b) < 4 {
		return nil, io.ErrUnexpectedEOF
	}
	v := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16 | int32(b[3])<<24
	*f.toInt32() = v
	return b[4:], nil
}

func unmarshalFixedS32Ptr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireFixed32 {
		return b, errInternalBadWireType
	}
	if len(b) < 4 {
		return nil, io.ErrUnexpectedEOF
	}
	v := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16 | int32(b[3])<<24
	f.setInt32Ptr(v)
	return b[4:], nil
}

func unmarshalFixedS32Slice(b []byte, f pointer, w int) ([]byte, error) {
	if w == WireBytes { // packed
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		res := b[x:]
		b = b[:x]
		for len(b) > 0 {
			if len(b) < 4 {
				return nil, io.ErrUnexpectedEOF
			}
			v := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16 | int32(b[3])<<24
			f.appendInt32Slice(v)
			b = b[4:]
		}
		return res, nil
	}
	if w != WireFixed32 {
		return b, errInternalBadWireType
	}
	if len(b) < 4 {
		return nil, io.ErrUnexpectedEOF
	}
	v := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16 | int32(b[3])<<24
	f.appendInt32Slice(v)
	return b[4:], nil
}

func unmarshalBoolValue(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	// Note: any length varint is allowed, even though any sane
	// encoder will use one byte.
	// See https://github.com/golang/protobuf/issues/76
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	// TODO: check if x>1? Tests seem to indicate no.
	v := x != 0
	*f.toBool() = v
	return b[n:], nil
}

func unmarshalBoolPtr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	v := x != 0
	*f.toBoolPtr() = &v
	return b[n:], nil
}

func unmarshalBoolSlice(b []byte, f pointer, w int) ([]byte, error) {
	if w == WireBytes { // packed
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		res := b[x:]
		b = b[:x]
		for len(b) > 0 {
			x, n = decodeVarint(b)
			if n == 0 {
				return nil, io.ErrUnexpectedEOF
			}
			v := x != 0
			s := f.toBoolSlice()
			*s = append(*s, v)
			b = b[n:]
		}
		return res, nil
	}
	if w != WireVarint {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	v := x != 0
	s := f.toBoolSlice()
	*s = append(*s, v)
	return b[n:], nil
}

func unmarshalFloat64Value(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireFixed64 {
		return b, errInternalBadWireType
	}
	if len(b) < 8 {
		return nil, io.ErrUnexpectedEOF
	}
	v := math.Float64frombits(uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56)
	*f.toFloat64() = v
	return b[8:], nil
}

func unmarshalFloat64Ptr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireFixed64 {
		return b, errInternalBadWireType
	}
	if len(b) < 8 {
		return nil, io.ErrUnexpectedEOF
	}
	v := math.Float64frombits(uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56)
	*f.toFloat64Ptr() = &v
	return b[8:], nil
}

func unmarshalFloat64Slice(b []byte, f pointer, w int) ([]byte, error) {
	if w == WireBytes { // packed
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		res := b[x:]
		b = b[:x]
		for len(b) > 0 {
			if len(b) < 8 {
				return nil, io.ErrUnexpectedEOF
			}
			v := math.Float64frombits(uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56)
			s := f.toFloat64Slice()
			*s = append(*s, v)
			b = b[8:]
		}
		return res, nil
	}
	if w != WireFixed64 {
		return b, errInternalBadWireType
	}
	if len(b) < 8 {
		return nil, io.ErrUnexpectedEOF
	}
	v := math.Float64frombits(uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56)
	s := f.toFloat64Slice()
	*s = append(*s, v)
	return b[8:], nil
}

func unmarshalFloat32Value(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireFixed32 {
		return b, errInternalBadWireType
	}
	if len(b) < 4 {
		return nil, io.ErrUnexpectedEOF
	}
	v := math.Float32frombits(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
	*f.toFloat32() = v
	return b[4:], nil
}

func unmarshalFloat32Ptr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireFixed32 {
		return b, errInternalBadWireType
	}
	if len(b) < 4 {
		return nil, io.ErrUnexpectedEOF
	}
	v := math.Float32frombits(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
	*f.toFloat32Ptr() = &v
	return b[4:], nil
}

func unmarshalFloat32Slice(b []byte, f pointer, w int) ([]byte, error) {
	if w == WireBytes { // packed
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		res := b[x:]
		b = b[:x]
		for len(b) > 0 {
			if len(b) < 4 {
				return nil, io.ErrUnexpectedEOF
			}
			v := math.Float32frombits(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
			s := f.toFloat32Slice()
			*s = append(*s, v)
			b = b[4:]
		}
		return res, nil
	}
	if w != WireFixed32 {
		return b, errInternalBadWireType
	}
	if len(b) < 4 {
		return nil, io.ErrUnexpectedEOF
	}
	v := math.Float32frombits(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
	s := f.toFloat32Slice()
	*s = append(*s, v)
	return b[4:], nil
}

func unmarshalStringValue(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireBytes {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	if x > uint64(len(b)) {
		return nil, io.ErrUnexpectedEOF
	}
	v := string(b[:x])
	*f.toString() = v
	return b[x:], nil
}

func unmarshalStringPtr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireBytes {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	if x > uint64(len(b)) {
		return nil, io.ErrUnexpectedEOF
	}
	v := string(b[:x])
	*f.toStringPtr() = &v
	return b[x:], nil
}

func unmarshalStringSlice(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireBytes {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	if x > uint64(len(b)) {
		return nil, io.ErrUnexpectedEOF
	}
	v := string(b[:x])
	s := f.toStringSlice()
	*s = append(*s, v)
	return b[x:], nil
}

func unmarshalUTF8StringValue(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireBytes {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	if x > uint64(len(b)) {
		return nil, io.ErrUnexpectedEOF
	}
	v := string(b[:x])
	*f.toString() = v
	if !utf8.ValidString(v) {
		return b[x:], errInvalidUTF8
	}
	return b[x:], nil
}

func unmarshalUTF8StringPtr(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireBytes {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	if x > uint64(len(b)) {
		return nil, io.ErrUnexpectedEOF
	}
	v := string(b[:x])
	*f.toStringPtr() = &v
	if !utf8.ValidString(v) {
		return b[x:], errInvalidUTF8
	}
	return b[x:], nil
}

func unmarshalUTF8StringSlice(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireBytes {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	if x > uint64(len(b)) {
		return nil, io.ErrUnexpectedEOF
	}
	v := string(b[:x])
	s := f.toStringSlice()
	*s = append(*s, v)
	if !utf8.ValidString(v) {
		return b[x:], errInvalidUTF8
	}
	return b[x:], nil
}

var emptyBuf [0]byte

func unmarshalBytesValue(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireBytes {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	if x > uint64(len(b)) {
		return nil, io.ErrUnexpectedEOF
	}
	// The use of append here is a trick which avoids the zeroing
	// that would be required if we used a make/copy pair.
	// We append to emptyBuf instead of nil because we want
	// a non-nil result even when the length is 0.
	v := append(emptyBuf[:], b[:x]...)
	*f.toBytes() = v
	return b[x:], nil
}

func unmarshalBytesSlice(b []byte, f pointer, w int) ([]byte, error) {
	if w != WireBytes {
		return b, errInternalBadWireType
	}
	x, n := decodeVarint(b)
	if n == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	b = b[n:]
	if x > uint64(len(b)) {
		return nil, io.ErrUnexpectedEOF
	}
	v := append(emptyBuf[:], b[:x]...)
	s := f.toBytesSlice()
	*s = append(*s, v)
	return b[x:], nil
}

func makeUnmarshalMessagePtr(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return b, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		// First read the message field to see if something is there.
		// The semantics of multiple submessages are weird.  Instead of
		// the last one winning (as it is for all other fields), multiple
		// submessages are merged.
		v := f.getPointer()
		if v.isNil() {
			v = valToPointer(reflect.New(sub.typ))
			f.setPointer(v)
		}
		err := sub.unmarshal(v, b[:x])
		if err != nil {
			if r, ok := err.(*RequiredNotSetError); ok {
				r.field = name + "." + r.field
			} else {
				return nil, err
			}
		}
		return b[x:], err
	}
}

func makeUnmarshalMessageSlicePtr(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return b, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		v := valToPointer(reflect.New(sub.typ))
		err := sub.unmarshal(v, b[:x])
		if err != nil {
			if r, ok := err.(*RequiredNotSetError); ok {
				r.field = name + "." + r.field
			} else {
				return nil, err
			}
		}
		f.appendPointer(v)
		return b[x:], err
	}
}

func makeUnmarshalGroupPtr(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireStartGroup {
			return b, errInternalBadWireType
		}
		x, y := findEndGroup(b)
		if x < 0 {
			return nil, io.ErrUnexpectedEOF
		}
		v := f.getPointer()
		if v.isNil() {
			v = valToPointer(reflect.New(sub.typ))
			f.setPointer(v)
		}
		err := sub.unmarshal(v, b[:x])
		if err != nil {
			if r, ok := err.(*RequiredNotSetError); ok {
				r.field = name + "." + r.field
			} else {
				return nil, err
			}
		}
		return b[y:], err
	}
}

func makeUnmarshalGroupSlicePtr(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireStartGroup {
			return b, errInternalBadWireType
		}
		x, y := findEndGroup(b)
		if x < 0 {
			return nil, io.ErrUnexpectedEOF
		}
		v := valToPointer(reflect.New(sub.typ))
		err := sub.unmarshal(v, b[:x])
		if err != nil {
			if r, ok := err.(*RequiredNotSetError); ok {
				r.field = name + "." + r.field
			} else {
				return nil, err
			}
		}
		f.appendPointer(v)
		return b[y:], err
	}
}

func makeUnmarshalMap(f *reflect.StructField) unmarshaler {
	t := f.Type
	kt := t.Key()
	vt := t.Elem()
	tagArray := strings.Split(f.Tag.Get("protobuf"), ",")
	valTags := strings.Split(f.Tag.Get("protobuf_val"), ",")
	for _, t := range tagArray {
		if strings.HasPrefix(t, "customtype=") {
			valTags = append(valTags, t)
		}
		if t == "stdtime" {
			valTags = append(valTags, t)
		}
		if t == "stdduration" {
			valTags = append(valTags, t)
		}
		if t == "wktptr" {
			valTags = append(valTags, t)
		}
	}
	unmarshalKey := typeUnmarshaler(kt, f.Tag.Get("protobuf_key"))
	unmarshalVal := typeUnmarshaler(vt, strings.Join(valTags, ","))
	return func(b []byte, f pointer, w int) ([]byte, error) {
		// The map entry is a submessage. Figure out how big it is.
		if w != WireBytes {
			return nil, fmt.Errorf("proto: bad wiretype for map field: got %d want %d", w, WireBytes)
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		r := b[x:] // unused data to return
		b = b[:x]  // data for map entry

		// Note: we could use #keys * #values ~= 200 functions
		// to do map decoding without reflection. Probably not worth it.
		// Maps will be somewhat slow. Oh well.

		// Read key and value from data.
		var nerr nonFatal
		k := reflect.New(kt)
		v := reflect.New(vt)
		for len(b) > 0 {
			x, n := decodeVarint(b)
			if n == 0 {
				return nil, io.ErrUnexpectedEOF
			}
			wire := int(x) & 7
			b = b[n:]

			var err error
			switch x >> 3 {
			case 1:
				b, err = unmarshalKey(b, valToPointer(k), wire)
			case 2:
				b, err = unmarshalVal(b, valToPointer(v), wire)
			default:
				err = errInternalBadWireType // skip unknown tag
			}

			if nerr.Merge(err) {
				continue
			}
			if err != errInternalBadWireType {
				return nil, err
			}

			// Skip past unknown fields.
			b, err = skipField(b, wire)
			if err != nil {
				return nil, err
			}
		}

		// Get map, allocate if needed.
		m := f.asPointerTo(t).Elem() // an addressable map[K]T
		if m.IsNil() {
			m.Set(reflect.MakeMap(t))
		}

		// Insert into map.
		m.SetMapIndex(k.Elem(), v.Elem())

		return r, nerr.E
	}
}

// makeUnmarshalOneof makes an unmarshaler for oneof fields.
// for:
// message Msg {
//   oneof F {
//     int64 X = 1;
//     float64 Y = 2;
//   }
// }
// typ is the type of the concrete entry for a oneof case (e.g. Msg_X).
// ityp is the interface type of the oneof field (e.g. isMsg_F).
// unmarshal is the unmarshaler for the base type of the oneof case (e.g. int64).
// Note that this function will be called once for each case in the oneof.
func makeUnmarshalOneof(typ, ityp reflect.Type, unmarshal unmarshaler) unmarshaler {
	sf := typ.Field(0)
	field0 := toField(&sf)
	return func(b []byte, f pointer, w int) ([]byte, error) {
		// Allocate holder for value.
		v := reflect.New(typ)

		// Unmarshal data into holder.
		// We unmarshal into the first field of the holder object.
		var err error
		var nerr nonFatal
		b, err = unmarshal(b, valToPointer(v).offset(field0), w)
		if !nerr.Merge(err) {
			return nil, err
		}

		// Write pointer to holder into target field.
		f.asPointerTo(ityp).Elem().Set(v)

		return b, nerr.E
	}
}

// Error used by decode internally.
var errInternalBadWireType = errors.New("proto: internal error: bad wiretype")

// skipField skips past a field of type wire and returns the remaining bytes.
func skipField(b []byte, wire int) ([]byte, error) {
	switch wire {
	case WireVarint:
		_, k := decodeVarint(b)
		if k == 0 {
			return b, io.ErrUnexpectedEOF
		}
		b = b[k:]
	case WireFixed32:
		if len(b) < 4 {
			return b, io.ErrUnexpectedEOF
		}
		b = b[4:]
	case WireFixed64:
		if len(b) < 8 {
			return b, io.ErrUnexpectedEOF
		}
		b = b[8:]
	case WireBytes:
		m, k := decodeVarint(b)
		if k == 0 || uint64(len(b)-k) < m {
			return b, io.ErrUnexpectedEOF
		}
		b = b[uint64(k)+m:]
	case WireStartGroup:
		_, i := findEndGroup(b)
		if i == -1 {
			return b, io.ErrUnexpectedEOF
		}
		b = b[i:]
	default:
		return b, fmt.Errorf("proto: can't skip unknown wire type %d", wire)
	}
	return b, nil
}

// findEndGroup finds the index of the next EndGroup tag.
// Groups may be nested, so the "next" EndGroup tag is the first
// unpaired EndGroup.
// findEndGroup returns the indexes of the start and end of the EndGroup tag.
// Returns (-1,-1) if it can't find one.
func findEndGroup(b []byte) (int, int) {
	depth := 1
	i := 0
	for {
		x, n := decodeVarint(b[i:])
		if n == 0 {
			return -1, -1
		}
		j := i
		i += n
		switch x & 7 {
		case WireVarint:
			_, k := decodeVarint(b[i:])
			if k == 0 {
				return -1, -1
			}
			i += k
		case WireFixed32:
			if len(b)-4 < i {
				return -1, -1
			}
			i += 4
		case WireFixed64:
			if len(b)-8 < i {
				return -1, -1
			}
			i += 8
		case WireBytes:
			m, k := decodeVarint(b[i:])
			if k == 0 {
				return -1, -1
			}
			i += k
			if uint64(len(b)-i) < m {
				return -1, -1
			}
			i += int(m)
		case WireStartGroup:
			depth++
		case WireEndGroup:
			depth--
			if depth == 0 {
				return j, i
			}
		default:
			return -1, -1
		}
	}
}

// encodeVarint appends a varint-encoded integer to b and returns the result.
func encodeVarint(b []byte, x uint64) []byte {
	for x >= 1<<7 {
		b = append(b, byte(x&0x7f|0x80))
		x >>= 7
	}
	return append(b, byte(x))
}

// decodeVarint reads a varint-encoded integer from b.
// Returns the decoded integer and the number of bytes read.
// If there is an error, it returns 0,0.
func decodeVarint(b []byte) (uint64, int) {
	var x, y uint64
	if len(b) == 0 {
		goto bad
	}
	x = uint64(b[0])
	if x < 0x80 {
		return x, 1
	}
	x -= 0x80

	if len(b) <= 1 {
		goto bad
	}
	y = uint64(b[1])
	x += y << 7
	if y < 0x80 {
		return x, 2
	}
	x -= 0x80 << 7

	if len(b) <= 2 {
		goto bad
	}
	y = uint64(b[2])
	x += y << 14
	if y < 0x80 {
		return x, 3
	}
	x -= 0x80 << 14

	if len(b) <= 3 {
		goto bad
	}
	y = uint64(b[3])
	x += y << 21
	if y < 0x80 {
		return x, 4
	}
	x -= 0x80 << 21

	if len(b) <= 4 {
		goto bad
	}
	y = uint64(b[4])
	x += y << 28
	if y < 0x80 {
		return x, 5
	}
	x -= 0x80 << 28

	if len(b) <= 5 {
		goto bad
	}
	y = uint64(b[5])
	x += y << 35
	if y < 0x80 {
		return x, 6
	}
	x -= 0x80 << 35

	if len(b) <= 6 {
		goto bad
	}
	y = uint64(b[6])
	x += y << 42
	if y < 0x80 {
		return x, 7
	}
	x -= 0x80 << 42

	if len(b) <= 7 {
		goto bad
	}
	y = uint64(b[7])
	x += y << 49
	if y < 0x80 {
		return x, 8
	}
	x -= 0x80 << 49

	if len(b) <= 8 {
		goto bad
	}
	y = uint64(b[8])
	x += y << 56
	if y < 0x80 {
		return x, 9
	}
	x -= 0x80 << 56

	if len(b) <= 9 {
		goto bad
	}
	y = uint64(b[9])
	x += y << 63
	if y < 2 {
		return x, 10
	}

bad:
	return 0, 0
}
