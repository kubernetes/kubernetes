// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2010 The Go Authors.  All rights reserved.
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

/*
Package proto converts data structures to and from the wire format of
protocol buffers.  It works in concert with the Go source code generated
for .proto files by the protocol compiler.

A summary of the properties of the protocol buffer interface
for a protocol buffer variable v:

  - Names are turned from camel_case to CamelCase for export.
  - There are no methods on v to set fields; just treat
	them as structure fields.
  - There are getters that return a field's value if set,
	and return the field's default value if unset.
	The getters work even if the receiver is a nil message.
  - The zero value for a struct is its correct initialization state.
	All desired fields must be set before marshaling.
  - A Reset() method will restore a protobuf struct to its zero state.
  - Non-repeated fields are pointers to the values; nil means unset.
	That is, optional or required field int32 f becomes F *int32.
  - Repeated fields are slices.
  - Helper functions are available to aid the setting of fields.
	msg.Foo = proto.String("hello") // set field
  - Constants are defined to hold the default values of all fields that
	have them.  They have the form Default_StructName_FieldName.
	Because the getter methods handle defaulted values,
	direct use of these constants should be rare.
  - Enums are given type names and maps from names to values.
	Enum values are prefixed by the enclosing message's name, or by the
	enum's type name if it is a top-level enum. Enum types have a String
	method, and a Enum method to assist in message construction.
  - Nested messages, groups and enums have type names prefixed with the name of
	the surrounding message type.
  - Extensions are given descriptor names that start with E_,
	followed by an underscore-delimited list of the nested messages
	that contain it (if any) followed by the CamelCased name of the
	extension field itself.  HasExtension, ClearExtension, GetExtension
	and SetExtension are functions for manipulating extensions.
  - Oneof field sets are given a single field in their message,
	with distinguished wrapper types for each possible field value.
  - Marshal and Unmarshal are functions to encode and decode the wire format.

When the .proto file specifies `syntax="proto3"`, there are some differences:

  - Non-repeated fields of non-message type are values instead of pointers.
  - Enum types do not get an Enum method.

The simplest way to describe this is to see an example.
Given file test.proto, containing

	package example;

	enum FOO { X = 17; }

	message Test {
	  required string label = 1;
	  optional int32 type = 2 [default=77];
	  repeated int64 reps = 3;
	  optional group OptionalGroup = 4 {
	    required string RequiredField = 5;
	  }
	  oneof union {
	    int32 number = 6;
	    string name = 7;
	  }
	}

The resulting file, test.pb.go, is:

	package example

	import proto "github.com/gogo/protobuf/proto"
	import math "math"

	type FOO int32
	const (
		FOO_X FOO = 17
	)
	var FOO_name = map[int32]string{
		17: "X",
	}
	var FOO_value = map[string]int32{
		"X": 17,
	}

	func (x FOO) Enum() *FOO {
		p := new(FOO)
		*p = x
		return p
	}
	func (x FOO) String() string {
		return proto.EnumName(FOO_name, int32(x))
	}
	func (x *FOO) UnmarshalJSON(data []byte) error {
		value, err := proto.UnmarshalJSONEnum(FOO_value, data)
		if err != nil {
			return err
		}
		*x = FOO(value)
		return nil
	}

	type Test struct {
		Label         *string             `protobuf:"bytes,1,req,name=label" json:"label,omitempty"`
		Type          *int32              `protobuf:"varint,2,opt,name=type,def=77" json:"type,omitempty"`
		Reps          []int64             `protobuf:"varint,3,rep,name=reps" json:"reps,omitempty"`
		Optionalgroup *Test_OptionalGroup `protobuf:"group,4,opt,name=OptionalGroup" json:"optionalgroup,omitempty"`
		// Types that are valid to be assigned to Union:
		//	*Test_Number
		//	*Test_Name
		Union            isTest_Union `protobuf_oneof:"union"`
		XXX_unrecognized []byte       `json:"-"`
	}
	func (m *Test) Reset()         { *m = Test{} }
	func (m *Test) String() string { return proto.CompactTextString(m) }
	func (*Test) ProtoMessage() {}

	type isTest_Union interface {
		isTest_Union()
	}

	type Test_Number struct {
		Number int32 `protobuf:"varint,6,opt,name=number"`
	}
	type Test_Name struct {
		Name string `protobuf:"bytes,7,opt,name=name"`
	}

	func (*Test_Number) isTest_Union() {}
	func (*Test_Name) isTest_Union()   {}

	func (m *Test) GetUnion() isTest_Union {
		if m != nil {
			return m.Union
		}
		return nil
	}
	const Default_Test_Type int32 = 77

	func (m *Test) GetLabel() string {
		if m != nil && m.Label != nil {
			return *m.Label
		}
		return ""
	}

	func (m *Test) GetType() int32 {
		if m != nil && m.Type != nil {
			return *m.Type
		}
		return Default_Test_Type
	}

	func (m *Test) GetOptionalgroup() *Test_OptionalGroup {
		if m != nil {
			return m.Optionalgroup
		}
		return nil
	}

	type Test_OptionalGroup struct {
		RequiredField *string `protobuf:"bytes,5,req" json:"RequiredField,omitempty"`
	}
	func (m *Test_OptionalGroup) Reset()         { *m = Test_OptionalGroup{} }
	func (m *Test_OptionalGroup) String() string { return proto.CompactTextString(m) }

	func (m *Test_OptionalGroup) GetRequiredField() string {
		if m != nil && m.RequiredField != nil {
			return *m.RequiredField
		}
		return ""
	}

	func (m *Test) GetNumber() int32 {
		if x, ok := m.GetUnion().(*Test_Number); ok {
			return x.Number
		}
		return 0
	}

	func (m *Test) GetName() string {
		if x, ok := m.GetUnion().(*Test_Name); ok {
			return x.Name
		}
		return ""
	}

	func init() {
		proto.RegisterEnum("example.FOO", FOO_name, FOO_value)
	}

To create and play with a Test object:

	package main

	import (
		"log"

		"github.com/gogo/protobuf/proto"
		pb "./example.pb"
	)

	func main() {
		test := &pb.Test{
			Label: proto.String("hello"),
			Type:  proto.Int32(17),
			Reps:  []int64{1, 2, 3},
			Optionalgroup: &pb.Test_OptionalGroup{
				RequiredField: proto.String("good bye"),
			},
			Union: &pb.Test_Name{"fred"},
		}
		data, err := proto.Marshal(test)
		if err != nil {
			log.Fatal("marshaling error: ", err)
		}
		newTest := &pb.Test{}
		err = proto.Unmarshal(data, newTest)
		if err != nil {
			log.Fatal("unmarshaling error: ", err)
		}
		// Now test and newTest contain the same data.
		if test.GetLabel() != newTest.GetLabel() {
			log.Fatalf("data mismatch %q != %q", test.GetLabel(), newTest.GetLabel())
		}
		// Use a type switch to determine which oneof was set.
		switch u := test.Union.(type) {
		case *pb.Test_Number: // u.Number contains the number.
		case *pb.Test_Name: // u.Name contains the string.
		}
		// etc.
	}
*/
package proto

import (
	"encoding/json"
	"fmt"
	"log"
	"reflect"
	"sort"
	"strconv"
	"sync"
)

// Message is implemented by generated protocol buffer messages.
type Message interface {
	Reset()
	String() string
	ProtoMessage()
}

// Stats records allocation details about the protocol buffer encoders
// and decoders.  Useful for tuning the library itself.
type Stats struct {
	Emalloc uint64 // mallocs in encode
	Dmalloc uint64 // mallocs in decode
	Encode  uint64 // number of encodes
	Decode  uint64 // number of decodes
	Chit    uint64 // number of cache hits
	Cmiss   uint64 // number of cache misses
	Size    uint64 // number of sizes
}

// Set to true to enable stats collection.
const collectStats = false

var stats Stats

// GetStats returns a copy of the global Stats structure.
func GetStats() Stats { return stats }

// A Buffer is a buffer manager for marshaling and unmarshaling
// protocol buffers.  It may be reused between invocations to
// reduce memory usage.  It is not necessary to use a Buffer;
// the global functions Marshal and Unmarshal create a
// temporary Buffer and are fine for most applications.
type Buffer struct {
	buf   []byte // encode/decode byte stream
	index int    // read point

	// pools of basic types to amortize allocation.
	bools   []bool
	uint32s []uint32
	uint64s []uint64

	// extra pools, only used with pointer_reflect.go
	int32s   []int32
	int64s   []int64
	float32s []float32
	float64s []float64
}

// NewBuffer allocates a new Buffer and initializes its internal data to
// the contents of the argument slice.
func NewBuffer(e []byte) *Buffer {
	return &Buffer{buf: e}
}

// Reset resets the Buffer, ready for marshaling a new protocol buffer.
func (p *Buffer) Reset() {
	p.buf = p.buf[0:0] // for reading/writing
	p.index = 0        // for reading
}

// SetBuf replaces the internal buffer with the slice,
// ready for unmarshaling the contents of the slice.
func (p *Buffer) SetBuf(s []byte) {
	p.buf = s
	p.index = 0
}

// Bytes returns the contents of the Buffer.
func (p *Buffer) Bytes() []byte { return p.buf }

/*
 * Helper routines for simplifying the creation of optional fields of basic type.
 */

// Bool is a helper routine that allocates a new bool value
// to store v and returns a pointer to it.
func Bool(v bool) *bool {
	return &v
}

// Int32 is a helper routine that allocates a new int32 value
// to store v and returns a pointer to it.
func Int32(v int32) *int32 {
	return &v
}

// Int is a helper routine that allocates a new int32 value
// to store v and returns a pointer to it, but unlike Int32
// its argument value is an int.
func Int(v int) *int32 {
	p := new(int32)
	*p = int32(v)
	return p
}

// Int64 is a helper routine that allocates a new int64 value
// to store v and returns a pointer to it.
func Int64(v int64) *int64 {
	return &v
}

// Float32 is a helper routine that allocates a new float32 value
// to store v and returns a pointer to it.
func Float32(v float32) *float32 {
	return &v
}

// Float64 is a helper routine that allocates a new float64 value
// to store v and returns a pointer to it.
func Float64(v float64) *float64 {
	return &v
}

// Uint32 is a helper routine that allocates a new uint32 value
// to store v and returns a pointer to it.
func Uint32(v uint32) *uint32 {
	return &v
}

// Uint64 is a helper routine that allocates a new uint64 value
// to store v and returns a pointer to it.
func Uint64(v uint64) *uint64 {
	return &v
}

// String is a helper routine that allocates a new string value
// to store v and returns a pointer to it.
func String(v string) *string {
	return &v
}

// EnumName is a helper function to simplify printing protocol buffer enums
// by name.  Given an enum map and a value, it returns a useful string.
func EnumName(m map[int32]string, v int32) string {
	s, ok := m[v]
	if ok {
		return s
	}
	return strconv.Itoa(int(v))
}

// UnmarshalJSONEnum is a helper function to simplify recovering enum int values
// from their JSON-encoded representation. Given a map from the enum's symbolic
// names to its int values, and a byte buffer containing the JSON-encoded
// value, it returns an int32 that can be cast to the enum type by the caller.
//
// The function can deal with both JSON representations, numeric and symbolic.
func UnmarshalJSONEnum(m map[string]int32, data []byte, enumName string) (int32, error) {
	if data[0] == '"' {
		// New style: enums are strings.
		var repr string
		if err := json.Unmarshal(data, &repr); err != nil {
			return -1, err
		}
		val, ok := m[repr]
		if !ok {
			return 0, fmt.Errorf("unrecognized enum %s value %q", enumName, repr)
		}
		return val, nil
	}
	// Old style: enums are ints.
	var val int32
	if err := json.Unmarshal(data, &val); err != nil {
		return 0, fmt.Errorf("cannot unmarshal %#q into enum %s", data, enumName)
	}
	return val, nil
}

// DebugPrint dumps the encoded data in b in a debugging format with a header
// including the string s. Used in testing but made available for general debugging.
func (p *Buffer) DebugPrint(s string, b []byte) {
	var u uint64

	obuf := p.buf
	sindex := p.index
	p.buf = b
	p.index = 0
	depth := 0

	fmt.Printf("\n--- %s ---\n", s)

out:
	for {
		for i := 0; i < depth; i++ {
			fmt.Print("  ")
		}

		index := p.index
		if index == len(p.buf) {
			break
		}

		op, err := p.DecodeVarint()
		if err != nil {
			fmt.Printf("%3d: fetching op err %v\n", index, err)
			break out
		}
		tag := op >> 3
		wire := op & 7

		switch wire {
		default:
			fmt.Printf("%3d: t=%3d unknown wire=%d\n",
				index, tag, wire)
			break out

		case WireBytes:
			var r []byte

			r, err = p.DecodeRawBytes(false)
			if err != nil {
				break out
			}
			fmt.Printf("%3d: t=%3d bytes [%d]", index, tag, len(r))
			if len(r) <= 6 {
				for i := 0; i < len(r); i++ {
					fmt.Printf(" %.2x", r[i])
				}
			} else {
				for i := 0; i < 3; i++ {
					fmt.Printf(" %.2x", r[i])
				}
				fmt.Printf(" ..")
				for i := len(r) - 3; i < len(r); i++ {
					fmt.Printf(" %.2x", r[i])
				}
			}
			fmt.Printf("\n")

		case WireFixed32:
			u, err = p.DecodeFixed32()
			if err != nil {
				fmt.Printf("%3d: t=%3d fix32 err %v\n", index, tag, err)
				break out
			}
			fmt.Printf("%3d: t=%3d fix32 %d\n", index, tag, u)

		case WireFixed64:
			u, err = p.DecodeFixed64()
			if err != nil {
				fmt.Printf("%3d: t=%3d fix64 err %v\n", index, tag, err)
				break out
			}
			fmt.Printf("%3d: t=%3d fix64 %d\n", index, tag, u)

		case WireVarint:
			u, err = p.DecodeVarint()
			if err != nil {
				fmt.Printf("%3d: t=%3d varint err %v\n", index, tag, err)
				break out
			}
			fmt.Printf("%3d: t=%3d varint %d\n", index, tag, u)

		case WireStartGroup:
			fmt.Printf("%3d: t=%3d start\n", index, tag)
			depth++

		case WireEndGroup:
			depth--
			fmt.Printf("%3d: t=%3d end\n", index, tag)
		}
	}

	if depth != 0 {
		fmt.Printf("%3d: start-end not balanced %d\n", p.index, depth)
	}
	fmt.Printf("\n")

	p.buf = obuf
	p.index = sindex
}

// SetDefaults sets unset protocol buffer fields to their default values.
// It only modifies fields that are both unset and have defined defaults.
// It recursively sets default values in any non-nil sub-messages.
func SetDefaults(pb Message) {
	setDefaults(reflect.ValueOf(pb), true, false)
}

// v is a pointer to a struct.
func setDefaults(v reflect.Value, recur, zeros bool) {
	v = v.Elem()

	defaultMu.RLock()
	dm, ok := defaults[v.Type()]
	defaultMu.RUnlock()
	if !ok {
		dm = buildDefaultMessage(v.Type())
		defaultMu.Lock()
		defaults[v.Type()] = dm
		defaultMu.Unlock()
	}

	for _, sf := range dm.scalars {
		f := v.Field(sf.index)
		if !f.IsNil() {
			// field already set
			continue
		}
		dv := sf.value
		if dv == nil && !zeros {
			// no explicit default, and don't want to set zeros
			continue
		}
		fptr := f.Addr().Interface() // **T
		// TODO: Consider batching the allocations we do here.
		switch sf.kind {
		case reflect.Bool:
			b := new(bool)
			if dv != nil {
				*b = dv.(bool)
			}
			*(fptr.(**bool)) = b
		case reflect.Float32:
			f := new(float32)
			if dv != nil {
				*f = dv.(float32)
			}
			*(fptr.(**float32)) = f
		case reflect.Float64:
			f := new(float64)
			if dv != nil {
				*f = dv.(float64)
			}
			*(fptr.(**float64)) = f
		case reflect.Int32:
			// might be an enum
			if ft := f.Type(); ft != int32PtrType {
				// enum
				f.Set(reflect.New(ft.Elem()))
				if dv != nil {
					f.Elem().SetInt(int64(dv.(int32)))
				}
			} else {
				// int32 field
				i := new(int32)
				if dv != nil {
					*i = dv.(int32)
				}
				*(fptr.(**int32)) = i
			}
		case reflect.Int64:
			i := new(int64)
			if dv != nil {
				*i = dv.(int64)
			}
			*(fptr.(**int64)) = i
		case reflect.String:
			s := new(string)
			if dv != nil {
				*s = dv.(string)
			}
			*(fptr.(**string)) = s
		case reflect.Uint8:
			// exceptional case: []byte
			var b []byte
			if dv != nil {
				db := dv.([]byte)
				b = make([]byte, len(db))
				copy(b, db)
			} else {
				b = []byte{}
			}
			*(fptr.(*[]byte)) = b
		case reflect.Uint32:
			u := new(uint32)
			if dv != nil {
				*u = dv.(uint32)
			}
			*(fptr.(**uint32)) = u
		case reflect.Uint64:
			u := new(uint64)
			if dv != nil {
				*u = dv.(uint64)
			}
			*(fptr.(**uint64)) = u
		default:
			log.Printf("proto: can't set default for field %v (sf.kind=%v)", f, sf.kind)
		}
	}

	for _, ni := range dm.nested {
		f := v.Field(ni)
		// f is *T or []*T or map[T]*T
		switch f.Kind() {
		case reflect.Ptr:
			if f.IsNil() {
				continue
			}
			setDefaults(f, recur, zeros)

		case reflect.Slice:
			for i := 0; i < f.Len(); i++ {
				e := f.Index(i)
				if e.IsNil() {
					continue
				}
				setDefaults(e, recur, zeros)
			}

		case reflect.Map:
			for _, k := range f.MapKeys() {
				e := f.MapIndex(k)
				if e.IsNil() {
					continue
				}
				setDefaults(e, recur, zeros)
			}
		}
	}
}

var (
	// defaults maps a protocol buffer struct type to a slice of the fields,
	// with its scalar fields set to their proto-declared non-zero default values.
	defaultMu sync.RWMutex
	defaults  = make(map[reflect.Type]defaultMessage)

	int32PtrType = reflect.TypeOf((*int32)(nil))
)

// defaultMessage represents information about the default values of a message.
type defaultMessage struct {
	scalars []scalarField
	nested  []int // struct field index of nested messages
}

type scalarField struct {
	index int          // struct field index
	kind  reflect.Kind // element type (the T in *T or []T)
	value interface{}  // the proto-declared default value, or nil
}

// t is a struct type.
func buildDefaultMessage(t reflect.Type) (dm defaultMessage) {
	sprop := GetProperties(t)
	for _, prop := range sprop.Prop {
		fi, ok := sprop.decoderTags.get(prop.Tag)
		if !ok {
			// XXX_unrecognized
			continue
		}
		ft := t.Field(fi).Type

		sf, nested, err := fieldDefault(ft, prop)
		switch {
		case err != nil:
			log.Print(err)
		case nested:
			dm.nested = append(dm.nested, fi)
		case sf != nil:
			sf.index = fi
			dm.scalars = append(dm.scalars, *sf)
		}
	}

	return dm
}

// fieldDefault returns the scalarField for field type ft.
// sf will be nil if the field can not have a default.
// nestedMessage will be true if this is a nested message.
// Note that sf.index is not set on return.
func fieldDefault(ft reflect.Type, prop *Properties) (sf *scalarField, nestedMessage bool, err error) {
	var canHaveDefault bool
	switch ft.Kind() {
	case reflect.Ptr:
		if ft.Elem().Kind() == reflect.Struct {
			nestedMessage = true
		} else {
			canHaveDefault = true // proto2 scalar field
		}

	case reflect.Slice:
		switch ft.Elem().Kind() {
		case reflect.Ptr:
			nestedMessage = true // repeated message
		case reflect.Uint8:
			canHaveDefault = true // bytes field
		}

	case reflect.Map:
		if ft.Elem().Kind() == reflect.Ptr {
			nestedMessage = true // map with message values
		}
	}

	if !canHaveDefault {
		if nestedMessage {
			return nil, true, nil
		}
		return nil, false, nil
	}

	// We now know that ft is a pointer or slice.
	sf = &scalarField{kind: ft.Elem().Kind()}

	// scalar fields without defaults
	if !prop.HasDefault {
		return sf, false, nil
	}

	// a scalar field: either *T or []byte
	switch ft.Elem().Kind() {
	case reflect.Bool:
		x, err := strconv.ParseBool(prop.Default)
		if err != nil {
			return nil, false, fmt.Errorf("proto: bad default bool %q: %v", prop.Default, err)
		}
		sf.value = x
	case reflect.Float32:
		x, err := strconv.ParseFloat(prop.Default, 32)
		if err != nil {
			return nil, false, fmt.Errorf("proto: bad default float32 %q: %v", prop.Default, err)
		}
		sf.value = float32(x)
	case reflect.Float64:
		x, err := strconv.ParseFloat(prop.Default, 64)
		if err != nil {
			return nil, false, fmt.Errorf("proto: bad default float64 %q: %v", prop.Default, err)
		}
		sf.value = x
	case reflect.Int32:
		x, err := strconv.ParseInt(prop.Default, 10, 32)
		if err != nil {
			return nil, false, fmt.Errorf("proto: bad default int32 %q: %v", prop.Default, err)
		}
		sf.value = int32(x)
	case reflect.Int64:
		x, err := strconv.ParseInt(prop.Default, 10, 64)
		if err != nil {
			return nil, false, fmt.Errorf("proto: bad default int64 %q: %v", prop.Default, err)
		}
		sf.value = x
	case reflect.String:
		sf.value = prop.Default
	case reflect.Uint8:
		// []byte (not *uint8)
		sf.value = []byte(prop.Default)
	case reflect.Uint32:
		x, err := strconv.ParseUint(prop.Default, 10, 32)
		if err != nil {
			return nil, false, fmt.Errorf("proto: bad default uint32 %q: %v", prop.Default, err)
		}
		sf.value = uint32(x)
	case reflect.Uint64:
		x, err := strconv.ParseUint(prop.Default, 10, 64)
		if err != nil {
			return nil, false, fmt.Errorf("proto: bad default uint64 %q: %v", prop.Default, err)
		}
		sf.value = x
	default:
		return nil, false, fmt.Errorf("proto: unhandled def kind %v", ft.Elem().Kind())
	}

	return sf, false, nil
}

// Map fields may have key types of non-float scalars, strings and enums.
// The easiest way to sort them in some deterministic order is to use fmt.
// If this turns out to be inefficient we can always consider other options,
// such as doing a Schwartzian transform.

func mapKeys(vs []reflect.Value) sort.Interface {
	s := mapKeySorter{
		vs: vs,
		// default Less function: textual comparison
		less: func(a, b reflect.Value) bool {
			return fmt.Sprint(a.Interface()) < fmt.Sprint(b.Interface())
		},
	}

	// Type specialization per https://developers.google.com/protocol-buffers/docs/proto#maps;
	// numeric keys are sorted numerically.
	if len(vs) == 0 {
		return s
	}
	switch vs[0].Kind() {
	case reflect.Int32, reflect.Int64:
		s.less = func(a, b reflect.Value) bool { return a.Int() < b.Int() }
	case reflect.Uint32, reflect.Uint64:
		s.less = func(a, b reflect.Value) bool { return a.Uint() < b.Uint() }
	}

	return s
}

type mapKeySorter struct {
	vs   []reflect.Value
	less func(a, b reflect.Value) bool
}

func (s mapKeySorter) Len() int      { return len(s.vs) }
func (s mapKeySorter) Swap(i, j int) { s.vs[i], s.vs[j] = s.vs[j], s.vs[i] }
func (s mapKeySorter) Less(i, j int) bool {
	return s.less(s.vs[i], s.vs[j])
}

// isProto3Zero reports whether v is a zero proto3 value.
func isProto3Zero(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint32, reflect.Uint64:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.String:
		return v.String() == ""
	}
	return false
}

// ProtoPackageIsVersion2 is referenced from generated protocol buffer files
// to assert that that code is compatible with this version of the proto package.
const GoGoProtoPackageIsVersion2 = true

// ProtoPackageIsVersion1 is referenced from generated protocol buffer files
// to assert that that code is compatible with this version of the proto package.
const GoGoProtoPackageIsVersion1 = true
