// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dynamicpb creates protocol buffer messages using runtime type information.
package dynamicpb

import (
	"math"

	"google.golang.org/protobuf/internal/errors"
	pref "google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoiface"
	"google.golang.org/protobuf/runtime/protoimpl"
)

// enum is a dynamic protoreflect.Enum.
type enum struct {
	num pref.EnumNumber
	typ pref.EnumType
}

func (e enum) Descriptor() pref.EnumDescriptor { return e.typ.Descriptor() }
func (e enum) Type() pref.EnumType             { return e.typ }
func (e enum) Number() pref.EnumNumber         { return e.num }

// enumType is a dynamic protoreflect.EnumType.
type enumType struct {
	desc pref.EnumDescriptor
}

// NewEnumType creates a new EnumType with the provided descriptor.
//
// EnumTypes created by this package are equal if their descriptors are equal.
// That is, if ed1 == ed2, then NewEnumType(ed1) == NewEnumType(ed2).
//
// Enum values created by the EnumType are equal if their numbers are equal.
func NewEnumType(desc pref.EnumDescriptor) pref.EnumType {
	return enumType{desc}
}

func (et enumType) New(n pref.EnumNumber) pref.Enum { return enum{n, et} }
func (et enumType) Descriptor() pref.EnumDescriptor { return et.desc }

// extensionType is a dynamic protoreflect.ExtensionType.
type extensionType struct {
	desc extensionTypeDescriptor
}

// A Message is a dynamically constructed protocol buffer message.
//
// Message implements the proto.Message interface, and may be used with all
// standard proto package functions such as Marshal, Unmarshal, and so forth.
//
// Message also implements the protoreflect.Message interface. See the protoreflect
// package documentation for that interface for how to get and set fields and
// otherwise interact with the contents of a Message.
//
// Reflection API functions which construct messages, such as NewField,
// return new dynamic messages of the appropriate type. Functions which take
// messages, such as Set for a message-value field, will accept any message
// with a compatible type.
//
// Operations which modify a Message are not safe for concurrent use.
type Message struct {
	typ     messageType
	known   map[pref.FieldNumber]pref.Value
	ext     map[pref.FieldNumber]pref.FieldDescriptor
	unknown pref.RawFields
}

var (
	_ pref.Message         = (*Message)(nil)
	_ pref.ProtoMessage    = (*Message)(nil)
	_ protoiface.MessageV1 = (*Message)(nil)
)

// NewMessage creates a new message with the provided descriptor.
func NewMessage(desc pref.MessageDescriptor) *Message {
	return &Message{
		typ:   messageType{desc},
		known: make(map[pref.FieldNumber]pref.Value),
		ext:   make(map[pref.FieldNumber]pref.FieldDescriptor),
	}
}

// ProtoMessage implements the legacy message interface.
func (m *Message) ProtoMessage() {}

// ProtoReflect implements the protoreflect.ProtoMessage interface.
func (m *Message) ProtoReflect() pref.Message {
	return m
}

// String returns a string representation of a message.
func (m *Message) String() string {
	return protoimpl.X.MessageStringOf(m)
}

// Reset clears the message to be empty, but preserves the dynamic message type.
func (m *Message) Reset() {
	m.known = make(map[pref.FieldNumber]pref.Value)
	m.ext = make(map[pref.FieldNumber]pref.FieldDescriptor)
	m.unknown = nil
}

// Descriptor returns the message descriptor.
func (m *Message) Descriptor() pref.MessageDescriptor {
	return m.typ.desc
}

// Type returns the message type.
func (m *Message) Type() pref.MessageType {
	return m.typ
}

// New returns a newly allocated empty message with the same descriptor.
// See protoreflect.Message for details.
func (m *Message) New() pref.Message {
	return m.Type().New()
}

// Interface returns the message.
// See protoreflect.Message for details.
func (m *Message) Interface() pref.ProtoMessage {
	return m
}

// ProtoMethods is an internal detail of the protoreflect.Message interface.
// Users should never call this directly.
func (m *Message) ProtoMethods() *protoiface.Methods {
	return nil
}

// Range visits every populated field in undefined order.
// See protoreflect.Message for details.
func (m *Message) Range(f func(pref.FieldDescriptor, pref.Value) bool) {
	for num, v := range m.known {
		fd := m.ext[num]
		if fd == nil {
			fd = m.Descriptor().Fields().ByNumber(num)
		}
		if !isSet(fd, v) {
			continue
		}
		if !f(fd, v) {
			return
		}
	}
}

// Has reports whether a field is populated.
// See protoreflect.Message for details.
func (m *Message) Has(fd pref.FieldDescriptor) bool {
	m.checkField(fd)
	if fd.IsExtension() && m.ext[fd.Number()] != fd {
		return false
	}
	v, ok := m.known[fd.Number()]
	if !ok {
		return false
	}
	return isSet(fd, v)
}

// Clear clears a field.
// See protoreflect.Message for details.
func (m *Message) Clear(fd pref.FieldDescriptor) {
	m.checkField(fd)
	num := fd.Number()
	delete(m.known, num)
	delete(m.ext, num)
}

// Get returns the value of a field.
// See protoreflect.Message for details.
func (m *Message) Get(fd pref.FieldDescriptor) pref.Value {
	m.checkField(fd)
	num := fd.Number()
	if fd.IsExtension() {
		if fd != m.ext[num] {
			return fd.(pref.ExtensionTypeDescriptor).Type().Zero()
		}
		return m.known[num]
	}
	if v, ok := m.known[num]; ok {
		switch {
		case fd.IsMap():
			if v.Map().Len() > 0 {
				return v
			}
		case fd.IsList():
			if v.List().Len() > 0 {
				return v
			}
		default:
			return v
		}
	}
	switch {
	case fd.IsMap():
		return pref.ValueOfMap(&dynamicMap{desc: fd})
	case fd.IsList():
		return pref.ValueOfList(emptyList{desc: fd})
	case fd.Message() != nil:
		return pref.ValueOfMessage(&Message{typ: messageType{fd.Message()}})
	case fd.Kind() == pref.BytesKind:
		return pref.ValueOfBytes(append([]byte(nil), fd.Default().Bytes()...))
	default:
		return fd.Default()
	}
}

// Mutable returns a mutable reference to a repeated, map, or message field.
// See protoreflect.Message for details.
func (m *Message) Mutable(fd pref.FieldDescriptor) pref.Value {
	m.checkField(fd)
	if !fd.IsMap() && !fd.IsList() && fd.Message() == nil {
		panic(errors.New("%v: getting mutable reference to non-composite type", fd.FullName()))
	}
	if m.known == nil {
		panic(errors.New("%v: modification of read-only message", fd.FullName()))
	}
	num := fd.Number()
	if fd.IsExtension() {
		if fd != m.ext[num] {
			m.ext[num] = fd
			m.known[num] = fd.(pref.ExtensionTypeDescriptor).Type().New()
		}
		return m.known[num]
	}
	if v, ok := m.known[num]; ok {
		return v
	}
	m.clearOtherOneofFields(fd)
	m.known[num] = m.NewField(fd)
	if fd.IsExtension() {
		m.ext[num] = fd
	}
	return m.known[num]
}

// Set stores a value in a field.
// See protoreflect.Message for details.
func (m *Message) Set(fd pref.FieldDescriptor, v pref.Value) {
	m.checkField(fd)
	if m.known == nil {
		panic(errors.New("%v: modification of read-only message", fd.FullName()))
	}
	if fd.IsExtension() {
		isValid := true
		switch {
		case !fd.(pref.ExtensionTypeDescriptor).Type().IsValidValue(v):
			isValid = false
		case fd.IsList():
			isValid = v.List().IsValid()
		case fd.IsMap():
			isValid = v.Map().IsValid()
		case fd.Message() != nil:
			isValid = v.Message().IsValid()
		}
		if !isValid {
			panic(errors.New("%v: assigning invalid type %T", fd.FullName(), v.Interface()))
		}
		m.ext[fd.Number()] = fd
	} else {
		typecheck(fd, v)
	}
	m.clearOtherOneofFields(fd)
	m.known[fd.Number()] = v
}

func (m *Message) clearOtherOneofFields(fd pref.FieldDescriptor) {
	od := fd.ContainingOneof()
	if od == nil {
		return
	}
	num := fd.Number()
	for i := 0; i < od.Fields().Len(); i++ {
		if n := od.Fields().Get(i).Number(); n != num {
			delete(m.known, n)
		}
	}
}

// NewField returns a new value for assignable to the field of a given descriptor.
// See protoreflect.Message for details.
func (m *Message) NewField(fd pref.FieldDescriptor) pref.Value {
	m.checkField(fd)
	switch {
	case fd.IsExtension():
		return fd.(pref.ExtensionTypeDescriptor).Type().New()
	case fd.IsMap():
		return pref.ValueOfMap(&dynamicMap{
			desc: fd,
			mapv: make(map[interface{}]pref.Value),
		})
	case fd.IsList():
		return pref.ValueOfList(&dynamicList{desc: fd})
	case fd.Message() != nil:
		return pref.ValueOfMessage(NewMessage(fd.Message()).ProtoReflect())
	default:
		return fd.Default()
	}
}

// WhichOneof reports which field in a oneof is populated, returning nil if none are populated.
// See protoreflect.Message for details.
func (m *Message) WhichOneof(od pref.OneofDescriptor) pref.FieldDescriptor {
	for i := 0; i < od.Fields().Len(); i++ {
		fd := od.Fields().Get(i)
		if m.Has(fd) {
			return fd
		}
	}
	return nil
}

// GetUnknown returns the raw unknown fields.
// See protoreflect.Message for details.
func (m *Message) GetUnknown() pref.RawFields {
	return m.unknown
}

// SetUnknown sets the raw unknown fields.
// See protoreflect.Message for details.
func (m *Message) SetUnknown(r pref.RawFields) {
	if m.known == nil {
		panic(errors.New("%v: modification of read-only message", m.typ.desc.FullName()))
	}
	m.unknown = r
}

// IsValid reports whether the message is valid.
// See protoreflect.Message for details.
func (m *Message) IsValid() bool {
	return m.known != nil
}

func (m *Message) checkField(fd pref.FieldDescriptor) {
	if fd.IsExtension() && fd.ContainingMessage().FullName() == m.Descriptor().FullName() {
		if _, ok := fd.(pref.ExtensionTypeDescriptor); !ok {
			panic(errors.New("%v: extension field descriptor does not implement ExtensionTypeDescriptor", fd.FullName()))
		}
		return
	}
	if fd.Parent() == m.Descriptor() {
		return
	}
	fields := m.Descriptor().Fields()
	index := fd.Index()
	if index >= fields.Len() || fields.Get(index) != fd {
		panic(errors.New("%v: field descriptor does not belong to this message", fd.FullName()))
	}
}

type messageType struct {
	desc pref.MessageDescriptor
}

// NewMessageType creates a new MessageType with the provided descriptor.
//
// MessageTypes created by this package are equal if their descriptors are equal.
// That is, if md1 == md2, then NewMessageType(md1) == NewMessageType(md2).
func NewMessageType(desc pref.MessageDescriptor) pref.MessageType {
	return messageType{desc}
}

func (mt messageType) New() pref.Message                  { return NewMessage(mt.desc) }
func (mt messageType) Zero() pref.Message                 { return &Message{typ: messageType{mt.desc}} }
func (mt messageType) Descriptor() pref.MessageDescriptor { return mt.desc }
func (mt messageType) Enum(i int) pref.EnumType {
	if ed := mt.desc.Fields().Get(i).Enum(); ed != nil {
		return NewEnumType(ed)
	}
	return nil
}
func (mt messageType) Message(i int) pref.MessageType {
	if md := mt.desc.Fields().Get(i).Message(); md != nil {
		return NewMessageType(md)
	}
	return nil
}

type emptyList struct {
	desc pref.FieldDescriptor
}

func (x emptyList) Len() int                  { return 0 }
func (x emptyList) Get(n int) pref.Value      { panic(errors.New("out of range")) }
func (x emptyList) Set(n int, v pref.Value)   { panic(errors.New("modification of immutable list")) }
func (x emptyList) Append(v pref.Value)       { panic(errors.New("modification of immutable list")) }
func (x emptyList) AppendMutable() pref.Value { panic(errors.New("modification of immutable list")) }
func (x emptyList) Truncate(n int)            { panic(errors.New("modification of immutable list")) }
func (x emptyList) NewElement() pref.Value    { return newListEntry(x.desc) }
func (x emptyList) IsValid() bool             { return false }

type dynamicList struct {
	desc pref.FieldDescriptor
	list []pref.Value
}

func (x *dynamicList) Len() int {
	return len(x.list)
}

func (x *dynamicList) Get(n int) pref.Value {
	return x.list[n]
}

func (x *dynamicList) Set(n int, v pref.Value) {
	typecheckSingular(x.desc, v)
	x.list[n] = v
}

func (x *dynamicList) Append(v pref.Value) {
	typecheckSingular(x.desc, v)
	x.list = append(x.list, v)
}

func (x *dynamicList) AppendMutable() pref.Value {
	if x.desc.Message() == nil {
		panic(errors.New("%v: invalid AppendMutable on list with non-message type", x.desc.FullName()))
	}
	v := x.NewElement()
	x.Append(v)
	return v
}

func (x *dynamicList) Truncate(n int) {
	// Zero truncated elements to avoid keeping data live.
	for i := n; i < len(x.list); i++ {
		x.list[i] = pref.Value{}
	}
	x.list = x.list[:n]
}

func (x *dynamicList) NewElement() pref.Value {
	return newListEntry(x.desc)
}

func (x *dynamicList) IsValid() bool {
	return true
}

type dynamicMap struct {
	desc pref.FieldDescriptor
	mapv map[interface{}]pref.Value
}

func (x *dynamicMap) Get(k pref.MapKey) pref.Value { return x.mapv[k.Interface()] }
func (x *dynamicMap) Set(k pref.MapKey, v pref.Value) {
	typecheckSingular(x.desc.MapKey(), k.Value())
	typecheckSingular(x.desc.MapValue(), v)
	x.mapv[k.Interface()] = v
}
func (x *dynamicMap) Has(k pref.MapKey) bool { return x.Get(k).IsValid() }
func (x *dynamicMap) Clear(k pref.MapKey)    { delete(x.mapv, k.Interface()) }
func (x *dynamicMap) Mutable(k pref.MapKey) pref.Value {
	if x.desc.MapValue().Message() == nil {
		panic(errors.New("%v: invalid Mutable on map with non-message value type", x.desc.FullName()))
	}
	v := x.Get(k)
	if !v.IsValid() {
		v = x.NewValue()
		x.Set(k, v)
	}
	return v
}
func (x *dynamicMap) Len() int { return len(x.mapv) }
func (x *dynamicMap) NewValue() pref.Value {
	if md := x.desc.MapValue().Message(); md != nil {
		return pref.ValueOfMessage(NewMessage(md).ProtoReflect())
	}
	return x.desc.MapValue().Default()
}
func (x *dynamicMap) IsValid() bool {
	return x.mapv != nil
}

func (x *dynamicMap) Range(f func(pref.MapKey, pref.Value) bool) {
	for k, v := range x.mapv {
		if !f(pref.ValueOf(k).MapKey(), v) {
			return
		}
	}
}

func isSet(fd pref.FieldDescriptor, v pref.Value) bool {
	switch {
	case fd.IsMap():
		return v.Map().Len() > 0
	case fd.IsList():
		return v.List().Len() > 0
	case fd.ContainingOneof() != nil:
		return true
	case fd.Syntax() == pref.Proto3 && !fd.IsExtension():
		switch fd.Kind() {
		case pref.BoolKind:
			return v.Bool()
		case pref.EnumKind:
			return v.Enum() != 0
		case pref.Int32Kind, pref.Sint32Kind, pref.Int64Kind, pref.Sint64Kind, pref.Sfixed32Kind, pref.Sfixed64Kind:
			return v.Int() != 0
		case pref.Uint32Kind, pref.Uint64Kind, pref.Fixed32Kind, pref.Fixed64Kind:
			return v.Uint() != 0
		case pref.FloatKind, pref.DoubleKind:
			return v.Float() != 0 || math.Signbit(v.Float())
		case pref.StringKind:
			return v.String() != ""
		case pref.BytesKind:
			return len(v.Bytes()) > 0
		}
	}
	return true
}

func typecheck(fd pref.FieldDescriptor, v pref.Value) {
	if err := typeIsValid(fd, v); err != nil {
		panic(err)
	}
}

func typeIsValid(fd pref.FieldDescriptor, v pref.Value) error {
	switch {
	case !v.IsValid():
		return errors.New("%v: assigning invalid value", fd.FullName())
	case fd.IsMap():
		if mapv, ok := v.Interface().(*dynamicMap); !ok || mapv.desc != fd || !mapv.IsValid() {
			return errors.New("%v: assigning invalid type %T", fd.FullName(), v.Interface())
		}
		return nil
	case fd.IsList():
		switch list := v.Interface().(type) {
		case *dynamicList:
			if list.desc == fd && list.IsValid() {
				return nil
			}
		case emptyList:
			if list.desc == fd && list.IsValid() {
				return nil
			}
		}
		return errors.New("%v: assigning invalid type %T", fd.FullName(), v.Interface())
	default:
		return singularTypeIsValid(fd, v)
	}
}

func typecheckSingular(fd pref.FieldDescriptor, v pref.Value) {
	if err := singularTypeIsValid(fd, v); err != nil {
		panic(err)
	}
}

func singularTypeIsValid(fd pref.FieldDescriptor, v pref.Value) error {
	vi := v.Interface()
	var ok bool
	switch fd.Kind() {
	case pref.BoolKind:
		_, ok = vi.(bool)
	case pref.EnumKind:
		// We could check against the valid set of enum values, but do not.
		_, ok = vi.(pref.EnumNumber)
	case pref.Int32Kind, pref.Sint32Kind, pref.Sfixed32Kind:
		_, ok = vi.(int32)
	case pref.Uint32Kind, pref.Fixed32Kind:
		_, ok = vi.(uint32)
	case pref.Int64Kind, pref.Sint64Kind, pref.Sfixed64Kind:
		_, ok = vi.(int64)
	case pref.Uint64Kind, pref.Fixed64Kind:
		_, ok = vi.(uint64)
	case pref.FloatKind:
		_, ok = vi.(float32)
	case pref.DoubleKind:
		_, ok = vi.(float64)
	case pref.StringKind:
		_, ok = vi.(string)
	case pref.BytesKind:
		_, ok = vi.([]byte)
	case pref.MessageKind, pref.GroupKind:
		var m pref.Message
		m, ok = vi.(pref.Message)
		if ok && m.Descriptor().FullName() != fd.Message().FullName() {
			return errors.New("%v: assigning invalid message type %v", fd.FullName(), m.Descriptor().FullName())
		}
		if dm, ok := vi.(*Message); ok && dm.known == nil {
			return errors.New("%v: assigning invalid zero-value message", fd.FullName())
		}
	}
	if !ok {
		return errors.New("%v: assigning invalid type %T", fd.FullName(), v.Interface())
	}
	return nil
}

func newListEntry(fd pref.FieldDescriptor) pref.Value {
	switch fd.Kind() {
	case pref.BoolKind:
		return pref.ValueOfBool(false)
	case pref.EnumKind:
		return pref.ValueOfEnum(fd.Enum().Values().Get(0).Number())
	case pref.Int32Kind, pref.Sint32Kind, pref.Sfixed32Kind:
		return pref.ValueOfInt32(0)
	case pref.Uint32Kind, pref.Fixed32Kind:
		return pref.ValueOfUint32(0)
	case pref.Int64Kind, pref.Sint64Kind, pref.Sfixed64Kind:
		return pref.ValueOfInt64(0)
	case pref.Uint64Kind, pref.Fixed64Kind:
		return pref.ValueOfUint64(0)
	case pref.FloatKind:
		return pref.ValueOfFloat32(0)
	case pref.DoubleKind:
		return pref.ValueOfFloat64(0)
	case pref.StringKind:
		return pref.ValueOfString("")
	case pref.BytesKind:
		return pref.ValueOfBytes(nil)
	case pref.MessageKind, pref.GroupKind:
		return pref.ValueOfMessage(NewMessage(fd.Message()).ProtoReflect())
	}
	panic(errors.New("%v: unknown kind %v", fd.FullName(), fd.Kind()))
}

// NewExtensionType creates a new ExtensionType with the provided descriptor.
//
// Dynamic ExtensionTypes with the same descriptor compare as equal. That is,
// if xd1 == xd2, then NewExtensionType(xd1) == NewExtensionType(xd2).
//
// The InterfaceOf and ValueOf methods of the extension type are defined as:
//
//	func (xt extensionType) ValueOf(iv interface{}) protoreflect.Value {
//		return protoreflect.ValueOf(iv)
//	}
//
//	func (xt extensionType) InterfaceOf(v protoreflect.Value) interface{} {
//		return v.Interface()
//	}
//
// The Go type used by the proto.GetExtension and proto.SetExtension functions
// is determined by these methods, and is therefore equivalent to the Go type
// used to represent a protoreflect.Value. See the protoreflect.Value
// documentation for more details.
func NewExtensionType(desc pref.ExtensionDescriptor) pref.ExtensionType {
	if xt, ok := desc.(pref.ExtensionTypeDescriptor); ok {
		desc = xt.Descriptor()
	}
	return extensionType{extensionTypeDescriptor{desc}}
}

func (xt extensionType) New() pref.Value {
	switch {
	case xt.desc.IsMap():
		return pref.ValueOfMap(&dynamicMap{
			desc: xt.desc,
			mapv: make(map[interface{}]pref.Value),
		})
	case xt.desc.IsList():
		return pref.ValueOfList(&dynamicList{desc: xt.desc})
	case xt.desc.Message() != nil:
		return pref.ValueOfMessage(NewMessage(xt.desc.Message()))
	default:
		return xt.desc.Default()
	}
}

func (xt extensionType) Zero() pref.Value {
	switch {
	case xt.desc.IsMap():
		return pref.ValueOfMap(&dynamicMap{desc: xt.desc})
	case xt.desc.Cardinality() == pref.Repeated:
		return pref.ValueOfList(emptyList{desc: xt.desc})
	case xt.desc.Message() != nil:
		return pref.ValueOfMessage(&Message{typ: messageType{xt.desc.Message()}})
	default:
		return xt.desc.Default()
	}
}

func (xt extensionType) TypeDescriptor() pref.ExtensionTypeDescriptor {
	return xt.desc
}

func (xt extensionType) ValueOf(iv interface{}) pref.Value {
	v := pref.ValueOf(iv)
	typecheck(xt.desc, v)
	return v
}

func (xt extensionType) InterfaceOf(v pref.Value) interface{} {
	typecheck(xt.desc, v)
	return v.Interface()
}

func (xt extensionType) IsValidInterface(iv interface{}) bool {
	return typeIsValid(xt.desc, pref.ValueOf(iv)) == nil
}

func (xt extensionType) IsValidValue(v pref.Value) bool {
	return typeIsValid(xt.desc, v) == nil
}

type extensionTypeDescriptor struct {
	pref.ExtensionDescriptor
}

func (xt extensionTypeDescriptor) Type() pref.ExtensionType {
	return extensionType{xt}
}

func (xt extensionTypeDescriptor) Descriptor() pref.ExtensionDescriptor {
	return xt.ExtensionDescriptor
}
