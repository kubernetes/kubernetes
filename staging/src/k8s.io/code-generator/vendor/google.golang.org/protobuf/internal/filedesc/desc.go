// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filedesc

import (
	"bytes"
	"fmt"
	"sync"
	"sync/atomic"

	"google.golang.org/protobuf/internal/descfmt"
	"google.golang.org/protobuf/internal/descopts"
	"google.golang.org/protobuf/internal/encoding/defval"
	"google.golang.org/protobuf/internal/encoding/messageset"
	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/internal/strs"
	pref "google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

// The types in this file may have a suffix:
//	• L0: Contains fields common to all descriptors (except File) and
//	must be initialized up front.
//	• L1: Contains fields specific to a descriptor and
//	must be initialized up front.
//	• L2: Contains fields that are lazily initialized when constructing
//	from the raw file descriptor. When constructing as a literal, the L2
//	fields must be initialized up front.
//
// The types are exported so that packages like reflect/protodesc can
// directly construct descriptors.

type (
	File struct {
		fileRaw
		L1 FileL1

		once uint32     // atomically set if L2 is valid
		mu   sync.Mutex // protects L2
		L2   *FileL2
	}
	FileL1 struct {
		Syntax  pref.Syntax
		Path    string
		Package pref.FullName

		Enums      Enums
		Messages   Messages
		Extensions Extensions
		Services   Services
	}
	FileL2 struct {
		Options   func() pref.ProtoMessage
		Imports   FileImports
		Locations SourceLocations
	}
)

func (fd *File) ParentFile() pref.FileDescriptor { return fd }
func (fd *File) Parent() pref.Descriptor         { return nil }
func (fd *File) Index() int                      { return 0 }
func (fd *File) Syntax() pref.Syntax             { return fd.L1.Syntax }
func (fd *File) Name() pref.Name                 { return fd.L1.Package.Name() }
func (fd *File) FullName() pref.FullName         { return fd.L1.Package }
func (fd *File) IsPlaceholder() bool             { return false }
func (fd *File) Options() pref.ProtoMessage {
	if f := fd.lazyInit().Options; f != nil {
		return f()
	}
	return descopts.File
}
func (fd *File) Path() string                          { return fd.L1.Path }
func (fd *File) Package() pref.FullName                { return fd.L1.Package }
func (fd *File) Imports() pref.FileImports             { return &fd.lazyInit().Imports }
func (fd *File) Enums() pref.EnumDescriptors           { return &fd.L1.Enums }
func (fd *File) Messages() pref.MessageDescriptors     { return &fd.L1.Messages }
func (fd *File) Extensions() pref.ExtensionDescriptors { return &fd.L1.Extensions }
func (fd *File) Services() pref.ServiceDescriptors     { return &fd.L1.Services }
func (fd *File) SourceLocations() pref.SourceLocations { return &fd.lazyInit().Locations }
func (fd *File) Format(s fmt.State, r rune)            { descfmt.FormatDesc(s, r, fd) }
func (fd *File) ProtoType(pref.FileDescriptor)         {}
func (fd *File) ProtoInternal(pragma.DoNotImplement)   {}

func (fd *File) lazyInit() *FileL2 {
	if atomic.LoadUint32(&fd.once) == 0 {
		fd.lazyInitOnce()
	}
	return fd.L2
}

func (fd *File) lazyInitOnce() {
	fd.mu.Lock()
	if fd.L2 == nil {
		fd.lazyRawInit() // recursively initializes all L2 structures
	}
	atomic.StoreUint32(&fd.once, 1)
	fd.mu.Unlock()
}

// GoPackagePath is a pseudo-internal API for determining the Go package path
// that this file descriptor is declared in.
//
// WARNING: This method is exempt from the compatibility promise and may be
// removed in the future without warning.
func (fd *File) GoPackagePath() string {
	return fd.builder.GoPackagePath
}

type (
	Enum struct {
		Base
		L1 EnumL1
		L2 *EnumL2 // protected by fileDesc.once
	}
	EnumL1 struct {
		eagerValues bool // controls whether EnumL2.Values is already populated
	}
	EnumL2 struct {
		Options        func() pref.ProtoMessage
		Values         EnumValues
		ReservedNames  Names
		ReservedRanges EnumRanges
	}

	EnumValue struct {
		Base
		L1 EnumValueL1
	}
	EnumValueL1 struct {
		Options func() pref.ProtoMessage
		Number  pref.EnumNumber
	}
)

func (ed *Enum) Options() pref.ProtoMessage {
	if f := ed.lazyInit().Options; f != nil {
		return f()
	}
	return descopts.Enum
}
func (ed *Enum) Values() pref.EnumValueDescriptors {
	if ed.L1.eagerValues {
		return &ed.L2.Values
	}
	return &ed.lazyInit().Values
}
func (ed *Enum) ReservedNames() pref.Names       { return &ed.lazyInit().ReservedNames }
func (ed *Enum) ReservedRanges() pref.EnumRanges { return &ed.lazyInit().ReservedRanges }
func (ed *Enum) Format(s fmt.State, r rune)      { descfmt.FormatDesc(s, r, ed) }
func (ed *Enum) ProtoType(pref.EnumDescriptor)   {}
func (ed *Enum) lazyInit() *EnumL2 {
	ed.L0.ParentFile.lazyInit() // implicitly initializes L2
	return ed.L2
}

func (ed *EnumValue) Options() pref.ProtoMessage {
	if f := ed.L1.Options; f != nil {
		return f()
	}
	return descopts.EnumValue
}
func (ed *EnumValue) Number() pref.EnumNumber            { return ed.L1.Number }
func (ed *EnumValue) Format(s fmt.State, r rune)         { descfmt.FormatDesc(s, r, ed) }
func (ed *EnumValue) ProtoType(pref.EnumValueDescriptor) {}

type (
	Message struct {
		Base
		L1 MessageL1
		L2 *MessageL2 // protected by fileDesc.once
	}
	MessageL1 struct {
		Enums        Enums
		Messages     Messages
		Extensions   Extensions
		IsMapEntry   bool // promoted from google.protobuf.MessageOptions
		IsMessageSet bool // promoted from google.protobuf.MessageOptions
	}
	MessageL2 struct {
		Options               func() pref.ProtoMessage
		Fields                Fields
		Oneofs                Oneofs
		ReservedNames         Names
		ReservedRanges        FieldRanges
		RequiredNumbers       FieldNumbers // must be consistent with Fields.Cardinality
		ExtensionRanges       FieldRanges
		ExtensionRangeOptions []func() pref.ProtoMessage // must be same length as ExtensionRanges
	}

	Field struct {
		Base
		L1 FieldL1
	}
	FieldL1 struct {
		Options          func() pref.ProtoMessage
		Number           pref.FieldNumber
		Cardinality      pref.Cardinality // must be consistent with Message.RequiredNumbers
		Kind             pref.Kind
		StringName       stringName
		IsProto3Optional bool // promoted from google.protobuf.FieldDescriptorProto
		IsWeak           bool // promoted from google.protobuf.FieldOptions
		HasPacked        bool // promoted from google.protobuf.FieldOptions
		IsPacked         bool // promoted from google.protobuf.FieldOptions
		HasEnforceUTF8   bool // promoted from google.protobuf.FieldOptions
		EnforceUTF8      bool // promoted from google.protobuf.FieldOptions
		Default          defaultValue
		ContainingOneof  pref.OneofDescriptor // must be consistent with Message.Oneofs.Fields
		Enum             pref.EnumDescriptor
		Message          pref.MessageDescriptor
	}

	Oneof struct {
		Base
		L1 OneofL1
	}
	OneofL1 struct {
		Options func() pref.ProtoMessage
		Fields  OneofFields // must be consistent with Message.Fields.ContainingOneof
	}
)

func (md *Message) Options() pref.ProtoMessage {
	if f := md.lazyInit().Options; f != nil {
		return f()
	}
	return descopts.Message
}
func (md *Message) IsMapEntry() bool                   { return md.L1.IsMapEntry }
func (md *Message) Fields() pref.FieldDescriptors      { return &md.lazyInit().Fields }
func (md *Message) Oneofs() pref.OneofDescriptors      { return &md.lazyInit().Oneofs }
func (md *Message) ReservedNames() pref.Names          { return &md.lazyInit().ReservedNames }
func (md *Message) ReservedRanges() pref.FieldRanges   { return &md.lazyInit().ReservedRanges }
func (md *Message) RequiredNumbers() pref.FieldNumbers { return &md.lazyInit().RequiredNumbers }
func (md *Message) ExtensionRanges() pref.FieldRanges  { return &md.lazyInit().ExtensionRanges }
func (md *Message) ExtensionRangeOptions(i int) pref.ProtoMessage {
	if f := md.lazyInit().ExtensionRangeOptions[i]; f != nil {
		return f()
	}
	return descopts.ExtensionRange
}
func (md *Message) Enums() pref.EnumDescriptors           { return &md.L1.Enums }
func (md *Message) Messages() pref.MessageDescriptors     { return &md.L1.Messages }
func (md *Message) Extensions() pref.ExtensionDescriptors { return &md.L1.Extensions }
func (md *Message) ProtoType(pref.MessageDescriptor)      {}
func (md *Message) Format(s fmt.State, r rune)            { descfmt.FormatDesc(s, r, md) }
func (md *Message) lazyInit() *MessageL2 {
	md.L0.ParentFile.lazyInit() // implicitly initializes L2
	return md.L2
}

// IsMessageSet is a pseudo-internal API for checking whether a message
// should serialize in the proto1 message format.
//
// WARNING: This method is exempt from the compatibility promise and may be
// removed in the future without warning.
func (md *Message) IsMessageSet() bool {
	return md.L1.IsMessageSet
}

func (fd *Field) Options() pref.ProtoMessage {
	if f := fd.L1.Options; f != nil {
		return f()
	}
	return descopts.Field
}
func (fd *Field) Number() pref.FieldNumber      { return fd.L1.Number }
func (fd *Field) Cardinality() pref.Cardinality { return fd.L1.Cardinality }
func (fd *Field) Kind() pref.Kind               { return fd.L1.Kind }
func (fd *Field) HasJSONName() bool             { return fd.L1.StringName.hasJSON }
func (fd *Field) JSONName() string              { return fd.L1.StringName.getJSON(fd) }
func (fd *Field) TextName() string              { return fd.L1.StringName.getText(fd) }
func (fd *Field) HasPresence() bool {
	return fd.L1.Cardinality != pref.Repeated && (fd.L0.ParentFile.L1.Syntax == pref.Proto2 || fd.L1.Message != nil || fd.L1.ContainingOneof != nil)
}
func (fd *Field) HasOptionalKeyword() bool {
	return (fd.L0.ParentFile.L1.Syntax == pref.Proto2 && fd.L1.Cardinality == pref.Optional && fd.L1.ContainingOneof == nil) || fd.L1.IsProto3Optional
}
func (fd *Field) IsPacked() bool {
	if !fd.L1.HasPacked && fd.L0.ParentFile.L1.Syntax != pref.Proto2 && fd.L1.Cardinality == pref.Repeated {
		switch fd.L1.Kind {
		case pref.StringKind, pref.BytesKind, pref.MessageKind, pref.GroupKind:
		default:
			return true
		}
	}
	return fd.L1.IsPacked
}
func (fd *Field) IsExtension() bool { return false }
func (fd *Field) IsWeak() bool      { return fd.L1.IsWeak }
func (fd *Field) IsList() bool      { return fd.Cardinality() == pref.Repeated && !fd.IsMap() }
func (fd *Field) IsMap() bool       { return fd.Message() != nil && fd.Message().IsMapEntry() }
func (fd *Field) MapKey() pref.FieldDescriptor {
	if !fd.IsMap() {
		return nil
	}
	return fd.Message().Fields().ByNumber(genid.MapEntry_Key_field_number)
}
func (fd *Field) MapValue() pref.FieldDescriptor {
	if !fd.IsMap() {
		return nil
	}
	return fd.Message().Fields().ByNumber(genid.MapEntry_Value_field_number)
}
func (fd *Field) HasDefault() bool                           { return fd.L1.Default.has }
func (fd *Field) Default() pref.Value                        { return fd.L1.Default.get(fd) }
func (fd *Field) DefaultEnumValue() pref.EnumValueDescriptor { return fd.L1.Default.enum }
func (fd *Field) ContainingOneof() pref.OneofDescriptor      { return fd.L1.ContainingOneof }
func (fd *Field) ContainingMessage() pref.MessageDescriptor {
	return fd.L0.Parent.(pref.MessageDescriptor)
}
func (fd *Field) Enum() pref.EnumDescriptor {
	return fd.L1.Enum
}
func (fd *Field) Message() pref.MessageDescriptor {
	if fd.L1.IsWeak {
		if d, _ := protoregistry.GlobalFiles.FindDescriptorByName(fd.L1.Message.FullName()); d != nil {
			return d.(pref.MessageDescriptor)
		}
	}
	return fd.L1.Message
}
func (fd *Field) Format(s fmt.State, r rune)     { descfmt.FormatDesc(s, r, fd) }
func (fd *Field) ProtoType(pref.FieldDescriptor) {}

// EnforceUTF8 is a pseudo-internal API to determine whether to enforce UTF-8
// validation for the string field. This exists for Google-internal use only
// since proto3 did not enforce UTF-8 validity prior to the open-source release.
// If this method does not exist, the default is to enforce valid UTF-8.
//
// WARNING: This method is exempt from the compatibility promise and may be
// removed in the future without warning.
func (fd *Field) EnforceUTF8() bool {
	if fd.L1.HasEnforceUTF8 {
		return fd.L1.EnforceUTF8
	}
	return fd.L0.ParentFile.L1.Syntax == pref.Proto3
}

func (od *Oneof) IsSynthetic() bool {
	return od.L0.ParentFile.L1.Syntax == pref.Proto3 && len(od.L1.Fields.List) == 1 && od.L1.Fields.List[0].HasOptionalKeyword()
}
func (od *Oneof) Options() pref.ProtoMessage {
	if f := od.L1.Options; f != nil {
		return f()
	}
	return descopts.Oneof
}
func (od *Oneof) Fields() pref.FieldDescriptors  { return &od.L1.Fields }
func (od *Oneof) Format(s fmt.State, r rune)     { descfmt.FormatDesc(s, r, od) }
func (od *Oneof) ProtoType(pref.OneofDescriptor) {}

type (
	Extension struct {
		Base
		L1 ExtensionL1
		L2 *ExtensionL2 // protected by fileDesc.once
	}
	ExtensionL1 struct {
		Number      pref.FieldNumber
		Extendee    pref.MessageDescriptor
		Cardinality pref.Cardinality
		Kind        pref.Kind
	}
	ExtensionL2 struct {
		Options          func() pref.ProtoMessage
		StringName       stringName
		IsProto3Optional bool // promoted from google.protobuf.FieldDescriptorProto
		IsPacked         bool // promoted from google.protobuf.FieldOptions
		Default          defaultValue
		Enum             pref.EnumDescriptor
		Message          pref.MessageDescriptor
	}
)

func (xd *Extension) Options() pref.ProtoMessage {
	if f := xd.lazyInit().Options; f != nil {
		return f()
	}
	return descopts.Field
}
func (xd *Extension) Number() pref.FieldNumber      { return xd.L1.Number }
func (xd *Extension) Cardinality() pref.Cardinality { return xd.L1.Cardinality }
func (xd *Extension) Kind() pref.Kind               { return xd.L1.Kind }
func (xd *Extension) HasJSONName() bool             { return xd.lazyInit().StringName.hasJSON }
func (xd *Extension) JSONName() string              { return xd.lazyInit().StringName.getJSON(xd) }
func (xd *Extension) TextName() string              { return xd.lazyInit().StringName.getText(xd) }
func (xd *Extension) HasPresence() bool             { return xd.L1.Cardinality != pref.Repeated }
func (xd *Extension) HasOptionalKeyword() bool {
	return (xd.L0.ParentFile.L1.Syntax == pref.Proto2 && xd.L1.Cardinality == pref.Optional) || xd.lazyInit().IsProto3Optional
}
func (xd *Extension) IsPacked() bool                             { return xd.lazyInit().IsPacked }
func (xd *Extension) IsExtension() bool                          { return true }
func (xd *Extension) IsWeak() bool                               { return false }
func (xd *Extension) IsList() bool                               { return xd.Cardinality() == pref.Repeated }
func (xd *Extension) IsMap() bool                                { return false }
func (xd *Extension) MapKey() pref.FieldDescriptor               { return nil }
func (xd *Extension) MapValue() pref.FieldDescriptor             { return nil }
func (xd *Extension) HasDefault() bool                           { return xd.lazyInit().Default.has }
func (xd *Extension) Default() pref.Value                        { return xd.lazyInit().Default.get(xd) }
func (xd *Extension) DefaultEnumValue() pref.EnumValueDescriptor { return xd.lazyInit().Default.enum }
func (xd *Extension) ContainingOneof() pref.OneofDescriptor      { return nil }
func (xd *Extension) ContainingMessage() pref.MessageDescriptor  { return xd.L1.Extendee }
func (xd *Extension) Enum() pref.EnumDescriptor                  { return xd.lazyInit().Enum }
func (xd *Extension) Message() pref.MessageDescriptor            { return xd.lazyInit().Message }
func (xd *Extension) Format(s fmt.State, r rune)                 { descfmt.FormatDesc(s, r, xd) }
func (xd *Extension) ProtoType(pref.FieldDescriptor)             {}
func (xd *Extension) ProtoInternal(pragma.DoNotImplement)        {}
func (xd *Extension) lazyInit() *ExtensionL2 {
	xd.L0.ParentFile.lazyInit() // implicitly initializes L2
	return xd.L2
}

type (
	Service struct {
		Base
		L1 ServiceL1
		L2 *ServiceL2 // protected by fileDesc.once
	}
	ServiceL1 struct{}
	ServiceL2 struct {
		Options func() pref.ProtoMessage
		Methods Methods
	}

	Method struct {
		Base
		L1 MethodL1
	}
	MethodL1 struct {
		Options           func() pref.ProtoMessage
		Input             pref.MessageDescriptor
		Output            pref.MessageDescriptor
		IsStreamingClient bool
		IsStreamingServer bool
	}
)

func (sd *Service) Options() pref.ProtoMessage {
	if f := sd.lazyInit().Options; f != nil {
		return f()
	}
	return descopts.Service
}
func (sd *Service) Methods() pref.MethodDescriptors     { return &sd.lazyInit().Methods }
func (sd *Service) Format(s fmt.State, r rune)          { descfmt.FormatDesc(s, r, sd) }
func (sd *Service) ProtoType(pref.ServiceDescriptor)    {}
func (sd *Service) ProtoInternal(pragma.DoNotImplement) {}
func (sd *Service) lazyInit() *ServiceL2 {
	sd.L0.ParentFile.lazyInit() // implicitly initializes L2
	return sd.L2
}

func (md *Method) Options() pref.ProtoMessage {
	if f := md.L1.Options; f != nil {
		return f()
	}
	return descopts.Method
}
func (md *Method) Input() pref.MessageDescriptor       { return md.L1.Input }
func (md *Method) Output() pref.MessageDescriptor      { return md.L1.Output }
func (md *Method) IsStreamingClient() bool             { return md.L1.IsStreamingClient }
func (md *Method) IsStreamingServer() bool             { return md.L1.IsStreamingServer }
func (md *Method) Format(s fmt.State, r rune)          { descfmt.FormatDesc(s, r, md) }
func (md *Method) ProtoType(pref.MethodDescriptor)     {}
func (md *Method) ProtoInternal(pragma.DoNotImplement) {}

// Surrogate files are can be used to create standalone descriptors
// where the syntax is only information derived from the parent file.
var (
	SurrogateProto2 = &File{L1: FileL1{Syntax: pref.Proto2}, L2: &FileL2{}}
	SurrogateProto3 = &File{L1: FileL1{Syntax: pref.Proto3}, L2: &FileL2{}}
)

type (
	Base struct {
		L0 BaseL0
	}
	BaseL0 struct {
		FullName   pref.FullName // must be populated
		ParentFile *File         // must be populated
		Parent     pref.Descriptor
		Index      int
	}
)

func (d *Base) Name() pref.Name         { return d.L0.FullName.Name() }
func (d *Base) FullName() pref.FullName { return d.L0.FullName }
func (d *Base) ParentFile() pref.FileDescriptor {
	if d.L0.ParentFile == SurrogateProto2 || d.L0.ParentFile == SurrogateProto3 {
		return nil // surrogate files are not real parents
	}
	return d.L0.ParentFile
}
func (d *Base) Parent() pref.Descriptor             { return d.L0.Parent }
func (d *Base) Index() int                          { return d.L0.Index }
func (d *Base) Syntax() pref.Syntax                 { return d.L0.ParentFile.Syntax() }
func (d *Base) IsPlaceholder() bool                 { return false }
func (d *Base) ProtoInternal(pragma.DoNotImplement) {}

type stringName struct {
	hasJSON  bool
	once     sync.Once
	nameJSON string
	nameText string
}

// InitJSON initializes the name. It is exported for use by other internal packages.
func (s *stringName) InitJSON(name string) {
	s.hasJSON = true
	s.nameJSON = name
}

func (s *stringName) lazyInit(fd pref.FieldDescriptor) *stringName {
	s.once.Do(func() {
		if fd.IsExtension() {
			// For extensions, JSON and text are formatted the same way.
			var name string
			if messageset.IsMessageSetExtension(fd) {
				name = string("[" + fd.FullName().Parent() + "]")
			} else {
				name = string("[" + fd.FullName() + "]")
			}
			s.nameJSON = name
			s.nameText = name
		} else {
			// Format the JSON name.
			if !s.hasJSON {
				s.nameJSON = strs.JSONCamelCase(string(fd.Name()))
			}

			// Format the text name.
			s.nameText = string(fd.Name())
			if fd.Kind() == pref.GroupKind {
				s.nameText = string(fd.Message().Name())
			}
		}
	})
	return s
}

func (s *stringName) getJSON(fd pref.FieldDescriptor) string { return s.lazyInit(fd).nameJSON }
func (s *stringName) getText(fd pref.FieldDescriptor) string { return s.lazyInit(fd).nameText }

func DefaultValue(v pref.Value, ev pref.EnumValueDescriptor) defaultValue {
	dv := defaultValue{has: v.IsValid(), val: v, enum: ev}
	if b, ok := v.Interface().([]byte); ok {
		// Store a copy of the default bytes, so that we can detect
		// accidental mutations of the original value.
		dv.bytes = append([]byte(nil), b...)
	}
	return dv
}

func unmarshalDefault(b []byte, k pref.Kind, pf *File, ed pref.EnumDescriptor) defaultValue {
	var evs pref.EnumValueDescriptors
	if k == pref.EnumKind {
		// If the enum is declared within the same file, be careful not to
		// blindly call the Values method, lest we bind ourselves in a deadlock.
		if e, ok := ed.(*Enum); ok && e.L0.ParentFile == pf {
			evs = &e.L2.Values
		} else {
			evs = ed.Values()
		}

		// If we are unable to resolve the enum dependency, use a placeholder
		// enum value since we will not be able to parse the default value.
		if ed.IsPlaceholder() && pref.Name(b).IsValid() {
			v := pref.ValueOfEnum(0)
			ev := PlaceholderEnumValue(ed.FullName().Parent().Append(pref.Name(b)))
			return DefaultValue(v, ev)
		}
	}

	v, ev, err := defval.Unmarshal(string(b), k, evs, defval.Descriptor)
	if err != nil {
		panic(err)
	}
	return DefaultValue(v, ev)
}

type defaultValue struct {
	has   bool
	val   pref.Value
	enum  pref.EnumValueDescriptor
	bytes []byte
}

func (dv *defaultValue) get(fd pref.FieldDescriptor) pref.Value {
	// Return the zero value as the default if unpopulated.
	if !dv.has {
		if fd.Cardinality() == pref.Repeated {
			return pref.Value{}
		}
		switch fd.Kind() {
		case pref.BoolKind:
			return pref.ValueOfBool(false)
		case pref.Int32Kind, pref.Sint32Kind, pref.Sfixed32Kind:
			return pref.ValueOfInt32(0)
		case pref.Int64Kind, pref.Sint64Kind, pref.Sfixed64Kind:
			return pref.ValueOfInt64(0)
		case pref.Uint32Kind, pref.Fixed32Kind:
			return pref.ValueOfUint32(0)
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
		case pref.EnumKind:
			if evs := fd.Enum().Values(); evs.Len() > 0 {
				return pref.ValueOfEnum(evs.Get(0).Number())
			}
			return pref.ValueOfEnum(0)
		}
	}

	if len(dv.bytes) > 0 && !bytes.Equal(dv.bytes, dv.val.Bytes()) {
		// TODO: Avoid panic if we're running with the race detector
		// and instead spawn a goroutine that periodically resets
		// this value back to the original to induce a race.
		panic(fmt.Sprintf("detected mutation on the default bytes for %v", fd.FullName()))
	}
	return dv.val
}
