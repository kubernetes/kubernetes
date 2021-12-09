// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filedesc

import (
	"google.golang.org/protobuf/internal/descopts"
	"google.golang.org/protobuf/internal/pragma"
	pref "google.golang.org/protobuf/reflect/protoreflect"
)

var (
	emptyNames           = new(Names)
	emptyEnumRanges      = new(EnumRanges)
	emptyFieldRanges     = new(FieldRanges)
	emptyFieldNumbers    = new(FieldNumbers)
	emptySourceLocations = new(SourceLocations)

	emptyFiles      = new(FileImports)
	emptyMessages   = new(Messages)
	emptyFields     = new(Fields)
	emptyOneofs     = new(Oneofs)
	emptyEnums      = new(Enums)
	emptyEnumValues = new(EnumValues)
	emptyExtensions = new(Extensions)
	emptyServices   = new(Services)
)

// PlaceholderFile is a placeholder, representing only the file path.
type PlaceholderFile string

func (f PlaceholderFile) ParentFile() pref.FileDescriptor       { return f }
func (f PlaceholderFile) Parent() pref.Descriptor               { return nil }
func (f PlaceholderFile) Index() int                            { return 0 }
func (f PlaceholderFile) Syntax() pref.Syntax                   { return 0 }
func (f PlaceholderFile) Name() pref.Name                       { return "" }
func (f PlaceholderFile) FullName() pref.FullName               { return "" }
func (f PlaceholderFile) IsPlaceholder() bool                   { return true }
func (f PlaceholderFile) Options() pref.ProtoMessage            { return descopts.File }
func (f PlaceholderFile) Path() string                          { return string(f) }
func (f PlaceholderFile) Package() pref.FullName                { return "" }
func (f PlaceholderFile) Imports() pref.FileImports             { return emptyFiles }
func (f PlaceholderFile) Messages() pref.MessageDescriptors     { return emptyMessages }
func (f PlaceholderFile) Enums() pref.EnumDescriptors           { return emptyEnums }
func (f PlaceholderFile) Extensions() pref.ExtensionDescriptors { return emptyExtensions }
func (f PlaceholderFile) Services() pref.ServiceDescriptors     { return emptyServices }
func (f PlaceholderFile) SourceLocations() pref.SourceLocations { return emptySourceLocations }
func (f PlaceholderFile) ProtoType(pref.FileDescriptor)         { return }
func (f PlaceholderFile) ProtoInternal(pragma.DoNotImplement)   { return }

// PlaceholderEnum is a placeholder, representing only the full name.
type PlaceholderEnum pref.FullName

func (e PlaceholderEnum) ParentFile() pref.FileDescriptor     { return nil }
func (e PlaceholderEnum) Parent() pref.Descriptor             { return nil }
func (e PlaceholderEnum) Index() int                          { return 0 }
func (e PlaceholderEnum) Syntax() pref.Syntax                 { return 0 }
func (e PlaceholderEnum) Name() pref.Name                     { return pref.FullName(e).Name() }
func (e PlaceholderEnum) FullName() pref.FullName             { return pref.FullName(e) }
func (e PlaceholderEnum) IsPlaceholder() bool                 { return true }
func (e PlaceholderEnum) Options() pref.ProtoMessage          { return descopts.Enum }
func (e PlaceholderEnum) Values() pref.EnumValueDescriptors   { return emptyEnumValues }
func (e PlaceholderEnum) ReservedNames() pref.Names           { return emptyNames }
func (e PlaceholderEnum) ReservedRanges() pref.EnumRanges     { return emptyEnumRanges }
func (e PlaceholderEnum) ProtoType(pref.EnumDescriptor)       { return }
func (e PlaceholderEnum) ProtoInternal(pragma.DoNotImplement) { return }

// PlaceholderEnumValue is a placeholder, representing only the full name.
type PlaceholderEnumValue pref.FullName

func (e PlaceholderEnumValue) ParentFile() pref.FileDescriptor     { return nil }
func (e PlaceholderEnumValue) Parent() pref.Descriptor             { return nil }
func (e PlaceholderEnumValue) Index() int                          { return 0 }
func (e PlaceholderEnumValue) Syntax() pref.Syntax                 { return 0 }
func (e PlaceholderEnumValue) Name() pref.Name                     { return pref.FullName(e).Name() }
func (e PlaceholderEnumValue) FullName() pref.FullName             { return pref.FullName(e) }
func (e PlaceholderEnumValue) IsPlaceholder() bool                 { return true }
func (e PlaceholderEnumValue) Options() pref.ProtoMessage          { return descopts.EnumValue }
func (e PlaceholderEnumValue) Number() pref.EnumNumber             { return 0 }
func (e PlaceholderEnumValue) ProtoType(pref.EnumValueDescriptor)  { return }
func (e PlaceholderEnumValue) ProtoInternal(pragma.DoNotImplement) { return }

// PlaceholderMessage is a placeholder, representing only the full name.
type PlaceholderMessage pref.FullName

func (m PlaceholderMessage) ParentFile() pref.FileDescriptor             { return nil }
func (m PlaceholderMessage) Parent() pref.Descriptor                     { return nil }
func (m PlaceholderMessage) Index() int                                  { return 0 }
func (m PlaceholderMessage) Syntax() pref.Syntax                         { return 0 }
func (m PlaceholderMessage) Name() pref.Name                             { return pref.FullName(m).Name() }
func (m PlaceholderMessage) FullName() pref.FullName                     { return pref.FullName(m) }
func (m PlaceholderMessage) IsPlaceholder() bool                         { return true }
func (m PlaceholderMessage) Options() pref.ProtoMessage                  { return descopts.Message }
func (m PlaceholderMessage) IsMapEntry() bool                            { return false }
func (m PlaceholderMessage) Fields() pref.FieldDescriptors               { return emptyFields }
func (m PlaceholderMessage) Oneofs() pref.OneofDescriptors               { return emptyOneofs }
func (m PlaceholderMessage) ReservedNames() pref.Names                   { return emptyNames }
func (m PlaceholderMessage) ReservedRanges() pref.FieldRanges            { return emptyFieldRanges }
func (m PlaceholderMessage) RequiredNumbers() pref.FieldNumbers          { return emptyFieldNumbers }
func (m PlaceholderMessage) ExtensionRanges() pref.FieldRanges           { return emptyFieldRanges }
func (m PlaceholderMessage) ExtensionRangeOptions(int) pref.ProtoMessage { panic("index out of range") }
func (m PlaceholderMessage) Messages() pref.MessageDescriptors           { return emptyMessages }
func (m PlaceholderMessage) Enums() pref.EnumDescriptors                 { return emptyEnums }
func (m PlaceholderMessage) Extensions() pref.ExtensionDescriptors       { return emptyExtensions }
func (m PlaceholderMessage) ProtoType(pref.MessageDescriptor)            { return }
func (m PlaceholderMessage) ProtoInternal(pragma.DoNotImplement)         { return }
