// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package filedesc provides functionality for constructing descriptors.
package filedesc

import (
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/fieldnum"
	"google.golang.org/protobuf/reflect/protoreflect"
	pref "google.golang.org/protobuf/reflect/protoreflect"
	preg "google.golang.org/protobuf/reflect/protoregistry"
)

// Builder construct a protoreflect.FileDescriptor from the raw descriptor.
type Builder struct {
	// GoPackagePath is the Go package path that is invoking this builder.
	GoPackagePath string

	// RawDescriptor is the wire-encoded bytes of FileDescriptorProto
	// and must be populated.
	RawDescriptor []byte

	// NumEnums is the total number of enums declared in the file.
	NumEnums int32
	// NumMessages is the total number of messages declared in the file.
	// It includes the implicit message declarations for map entries.
	NumMessages int32
	// NumExtensions is the total number of extensions declared in the file.
	NumExtensions int32
	// NumServices is the total number of services declared in the file.
	NumServices int32

	// TypeResolver resolves extension field types for descriptor options.
	// If nil, it uses protoregistry.GlobalTypes.
	TypeResolver interface {
		preg.ExtensionTypeResolver
	}

	// FileRegistry is use to lookup file, enum, and message dependencies.
	// Once constructed, the file descriptor is registered here.
	// If nil, it uses protoregistry.GlobalFiles.
	FileRegistry interface {
		FindFileByPath(string) (protoreflect.FileDescriptor, error)
		FindDescriptorByName(pref.FullName) (pref.Descriptor, error)
		RegisterFile(pref.FileDescriptor) error
	}
}

// resolverByIndex is an interface Builder.FileRegistry may implement.
// If so, it permits looking up an enum or message dependency based on the
// sub-list and element index into filetype.Builder.DependencyIndexes.
type resolverByIndex interface {
	FindEnumByIndex(int32, int32, []Enum, []Message) pref.EnumDescriptor
	FindMessageByIndex(int32, int32, []Enum, []Message) pref.MessageDescriptor
}

// Indexes of each sub-list in filetype.Builder.DependencyIndexes.
const (
	listFieldDeps int32 = iota
	listExtTargets
	listExtDeps
	listMethInDeps
	listMethOutDeps
)

// Out is the output of the Builder.
type Out struct {
	File pref.FileDescriptor

	// Enums is all enum descriptors in "flattened ordering".
	Enums []Enum
	// Messages is all message descriptors in "flattened ordering".
	// It includes the implicit message declarations for map entries.
	Messages []Message
	// Extensions is all extension descriptors in "flattened ordering".
	Extensions []Extension
	// Service is all service descriptors in "flattened ordering".
	Services []Service
}

// Build constructs a FileDescriptor given the parameters set in Builder.
// It assumes that the inputs are well-formed and panics if any inconsistencies
// are encountered.
//
// If NumEnums+NumMessages+NumExtensions+NumServices is zero,
// then Build automatically derives them from the raw descriptor.
func (db Builder) Build() (out Out) {
	// Populate the counts if uninitialized.
	if db.NumEnums+db.NumMessages+db.NumExtensions+db.NumServices == 0 {
		db.unmarshalCounts(db.RawDescriptor, true)
	}

	// Initialize resolvers and registries if unpopulated.
	if db.TypeResolver == nil {
		db.TypeResolver = preg.GlobalTypes
	}
	if db.FileRegistry == nil {
		db.FileRegistry = preg.GlobalFiles
	}

	fd := newRawFile(db)
	out.File = fd
	out.Enums = fd.allEnums
	out.Messages = fd.allMessages
	out.Extensions = fd.allExtensions
	out.Services = fd.allServices

	if err := db.FileRegistry.RegisterFile(fd); err != nil {
		panic(err)
	}
	return out
}

// unmarshalCounts counts the number of enum, message, extension, and service
// declarations in the raw message, which is either a FileDescriptorProto
// or a MessageDescriptorProto depending on whether isFile is set.
func (db *Builder) unmarshalCounts(b []byte, isFile bool) {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			if isFile {
				switch num {
				case fieldnum.FileDescriptorProto_EnumType:
					db.NumEnums++
				case fieldnum.FileDescriptorProto_MessageType:
					db.unmarshalCounts(v, false)
					db.NumMessages++
				case fieldnum.FileDescriptorProto_Extension:
					db.NumExtensions++
				case fieldnum.FileDescriptorProto_Service:
					db.NumServices++
				}
			} else {
				switch num {
				case fieldnum.DescriptorProto_EnumType:
					db.NumEnums++
				case fieldnum.DescriptorProto_NestedType:
					db.unmarshalCounts(v, false)
					db.NumMessages++
				case fieldnum.DescriptorProto_Extension:
					db.NumExtensions++
				}
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
}
