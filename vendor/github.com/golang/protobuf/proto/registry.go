// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io/ioutil"
	"reflect"
	"strings"
	"sync"

	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/runtime/protoimpl"
)

// filePath is the path to the proto source file.
type filePath = string // e.g., "google/protobuf/descriptor.proto"

// fileDescGZIP is the compressed contents of the encoded FileDescriptorProto.
type fileDescGZIP = []byte

var fileCache sync.Map // map[filePath]fileDescGZIP

// RegisterFile is called from generated code to register the compressed
// FileDescriptorProto with the file path for a proto source file.
//
// Deprecated: Use protoregistry.GlobalFiles.RegisterFile instead.
func RegisterFile(s filePath, d fileDescGZIP) {
	// Decompress the descriptor.
	zr, err := gzip.NewReader(bytes.NewReader(d))
	if err != nil {
		panic(fmt.Sprintf("proto: invalid compressed file descriptor: %v", err))
	}
	b, err := ioutil.ReadAll(zr)
	if err != nil {
		panic(fmt.Sprintf("proto: invalid compressed file descriptor: %v", err))
	}

	// Construct a protoreflect.FileDescriptor from the raw descriptor.
	// Note that DescBuilder.Build automatically registers the constructed
	// file descriptor with the v2 registry.
	protoimpl.DescBuilder{RawDescriptor: b}.Build()

	// Locally cache the raw descriptor form for the file.
	fileCache.Store(s, d)
}

// FileDescriptor returns the compressed FileDescriptorProto given the file path
// for a proto source file. It returns nil if not found.
//
// Deprecated: Use protoregistry.GlobalFiles.FindFileByPath instead.
func FileDescriptor(s filePath) fileDescGZIP {
	if v, ok := fileCache.Load(s); ok {
		return v.(fileDescGZIP)
	}

	// Find the descriptor in the v2 registry.
	var b []byte
	if fd, _ := protoregistry.GlobalFiles.FindFileByPath(s); fd != nil {
		b, _ = Marshal(protodesc.ToFileDescriptorProto(fd))
	}

	// Locally cache the raw descriptor form for the file.
	if len(b) > 0 {
		v, _ := fileCache.LoadOrStore(s, protoimpl.X.CompressGZIP(b))
		return v.(fileDescGZIP)
	}
	return nil
}

// enumName is the name of an enum. For historical reasons, the enum name is
// neither the full Go name nor the full protobuf name of the enum.
// The name is the dot-separated combination of just the proto package that the
// enum is declared within followed by the Go type name of the generated enum.
type enumName = string // e.g., "my.proto.package.GoMessage_GoEnum"

// enumsByName maps enum values by name to their numeric counterpart.
type enumsByName = map[string]int32

// enumsByNumber maps enum values by number to their name counterpart.
type enumsByNumber = map[int32]string

var enumCache sync.Map     // map[enumName]enumsByName
var numFilesCache sync.Map // map[protoreflect.FullName]int

// RegisterEnum is called from the generated code to register the mapping of
// enum value names to enum numbers for the enum identified by s.
//
// Deprecated: Use protoregistry.GlobalTypes.RegisterEnum instead.
func RegisterEnum(s enumName, _ enumsByNumber, m enumsByName) {
	if _, ok := enumCache.Load(s); ok {
		panic("proto: duplicate enum registered: " + s)
	}
	enumCache.Store(s, m)

	// This does not forward registration to the v2 registry since this API
	// lacks sufficient information to construct a complete v2 enum descriptor.
}

// EnumValueMap returns the mapping from enum value names to enum numbers for
// the enum of the given name. It returns nil if not found.
//
// Deprecated: Use protoregistry.GlobalTypes.FindEnumByName instead.
func EnumValueMap(s enumName) enumsByName {
	if v, ok := enumCache.Load(s); ok {
		return v.(enumsByName)
	}

	// Check whether the cache is stale. If the number of files in the current
	// package differs, then it means that some enums may have been recently
	// registered upstream that we do not know about.
	var protoPkg protoreflect.FullName
	if i := strings.LastIndexByte(s, '.'); i >= 0 {
		protoPkg = protoreflect.FullName(s[:i])
	}
	v, _ := numFilesCache.Load(protoPkg)
	numFiles, _ := v.(int)
	if protoregistry.GlobalFiles.NumFilesByPackage(protoPkg) == numFiles {
		return nil // cache is up-to-date; was not found earlier
	}

	// Update the enum cache for all enums declared in the given proto package.
	numFiles = 0
	protoregistry.GlobalFiles.RangeFilesByPackage(protoPkg, func(fd protoreflect.FileDescriptor) bool {
		walkEnums(fd, func(ed protoreflect.EnumDescriptor) {
			name := protoimpl.X.LegacyEnumName(ed)
			if _, ok := enumCache.Load(name); !ok {
				m := make(enumsByName)
				evs := ed.Values()
				for i := evs.Len() - 1; i >= 0; i-- {
					ev := evs.Get(i)
					m[string(ev.Name())] = int32(ev.Number())
				}
				enumCache.LoadOrStore(name, m)
			}
		})
		numFiles++
		return true
	})
	numFilesCache.Store(protoPkg, numFiles)

	// Check cache again for enum map.
	if v, ok := enumCache.Load(s); ok {
		return v.(enumsByName)
	}
	return nil
}

// walkEnums recursively walks all enums declared in d.
func walkEnums(d interface {
	Enums() protoreflect.EnumDescriptors
	Messages() protoreflect.MessageDescriptors
}, f func(protoreflect.EnumDescriptor)) {
	eds := d.Enums()
	for i := eds.Len() - 1; i >= 0; i-- {
		f(eds.Get(i))
	}
	mds := d.Messages()
	for i := mds.Len() - 1; i >= 0; i-- {
		walkEnums(mds.Get(i), f)
	}
}

// messageName is the full name of protobuf message.
type messageName = string

var messageTypeCache sync.Map // map[messageName]reflect.Type

// RegisterType is called from generated code to register the message Go type
// for a message of the given name.
//
// Deprecated: Use protoregistry.GlobalTypes.RegisterMessage instead.
func RegisterType(m Message, s messageName) {
	mt := protoimpl.X.LegacyMessageTypeOf(m, protoreflect.FullName(s))
	if err := protoregistry.GlobalTypes.RegisterMessage(mt); err != nil {
		panic(err)
	}
	messageTypeCache.Store(s, reflect.TypeOf(m))
}

// RegisterMapType is called from generated code to register the Go map type
// for a protobuf message representing a map entry.
//
// Deprecated: Do not use.
func RegisterMapType(m interface{}, s messageName) {
	t := reflect.TypeOf(m)
	if t.Kind() != reflect.Map {
		panic(fmt.Sprintf("invalid map kind: %v", t))
	}
	if _, ok := messageTypeCache.Load(s); ok {
		panic(fmt.Errorf("proto: duplicate proto message registered: %s", s))
	}
	messageTypeCache.Store(s, t)
}

// MessageType returns the message type for a named message.
// It returns nil if not found.
//
// Deprecated: Use protoregistry.GlobalTypes.FindMessageByName instead.
func MessageType(s messageName) reflect.Type {
	if v, ok := messageTypeCache.Load(s); ok {
		return v.(reflect.Type)
	}

	// Derive the message type from the v2 registry.
	var t reflect.Type
	if mt, _ := protoregistry.GlobalTypes.FindMessageByName(protoreflect.FullName(s)); mt != nil {
		t = messageGoType(mt)
	}

	// If we could not get a concrete type, it is possible that it is a
	// pseudo-message for a map entry.
	if t == nil {
		d, _ := protoregistry.GlobalFiles.FindDescriptorByName(protoreflect.FullName(s))
		if md, _ := d.(protoreflect.MessageDescriptor); md != nil && md.IsMapEntry() {
			kt := goTypeForField(md.Fields().ByNumber(1))
			vt := goTypeForField(md.Fields().ByNumber(2))
			t = reflect.MapOf(kt, vt)
		}
	}

	// Locally cache the message type for the given name.
	if t != nil {
		v, _ := messageTypeCache.LoadOrStore(s, t)
		return v.(reflect.Type)
	}
	return nil
}

func goTypeForField(fd protoreflect.FieldDescriptor) reflect.Type {
	switch k := fd.Kind(); k {
	case protoreflect.EnumKind:
		if et, _ := protoregistry.GlobalTypes.FindEnumByName(fd.Enum().FullName()); et != nil {
			return enumGoType(et)
		}
		return reflect.TypeOf(protoreflect.EnumNumber(0))
	case protoreflect.MessageKind, protoreflect.GroupKind:
		if mt, _ := protoregistry.GlobalTypes.FindMessageByName(fd.Message().FullName()); mt != nil {
			return messageGoType(mt)
		}
		return reflect.TypeOf((*protoreflect.Message)(nil)).Elem()
	default:
		return reflect.TypeOf(fd.Default().Interface())
	}
}

func enumGoType(et protoreflect.EnumType) reflect.Type {
	return reflect.TypeOf(et.New(0))
}

func messageGoType(mt protoreflect.MessageType) reflect.Type {
	return reflect.TypeOf(MessageV1(mt.Zero().Interface()))
}

// MessageName returns the full protobuf name for the given message type.
//
// Deprecated: Use protoreflect.MessageDescriptor.FullName instead.
func MessageName(m Message) messageName {
	if m == nil {
		return ""
	}
	if m, ok := m.(interface{ XXX_MessageName() messageName }); ok {
		return m.XXX_MessageName()
	}
	return messageName(protoimpl.X.MessageDescriptorOf(m).FullName())
}

// RegisterExtension is called from the generated code to register
// the extension descriptor.
//
// Deprecated: Use protoregistry.GlobalTypes.RegisterExtension instead.
func RegisterExtension(d *ExtensionDesc) {
	if err := protoregistry.GlobalTypes.RegisterExtension(d); err != nil {
		panic(err)
	}
}

type extensionsByNumber = map[int32]*ExtensionDesc

var extensionCache sync.Map // map[messageName]extensionsByNumber

// RegisteredExtensions returns a map of the registered extensions for the
// provided protobuf message, indexed by the extension field number.
//
// Deprecated: Use protoregistry.GlobalTypes.RangeExtensionsByMessage instead.
func RegisteredExtensions(m Message) extensionsByNumber {
	// Check whether the cache is stale. If the number of extensions for
	// the given message differs, then it means that some extensions were
	// recently registered upstream that we do not know about.
	s := MessageName(m)
	v, _ := extensionCache.Load(s)
	xs, _ := v.(extensionsByNumber)
	if protoregistry.GlobalTypes.NumExtensionsByMessage(protoreflect.FullName(s)) == len(xs) {
		return xs // cache is up-to-date
	}

	// Cache is stale, re-compute the extensions map.
	xs = make(extensionsByNumber)
	protoregistry.GlobalTypes.RangeExtensionsByMessage(protoreflect.FullName(s), func(xt protoreflect.ExtensionType) bool {
		if xd, ok := xt.(*ExtensionDesc); ok {
			xs[int32(xt.TypeDescriptor().Number())] = xd
		} else {
			// TODO: This implies that the protoreflect.ExtensionType is a
			// custom type not generated by protoc-gen-go. We could try and
			// convert the type to an ExtensionDesc.
		}
		return true
	})
	extensionCache.Store(s, xs)
	return xs
}
