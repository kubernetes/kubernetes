// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package pb reflects over protocol buffer descriptors to generate objects
// that simplify type, enum, and field lookup.
package pb

import (
	"fmt"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"

	anypb "google.golang.org/protobuf/types/known/anypb"
	durpb "google.golang.org/protobuf/types/known/durationpb"
	emptypb "google.golang.org/protobuf/types/known/emptypb"
	structpb "google.golang.org/protobuf/types/known/structpb"
	tspb "google.golang.org/protobuf/types/known/timestamppb"
	wrapperspb "google.golang.org/protobuf/types/known/wrapperspb"
)

// Db maps from file / message / enum name to file description.
//
// Each Db is isolated from each other, and while information about protobuf descriptors may be
// fetched from the global protobuf registry, no descriptors are added to this registry, else
// the isolation guarantees of the Db object would be violated.
type Db struct {
	revFileDescriptorMap map[string]*FileDescription
	// files contains the deduped set of FileDescriptions whose types are contained in the pb.Db.
	files []*FileDescription
	// extensions contains the mapping between a given type name, extension name and its FieldDescription
	extensions map[string]map[string]*FieldDescription
}

// extensionsMap is a type alias to a map[typeName]map[extensionName]*FieldDescription
type extensionMap = map[string]map[string]*FieldDescription

var (
	// DefaultDb used at evaluation time or unless overridden at check time.
	DefaultDb = &Db{
		revFileDescriptorMap: make(map[string]*FileDescription),
		files:                []*FileDescription{},
		extensions:           make(extensionMap),
	}
)

// Merge will copy the source proto message into the destination, or error if the merge cannot be completed.
//
// Unlike the proto.Merge, this method will fallback to proto.Marshal/Unmarshal of the two proto messages do not
// share the same instance of their type descriptor.
func Merge(dstPB, srcPB proto.Message) error {
	src, dst := srcPB.ProtoReflect(), dstPB.ProtoReflect()
	if src.Descriptor() == dst.Descriptor() {
		proto.Merge(dstPB, srcPB)
		return nil
	}
	if src.Descriptor().FullName() != dst.Descriptor().FullName() {
		return fmt.Errorf("pb.Merge() arguments must be the same type. got: %v, %v",
			dst.Descriptor().FullName(), src.Descriptor().FullName())
	}
	bytes, err := proto.Marshal(srcPB)
	if err != nil {
		return fmt.Errorf("pb.Merge(dstPB, srcPB) failed to marshal source proto: %v", err)
	}
	err = proto.Unmarshal(bytes, dstPB)
	if err != nil {
		return fmt.Errorf("pb.Merge(dstPB, srcPB) failed to unmarshal to dest proto: %v", err)
	}
	return nil
}

// NewDb creates a new `pb.Db` with an empty type name to file description map.
func NewDb() *Db {
	pbdb := &Db{
		revFileDescriptorMap: make(map[string]*FileDescription),
		files:                []*FileDescription{},
		extensions:           make(extensionMap),
	}
	// The FileDescription objects in the default db contain lazily initialized TypeDescription
	// values which may point to the state contained in the DefaultDb irrespective of this shallow
	// copy; however, the type graph for a field is idempotently computed, and is guaranteed to
	// only be initialized once thanks to atomic values within the TypeDescription objects, so it
	// is safe to share these values across instances.
	for k, v := range DefaultDb.revFileDescriptorMap {
		pbdb.revFileDescriptorMap[k] = v
	}
	pbdb.files = append(pbdb.files, DefaultDb.files...)
	return pbdb
}

// Copy creates a copy of the current database with its own internal descriptor mapping.
func (pbdb *Db) Copy() *Db {
	copy := NewDb()
	for _, fd := range pbdb.files {
		hasFile := false
		for _, fd2 := range copy.files {
			if fd2 == fd {
				hasFile = true
			}
		}
		if !hasFile {
			fd = fd.Copy(copy)
			copy.files = append(copy.files, fd)
		}
		for _, enumValName := range fd.GetEnumNames() {
			copy.revFileDescriptorMap[enumValName] = fd
		}
		for _, msgTypeName := range fd.GetTypeNames() {
			copy.revFileDescriptorMap[msgTypeName] = fd
		}
		copy.revFileDescriptorMap[fd.GetName()] = fd
	}
	for typeName, extFieldMap := range pbdb.extensions {
		copyExtFieldMap, found := copy.extensions[typeName]
		if !found {
			copyExtFieldMap = make(map[string]*FieldDescription, len(extFieldMap))
		}
		for extFieldName, fd := range extFieldMap {
			copyExtFieldMap[extFieldName] = fd
		}
		copy.extensions[typeName] = copyExtFieldMap
	}
	return copy
}

// FileDescriptions returns the set of file descriptions associated with this db.
func (pbdb *Db) FileDescriptions() []*FileDescription {
	return pbdb.files
}

// RegisterDescriptor produces a `FileDescription` from a `FileDescriptor` and registers the
// message and enum types into the `pb.Db`.
func (pbdb *Db) RegisterDescriptor(fileDesc protoreflect.FileDescriptor) (*FileDescription, error) {
	fd, found := pbdb.revFileDescriptorMap[fileDesc.Path()]
	if found {
		return fd, nil
	}
	// Make sure to search the global registry to see if a protoreflect.FileDescriptor for
	// the file specified has been linked into the binary. If so, use the copy of the descriptor
	// from the global cache.
	//
	// Note: Proto reflection relies on descriptor values being object equal rather than object
	// equivalence. This choice means that a FieldDescriptor generated from a FileDescriptorProto
	// will be incompatible with the FieldDescriptor in the global registry and any message created
	// from that global registry.
	globalFD, err := protoregistry.GlobalFiles.FindFileByPath(fileDesc.Path())
	if err == nil {
		fileDesc = globalFD
	}
	var fileExtMap extensionMap
	fd, fileExtMap = newFileDescription(fileDesc, pbdb)
	for _, enumValName := range fd.GetEnumNames() {
		pbdb.revFileDescriptorMap[enumValName] = fd
	}
	for _, msgTypeName := range fd.GetTypeNames() {
		pbdb.revFileDescriptorMap[msgTypeName] = fd
	}
	pbdb.revFileDescriptorMap[fd.GetName()] = fd

	// Return the specific file descriptor registered.
	pbdb.files = append(pbdb.files, fd)

	// Index the protobuf message extensions from the file into the pbdb
	for typeName, extMap := range fileExtMap {
		typeExtMap, found := pbdb.extensions[typeName]
		if !found {
			pbdb.extensions[typeName] = extMap
			continue
		}
		for extName, field := range extMap {
			typeExtMap[extName] = field
		}
	}
	return fd, nil
}

// RegisterMessage produces a `FileDescription` from a `message` and registers the message and all
// other definitions within the message file into the `pb.Db`.
func (pbdb *Db) RegisterMessage(message proto.Message) (*FileDescription, error) {
	msgDesc := message.ProtoReflect().Descriptor()
	msgName := msgDesc.FullName()
	typeName := sanitizeProtoName(string(msgName))
	if fd, found := pbdb.revFileDescriptorMap[typeName]; found {
		return fd, nil
	}
	return pbdb.RegisterDescriptor(msgDesc.ParentFile())
}

// DescribeEnum takes a qualified enum name and returns an `EnumDescription` if it exists in the
// `pb.Db`.
func (pbdb *Db) DescribeEnum(enumName string) (*EnumValueDescription, bool) {
	enumName = sanitizeProtoName(enumName)
	if fd, found := pbdb.revFileDescriptorMap[enumName]; found {
		return fd.GetEnumDescription(enumName)
	}
	return nil, false
}

// DescribeType returns a `TypeDescription` for the `typeName` if it exists in the `pb.Db`.
func (pbdb *Db) DescribeType(typeName string) (*TypeDescription, bool) {
	typeName = sanitizeProtoName(typeName)
	if fd, found := pbdb.revFileDescriptorMap[typeName]; found {
		return fd.GetTypeDescription(typeName)
	}
	return nil, false
}

// CollectFileDescriptorSet builds a file descriptor set associated with the file where the input
// message is declared.
func CollectFileDescriptorSet(message proto.Message) map[string]protoreflect.FileDescriptor {
	fdMap := map[string]protoreflect.FileDescriptor{}
	parentFile := message.ProtoReflect().Descriptor().ParentFile()
	fdMap[parentFile.Path()] = parentFile
	// Initialize list of dependencies
	deps := make([]protoreflect.FileImport, parentFile.Imports().Len())
	for i := 0; i < parentFile.Imports().Len(); i++ {
		deps[i] = parentFile.Imports().Get(i)
	}
	// Expand list for new dependencies
	for i := 0; i < len(deps); i++ {
		dep := deps[i]
		if _, found := fdMap[dep.Path()]; found {
			continue
		}
		fdMap[dep.Path()] = dep.FileDescriptor
		for j := 0; j < dep.FileDescriptor.Imports().Len(); j++ {
			deps = append(deps, dep.FileDescriptor.Imports().Get(j))
		}
	}
	return fdMap
}

func init() {
	// Describe well-known types to ensure they can always be resolved by the check and interpret
	// execution phases.
	//
	// The following subset of message types is enough to ensure that all well-known types can
	// resolved in the runtime, since describing the value results in describing the whole file
	// where the message is declared.
	DefaultDb.RegisterMessage(&anypb.Any{})
	DefaultDb.RegisterMessage(&durpb.Duration{})
	DefaultDb.RegisterMessage(&emptypb.Empty{})
	DefaultDb.RegisterMessage(&tspb.Timestamp{})
	DefaultDb.RegisterMessage(&structpb.Value{})
	DefaultDb.RegisterMessage(&wrapperspb.BoolValue{})
}
