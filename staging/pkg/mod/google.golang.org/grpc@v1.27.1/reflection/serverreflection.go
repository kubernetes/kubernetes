/*
 *
 * Copyright 2016 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

//go:generate protoc --go_out=plugins=grpc:. grpc_reflection_v1alpha/reflection.proto

/*
Package reflection implements server reflection service.

The service implemented is defined in:
https://github.com/grpc/grpc/blob/master/src/proto/grpc/reflection/v1alpha/reflection.proto.

To register server reflection on a gRPC server:
	import "google.golang.org/grpc/reflection"

	s := grpc.NewServer()
	pb.RegisterYourOwnServer(s, &server{})

	// Register reflection service on gRPC server.
	reflection.Register(s)

	s.Serve(lis)

*/
package reflection // import "google.golang.org/grpc/reflection"

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"io/ioutil"
	"reflect"
	"sort"
	"sync"

	"github.com/golang/protobuf/proto"
	dpb "github.com/golang/protobuf/protoc-gen-go/descriptor"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	rpb "google.golang.org/grpc/reflection/grpc_reflection_v1alpha"
	"google.golang.org/grpc/status"
)

type serverReflectionServer struct {
	s *grpc.Server

	initSymbols  sync.Once
	serviceNames []string
	symbols      map[string]*dpb.FileDescriptorProto // map of fully-qualified names to files
}

// Register registers the server reflection service on the given gRPC server.
func Register(s *grpc.Server) {
	rpb.RegisterServerReflectionServer(s, &serverReflectionServer{
		s: s,
	})
}

// protoMessage is used for type assertion on proto messages.
// Generated proto message implements function Descriptor(), but Descriptor()
// is not part of interface proto.Message. This interface is needed to
// call Descriptor().
type protoMessage interface {
	Descriptor() ([]byte, []int)
}

func (s *serverReflectionServer) getSymbols() (svcNames []string, symbolIndex map[string]*dpb.FileDescriptorProto) {
	s.initSymbols.Do(func() {
		serviceInfo := s.s.GetServiceInfo()

		s.symbols = map[string]*dpb.FileDescriptorProto{}
		s.serviceNames = make([]string, 0, len(serviceInfo))
		processed := map[string]struct{}{}
		for svc, info := range serviceInfo {
			s.serviceNames = append(s.serviceNames, svc)
			fdenc, ok := parseMetadata(info.Metadata)
			if !ok {
				continue
			}
			fd, err := decodeFileDesc(fdenc)
			if err != nil {
				continue
			}
			s.processFile(fd, processed)
		}
		sort.Strings(s.serviceNames)
	})

	return s.serviceNames, s.symbols
}

func (s *serverReflectionServer) processFile(fd *dpb.FileDescriptorProto, processed map[string]struct{}) {
	filename := fd.GetName()
	if _, ok := processed[filename]; ok {
		return
	}
	processed[filename] = struct{}{}

	prefix := fd.GetPackage()

	for _, msg := range fd.MessageType {
		s.processMessage(fd, prefix, msg)
	}
	for _, en := range fd.EnumType {
		s.processEnum(fd, prefix, en)
	}
	for _, ext := range fd.Extension {
		s.processField(fd, prefix, ext)
	}
	for _, svc := range fd.Service {
		svcName := fqn(prefix, svc.GetName())
		s.symbols[svcName] = fd
		for _, meth := range svc.Method {
			name := fqn(svcName, meth.GetName())
			s.symbols[name] = fd
		}
	}

	for _, dep := range fd.Dependency {
		fdenc := proto.FileDescriptor(dep)
		fdDep, err := decodeFileDesc(fdenc)
		if err != nil {
			continue
		}
		s.processFile(fdDep, processed)
	}
}

func (s *serverReflectionServer) processMessage(fd *dpb.FileDescriptorProto, prefix string, msg *dpb.DescriptorProto) {
	msgName := fqn(prefix, msg.GetName())
	s.symbols[msgName] = fd

	for _, nested := range msg.NestedType {
		s.processMessage(fd, msgName, nested)
	}
	for _, en := range msg.EnumType {
		s.processEnum(fd, msgName, en)
	}
	for _, ext := range msg.Extension {
		s.processField(fd, msgName, ext)
	}
	for _, fld := range msg.Field {
		s.processField(fd, msgName, fld)
	}
	for _, oneof := range msg.OneofDecl {
		oneofName := fqn(msgName, oneof.GetName())
		s.symbols[oneofName] = fd
	}
}

func (s *serverReflectionServer) processEnum(fd *dpb.FileDescriptorProto, prefix string, en *dpb.EnumDescriptorProto) {
	enName := fqn(prefix, en.GetName())
	s.symbols[enName] = fd

	for _, val := range en.Value {
		valName := fqn(enName, val.GetName())
		s.symbols[valName] = fd
	}
}

func (s *serverReflectionServer) processField(fd *dpb.FileDescriptorProto, prefix string, fld *dpb.FieldDescriptorProto) {
	fldName := fqn(prefix, fld.GetName())
	s.symbols[fldName] = fd
}

func fqn(prefix, name string) string {
	if prefix == "" {
		return name
	}
	return prefix + "." + name
}

// fileDescForType gets the file descriptor for the given type.
// The given type should be a proto message.
func (s *serverReflectionServer) fileDescForType(st reflect.Type) (*dpb.FileDescriptorProto, error) {
	m, ok := reflect.Zero(reflect.PtrTo(st)).Interface().(protoMessage)
	if !ok {
		return nil, fmt.Errorf("failed to create message from type: %v", st)
	}
	enc, _ := m.Descriptor()

	return decodeFileDesc(enc)
}

// decodeFileDesc does decompression and unmarshalling on the given
// file descriptor byte slice.
func decodeFileDesc(enc []byte) (*dpb.FileDescriptorProto, error) {
	raw, err := decompress(enc)
	if err != nil {
		return nil, fmt.Errorf("failed to decompress enc: %v", err)
	}

	fd := new(dpb.FileDescriptorProto)
	if err := proto.Unmarshal(raw, fd); err != nil {
		return nil, fmt.Errorf("bad descriptor: %v", err)
	}
	return fd, nil
}

// decompress does gzip decompression.
func decompress(b []byte) ([]byte, error) {
	r, err := gzip.NewReader(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("bad gzipped descriptor: %v", err)
	}
	out, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("bad gzipped descriptor: %v", err)
	}
	return out, nil
}

func typeForName(name string) (reflect.Type, error) {
	pt := proto.MessageType(name)
	if pt == nil {
		return nil, fmt.Errorf("unknown type: %q", name)
	}
	st := pt.Elem()

	return st, nil
}

func fileDescContainingExtension(st reflect.Type, ext int32) (*dpb.FileDescriptorProto, error) {
	m, ok := reflect.Zero(reflect.PtrTo(st)).Interface().(proto.Message)
	if !ok {
		return nil, fmt.Errorf("failed to create message from type: %v", st)
	}

	var extDesc *proto.ExtensionDesc
	for id, desc := range proto.RegisteredExtensions(m) {
		if id == ext {
			extDesc = desc
			break
		}
	}

	if extDesc == nil {
		return nil, fmt.Errorf("failed to find registered extension for extension number %v", ext)
	}

	return decodeFileDesc(proto.FileDescriptor(extDesc.Filename))
}

func (s *serverReflectionServer) allExtensionNumbersForType(st reflect.Type) ([]int32, error) {
	m, ok := reflect.Zero(reflect.PtrTo(st)).Interface().(proto.Message)
	if !ok {
		return nil, fmt.Errorf("failed to create message from type: %v", st)
	}

	exts := proto.RegisteredExtensions(m)
	out := make([]int32, 0, len(exts))
	for id := range exts {
		out = append(out, id)
	}
	return out, nil
}

// fileDescEncodingByFilename finds the file descriptor for given filename,
// does marshalling on it and returns the marshalled result.
func (s *serverReflectionServer) fileDescEncodingByFilename(name string) ([]byte, error) {
	enc := proto.FileDescriptor(name)
	if enc == nil {
		return nil, fmt.Errorf("unknown file: %v", name)
	}
	fd, err := decodeFileDesc(enc)
	if err != nil {
		return nil, err
	}
	return proto.Marshal(fd)
}

// parseMetadata finds the file descriptor bytes specified meta.
// For SupportPackageIsVersion4, m is the name of the proto file, we
// call proto.FileDescriptor to get the byte slice.
// For SupportPackageIsVersion3, m is a byte slice itself.
func parseMetadata(meta interface{}) ([]byte, bool) {
	// Check if meta is the file name.
	if fileNameForMeta, ok := meta.(string); ok {
		return proto.FileDescriptor(fileNameForMeta), true
	}

	// Check if meta is the byte slice.
	if enc, ok := meta.([]byte); ok {
		return enc, true
	}

	return nil, false
}

// fileDescEncodingContainingSymbol finds the file descriptor containing the given symbol,
// does marshalling on it and returns the marshalled result.
// The given symbol can be a type, a service or a method.
func (s *serverReflectionServer) fileDescEncodingContainingSymbol(name string) ([]byte, error) {
	_, symbols := s.getSymbols()
	fd := symbols[name]
	if fd == nil {
		// Check if it's a type name that was not present in the
		// transitive dependencies of the registered services.
		if st, err := typeForName(name); err == nil {
			fd, err = s.fileDescForType(st)
			if err != nil {
				return nil, err
			}
		}
	}

	if fd == nil {
		return nil, fmt.Errorf("unknown symbol: %v", name)
	}

	return proto.Marshal(fd)
}

// fileDescEncodingContainingExtension finds the file descriptor containing given extension,
// does marshalling on it and returns the marshalled result.
func (s *serverReflectionServer) fileDescEncodingContainingExtension(typeName string, extNum int32) ([]byte, error) {
	st, err := typeForName(typeName)
	if err != nil {
		return nil, err
	}
	fd, err := fileDescContainingExtension(st, extNum)
	if err != nil {
		return nil, err
	}
	return proto.Marshal(fd)
}

// allExtensionNumbersForTypeName returns all extension numbers for the given type.
func (s *serverReflectionServer) allExtensionNumbersForTypeName(name string) ([]int32, error) {
	st, err := typeForName(name)
	if err != nil {
		return nil, err
	}
	extNums, err := s.allExtensionNumbersForType(st)
	if err != nil {
		return nil, err
	}
	return extNums, nil
}

// ServerReflectionInfo is the reflection service handler.
func (s *serverReflectionServer) ServerReflectionInfo(stream rpb.ServerReflection_ServerReflectionInfoServer) error {
	for {
		in, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		out := &rpb.ServerReflectionResponse{
			ValidHost:       in.Host,
			OriginalRequest: in,
		}
		switch req := in.MessageRequest.(type) {
		case *rpb.ServerReflectionRequest_FileByFilename:
			b, err := s.fileDescEncodingByFilename(req.FileByFilename)
			if err != nil {
				out.MessageResponse = &rpb.ServerReflectionResponse_ErrorResponse{
					ErrorResponse: &rpb.ErrorResponse{
						ErrorCode:    int32(codes.NotFound),
						ErrorMessage: err.Error(),
					},
				}
			} else {
				out.MessageResponse = &rpb.ServerReflectionResponse_FileDescriptorResponse{
					FileDescriptorResponse: &rpb.FileDescriptorResponse{FileDescriptorProto: [][]byte{b}},
				}
			}
		case *rpb.ServerReflectionRequest_FileContainingSymbol:
			b, err := s.fileDescEncodingContainingSymbol(req.FileContainingSymbol)
			if err != nil {
				out.MessageResponse = &rpb.ServerReflectionResponse_ErrorResponse{
					ErrorResponse: &rpb.ErrorResponse{
						ErrorCode:    int32(codes.NotFound),
						ErrorMessage: err.Error(),
					},
				}
			} else {
				out.MessageResponse = &rpb.ServerReflectionResponse_FileDescriptorResponse{
					FileDescriptorResponse: &rpb.FileDescriptorResponse{FileDescriptorProto: [][]byte{b}},
				}
			}
		case *rpb.ServerReflectionRequest_FileContainingExtension:
			typeName := req.FileContainingExtension.ContainingType
			extNum := req.FileContainingExtension.ExtensionNumber
			b, err := s.fileDescEncodingContainingExtension(typeName, extNum)
			if err != nil {
				out.MessageResponse = &rpb.ServerReflectionResponse_ErrorResponse{
					ErrorResponse: &rpb.ErrorResponse{
						ErrorCode:    int32(codes.NotFound),
						ErrorMessage: err.Error(),
					},
				}
			} else {
				out.MessageResponse = &rpb.ServerReflectionResponse_FileDescriptorResponse{
					FileDescriptorResponse: &rpb.FileDescriptorResponse{FileDescriptorProto: [][]byte{b}},
				}
			}
		case *rpb.ServerReflectionRequest_AllExtensionNumbersOfType:
			extNums, err := s.allExtensionNumbersForTypeName(req.AllExtensionNumbersOfType)
			if err != nil {
				out.MessageResponse = &rpb.ServerReflectionResponse_ErrorResponse{
					ErrorResponse: &rpb.ErrorResponse{
						ErrorCode:    int32(codes.NotFound),
						ErrorMessage: err.Error(),
					},
				}
			} else {
				out.MessageResponse = &rpb.ServerReflectionResponse_AllExtensionNumbersResponse{
					AllExtensionNumbersResponse: &rpb.ExtensionNumberResponse{
						BaseTypeName:    req.AllExtensionNumbersOfType,
						ExtensionNumber: extNums,
					},
				}
			}
		case *rpb.ServerReflectionRequest_ListServices:
			svcNames, _ := s.getSymbols()
			serviceResponses := make([]*rpb.ServiceResponse, len(svcNames))
			for i, n := range svcNames {
				serviceResponses[i] = &rpb.ServiceResponse{
					Name: n,
				}
			}
			out.MessageResponse = &rpb.ServerReflectionResponse_ListServicesResponse{
				ListServicesResponse: &rpb.ListServiceResponse{
					Service: serviceResponses,
				},
			}
		default:
			return status.Errorf(codes.InvalidArgument, "invalid MessageRequest: %v", in.MessageRequest)
		}

		if err := stream.Send(out); err != nil {
			return err
		}
	}
}
