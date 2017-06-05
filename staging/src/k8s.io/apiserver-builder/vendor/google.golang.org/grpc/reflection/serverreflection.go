/*
 *
 * Copyright 2016, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

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
	"strings"

	"github.com/golang/protobuf/proto"
	dpb "github.com/golang/protobuf/protoc-gen-go/descriptor"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	rpb "google.golang.org/grpc/reflection/grpc_reflection_v1alpha"
)

type serverReflectionServer struct {
	s *grpc.Server
	// TODO add more cache if necessary
	serviceInfo map[string]grpc.ServiceInfo // cache for s.GetServiceInfo()
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

// fileDescForType gets the file descriptor for the given type.
// The given type should be a proto message.
func (s *serverReflectionServer) fileDescForType(st reflect.Type) (*dpb.FileDescriptorProto, error) {
	m, ok := reflect.Zero(reflect.PtrTo(st)).Interface().(protoMessage)
	if !ok {
		return nil, fmt.Errorf("failed to create message from type: %v", st)
	}
	enc, _ := m.Descriptor()

	return s.decodeFileDesc(enc)
}

// decodeFileDesc does decompression and unmarshalling on the given
// file descriptor byte slice.
func (s *serverReflectionServer) decodeFileDesc(enc []byte) (*dpb.FileDescriptorProto, error) {
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
		return nil, fmt.Errorf("bad gzipped descriptor: %v\n", err)
	}
	out, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("bad gzipped descriptor: %v\n", err)
	}
	return out, nil
}

func (s *serverReflectionServer) typeForName(name string) (reflect.Type, error) {
	pt := proto.MessageType(name)
	if pt == nil {
		return nil, fmt.Errorf("unknown type: %q", name)
	}
	st := pt.Elem()

	return st, nil
}

func (s *serverReflectionServer) fileDescContainingExtension(st reflect.Type, ext int32) (*dpb.FileDescriptorProto, error) {
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

	extT := reflect.TypeOf(extDesc.ExtensionType).Elem()

	return s.fileDescForType(extT)
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
	fd, err := s.decodeFileDesc(enc)
	if err != nil {
		return nil, err
	}
	return proto.Marshal(fd)
}

// serviceMetadataForSymbol finds the metadata for name in s.serviceInfo.
// name should be a service name or a method name.
func (s *serverReflectionServer) serviceMetadataForSymbol(name string) (interface{}, error) {
	if s.serviceInfo == nil {
		s.serviceInfo = s.s.GetServiceInfo()
	}

	// Check if it's a service name.
	if info, ok := s.serviceInfo[name]; ok {
		return info.Metadata, nil
	}

	// Check if it's a method name.
	pos := strings.LastIndex(name, ".")
	// Not a valid method name.
	if pos == -1 {
		return nil, fmt.Errorf("unknown symbol: %v", name)
	}

	info, ok := s.serviceInfo[name[:pos]]
	// Substring before last "." is not a service name.
	if !ok {
		return nil, fmt.Errorf("unknown symbol: %v", name)
	}

	// Search the method name in info.Methods.
	var found bool
	for _, m := range info.Methods {
		if m.Name == name[pos+1:] {
			found = true
			break
		}
	}
	if found {
		return info.Metadata, nil
	}

	return nil, fmt.Errorf("unknown symbol: %v", name)
}

// fileDescEncodingContainingSymbol finds the file descriptor containing the given symbol,
// does marshalling on it and returns the marshalled result.
// The given symbol can be a type, a service or a method.
func (s *serverReflectionServer) fileDescEncodingContainingSymbol(name string) ([]byte, error) {
	var (
		fd *dpb.FileDescriptorProto
	)
	// Check if it's a type name.
	if st, err := s.typeForName(name); err == nil {
		fd, err = s.fileDescForType(st)
		if err != nil {
			return nil, err
		}
	} else { // Check if it's a service name or a method name.
		meta, err := s.serviceMetadataForSymbol(name)

		// Metadata not found.
		if err != nil {
			return nil, err
		}

		// Metadata not valid.
		enc, ok := meta.([]byte)
		if !ok {
			return nil, fmt.Errorf("invalid file descriptor for symbol: %v", name)
		}

		fd, err = s.decodeFileDesc(enc)
		if err != nil {
			return nil, err
		}
	}

	return proto.Marshal(fd)
}

// fileDescEncodingContainingExtension finds the file descriptor containing given extension,
// does marshalling on it and returns the marshalled result.
func (s *serverReflectionServer) fileDescEncodingContainingExtension(typeName string, extNum int32) ([]byte, error) {
	st, err := s.typeForName(typeName)
	if err != nil {
		return nil, err
	}
	fd, err := s.fileDescContainingExtension(st, extNum)
	if err != nil {
		return nil, err
	}
	return proto.Marshal(fd)
}

// allExtensionNumbersForTypeName returns all extension numbers for the given type.
func (s *serverReflectionServer) allExtensionNumbersForTypeName(name string) ([]int32, error) {
	st, err := s.typeForName(name)
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
			if s.serviceInfo == nil {
				s.serviceInfo = s.s.GetServiceInfo()
			}
			serviceResponses := make([]*rpb.ServiceResponse, 0, len(s.serviceInfo))
			for n := range s.serviceInfo {
				serviceResponses = append(serviceResponses, &rpb.ServiceResponse{
					Name: n,
				})
			}
			out.MessageResponse = &rpb.ServerReflectionResponse_ListServicesResponse{
				ListServicesResponse: &rpb.ListServiceResponse{
					Service: serviceResponses,
				},
			}
		default:
			return grpc.Errorf(codes.InvalidArgument, "invalid MessageRequest: %v", in.MessageRequest)
		}

		if err := stream.Send(out); err != nil {
			return err
		}
	}
}
