/*
 *
 * Copyright 2024 gRPC authors.
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

// Package internal contains code that is shared by both reflection package and
// the test package. The packages are split in this way inorder to avoid
// dependency to deprecated package github.com/golang/protobuf.
package internal

import (
	"io"
	"sort"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"

	v1reflectiongrpc "google.golang.org/grpc/reflection/grpc_reflection_v1"
	v1reflectionpb "google.golang.org/grpc/reflection/grpc_reflection_v1"
	v1alphareflectiongrpc "google.golang.org/grpc/reflection/grpc_reflection_v1alpha"
	v1alphareflectionpb "google.golang.org/grpc/reflection/grpc_reflection_v1alpha"
)

// ServiceInfoProvider is an interface used to retrieve metadata about the
// services to expose.
type ServiceInfoProvider interface {
	GetServiceInfo() map[string]grpc.ServiceInfo
}

// ExtensionResolver is the interface used to query details about extensions.
// This interface is satisfied by protoregistry.GlobalTypes.
type ExtensionResolver interface {
	protoregistry.ExtensionTypeResolver
	RangeExtensionsByMessage(message protoreflect.FullName, f func(protoreflect.ExtensionType) bool)
}

// ServerReflectionServer is the server API for ServerReflection service.
type ServerReflectionServer struct {
	v1alphareflectiongrpc.UnimplementedServerReflectionServer
	S            ServiceInfoProvider
	DescResolver protodesc.Resolver
	ExtResolver  ExtensionResolver
}

// FileDescWithDependencies returns a slice of serialized fileDescriptors in
// wire format ([]byte). The fileDescriptors will include fd and all the
// transitive dependencies of fd with names not in sentFileDescriptors.
func (s *ServerReflectionServer) FileDescWithDependencies(fd protoreflect.FileDescriptor, sentFileDescriptors map[string]bool) ([][]byte, error) {
	if fd.IsPlaceholder() {
		// If the given root file is a placeholder, treat it
		// as missing instead of serializing it.
		return nil, protoregistry.NotFound
	}
	var r [][]byte
	queue := []protoreflect.FileDescriptor{fd}
	for len(queue) > 0 {
		currentfd := queue[0]
		queue = queue[1:]
		if currentfd.IsPlaceholder() {
			// Skip any missing files in the dependency graph.
			continue
		}
		if sent := sentFileDescriptors[currentfd.Path()]; len(r) == 0 || !sent {
			sentFileDescriptors[currentfd.Path()] = true
			fdProto := protodesc.ToFileDescriptorProto(currentfd)
			currentfdEncoded, err := proto.Marshal(fdProto)
			if err != nil {
				return nil, err
			}
			r = append(r, currentfdEncoded)
		}
		for i := 0; i < currentfd.Imports().Len(); i++ {
			queue = append(queue, currentfd.Imports().Get(i))
		}
	}
	return r, nil
}

// FileDescEncodingContainingSymbol finds the file descriptor containing the
// given symbol, finds all of its previously unsent transitive dependencies,
// does marshalling on them, and returns the marshalled result. The given symbol
// can be a type, a service or a method.
func (s *ServerReflectionServer) FileDescEncodingContainingSymbol(name string, sentFileDescriptors map[string]bool) ([][]byte, error) {
	d, err := s.DescResolver.FindDescriptorByName(protoreflect.FullName(name))
	if err != nil {
		return nil, err
	}
	return s.FileDescWithDependencies(d.ParentFile(), sentFileDescriptors)
}

// FileDescEncodingContainingExtension finds the file descriptor containing
// given extension, finds all of its previously unsent transitive dependencies,
// does marshalling on them, and returns the marshalled result.
func (s *ServerReflectionServer) FileDescEncodingContainingExtension(typeName string, extNum int32, sentFileDescriptors map[string]bool) ([][]byte, error) {
	xt, err := s.ExtResolver.FindExtensionByNumber(protoreflect.FullName(typeName), protoreflect.FieldNumber(extNum))
	if err != nil {
		return nil, err
	}
	return s.FileDescWithDependencies(xt.TypeDescriptor().ParentFile(), sentFileDescriptors)
}

// AllExtensionNumbersForTypeName returns all extension numbers for the given type.
func (s *ServerReflectionServer) AllExtensionNumbersForTypeName(name string) ([]int32, error) {
	var numbers []int32
	s.ExtResolver.RangeExtensionsByMessage(protoreflect.FullName(name), func(xt protoreflect.ExtensionType) bool {
		numbers = append(numbers, int32(xt.TypeDescriptor().Number()))
		return true
	})
	sort.Slice(numbers, func(i, j int) bool {
		return numbers[i] < numbers[j]
	})
	if len(numbers) == 0 {
		// maybe return an error if given type name is not known
		if _, err := s.DescResolver.FindDescriptorByName(protoreflect.FullName(name)); err != nil {
			return nil, err
		}
	}
	return numbers, nil
}

// ListServices returns the names of services this server exposes.
func (s *ServerReflectionServer) ListServices() []*v1reflectionpb.ServiceResponse {
	serviceInfo := s.S.GetServiceInfo()
	resp := make([]*v1reflectionpb.ServiceResponse, 0, len(serviceInfo))
	for svc := range serviceInfo {
		resp = append(resp, &v1reflectionpb.ServiceResponse{Name: svc})
	}
	sort.Slice(resp, func(i, j int) bool {
		return resp[i].Name < resp[j].Name
	})
	return resp
}

// ServerReflectionInfo is the reflection service handler.
func (s *ServerReflectionServer) ServerReflectionInfo(stream v1reflectiongrpc.ServerReflection_ServerReflectionInfoServer) error {
	sentFileDescriptors := make(map[string]bool)
	for {
		in, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		out := &v1reflectionpb.ServerReflectionResponse{
			ValidHost:       in.Host,
			OriginalRequest: in,
		}
		switch req := in.MessageRequest.(type) {
		case *v1reflectionpb.ServerReflectionRequest_FileByFilename:
			var b [][]byte
			fd, err := s.DescResolver.FindFileByPath(req.FileByFilename)
			if err == nil {
				b, err = s.FileDescWithDependencies(fd, sentFileDescriptors)
			}
			if err != nil {
				out.MessageResponse = &v1reflectionpb.ServerReflectionResponse_ErrorResponse{
					ErrorResponse: &v1reflectionpb.ErrorResponse{
						ErrorCode:    int32(codes.NotFound),
						ErrorMessage: err.Error(),
					},
				}
			} else {
				out.MessageResponse = &v1reflectionpb.ServerReflectionResponse_FileDescriptorResponse{
					FileDescriptorResponse: &v1reflectionpb.FileDescriptorResponse{FileDescriptorProto: b},
				}
			}
		case *v1reflectionpb.ServerReflectionRequest_FileContainingSymbol:
			b, err := s.FileDescEncodingContainingSymbol(req.FileContainingSymbol, sentFileDescriptors)
			if err != nil {
				out.MessageResponse = &v1reflectionpb.ServerReflectionResponse_ErrorResponse{
					ErrorResponse: &v1reflectionpb.ErrorResponse{
						ErrorCode:    int32(codes.NotFound),
						ErrorMessage: err.Error(),
					},
				}
			} else {
				out.MessageResponse = &v1reflectionpb.ServerReflectionResponse_FileDescriptorResponse{
					FileDescriptorResponse: &v1reflectionpb.FileDescriptorResponse{FileDescriptorProto: b},
				}
			}
		case *v1reflectionpb.ServerReflectionRequest_FileContainingExtension:
			typeName := req.FileContainingExtension.ContainingType
			extNum := req.FileContainingExtension.ExtensionNumber
			b, err := s.FileDescEncodingContainingExtension(typeName, extNum, sentFileDescriptors)
			if err != nil {
				out.MessageResponse = &v1reflectionpb.ServerReflectionResponse_ErrorResponse{
					ErrorResponse: &v1reflectionpb.ErrorResponse{
						ErrorCode:    int32(codes.NotFound),
						ErrorMessage: err.Error(),
					},
				}
			} else {
				out.MessageResponse = &v1reflectionpb.ServerReflectionResponse_FileDescriptorResponse{
					FileDescriptorResponse: &v1reflectionpb.FileDescriptorResponse{FileDescriptorProto: b},
				}
			}
		case *v1reflectionpb.ServerReflectionRequest_AllExtensionNumbersOfType:
			extNums, err := s.AllExtensionNumbersForTypeName(req.AllExtensionNumbersOfType)
			if err != nil {
				out.MessageResponse = &v1reflectionpb.ServerReflectionResponse_ErrorResponse{
					ErrorResponse: &v1reflectionpb.ErrorResponse{
						ErrorCode:    int32(codes.NotFound),
						ErrorMessage: err.Error(),
					},
				}
			} else {
				out.MessageResponse = &v1reflectionpb.ServerReflectionResponse_AllExtensionNumbersResponse{
					AllExtensionNumbersResponse: &v1reflectionpb.ExtensionNumberResponse{
						BaseTypeName:    req.AllExtensionNumbersOfType,
						ExtensionNumber: extNums,
					},
				}
			}
		case *v1reflectionpb.ServerReflectionRequest_ListServices:
			out.MessageResponse = &v1reflectionpb.ServerReflectionResponse_ListServicesResponse{
				ListServicesResponse: &v1reflectionpb.ListServiceResponse{
					Service: s.ListServices(),
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

// V1ToV1AlphaResponse converts a v1 ServerReflectionResponse to a v1alpha.
func V1ToV1AlphaResponse(v1 *v1reflectionpb.ServerReflectionResponse) *v1alphareflectionpb.ServerReflectionResponse {
	var v1alpha v1alphareflectionpb.ServerReflectionResponse
	v1alpha.ValidHost = v1.ValidHost
	if v1.OriginalRequest != nil {
		v1alpha.OriginalRequest = V1ToV1AlphaRequest(v1.OriginalRequest)
	}
	switch mr := v1.MessageResponse.(type) {
	case *v1reflectionpb.ServerReflectionResponse_FileDescriptorResponse:
		if mr != nil {
			v1alpha.MessageResponse = &v1alphareflectionpb.ServerReflectionResponse_FileDescriptorResponse{
				FileDescriptorResponse: &v1alphareflectionpb.FileDescriptorResponse{
					FileDescriptorProto: mr.FileDescriptorResponse.GetFileDescriptorProto(),
				},
			}
		}
	case *v1reflectionpb.ServerReflectionResponse_AllExtensionNumbersResponse:
		if mr != nil {
			v1alpha.MessageResponse = &v1alphareflectionpb.ServerReflectionResponse_AllExtensionNumbersResponse{
				AllExtensionNumbersResponse: &v1alphareflectionpb.ExtensionNumberResponse{
					BaseTypeName:    mr.AllExtensionNumbersResponse.GetBaseTypeName(),
					ExtensionNumber: mr.AllExtensionNumbersResponse.GetExtensionNumber(),
				},
			}
		}
	case *v1reflectionpb.ServerReflectionResponse_ListServicesResponse:
		if mr != nil {
			svcs := make([]*v1alphareflectionpb.ServiceResponse, len(mr.ListServicesResponse.GetService()))
			for i, svc := range mr.ListServicesResponse.GetService() {
				svcs[i] = &v1alphareflectionpb.ServiceResponse{
					Name: svc.GetName(),
				}
			}
			v1alpha.MessageResponse = &v1alphareflectionpb.ServerReflectionResponse_ListServicesResponse{
				ListServicesResponse: &v1alphareflectionpb.ListServiceResponse{
					Service: svcs,
				},
			}
		}
	case *v1reflectionpb.ServerReflectionResponse_ErrorResponse:
		if mr != nil {
			v1alpha.MessageResponse = &v1alphareflectionpb.ServerReflectionResponse_ErrorResponse{
				ErrorResponse: &v1alphareflectionpb.ErrorResponse{
					ErrorCode:    mr.ErrorResponse.GetErrorCode(),
					ErrorMessage: mr.ErrorResponse.GetErrorMessage(),
				},
			}
		}
	default:
		// no value set
	}
	return &v1alpha
}

// V1AlphaToV1Request converts a v1alpha ServerReflectionRequest to a v1.
func V1AlphaToV1Request(v1alpha *v1alphareflectionpb.ServerReflectionRequest) *v1reflectionpb.ServerReflectionRequest {
	var v1 v1reflectionpb.ServerReflectionRequest
	v1.Host = v1alpha.Host
	switch mr := v1alpha.MessageRequest.(type) {
	case *v1alphareflectionpb.ServerReflectionRequest_FileByFilename:
		v1.MessageRequest = &v1reflectionpb.ServerReflectionRequest_FileByFilename{
			FileByFilename: mr.FileByFilename,
		}
	case *v1alphareflectionpb.ServerReflectionRequest_FileContainingSymbol:
		v1.MessageRequest = &v1reflectionpb.ServerReflectionRequest_FileContainingSymbol{
			FileContainingSymbol: mr.FileContainingSymbol,
		}
	case *v1alphareflectionpb.ServerReflectionRequest_FileContainingExtension:
		if mr.FileContainingExtension != nil {
			v1.MessageRequest = &v1reflectionpb.ServerReflectionRequest_FileContainingExtension{
				FileContainingExtension: &v1reflectionpb.ExtensionRequest{
					ContainingType:  mr.FileContainingExtension.GetContainingType(),
					ExtensionNumber: mr.FileContainingExtension.GetExtensionNumber(),
				},
			}
		}
	case *v1alphareflectionpb.ServerReflectionRequest_AllExtensionNumbersOfType:
		v1.MessageRequest = &v1reflectionpb.ServerReflectionRequest_AllExtensionNumbersOfType{
			AllExtensionNumbersOfType: mr.AllExtensionNumbersOfType,
		}
	case *v1alphareflectionpb.ServerReflectionRequest_ListServices:
		v1.MessageRequest = &v1reflectionpb.ServerReflectionRequest_ListServices{
			ListServices: mr.ListServices,
		}
	default:
		// no value set
	}
	return &v1
}

// V1ToV1AlphaRequest converts a v1 ServerReflectionRequest to a v1alpha.
func V1ToV1AlphaRequest(v1 *v1reflectionpb.ServerReflectionRequest) *v1alphareflectionpb.ServerReflectionRequest {
	var v1alpha v1alphareflectionpb.ServerReflectionRequest
	v1alpha.Host = v1.Host
	switch mr := v1.MessageRequest.(type) {
	case *v1reflectionpb.ServerReflectionRequest_FileByFilename:
		if mr != nil {
			v1alpha.MessageRequest = &v1alphareflectionpb.ServerReflectionRequest_FileByFilename{
				FileByFilename: mr.FileByFilename,
			}
		}
	case *v1reflectionpb.ServerReflectionRequest_FileContainingSymbol:
		if mr != nil {
			v1alpha.MessageRequest = &v1alphareflectionpb.ServerReflectionRequest_FileContainingSymbol{
				FileContainingSymbol: mr.FileContainingSymbol,
			}
		}
	case *v1reflectionpb.ServerReflectionRequest_FileContainingExtension:
		if mr != nil {
			v1alpha.MessageRequest = &v1alphareflectionpb.ServerReflectionRequest_FileContainingExtension{
				FileContainingExtension: &v1alphareflectionpb.ExtensionRequest{
					ContainingType:  mr.FileContainingExtension.GetContainingType(),
					ExtensionNumber: mr.FileContainingExtension.GetExtensionNumber(),
				},
			}
		}
	case *v1reflectionpb.ServerReflectionRequest_AllExtensionNumbersOfType:
		if mr != nil {
			v1alpha.MessageRequest = &v1alphareflectionpb.ServerReflectionRequest_AllExtensionNumbersOfType{
				AllExtensionNumbersOfType: mr.AllExtensionNumbersOfType,
			}
		}
	case *v1reflectionpb.ServerReflectionRequest_ListServices:
		if mr != nil {
			v1alpha.MessageRequest = &v1alphareflectionpb.ServerReflectionRequest_ListServices{
				ListServices: mr.ListServices,
			}
		}
	default:
		// no value set
	}
	return &v1alpha
}

// V1AlphaToV1Response converts a v1alpha ServerReflectionResponse to a v1.
func V1AlphaToV1Response(v1alpha *v1alphareflectionpb.ServerReflectionResponse) *v1reflectionpb.ServerReflectionResponse {
	var v1 v1reflectionpb.ServerReflectionResponse
	v1.ValidHost = v1alpha.ValidHost
	if v1alpha.OriginalRequest != nil {
		v1.OriginalRequest = V1AlphaToV1Request(v1alpha.OriginalRequest)
	}
	switch mr := v1alpha.MessageResponse.(type) {
	case *v1alphareflectionpb.ServerReflectionResponse_FileDescriptorResponse:
		if mr != nil {
			v1.MessageResponse = &v1reflectionpb.ServerReflectionResponse_FileDescriptorResponse{
				FileDescriptorResponse: &v1reflectionpb.FileDescriptorResponse{
					FileDescriptorProto: mr.FileDescriptorResponse.GetFileDescriptorProto(),
				},
			}
		}
	case *v1alphareflectionpb.ServerReflectionResponse_AllExtensionNumbersResponse:
		if mr != nil {
			v1.MessageResponse = &v1reflectionpb.ServerReflectionResponse_AllExtensionNumbersResponse{
				AllExtensionNumbersResponse: &v1reflectionpb.ExtensionNumberResponse{
					BaseTypeName:    mr.AllExtensionNumbersResponse.GetBaseTypeName(),
					ExtensionNumber: mr.AllExtensionNumbersResponse.GetExtensionNumber(),
				},
			}
		}
	case *v1alphareflectionpb.ServerReflectionResponse_ListServicesResponse:
		if mr != nil {
			svcs := make([]*v1reflectionpb.ServiceResponse, len(mr.ListServicesResponse.GetService()))
			for i, svc := range mr.ListServicesResponse.GetService() {
				svcs[i] = &v1reflectionpb.ServiceResponse{
					Name: svc.GetName(),
				}
			}
			v1.MessageResponse = &v1reflectionpb.ServerReflectionResponse_ListServicesResponse{
				ListServicesResponse: &v1reflectionpb.ListServiceResponse{
					Service: svcs,
				},
			}
		}
	case *v1alphareflectionpb.ServerReflectionResponse_ErrorResponse:
		if mr != nil {
			v1.MessageResponse = &v1reflectionpb.ServerReflectionResponse_ErrorResponse{
				ErrorResponse: &v1reflectionpb.ErrorResponse{
					ErrorCode:    mr.ErrorResponse.GetErrorCode(),
					ErrorMessage: mr.ErrorResponse.GetErrorMessage(),
				},
			}
		}
	default:
		// no value set
	}
	return &v1
}
