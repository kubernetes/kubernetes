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

package reflection

import (
	"fmt"
	"net"
	"reflect"
	"sort"
	"testing"

	"github.com/golang/protobuf/proto"
	dpb "github.com/golang/protobuf/protoc-gen-go/descriptor"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	rpb "google.golang.org/grpc/reflection/grpc_reflection_v1alpha"
	pb "google.golang.org/grpc/reflection/grpc_testing"
)

var (
	s = &serverReflectionServer{}
	// fileDescriptor of each test proto file.
	fdTest       *dpb.FileDescriptorProto
	fdProto2     *dpb.FileDescriptorProto
	fdProto2Ext  *dpb.FileDescriptorProto
	fdProto2Ext2 *dpb.FileDescriptorProto
	// fileDescriptor marshalled.
	fdTestByte       []byte
	fdProto2Byte     []byte
	fdProto2ExtByte  []byte
	fdProto2Ext2Byte []byte
)

func loadFileDesc(filename string) (*dpb.FileDescriptorProto, []byte) {
	enc := proto.FileDescriptor(filename)
	if enc == nil {
		panic(fmt.Sprintf("failed to find fd for file: %v", filename))
	}
	fd, err := s.decodeFileDesc(enc)
	if err != nil {
		panic(fmt.Sprintf("failed to decode enc: %v", err))
	}
	b, err := proto.Marshal(fd)
	if err != nil {
		panic(fmt.Sprintf("failed to marshal fd: %v", err))
	}
	return fd, b
}

func init() {
	fdTest, fdTestByte = loadFileDesc("test.proto")
	fdProto2, fdProto2Byte = loadFileDesc("proto2.proto")
	fdProto2Ext, fdProto2ExtByte = loadFileDesc("proto2_ext.proto")
	fdProto2Ext2, fdProto2Ext2Byte = loadFileDesc("proto2_ext2.proto")
}

func TestFileDescForType(t *testing.T) {
	for _, test := range []struct {
		st     reflect.Type
		wantFd *dpb.FileDescriptorProto
	}{
		{reflect.TypeOf(pb.SearchResponse_Result{}), fdTest},
		{reflect.TypeOf(pb.ToBeExtended{}), fdProto2},
	} {
		fd, err := s.fileDescForType(test.st)
		if err != nil || !proto.Equal(fd, test.wantFd) {
			t.Errorf("fileDescForType(%q) = %q, %v, want %q, <nil>", test.st, fd, err, test.wantFd)
		}
	}
}

func TestTypeForName(t *testing.T) {
	for _, test := range []struct {
		name string
		want reflect.Type
	}{
		{"grpc.testing.SearchResponse", reflect.TypeOf(pb.SearchResponse{})},
	} {
		r, err := s.typeForName(test.name)
		if err != nil || r != test.want {
			t.Errorf("typeForName(%q) = %q, %v, want %q, <nil>", test.name, r, err, test.want)
		}
	}
}

func TestTypeForNameNotFound(t *testing.T) {
	for _, test := range []string{
		"grpc.testing.not_exiting",
	} {
		_, err := s.typeForName(test)
		if err == nil {
			t.Errorf("typeForName(%q) = _, %v, want _, <non-nil>", test, err)
		}
	}
}

func TestFileDescContainingExtension(t *testing.T) {
	for _, test := range []struct {
		st     reflect.Type
		extNum int32
		want   *dpb.FileDescriptorProto
	}{
		{reflect.TypeOf(pb.ToBeExtended{}), 13, fdProto2Ext},
		{reflect.TypeOf(pb.ToBeExtended{}), 17, fdProto2Ext},
		{reflect.TypeOf(pb.ToBeExtended{}), 19, fdProto2Ext},
		{reflect.TypeOf(pb.ToBeExtended{}), 23, fdProto2Ext2},
		{reflect.TypeOf(pb.ToBeExtended{}), 29, fdProto2Ext2},
	} {
		fd, err := s.fileDescContainingExtension(test.st, test.extNum)
		if err != nil || !proto.Equal(fd, test.want) {
			t.Errorf("fileDescContainingExtension(%q) = %q, %v, want %q, <nil>", test.st, fd, err, test.want)
		}
	}
}

// intArray is used to sort []int32
type intArray []int32

func (s intArray) Len() int           { return len(s) }
func (s intArray) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s intArray) Less(i, j int) bool { return s[i] < s[j] }

func TestAllExtensionNumbersForType(t *testing.T) {
	for _, test := range []struct {
		st   reflect.Type
		want []int32
	}{
		{reflect.TypeOf(pb.ToBeExtended{}), []int32{13, 17, 19, 23, 29}},
	} {
		r, err := s.allExtensionNumbersForType(test.st)
		sort.Sort(intArray(r))
		if err != nil || !reflect.DeepEqual(r, test.want) {
			t.Errorf("allExtensionNumbersForType(%q) = %v, %v, want %v, <nil>", test.st, r, err, test.want)
		}
	}
}

// Do end2end tests.

type server struct{}

func (s *server) Search(ctx context.Context, in *pb.SearchRequest) (*pb.SearchResponse, error) {
	return &pb.SearchResponse{}, nil
}

func (s *server) StreamingSearch(stream pb.SearchService_StreamingSearchServer) error {
	return nil
}

func TestReflectionEnd2end(t *testing.T) {
	// Start server.
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterSearchServiceServer(s, &server{})
	// Register reflection service on s.
	Register(s)
	go s.Serve(lis)

	// Create client.
	conn, err := grpc.Dial(lis.Addr().String(), grpc.WithInsecure())
	if err != nil {
		t.Fatalf("cannot connect to server: %v", err)
	}
	defer conn.Close()

	c := rpb.NewServerReflectionClient(conn)
	stream, err := c.ServerReflectionInfo(context.Background())
	if err != nil {
		t.Fatalf("cannot get ServerReflectionInfo: %v", err)
	}

	testFileByFilename(t, stream)
	testFileByFilenameError(t, stream)
	testFileContainingSymbol(t, stream)
	testFileContainingSymbolError(t, stream)
	testFileContainingExtension(t, stream)
	testFileContainingExtensionError(t, stream)
	testAllExtensionNumbersOfType(t, stream)
	testAllExtensionNumbersOfTypeError(t, stream)
	testListServices(t, stream)

	s.Stop()
}

func testFileByFilename(t *testing.T, stream rpb.ServerReflection_ServerReflectionInfoClient) {
	for _, test := range []struct {
		filename string
		want     []byte
	}{
		{"test.proto", fdTestByte},
		{"proto2.proto", fdProto2Byte},
		{"proto2_ext.proto", fdProto2ExtByte},
	} {
		if err := stream.Send(&rpb.ServerReflectionRequest{
			MessageRequest: &rpb.ServerReflectionRequest_FileByFilename{
				FileByFilename: test.filename,
			},
		}); err != nil {
			t.Fatalf("failed to send request: %v", err)
		}
		r, err := stream.Recv()
		if err != nil {
			// io.EOF is not ok.
			t.Fatalf("failed to recv response: %v", err)
		}

		switch r.MessageResponse.(type) {
		case *rpb.ServerReflectionResponse_FileDescriptorResponse:
			if !reflect.DeepEqual(r.GetFileDescriptorResponse().FileDescriptorProto[0], test.want) {
				t.Errorf("FileByFilename(%v)\nreceived: %q,\nwant: %q", test.filename, r.GetFileDescriptorResponse().FileDescriptorProto[0], test.want)
			}
		default:
			t.Errorf("FileByFilename(%v) = %v, want type <ServerReflectionResponse_FileDescriptorResponse>", test.filename, r.MessageResponse)
		}
	}
}

func testFileByFilenameError(t *testing.T, stream rpb.ServerReflection_ServerReflectionInfoClient) {
	for _, test := range []string{
		"test.poto",
		"proo2.proto",
		"proto2_et.proto",
	} {
		if err := stream.Send(&rpb.ServerReflectionRequest{
			MessageRequest: &rpb.ServerReflectionRequest_FileByFilename{
				FileByFilename: test,
			},
		}); err != nil {
			t.Fatalf("failed to send request: %v", err)
		}
		r, err := stream.Recv()
		if err != nil {
			// io.EOF is not ok.
			t.Fatalf("failed to recv response: %v", err)
		}

		switch r.MessageResponse.(type) {
		case *rpb.ServerReflectionResponse_ErrorResponse:
		default:
			t.Errorf("FileByFilename(%v) = %v, want type <ServerReflectionResponse_ErrorResponse>", test, r.MessageResponse)
		}
	}
}

func testFileContainingSymbol(t *testing.T, stream rpb.ServerReflection_ServerReflectionInfoClient) {
	for _, test := range []struct {
		symbol string
		want   []byte
	}{
		{"grpc.testing.SearchService", fdTestByte},
		{"grpc.testing.SearchService.Search", fdTestByte},
		{"grpc.testing.SearchService.StreamingSearch", fdTestByte},
		{"grpc.testing.SearchResponse", fdTestByte},
		{"grpc.testing.ToBeExtended", fdProto2Byte},
	} {
		if err := stream.Send(&rpb.ServerReflectionRequest{
			MessageRequest: &rpb.ServerReflectionRequest_FileContainingSymbol{
				FileContainingSymbol: test.symbol,
			},
		}); err != nil {
			t.Fatalf("failed to send request: %v", err)
		}
		r, err := stream.Recv()
		if err != nil {
			// io.EOF is not ok.
			t.Fatalf("failed to recv response: %v", err)
		}

		switch r.MessageResponse.(type) {
		case *rpb.ServerReflectionResponse_FileDescriptorResponse:
			if !reflect.DeepEqual(r.GetFileDescriptorResponse().FileDescriptorProto[0], test.want) {
				t.Errorf("FileContainingSymbol(%v)\nreceived: %q,\nwant: %q", test.symbol, r.GetFileDescriptorResponse().FileDescriptorProto[0], test.want)
			}
		default:
			t.Errorf("FileContainingSymbol(%v) = %v, want type <ServerReflectionResponse_FileDescriptorResponse>", test.symbol, r.MessageResponse)
		}
	}
}

func testFileContainingSymbolError(t *testing.T, stream rpb.ServerReflection_ServerReflectionInfoClient) {
	for _, test := range []string{
		"grpc.testing.SerchService",
		"grpc.testing.SearchService.SearchE",
		"grpc.tesing.SearchResponse",
		"gpc.testing.ToBeExtended",
	} {
		if err := stream.Send(&rpb.ServerReflectionRequest{
			MessageRequest: &rpb.ServerReflectionRequest_FileContainingSymbol{
				FileContainingSymbol: test,
			},
		}); err != nil {
			t.Fatalf("failed to send request: %v", err)
		}
		r, err := stream.Recv()
		if err != nil {
			// io.EOF is not ok.
			t.Fatalf("failed to recv response: %v", err)
		}

		switch r.MessageResponse.(type) {
		case *rpb.ServerReflectionResponse_ErrorResponse:
		default:
			t.Errorf("FileContainingSymbol(%v) = %v, want type <ServerReflectionResponse_ErrorResponse>", test, r.MessageResponse)
		}
	}
}

func testFileContainingExtension(t *testing.T, stream rpb.ServerReflection_ServerReflectionInfoClient) {
	for _, test := range []struct {
		typeName string
		extNum   int32
		want     []byte
	}{
		{"grpc.testing.ToBeExtended", 13, fdProto2ExtByte},
		{"grpc.testing.ToBeExtended", 17, fdProto2ExtByte},
		{"grpc.testing.ToBeExtended", 19, fdProto2ExtByte},
		{"grpc.testing.ToBeExtended", 23, fdProto2Ext2Byte},
		{"grpc.testing.ToBeExtended", 29, fdProto2Ext2Byte},
	} {
		if err := stream.Send(&rpb.ServerReflectionRequest{
			MessageRequest: &rpb.ServerReflectionRequest_FileContainingExtension{
				FileContainingExtension: &rpb.ExtensionRequest{
					ContainingType:  test.typeName,
					ExtensionNumber: test.extNum,
				},
			},
		}); err != nil {
			t.Fatalf("failed to send request: %v", err)
		}
		r, err := stream.Recv()
		if err != nil {
			// io.EOF is not ok.
			t.Fatalf("failed to recv response: %v", err)
		}

		switch r.MessageResponse.(type) {
		case *rpb.ServerReflectionResponse_FileDescriptorResponse:
			if !reflect.DeepEqual(r.GetFileDescriptorResponse().FileDescriptorProto[0], test.want) {
				t.Errorf("FileContainingExtension(%v, %v)\nreceived: %q,\nwant: %q", test.typeName, test.extNum, r.GetFileDescriptorResponse().FileDescriptorProto[0], test.want)
			}
		default:
			t.Errorf("FileContainingExtension(%v, %v) = %v, want type <ServerReflectionResponse_FileDescriptorResponse>", test.typeName, test.extNum, r.MessageResponse)
		}
	}
}

func testFileContainingExtensionError(t *testing.T, stream rpb.ServerReflection_ServerReflectionInfoClient) {
	for _, test := range []struct {
		typeName string
		extNum   int32
	}{
		{"grpc.testing.ToBExtended", 17},
		{"grpc.testing.ToBeExtended", 15},
	} {
		if err := stream.Send(&rpb.ServerReflectionRequest{
			MessageRequest: &rpb.ServerReflectionRequest_FileContainingExtension{
				FileContainingExtension: &rpb.ExtensionRequest{
					ContainingType:  test.typeName,
					ExtensionNumber: test.extNum,
				},
			},
		}); err != nil {
			t.Fatalf("failed to send request: %v", err)
		}
		r, err := stream.Recv()
		if err != nil {
			// io.EOF is not ok.
			t.Fatalf("failed to recv response: %v", err)
		}

		switch r.MessageResponse.(type) {
		case *rpb.ServerReflectionResponse_ErrorResponse:
		default:
			t.Errorf("FileContainingExtension(%v, %v) = %v, want type <ServerReflectionResponse_FileDescriptorResponse>", test.typeName, test.extNum, r.MessageResponse)
		}
	}
}

func testAllExtensionNumbersOfType(t *testing.T, stream rpb.ServerReflection_ServerReflectionInfoClient) {
	for _, test := range []struct {
		typeName string
		want     []int32
	}{
		{"grpc.testing.ToBeExtended", []int32{13, 17, 19, 23, 29}},
	} {
		if err := stream.Send(&rpb.ServerReflectionRequest{
			MessageRequest: &rpb.ServerReflectionRequest_AllExtensionNumbersOfType{
				AllExtensionNumbersOfType: test.typeName,
			},
		}); err != nil {
			t.Fatalf("failed to send request: %v", err)
		}
		r, err := stream.Recv()
		if err != nil {
			// io.EOF is not ok.
			t.Fatalf("failed to recv response: %v", err)
		}

		switch r.MessageResponse.(type) {
		case *rpb.ServerReflectionResponse_AllExtensionNumbersResponse:
			extNum := r.GetAllExtensionNumbersResponse().ExtensionNumber
			sort.Sort(intArray(extNum))
			if r.GetAllExtensionNumbersResponse().BaseTypeName != test.typeName ||
				!reflect.DeepEqual(extNum, test.want) {
				t.Errorf("AllExtensionNumbersOfType(%v)\nreceived: %v,\nwant: {%q %v}", r.GetAllExtensionNumbersResponse(), test.typeName, test.typeName, test.want)
			}
		default:
			t.Errorf("AllExtensionNumbersOfType(%v) = %v, want type <ServerReflectionResponse_AllExtensionNumbersResponse>", test.typeName, r.MessageResponse)
		}
	}
}

func testAllExtensionNumbersOfTypeError(t *testing.T, stream rpb.ServerReflection_ServerReflectionInfoClient) {
	for _, test := range []string{
		"grpc.testing.ToBeExtendedE",
	} {
		if err := stream.Send(&rpb.ServerReflectionRequest{
			MessageRequest: &rpb.ServerReflectionRequest_AllExtensionNumbersOfType{
				AllExtensionNumbersOfType: test,
			},
		}); err != nil {
			t.Fatalf("failed to send request: %v", err)
		}
		r, err := stream.Recv()
		if err != nil {
			// io.EOF is not ok.
			t.Fatalf("failed to recv response: %v", err)
		}

		switch r.MessageResponse.(type) {
		case *rpb.ServerReflectionResponse_ErrorResponse:
		default:
			t.Errorf("AllExtensionNumbersOfType(%v) = %v, want type <ServerReflectionResponse_ErrorResponse>", test, r.MessageResponse)
		}
	}
}

func testListServices(t *testing.T, stream rpb.ServerReflection_ServerReflectionInfoClient) {
	if err := stream.Send(&rpb.ServerReflectionRequest{
		MessageRequest: &rpb.ServerReflectionRequest_ListServices{},
	}); err != nil {
		t.Fatalf("failed to send request: %v", err)
	}
	r, err := stream.Recv()
	if err != nil {
		// io.EOF is not ok.
		t.Fatalf("failed to recv response: %v", err)
	}

	switch r.MessageResponse.(type) {
	case *rpb.ServerReflectionResponse_ListServicesResponse:
		services := r.GetListServicesResponse().Service
		want := []string{"grpc.testing.SearchService", "grpc.reflection.v1alpha.ServerReflection"}
		// Compare service names in response with want.
		if len(services) != len(want) {
			t.Errorf("= %v, want service names: %v", services, want)
		}
		m := make(map[string]int)
		for _, e := range services {
			m[e.Name]++
		}
		for _, e := range want {
			if m[e] > 0 {
				m[e]--
				continue
			}
			t.Errorf("ListService\nreceived: %v,\nwant: %q", services, want)
		}
	default:
		t.Errorf("ListServices = %v, want type <ServerReflectionResponse_ListServicesResponse>", r.MessageResponse)
	}
}
