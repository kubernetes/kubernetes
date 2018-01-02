/*
 *
 * Copyright 2014, Google Inc.
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
Package benchmark implements the building blocks to setup end-to-end gRPC benchmarks.
*/
package benchmark

import (
	"fmt"
	"io"
	"net"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	testpb "google.golang.org/grpc/benchmark/grpc_testing"
	"google.golang.org/grpc/grpclog"
)

// Allows reuse of the same testpb.Payload object.
func setPayload(p *testpb.Payload, t testpb.PayloadType, size int) {
	if size < 0 {
		grpclog.Fatalf("Requested a response with invalid length %d", size)
	}
	body := make([]byte, size)
	switch t {
	case testpb.PayloadType_COMPRESSABLE:
	case testpb.PayloadType_UNCOMPRESSABLE:
		grpclog.Fatalf("PayloadType UNCOMPRESSABLE is not supported")
	default:
		grpclog.Fatalf("Unsupported payload type: %d", t)
	}
	p.Type = t
	p.Body = body
	return
}

func newPayload(t testpb.PayloadType, size int) *testpb.Payload {
	p := new(testpb.Payload)
	setPayload(p, t, size)
	return p
}

type testServer struct {
}

func (s *testServer) UnaryCall(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
	return &testpb.SimpleResponse{
		Payload: newPayload(in.ResponseType, int(in.ResponseSize)),
	}, nil
}

func (s *testServer) StreamingCall(stream testpb.BenchmarkService_StreamingCallServer) error {
	response := &testpb.SimpleResponse{
		Payload: new(testpb.Payload),
	}
	in := new(testpb.SimpleRequest)
	for {
		// use ServerStream directly to reuse the same testpb.SimpleRequest object
		err := stream.(grpc.ServerStream).RecvMsg(in)
		if err == io.EOF {
			// read done.
			return nil
		}
		if err != nil {
			return err
		}
		setPayload(response.Payload, in.ResponseType, int(in.ResponseSize))
		if err := stream.Send(response); err != nil {
			return err
		}
	}
}

// byteBufServer is a gRPC server that sends and receives byte buffer.
// The purpose is to benchmark the gRPC performance without protobuf serialization/deserialization overhead.
type byteBufServer struct {
	respSize int32
}

// UnaryCall is an empty function and is not used for benchmark.
// If bytebuf UnaryCall benchmark is needed later, the function body needs to be updated.
func (s *byteBufServer) UnaryCall(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
	return &testpb.SimpleResponse{}, nil
}

func (s *byteBufServer) StreamingCall(stream testpb.BenchmarkService_StreamingCallServer) error {
	for {
		var in []byte
		err := stream.(grpc.ServerStream).RecvMsg(&in)
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		out := make([]byte, s.respSize)
		if err := stream.(grpc.ServerStream).SendMsg(&out); err != nil {
			return err
		}
	}
}

// ServerInfo contains the information to create a gRPC benchmark server.
type ServerInfo struct {
	// Addr is the address of the server.
	Addr string

	// Type is the type of the server.
	// It should be "protobuf" or "bytebuf".
	Type string

	// Metadata is an optional configuration.
	// For "protobuf", it's ignored.
	// For "bytebuf", it should be an int representing response size.
	Metadata interface{}
}

// StartServer starts a gRPC server serving a benchmark service according to info.
// It returns its listen address and a function to stop the server.
func StartServer(info ServerInfo, opts ...grpc.ServerOption) (string, func()) {
	lis, err := net.Listen("tcp", info.Addr)
	if err != nil {
		grpclog.Fatalf("Failed to listen: %v", err)
	}
	s := grpc.NewServer(opts...)
	switch info.Type {
	case "protobuf":
		testpb.RegisterBenchmarkServiceServer(s, &testServer{})
	case "bytebuf":
		respSize, ok := info.Metadata.(int32)
		if !ok {
			grpclog.Fatalf("failed to StartServer, invalid metadata: %v, for Type: %v", info.Metadata, info.Type)
		}
		testpb.RegisterBenchmarkServiceServer(s, &byteBufServer{respSize: respSize})
	default:
		grpclog.Fatalf("failed to StartServer, unknown Type: %v", info.Type)
	}
	go s.Serve(lis)
	return lis.Addr().String(), func() {
		s.Stop()
	}
}

// DoUnaryCall performs an unary RPC with given stub and request and response sizes.
func DoUnaryCall(tc testpb.BenchmarkServiceClient, reqSize, respSize int) error {
	pl := newPayload(testpb.PayloadType_COMPRESSABLE, reqSize)
	req := &testpb.SimpleRequest{
		ResponseType: pl.Type,
		ResponseSize: int32(respSize),
		Payload:      pl,
	}
	if _, err := tc.UnaryCall(context.Background(), req); err != nil {
		return fmt.Errorf("/BenchmarkService/UnaryCall(_, _) = _, %v, want _, <nil>", err)
	}
	return nil
}

// DoStreamingRoundTrip performs a round trip for a single streaming rpc.
func DoStreamingRoundTrip(stream testpb.BenchmarkService_StreamingCallClient, reqSize, respSize int) error {
	pl := newPayload(testpb.PayloadType_COMPRESSABLE, reqSize)
	req := &testpb.SimpleRequest{
		ResponseType: pl.Type,
		ResponseSize: int32(respSize),
		Payload:      pl,
	}
	if err := stream.Send(req); err != nil {
		return fmt.Errorf("/BenchmarkService/StreamingCall.Send(_) = %v, want <nil>", err)
	}
	if _, err := stream.Recv(); err != nil {
		// EOF is a valid error here.
		if err == io.EOF {
			return nil
		}
		return fmt.Errorf("/BenchmarkService/StreamingCall.Recv(_) = %v, want <nil>", err)
	}
	return nil
}

// DoByteBufStreamingRoundTrip performs a round trip for a single streaming rpc, using a custom codec for byte buffer.
func DoByteBufStreamingRoundTrip(stream testpb.BenchmarkService_StreamingCallClient, reqSize, respSize int) error {
	out := make([]byte, reqSize)
	if err := stream.(grpc.ClientStream).SendMsg(&out); err != nil {
		return fmt.Errorf("/BenchmarkService/StreamingCall.(ClientStream).SendMsg(_) = %v, want <nil>", err)
	}
	var in []byte
	if err := stream.(grpc.ClientStream).RecvMsg(&in); err != nil {
		// EOF is a valid error here.
		if err == io.EOF {
			return nil
		}
		return fmt.Errorf("/BenchmarkService/StreamingCall.(ClientStream).RecvMsg(_) = %v, want <nil>", err)
	}
	return nil
}

// NewClientConn creates a gRPC client connection to addr.
func NewClientConn(addr string, opts ...grpc.DialOption) *grpc.ClientConn {
	conn, err := grpc.Dial(addr, opts...)
	if err != nil {
		grpclog.Fatalf("NewClientConn(%q) failed to create a ClientConn %v", addr, err)
	}
	return conn
}
