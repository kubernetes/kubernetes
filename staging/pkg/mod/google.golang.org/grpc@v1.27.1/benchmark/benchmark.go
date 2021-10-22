/*
 *
 * Copyright 2014 gRPC authors.
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

//go:generate protoc -I grpc_testing --go_out=plugins=grpc:grpc_testing grpc_testing/control.proto grpc_testing/messages.proto grpc_testing/payloads.proto grpc_testing/services.proto grpc_testing/stats.proto

/*
Package benchmark implements the building blocks to setup end-to-end gRPC benchmarks.
*/
package benchmark

import (
	"context"
	"fmt"
	"io"
	"log"
	"net"

	"google.golang.org/grpc"
	testpb "google.golang.org/grpc/benchmark/grpc_testing"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/status"
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
}

// NewPayload creates a payload with the given type and size.
func NewPayload(t testpb.PayloadType, size int) *testpb.Payload {
	p := new(testpb.Payload)
	setPayload(p, t, size)
	return p
}

type testServer struct {
}

func (s *testServer) UnaryCall(ctx context.Context, in *testpb.SimpleRequest) (*testpb.SimpleResponse, error) {
	return &testpb.SimpleResponse{
		Payload: NewPayload(in.ResponseType, int(in.ResponseSize)),
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

func (s *testServer) UnconstrainedStreamingCall(stream testpb.BenchmarkService_UnconstrainedStreamingCallServer) error {
	in := new(testpb.SimpleRequest)
	// Receive a message to learn response type and size.
	err := stream.RecvMsg(in)
	if err == io.EOF {
		// read done.
		return nil
	}
	if err != nil {
		return err
	}

	response := &testpb.SimpleResponse{
		Payload: new(testpb.Payload),
	}
	setPayload(response.Payload, in.ResponseType, int(in.ResponseSize))

	go func() {
		for {
			// Using RecvMsg rather than Recv to prevent reallocation of SimpleRequest.
			err := stream.RecvMsg(in)
			switch status.Code(err) {
			case codes.Canceled:
			case codes.OK:
			default:
				log.Fatalf("server recv error: %v", err)
			}
		}
	}()

	go func() {
		for {
			err := stream.Send(response)
			switch status.Code(err) {
			case codes.Unavailable:
			case codes.OK:
			default:
				log.Fatalf("server send error: %v", err)
			}
		}
	}()

	<-stream.Context().Done()
	return stream.Context().Err()
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

func (s *byteBufServer) UnconstrainedStreamingCall(stream testpb.BenchmarkService_UnconstrainedStreamingCallServer) error {
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
	// Type is the type of the server.
	// It should be "protobuf" or "bytebuf".
	Type string

	// Metadata is an optional configuration.
	// For "protobuf", it's ignored.
	// For "bytebuf", it should be an int representing response size.
	Metadata interface{}

	// Listener is the network listener for the server to use
	Listener net.Listener
}

// StartServer starts a gRPC server serving a benchmark service according to info.
// It returns a function to stop the server.
func StartServer(info ServerInfo, opts ...grpc.ServerOption) func() {
	opts = append(opts, grpc.WriteBufferSize(128*1024))
	opts = append(opts, grpc.ReadBufferSize(128*1024))
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
	go s.Serve(info.Listener)
	return func() {
		s.Stop()
	}
}

// DoUnaryCall performs an unary RPC with given stub and request and response sizes.
func DoUnaryCall(tc testpb.BenchmarkServiceClient, reqSize, respSize int) error {
	pl := NewPayload(testpb.PayloadType_COMPRESSABLE, reqSize)
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
	pl := NewPayload(testpb.PayloadType_COMPRESSABLE, reqSize)
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
	return NewClientConnWithContext(context.Background(), addr, opts...)
}

// NewClientConnWithContext creates a gRPC client connection to addr using ctx.
func NewClientConnWithContext(ctx context.Context, addr string, opts ...grpc.DialOption) *grpc.ClientConn {
	opts = append(opts, grpc.WithWriteBufferSize(128*1024))
	opts = append(opts, grpc.WithReadBufferSize(128*1024))
	conn, err := grpc.DialContext(ctx, addr, opts...)
	if err != nil {
		grpclog.Fatalf("NewClientConn(%q) failed to create a ClientConn %v", addr, err)
	}
	return conn
}
