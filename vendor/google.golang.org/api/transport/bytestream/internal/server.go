// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

// This file contains the server implementation of Bytestream declared at:
// https://github.com/googleapis/googleapis/blob/master/google/bytestream/bytestream.proto
//
// Bytestream uses bidirectional streaming (http://grpc.io/docs/guides/concepts.html#bidirectional-streaming-rpc).

import (
	"fmt"
	"io"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"

	pb "google.golang.org/genproto/googleapis/bytestream"
)

// ReadHandler reads from the Bytestream.
// Note: error returns must return an instance of grpc.rpcError unless otherwise handled in grpc-go/rpc_util.go.
// http://google.golang.org/grpc provides Errorf(code, fmt, ...) to create instances of grpc.rpcError.
// Note: Cancelling the context will abort the stream ("drop the connection"). Consider returning a non-nil error instead.
type ReadHandler interface {
	// GetReader provides an io.ReaderAt, which will not be retained by the Server after the pb.ReadRequest.
	GetReader(ctx context.Context, name string) (io.ReaderAt, error)
	// Close does not have to do anything, but is here for if the io.ReaderAt wants to call Close().
	Close(ctx context.Context, name string) error
}

// WriteHandler handles writes from the Bytestream. For example:
// Note: error returns must return an instance of grpc.rpcError unless otherwise handled in grpc-go/rpc_util.go.
// grpc-go/rpc_util.go provides the helper func Errorf(code, fmt, ...) to create instances of grpc.rpcError.
// Note: Cancelling the context will abort the stream ("drop the connection"). Consider returning a non-nil error instead.
type WriteHandler interface {
	// GetWriter provides an io.Writer that is ready to write at initOffset.
	// The io.Writer will not be retained by the Server after the pb.WriteRequest.
	GetWriter(ctx context.Context, name string, initOffset int64) (io.Writer, error)
	// Close does not have to do anything, but is related to Server.AllowOverwrite. Or if the io.Writer simply wants a Close() call.
	// Close is called when the server receives a pb.WriteRequest with finish_write = true.
	// If Server.AllowOverwrite == true then Close() followed by GetWriter() for the same name indicates the name is being overwritten, even if the initOffset is different.
	Close(ctx context.Context, name string) error
}

// Internal service that implements pb.ByteStreamServer. Because the methods Write() and Read() are exported for grpc to link against,
// grpcService is deliberately not exported so go code cannot call grpcService.Write() or grpcService.Read().
type grpcService struct {
	parent *Server
}

// Server wraps the RPCs in pb. Use bytestream.NewServer() to create a Server.
type Server struct {
	status       map[string]*pb.QueryWriteStatusResponse
	readHandler  ReadHandler
	writeHandler WriteHandler
	rpc          *grpcService

	// AllowOverwrite controls Server behavior when a WriteRequest with finish_write = true is followed by another WriteRequest.
	AllowOverwrite bool

	// Bytestream allows a WriteRequest to omit the resource name, in which case it will be appended to the last WriteRequest.
	LastWrittenResource string
}

// NewServer creates a new bytestream.Server using gRPC.
// gsrv is the *grpc.Server this bytestream.Server will listen on.
// readHandler handles any incoming pb.ReadRequest or nil which means all pb.ReadRequests will be rejected.
// writeHandler handles any incoming pb.WriteRequest or nil which means all pb.WriteRequests will be rejected.
// readHandler and writeHandler cannot both be nil.
func NewServer(gsrv *grpc.Server, readHandler ReadHandler, writeHandler WriteHandler) (*Server, error) {
	if readHandler == nil && writeHandler == nil {
		return nil, fmt.Errorf("readHandler and writeHandler cannot both be nil")
	}

	server := &Server{
		status:       make(map[string]*pb.QueryWriteStatusResponse),
		readHandler:  readHandler,
		writeHandler: writeHandler,
		rpc:          &grpcService{},
	}
	server.rpc.parent = server

	// Register a server.
	pb.RegisterByteStreamServer(gsrv, server.rpc)

	return server, nil
}

// Write handles the pb.ByteStream_WriteServer and sends a pb.WriteResponse
// Implements bytestream.proto "rpc Write(stream WriteRequest) returns (WriteResponse)".
func (rpc *grpcService) Write(stream pb.ByteStream_WriteServer) error {
	for {
		writeReq, err := stream.Recv()
		if err == io.EOF {
			// io.EOF errors are a non-error for the Write() caller.
			return nil
		} else if err != nil {
			return grpc.Errorf(codes.Unknown, "stream.Recv() failed: %v", err)
		}
		if rpc.parent.writeHandler == nil {
			return grpc.Errorf(codes.Unimplemented, "instance of NewServer(writeHandler = nil) rejects all writes")
		}

		status, ok := rpc.parent.status[writeReq.ResourceName]
		if !ok {
			// writeReq.ResourceName is a new resource name.
			if writeReq.ResourceName == "" {
				return grpc.Errorf(codes.InvalidArgument, "WriteRequest: empty or missing resource_name")
			}
			status = &pb.QueryWriteStatusResponse{
				CommittedSize: writeReq.WriteOffset,
			}
			rpc.parent.status[writeReq.ResourceName] = status
		} else {
			// writeReq.ResourceName has already been seen by this server.
			if status.Complete {
				if !rpc.parent.AllowOverwrite {
					return grpc.Errorf(codes.InvalidArgument, "%q finish_write = true already, got %d byte WriteRequest and Server.AllowOverwrite = false",
						writeReq.ResourceName, len(writeReq.Data))
				}
				// Truncate the resource stream.
				status.Complete = false
				status.CommittedSize = writeReq.WriteOffset
			}
		}

		if writeReq.WriteOffset != status.CommittedSize {
			return grpc.Errorf(codes.FailedPrecondition, "%q write_offset=%d differs from server internal committed_size=%d",
				writeReq.ResourceName, writeReq.WriteOffset, status.CommittedSize)
		}

		// WriteRequest with empty data is ok.
		if len(writeReq.Data) != 0 {
			writer, err := rpc.parent.writeHandler.GetWriter(stream.Context(), writeReq.ResourceName, status.CommittedSize)
			if err != nil {
				return grpc.Errorf(codes.Internal, "GetWriter(%q): %v", writeReq.ResourceName, err)
			}
			wroteLen, err := writer.Write(writeReq.Data)
			if err != nil {
				return grpc.Errorf(codes.Internal, "Write(%q): %v", writeReq.ResourceName, err)
			}
			status.CommittedSize += int64(wroteLen)
		}

		if writeReq.FinishWrite {
			r := &pb.WriteResponse{CommittedSize: status.CommittedSize}
			// Note: SendAndClose does NOT close the server stream.
			if err = stream.SendAndClose(r); err != nil {
				return grpc.Errorf(codes.Internal, "stream.SendAndClose(%q, WriteResponse{ %d }): %v", writeReq.ResourceName, status.CommittedSize, err)
			}
			status.Complete = true
			if status.CommittedSize == 0 {
				return grpc.Errorf(codes.FailedPrecondition, "writeHandler.Close(%q): 0 bytes written", writeReq.ResourceName)
			}
			if err = rpc.parent.writeHandler.Close(stream.Context(), writeReq.ResourceName); err != nil {
				return grpc.Errorf(codes.Internal, "writeHandler.Close(%q): %v", writeReq.ResourceName, err)
			}
		}
	}
}

// QueryWriteStatus implements bytestream.proto "rpc QueryWriteStatus(QueryWriteStatusRequest) returns (QueryWriteStatusResponse)".
// QueryWriteStatus returns the CommittedSize known to the server.
func (rpc *grpcService) QueryWriteStatus(ctx context.Context, request *pb.QueryWriteStatusRequest) (*pb.QueryWriteStatusResponse, error) {
	s, ok := rpc.parent.status[request.ResourceName]
	if !ok {
		return nil, grpc.Errorf(codes.NotFound, "resource_name not found: QueryWriteStatusRequest %v", request)
	}
	return s, nil
}

func (rpc *grpcService) readFrom(request *pb.ReadRequest, reader io.ReaderAt, stream pb.ByteStream_ReadServer) error {
	limit := int(request.ReadLimit)
	if limit < 0 {
		return grpc.Errorf(codes.InvalidArgument, "Read(): read_limit=%d is invalid", limit)
	}
	offset := request.ReadOffset
	if offset < 0 {
		return grpc.Errorf(codes.InvalidArgument, "Read(): offset=%d is invalid", offset)
	}

	var buf []byte
	if limit > 0 {
		buf = make([]byte, limit)
	} else {
		buf = make([]byte, 1024*1024) // 1M buffer is reasonable.
	}
	bytesSent := 0
	for limit == 0 || bytesSent < limit {
		n, err := reader.ReadAt(buf, offset)
		if n > 0 {
			if err := stream.Send(&pb.ReadResponse{Data: buf[:n]}); err != nil {
				return grpc.Errorf(grpc.Code(err), "Send(resourceName=%q offset=%d): %v", request.ResourceName, offset, grpc.ErrorDesc(err))
			}
		} else if err == nil {
			return grpc.Errorf(codes.Internal, "nil error on empty read: io.ReaderAt contract violated")
		}
		offset += int64(n)
		bytesSent += n
		if err == io.EOF {
			break
		}
		if err != nil {
			return grpc.Errorf(codes.Unknown, "ReadAt(resourceName=%q offset=%d): %v", request.ResourceName, offset, err)
		}
	}
	return nil
}

// Read handles a pb.ReadRequest sending bytes to the pb.ByteStream_ReadServer
// Implements bytestream.proto "rpc Read(ReadRequest) returns (stream ReadResponse)"
func (rpc *grpcService) Read(request *pb.ReadRequest, stream pb.ByteStream_ReadServer) error {
	if rpc.parent.readHandler == nil {
		return grpc.Errorf(codes.Unimplemented, "instance of NewServer(readHandler = nil) rejects all reads")
	}
	if request == nil {
		return grpc.Errorf(codes.Internal, "Read(ReadRequest == nil)")
	}
	if request.ResourceName == "" {
		return grpc.Errorf(codes.InvalidArgument, "ReadRequest: empty or missing resource_name")
	}

	reader, err := rpc.parent.readHandler.GetReader(stream.Context(), request.ResourceName)
	if err != nil {
		return err
	}
	if err = rpc.readFrom(request, reader, stream); err != nil {
		rpc.parent.readHandler.Close(stream.Context(), request.ResourceName)
		return err
	}
	if err = rpc.parent.readHandler.Close(stream.Context(), request.ResourceName); err != nil {
		return err
	}
	return nil
}
