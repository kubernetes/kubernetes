/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

package ttrpc

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"unsafe"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

type Method func(ctx context.Context, unmarshal func(interface{}) error) (interface{}, error)

type StreamHandler func(context.Context, StreamServer) (interface{}, error)

type Stream struct {
	Handler         StreamHandler
	StreamingClient bool
	StreamingServer bool
}

type ServiceDesc struct {
	Methods map[string]Method
	Streams map[string]Stream
}

type serviceSet struct {
	services          map[string]*ServiceDesc
	unaryInterceptor  UnaryServerInterceptor
	streamInterceptor StreamServerInterceptor
}

func newServiceSet(interceptor UnaryServerInterceptor) *serviceSet {
	return &serviceSet{
		services:          make(map[string]*ServiceDesc),
		unaryInterceptor:  interceptor,
		streamInterceptor: defaultStreamServerInterceptor,
	}
}

func (s *serviceSet) register(name string, desc *ServiceDesc) {
	if _, ok := s.services[name]; ok {
		panic(fmt.Errorf("duplicate service %v registered", name))
	}

	s.services[name] = desc
}

func (s *serviceSet) unaryCall(ctx context.Context, method Method, info *UnaryServerInfo, data []byte) (p []byte, st *status.Status) {
	unmarshal := func(obj interface{}) error {
		return protoUnmarshal(data, obj)
	}

	resp, err := s.unaryInterceptor(ctx, unmarshal, info, method)
	if err == nil {
		if isNil(resp) {
			err = errors.New("ttrpc: marshal called with nil")
		} else {
			p, err = protoMarshal(resp)
		}
	}

	st, ok := status.FromError(err)
	if !ok {
		st = status.New(convertCode(err), err.Error())
	}

	return p, st
}

func (s *serviceSet) streamCall(ctx context.Context, stream StreamHandler, info *StreamServerInfo, ss StreamServer) (p []byte, st *status.Status) {
	resp, err := s.streamInterceptor(ctx, ss, info, stream)
	if err == nil {
		p, err = protoMarshal(resp)
	}
	st, ok := status.FromError(err)
	if !ok {
		st = status.New(convertCode(err), err.Error())
	}
	return
}

func (s *serviceSet) handle(ctx context.Context, req *Request, respond func(*status.Status, []byte, bool, bool) error) (*streamHandler, error) {
	srv, ok := s.services[req.Service]
	if !ok {
		return nil, status.Errorf(codes.Unimplemented, "service %v", req.Service)
	}

	if method, ok := srv.Methods[req.Method]; ok {
		go func() {
			ctx, cancel := getRequestContext(ctx, req)
			defer cancel()

			info := &UnaryServerInfo{
				FullMethod: fullPath(req.Service, req.Method),
			}
			p, st := s.unaryCall(ctx, method, info, req.Payload)

			respond(st, p, false, true)
		}()
		return nil, nil
	}
	if stream, ok := srv.Streams[req.Method]; ok {
		ctx, cancel := getRequestContext(ctx, req)
		info := &StreamServerInfo{
			FullMethod:      fullPath(req.Service, req.Method),
			StreamingClient: stream.StreamingClient,
			StreamingServer: stream.StreamingServer,
		}
		sh := &streamHandler{
			ctx:     ctx,
			respond: respond,
			recv:    make(chan Unmarshaler, 5),
			info:    info,
		}
		go func() {
			defer cancel()
			p, st := s.streamCall(ctx, stream.Handler, info, sh)
			respond(st, p, stream.StreamingServer, true)
		}()

		if req.Payload != nil {
			unmarshal := func(obj interface{}) error {
				return protoUnmarshal(req.Payload, obj)
			}
			if err := sh.data(unmarshal); err != nil {
				return nil, err
			}
		}

		return sh, nil
	}
	return nil, status.Errorf(codes.Unimplemented, "method %v", req.Method)
}

type streamHandler struct {
	ctx     context.Context
	respond func(*status.Status, []byte, bool, bool) error
	recv    chan Unmarshaler
	info    *StreamServerInfo

	remoteClosed bool
	localClosed  bool
}

func (s *streamHandler) closeSend() {
	if !s.remoteClosed {
		s.remoteClosed = true
		close(s.recv)
	}
}

func (s *streamHandler) data(unmarshal Unmarshaler) error {
	if s.remoteClosed {
		return ErrStreamClosed
	}
	select {
	case s.recv <- unmarshal:
		return nil
	case <-s.ctx.Done():
		return s.ctx.Err()
	}
}

func (s *streamHandler) SendMsg(m interface{}) error {
	if s.localClosed {
		return ErrStreamClosed
	}
	p, err := protoMarshal(m)
	if err != nil {
		return err
	}
	return s.respond(nil, p, true, false)
}

func (s *streamHandler) RecvMsg(m interface{}) error {
	select {
	case unmarshal, ok := <-s.recv:
		if !ok {
			return io.EOF
		}
		return unmarshal(m)
	case <-s.ctx.Done():
		return s.ctx.Err()

	}
}

func protoUnmarshal(p []byte, obj interface{}) error {
	switch v := obj.(type) {
	case proto.Message:
		if err := proto.Unmarshal(p, v); err != nil {
			return status.Errorf(codes.Internal, "ttrpc: error unmarshalling payload: %v", err.Error())
		}
	default:
		return status.Errorf(codes.Internal, "ttrpc: error unsupported request type: %T", v)
	}
	return nil
}

func protoMarshal(obj interface{}) ([]byte, error) {
	if obj == nil {
		return nil, nil
	}

	switch v := obj.(type) {
	case proto.Message:
		r, err := proto.Marshal(v)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "ttrpc: error marshaling payload: %v", err.Error())
		}

		return r, nil
	default:
		return nil, status.Errorf(codes.Internal, "ttrpc: error unsupported response type: %T", v)
	}
}

// convertCode maps stdlib go errors into grpc space.
//
// This is ripped from the grpc-go code base.
func convertCode(err error) codes.Code {
	switch err {
	case nil:
		return codes.OK
	case io.EOF:
		return codes.OutOfRange
	case io.ErrClosedPipe, io.ErrNoProgress, io.ErrShortBuffer, io.ErrShortWrite, io.ErrUnexpectedEOF:
		return codes.FailedPrecondition
	case os.ErrInvalid:
		return codes.InvalidArgument
	case context.Canceled:
		return codes.Canceled
	case context.DeadlineExceeded:
		return codes.DeadlineExceeded
	}
	switch {
	case os.IsExist(err):
		return codes.AlreadyExists
	case os.IsNotExist(err):
		return codes.NotFound
	case os.IsPermission(err):
		return codes.PermissionDenied
	}
	return codes.Unknown
}

func fullPath(service, method string) string {
	return "/" + path.Join(service, method)
}

func isNil(resp interface{}) bool {
	return (*[2]uintptr)(unsafe.Pointer(&resp))[1] == 0
}
