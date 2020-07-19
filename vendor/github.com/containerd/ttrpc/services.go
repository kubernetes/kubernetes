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
	"io"
	"os"
	"path"

	"github.com/gogo/protobuf/proto"
	"github.com/pkg/errors"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type Method func(ctx context.Context, unmarshal func(interface{}) error) (interface{}, error)

type ServiceDesc struct {
	Methods map[string]Method

	// TODO(stevvooe): Add stream support.
}

type serviceSet struct {
	services    map[string]ServiceDesc
	interceptor UnaryServerInterceptor
}

func newServiceSet(interceptor UnaryServerInterceptor) *serviceSet {
	return &serviceSet{
		services:    make(map[string]ServiceDesc),
		interceptor: interceptor,
	}
}

func (s *serviceSet) register(name string, methods map[string]Method) {
	if _, ok := s.services[name]; ok {
		panic(errors.Errorf("duplicate service %v registered", name))
	}

	s.services[name] = ServiceDesc{
		Methods: methods,
	}
}

func (s *serviceSet) call(ctx context.Context, serviceName, methodName string, p []byte) ([]byte, *status.Status) {
	p, err := s.dispatch(ctx, serviceName, methodName, p)
	st, ok := status.FromError(err)
	if !ok {
		st = status.New(convertCode(err), err.Error())
	}

	return p, st
}

func (s *serviceSet) dispatch(ctx context.Context, serviceName, methodName string, p []byte) ([]byte, error) {
	method, err := s.resolve(serviceName, methodName)
	if err != nil {
		return nil, err
	}

	unmarshal := func(obj interface{}) error {
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

	info := &UnaryServerInfo{
		FullMethod: fullPath(serviceName, methodName),
	}

	resp, err := s.interceptor(ctx, unmarshal, info, method)
	if err != nil {
		return nil, err
	}

	switch v := resp.(type) {
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

func (s *serviceSet) resolve(service, method string) (Method, error) {
	srv, ok := s.services[service]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "service %v", service)
	}

	mthd, ok := srv.Methods[method]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "method %v", method)
	}

	return mthd, nil
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
