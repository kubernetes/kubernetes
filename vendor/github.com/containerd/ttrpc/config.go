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
)

type serverConfig struct {
	handshaker  Handshaker
	interceptor UnaryServerInterceptor
}

// ServerOpt for configuring a ttrpc server
type ServerOpt func(*serverConfig) error

// WithServerHandshaker can be passed to NewServer to ensure that the
// handshaker is called before every connection attempt.
//
// Only one handshaker is allowed per server.
func WithServerHandshaker(handshaker Handshaker) ServerOpt {
	return func(c *serverConfig) error {
		if c.handshaker != nil {
			return errors.New("only one handshaker allowed per server")
		}
		c.handshaker = handshaker
		return nil
	}
}

// WithUnaryServerInterceptor sets the provided interceptor on the server
func WithUnaryServerInterceptor(i UnaryServerInterceptor) ServerOpt {
	return func(c *serverConfig) error {
		if c.interceptor != nil {
			return errors.New("only one unchained interceptor allowed per server")
		}
		c.interceptor = i
		return nil
	}
}

// WithChainUnaryServerInterceptor sets the provided chain of server interceptors
func WithChainUnaryServerInterceptor(interceptors ...UnaryServerInterceptor) ServerOpt {
	return func(c *serverConfig) error {
		if len(interceptors) == 0 {
			return nil
		}
		if c.interceptor != nil {
			interceptors = append([]UnaryServerInterceptor{c.interceptor}, interceptors...)
		}
		c.interceptor = func(
			ctx context.Context,
			unmarshal Unmarshaler,
			info *UnaryServerInfo,
			method Method) (interface{}, error) {
			return interceptors[0](ctx, unmarshal, info,
				chainUnaryServerInterceptors(info, method, interceptors[1:]))
		}
		return nil
	}
}

func chainUnaryServerInterceptors(info *UnaryServerInfo, method Method, interceptors []UnaryServerInterceptor) Method {
	if len(interceptors) == 0 {
		return method
	}
	return func(ctx context.Context, unmarshal func(interface{}) error) (interface{}, error) {
		return interceptors[0](ctx, unmarshal, info,
			chainUnaryServerInterceptors(info, method, interceptors[1:]))
	}
}
