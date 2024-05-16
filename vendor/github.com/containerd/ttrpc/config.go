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

import "errors"

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
			return errors.New("only one interceptor allowed per server")
		}
		c.interceptor = i
		return nil
	}
}
