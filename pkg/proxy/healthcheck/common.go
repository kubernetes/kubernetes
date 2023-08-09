/*
Copyright 2016 The Kubernetes Authors.

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

package healthcheck

import (
	"net"
	"net/http"
)

// listener allows for testing of ServiceHealthServer and ProxierHealthServer.
type listener interface {
	// Listen is very much like net.Listen, except the first arg (network) is
	// fixed to be "tcp".
	Listen(addr string) (net.Listener, error)
}

// httpServerFactory allows for testing of ServiceHealthServer and ProxierHealthServer.
type httpServerFactory interface {
	// New creates an instance of a type satisfying HTTPServer.  This is
	// designed to include http.Server.
	New(addr string, handler http.Handler) httpServer
}

// httpServer allows for testing of ServiceHealthServer and ProxierHealthServer.
// It is designed so that http.Server satisfies this interface,
type httpServer interface {
	Serve(listener net.Listener) error
}

// Implement listener in terms of net.Listen.
type stdNetListener struct{}

func (stdNetListener) Listen(addr string) (net.Listener, error) {
	return net.Listen("tcp", addr)
}

var _ listener = stdNetListener{}

// Implement httpServerFactory in terms of http.Server.
type stdHTTPServerFactory struct{}

func (stdHTTPServerFactory) New(addr string, handler http.Handler) httpServer {
	return &http.Server{
		Addr:    addr,
		Handler: handler,
	}
}

var _ httpServerFactory = stdHTTPServerFactory{}
