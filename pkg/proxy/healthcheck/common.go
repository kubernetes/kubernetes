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
	"context"
	"net"
	"net/http"

	netutils "k8s.io/utils/net"
)

// listener allows for testing of ServiceHealthServer and ProxierHealthServer.
type listener interface {
	// Listen is very much like netutils.MultiListen, except the second arg (network) is
	// fixed to be "tcp".
	Listen(ctx context.Context, addrs ...string) (net.Listener, error)
}

// httpServerFactory allows for testing of ServiceHealthServer and ProxierHealthServer.
type httpServerFactory interface {
	// New creates an instance of a type satisfying HTTPServer.  This is
	// designed to include http.Server.
	New(handler http.Handler) httpServer
}

// httpServer allows for testing of ServiceHealthServer and ProxierHealthServer.
// It is designed so that http.Server satisfies this interface,
type httpServer interface {
	Serve(listener net.Listener) error
	Close() error
}

// Implement listener in terms of net.Listen.
type stdNetListener struct{}

func (stdNetListener) Listen(ctx context.Context, addrs ...string) (net.Listener, error) {
	return netutils.MultiListen(ctx, "tcp", addrs...)
}

var _ listener = stdNetListener{}

// Implement httpServerFactory in terms of http.Server.
type stdHTTPServerFactory struct{}

func (stdHTTPServerFactory) New(handler http.Handler) httpServer {
	return &http.Server{
		Handler: handler,
	}
}

var _ httpServerFactory = stdHTTPServerFactory{}
