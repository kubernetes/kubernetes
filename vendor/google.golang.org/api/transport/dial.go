// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package transport supports network connections to HTTP and GRPC servers.
// This package is not intended for use by end developers. Use the
// google.golang.org/api/option package to configure API clients.
package transport

import (
	"net/http"

	"golang.org/x/net/context"
	"google.golang.org/grpc"

	"google.golang.org/api/option"
	gtransport "google.golang.org/api/transport/grpc"
	htransport "google.golang.org/api/transport/http"
)

// NewHTTPClient returns an HTTP client for use communicating with a Google cloud
// service, configured with the given ClientOptions. It also returns the endpoint
// for the service as specified in the options.
func NewHTTPClient(ctx context.Context, opts ...option.ClientOption) (*http.Client, string, error) {
	return htransport.NewClient(ctx, opts...)
}

// DialGRPC returns a GRPC connection for use communicating with a Google cloud
// service, configured with the given ClientOptions.
func DialGRPC(ctx context.Context, opts ...option.ClientOption) (*grpc.ClientConn, error) {
	return gtransport.Dial(ctx, opts...)
}

// DialGRPCInsecure returns an insecure GRPC connection for use communicating
// with fake or mock Google cloud service implementations, such as emulators.
// The connection is configured with the given ClientOptions.
func DialGRPCInsecure(ctx context.Context, opts ...option.ClientOption) (*grpc.ClientConn, error) {
	return gtransport.DialInsecure(ctx, opts...)
}
