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

package cloud

import (
	"net/http"

	"golang.org/x/oauth2"
	"google.golang.org/cloud/internal/opts"
	"google.golang.org/grpc"
)

// ClientOption is used when construct clients for each cloud service.
type ClientOption interface {
	// Resolve configures the given DialOpts for this option.
	Resolve(*opts.DialOpt)
}

// WithTokenSource returns a ClientOption that specifies an OAuth2 token
// source to be used as the basis for authentication.
func WithTokenSource(s oauth2.TokenSource) ClientOption {
	return withTokenSource{s}
}

type withTokenSource struct{ ts oauth2.TokenSource }

func (w withTokenSource) Resolve(o *opts.DialOpt) {
	o.TokenSource = w.ts
}

// WithEndpoint returns a ClientOption that overrides the default endpoint
// to be used for a service.
func WithEndpoint(url string) ClientOption {
	return withEndpoint(url)
}

type withEndpoint string

func (w withEndpoint) Resolve(o *opts.DialOpt) {
	o.Endpoint = string(w)
}

// WithScopes returns a ClientOption that overrides the default OAuth2 scopes
// to be used for a service.
func WithScopes(scope ...string) ClientOption {
	return withScopes(scope)
}

type withScopes []string

func (w withScopes) Resolve(o *opts.DialOpt) {
	s := make([]string, len(w))
	copy(s, w)
	o.Scopes = s
}

// WithUserAgent returns a ClientOption that sets the User-Agent.
func WithUserAgent(ua string) ClientOption {
	return withUA(ua)
}

type withUA string

func (w withUA) Resolve(o *opts.DialOpt) { o.UserAgent = string(w) }

// WithBaseHTTP returns a ClientOption that specifies the HTTP client to
// use as the basis of communications. This option may only be used with
// services that support HTTP as their communication transport.
func WithBaseHTTP(client *http.Client) ClientOption {
	return withBaseHTTP{client}
}

type withBaseHTTP struct{ client *http.Client }

func (w withBaseHTTP) Resolve(o *opts.DialOpt) {
	o.HTTPClient = w.client
}

// WithBaseGRPC returns a ClientOption that specifies the gRPC client
// connection to use as the basis of communications. This option many only be
// used with services that support gRPC as their communication transport.
func WithBaseGRPC(client *grpc.ClientConn) ClientOption {
	return withBaseGRPC{client}
}

type withBaseGRPC struct{ client *grpc.ClientConn }

func (w withBaseGRPC) Resolve(o *opts.DialOpt) {
	o.GRPCClient = w.client
}

// WithGRPCDialOption returns a ClientOption that appends a new grpc.DialOption
// to an underlying gRPC dial. It does not work with WithBaseGRPC.
func WithGRPCDialOption(opt grpc.DialOption) ClientOption {
	return withGRPCDialOption{opt}
}

type withGRPCDialOption struct{ opt grpc.DialOption }

func (w withGRPCDialOption) Resolve(o *opts.DialOpt) {
	o.GRPCDialOpts = append(o.GRPCDialOpts, w.opt)
}
