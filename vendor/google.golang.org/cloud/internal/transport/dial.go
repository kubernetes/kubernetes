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

package transport

import (
	"errors"
	"fmt"
	"net/http"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/cloud"
	"google.golang.org/cloud/internal/opts"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/oauth"
)

// ErrHTTP is returned when on a non-200 HTTP response.
type ErrHTTP struct {
	StatusCode int
	Body       []byte
	err        error
}

func (e *ErrHTTP) Error() string {
	if e.err == nil {
		return fmt.Sprintf("error during call, http status code: %v %s", e.StatusCode, e.Body)
	}
	return e.err.Error()
}

// NewHTTPClient returns an HTTP client for use communicating with a Google cloud
// service, configured with the given ClientOptions. It also returns the endpoint
// for the service as specified in the options.
func NewHTTPClient(ctx context.Context, opt ...cloud.ClientOption) (*http.Client, string, error) {
	var o opts.DialOpt
	for _, opt := range opt {
		opt.Resolve(&o)
	}
	if o.GRPCClient != nil {
		return nil, "", errors.New("unsupported GRPC base transport specified")
	}
	// TODO(djd): Wrap all http.Clients with appropriate internal version to add
	// UserAgent header and prepend correct endpoint.
	if o.HTTPClient != nil {
		return o.HTTPClient, o.Endpoint, nil
	}
	if o.TokenSource == nil {
		var err error
		o.TokenSource, err = google.DefaultTokenSource(ctx, o.Scopes...)
		if err != nil {
			return nil, "", fmt.Errorf("google.DefaultTokenSource: %v", err)
		}
	}
	return oauth2.NewClient(ctx, o.TokenSource), o.Endpoint, nil
}

// NewProtoClient returns a ProtoClient for communicating with a Google cloud service,
// configured with the given ClientOptions.
func NewProtoClient(ctx context.Context, opt ...cloud.ClientOption) (*ProtoClient, error) {
	var o opts.DialOpt
	for _, opt := range opt {
		opt.Resolve(&o)
	}
	if o.GRPCClient != nil {
		return nil, errors.New("unsupported GRPC base transport specified")
	}
	var client *http.Client
	switch {
	case o.HTTPClient != nil:
		if o.TokenSource != nil {
			return nil, errors.New("at most one of WithTokenSource or WithBaseHTTP may be provided")
		}
		client = o.HTTPClient
	case o.TokenSource != nil:
		client = oauth2.NewClient(ctx, o.TokenSource)
	default:
		var err error
		client, err = google.DefaultClient(ctx, o.Scopes...)
		if err != nil {
			return nil, err
		}
	}

	return &ProtoClient{
		client:    client,
		endpoint:  o.Endpoint,
		userAgent: o.UserAgent,
	}, nil
}

// DialGRPC returns a GRPC connection for use communicating with a Google cloud
// service, configured with the given ClientOptions.
func DialGRPC(ctx context.Context, opt ...cloud.ClientOption) (*grpc.ClientConn, error) {
	var o opts.DialOpt
	for _, opt := range opt {
		opt.Resolve(&o)
	}
	if o.HTTPClient != nil {
		return nil, errors.New("unsupported HTTP base transport specified")
	}
	if o.GRPCClient != nil {
		return o.GRPCClient, nil
	}
	if o.TokenSource == nil {
		var err error
		o.TokenSource, err = google.DefaultTokenSource(ctx, o.Scopes...)
		if err != nil {
			return nil, fmt.Errorf("google.DefaultTokenSource: %v", err)
		}
	}
	grpcOpts := []grpc.DialOption{
		grpc.WithPerRPCCredentials(oauth.TokenSource{o.TokenSource}),
		grpc.WithTransportCredentials(credentials.NewClientTLSFromCert(nil, "")),
	}
	grpcOpts = append(grpcOpts, o.GRPCDialOpts...)
	if o.UserAgent != "" {
		grpcOpts = append(grpcOpts, grpc.WithUserAgent(o.UserAgent))
	}
	return grpc.Dial(o.Endpoint, grpcOpts...)
}
