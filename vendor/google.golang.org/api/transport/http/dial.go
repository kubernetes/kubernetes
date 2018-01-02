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

// Package transport/http supports network connections to HTTP servers.
// This package is not intended for use by end developers. Use the
// google.golang.org/api/option package to configure API clients.
package http

import (
	"errors"
	"net/http"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"google.golang.org/api/googleapi/transport"
	"google.golang.org/api/internal"
	"google.golang.org/api/option"
)

// NewClient returns an HTTP client for use communicating with a Google cloud
// service, configured with the given ClientOptions. It also returns the endpoint
// for the service as specified in the options.
func NewClient(ctx context.Context, opts ...option.ClientOption) (*http.Client, string, error) {
	var o internal.DialSettings
	for _, opt := range opts {
		opt.Apply(&o)
	}
	if o.GRPCConn != nil {
		return nil, "", errors.New("unsupported gRPC connection specified")
	}
	// TODO(cbro): consider injecting the User-Agent even if an explicit HTTP client is provided?
	if o.HTTPClient != nil {
		return o.HTTPClient, o.Endpoint, nil
	}
	if o.APIKey != "" {
		hc := &http.Client{
			Transport: &transport.APIKey{
				Key: o.APIKey,
				Transport: userAgentTransport{
					base:      baseTransport(ctx),
					userAgent: o.UserAgent,
				},
			},
		}
		return hc, o.Endpoint, nil
	}
	creds, err := internal.Creds(ctx, &o)
	if err != nil {
		return nil, "", err
	}
	hc := &http.Client{
		Transport: &oauth2.Transport{
			Source: creds.TokenSource,
			Base: userAgentTransport{
				base:      baseTransport(ctx),
				userAgent: o.UserAgent,
			},
		},
	}
	return hc, o.Endpoint, nil
}

type userAgentTransport struct {
	userAgent string
	base      http.RoundTripper
}

func (t userAgentTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	rt := t.base
	if rt == nil {
		return nil, errors.New("transport: no Transport specified")
	}
	if t.userAgent == "" {
		return rt.RoundTrip(req)
	}
	newReq := *req
	newReq.Header = make(http.Header)
	for k, vv := range req.Header {
		newReq.Header[k] = vv
	}
	// TODO(cbro): append to existing User-Agent header?
	newReq.Header["User-Agent"] = []string{t.userAgent}
	return rt.RoundTrip(&newReq)
}

// Set at init time by dial_appengine.go. If nil, we're not on App Engine.
var appengineUrlfetchHook func(context.Context) http.RoundTripper

// baseTransport returns the base HTTP transport.
// On App Engine, this is urlfetch.Transport, otherwise it's http.DefaultTransport.
func baseTransport(ctx context.Context) http.RoundTripper {
	if appengineUrlfetchHook != nil {
		return appengineUrlfetchHook(ctx)
	}
	return http.DefaultTransport
}
