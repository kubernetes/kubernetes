// Copyright 2012 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package transport contains HTTP transports used to make
// authenticated API requests.
package transport

import (
	"errors"
	"net/http"
)

// APIKey is an HTTP Transport which wraps an underlying transport and
// appends an API Key "key" parameter to the URL of outgoing requests.
type APIKey struct {
	// Key is the API Key to set on requests.
	Key string

	// Transport is the underlying HTTP transport.
	// If nil, http.DefaultTransport is used.
	Transport http.RoundTripper
}

func (t *APIKey) RoundTrip(req *http.Request) (*http.Response, error) {
	rt := t.Transport
	if rt == nil {
		rt = http.DefaultTransport
		if rt == nil {
			return nil, errors.New("googleapi/transport: no Transport specified or available")
		}
	}
	newReq := *req
	args := newReq.URL.Query()
	args.Set("key", t.Key)
	newReq.URL.RawQuery = args.Encode()
	return rt.RoundTrip(&newReq)
}
