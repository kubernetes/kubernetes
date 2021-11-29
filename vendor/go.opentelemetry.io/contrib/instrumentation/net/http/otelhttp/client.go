// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package otelhttp

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// DefaultClient is the default Client and is used by Get, Head, Post and PostForm.
// Please be careful of intitialization order - for example, if you change
// the global propagator, the DefaultClient might still be using the old one
var DefaultClient = &http.Client{Transport: NewTransport(http.DefaultTransport)}

// Get is a convenient replacement for http.Get that adds a span around the request.
func Get(ctx context.Context, url string) (resp *http.Response, err error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	return DefaultClient.Do(req)
}

// Head is a convenient replacement for http.Head that adds a span around the request.
func Head(ctx context.Context, url string) (resp *http.Response, err error) {
	req, err := http.NewRequestWithContext(ctx, "HEAD", url, nil)
	if err != nil {
		return nil, err
	}
	return DefaultClient.Do(req)
}

// Post is a convenient replacement for http.Post that adds a span around the request.
func Post(ctx context.Context, url, contentType string, body io.Reader) (resp *http.Response, err error) {
	req, err := http.NewRequestWithContext(ctx, "POST", url, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", contentType)
	return DefaultClient.Do(req)
}

// PostForm is a convenient replacement for http.PostForm that adds a span around the request.
func PostForm(ctx context.Context, url string, data url.Values) (resp *http.Response, err error) {
	return Post(ctx, url, "application/x-www-form-urlencoded", strings.NewReader(data.Encode()))
}
