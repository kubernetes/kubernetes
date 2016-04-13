// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ctxhttp provides helper functions for performing context-aware HTTP requests.
package ctxhttp // import "golang.org/x/net/context/ctxhttp"

import (
	"io"
	"net/http"
	"net/url"
	"strings"

	"golang.org/x/net/context"
)

// Do sends an HTTP request with the provided http.Client and returns an HTTP response.
// If the client is nil, http.DefaultClient is used.
// If the context is canceled or times out, ctx.Err() will be returned.
func Do(ctx context.Context, client *http.Client, req *http.Request) (*http.Response, error) {
	if client == nil {
		client = http.DefaultClient
	}

	// Request cancelation changed in Go 1.5, see cancelreq.go and cancelreq_go14.go.
	cancel := canceler(client, req)

	type responseAndError struct {
		resp *http.Response
		err  error
	}
	result := make(chan responseAndError, 1)

	go func() {
		resp, err := client.Do(req)
		result <- responseAndError{resp, err}
	}()

	select {
	case <-ctx.Done():
		cancel()
		return nil, ctx.Err()
	case r := <-result:
		return r.resp, r.err
	}
}

// Get issues a GET request via the Do function.
func Get(ctx context.Context, client *http.Client, url string) (*http.Response, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	return Do(ctx, client, req)
}

// Head issues a HEAD request via the Do function.
func Head(ctx context.Context, client *http.Client, url string) (*http.Response, error) {
	req, err := http.NewRequest("HEAD", url, nil)
	if err != nil {
		return nil, err
	}
	return Do(ctx, client, req)
}

// Post issues a POST request via the Do function.
func Post(ctx context.Context, client *http.Client, url string, bodyType string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", bodyType)
	return Do(ctx, client, req)
}

// PostForm issues a POST request via the Do function.
func PostForm(ctx context.Context, client *http.Client, url string, data url.Values) (*http.Response, error) {
	return Post(ctx, client, url, "application/x-www-form-urlencoded", strings.NewReader(data.Encode()))
}
