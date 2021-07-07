// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"time"
)

// Hook is the type of a function that is called once before each HTTP request
// that is sent by a generated API.  It returns a function that is called after
// the request returns.
// Hooks are not called if the context is nil.
type Hook func(ctx context.Context, req *http.Request) func(resp *http.Response)

var hooks []Hook

// RegisterHook registers a Hook to be called before each HTTP request by a
// generated API.  Hooks are called in the order they are registered.  Each
// hook can return a function; if it is non-nil, it is called after the HTTP
// request returns.  These functions are called in the reverse order.
// RegisterHook should not be called concurrently with itself or SendRequest.
func RegisterHook(h Hook) {
	hooks = append(hooks, h)
}

// SendRequest sends a single HTTP request using the given client.
// If ctx is non-nil, it calls all hooks, then sends the request with
// req.WithContext, then calls any functions returned by the hooks in
// reverse order.
func SendRequest(ctx context.Context, client *http.Client, req *http.Request) (*http.Response, error) {
	// Disallow Accept-Encoding because it interferes with the automatic gzip handling
	// done by the default http.Transport. See https://github.com/google/google-api-go-client/issues/219.
	if _, ok := req.Header["Accept-Encoding"]; ok {
		return nil, errors.New("google api: custom Accept-Encoding headers not allowed")
	}
	if ctx == nil {
		return client.Do(req)
	}
	// Call hooks in order of registration, store returned funcs.
	post := make([]func(resp *http.Response), len(hooks))
	for i, h := range hooks {
		fn := h(ctx, req)
		post[i] = fn
	}

	// Send request.
	resp, err := send(ctx, client, req)

	// Call returned funcs in reverse order.
	for i := len(post) - 1; i >= 0; i-- {
		if fn := post[i]; fn != nil {
			fn(resp)
		}
	}
	return resp, err
}

func send(ctx context.Context, client *http.Client, req *http.Request) (*http.Response, error) {
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req.WithContext(ctx))
	// If we got an error, and the context has been canceled,
	// the context's error is probably more useful.
	if err != nil {
		select {
		case <-ctx.Done():
			err = ctx.Err()
		default:
		}
	}
	return resp, err
}

// SendRequestWithRetry sends a single HTTP request using the given client,
// with retries if a retryable error is returned.
// If ctx is non-nil, it calls all hooks, then sends the request with
// req.WithContext, then calls any functions returned by the hooks in
// reverse order.
func SendRequestWithRetry(ctx context.Context, client *http.Client, req *http.Request) (*http.Response, error) {
	// Disallow Accept-Encoding because it interferes with the automatic gzip handling
	// done by the default http.Transport. See https://github.com/google/google-api-go-client/issues/219.
	if _, ok := req.Header["Accept-Encoding"]; ok {
		return nil, errors.New("google api: custom Accept-Encoding headers not allowed")
	}
	if ctx == nil {
		return client.Do(req)
	}
	// Call hooks in order of registration, store returned funcs.
	post := make([]func(resp *http.Response), len(hooks))
	for i, h := range hooks {
		fn := h(ctx, req)
		post[i] = fn
	}

	// Send request with retry.
	resp, err := sendAndRetry(ctx, client, req)

	// Call returned funcs in reverse order.
	for i := len(post) - 1; i >= 0; i-- {
		if fn := post[i]; fn != nil {
			fn(resp)
		}
	}
	return resp, err
}

func sendAndRetry(ctx context.Context, client *http.Client, req *http.Request) (*http.Response, error) {
	if client == nil {
		client = http.DefaultClient
	}

	var resp *http.Response
	var err error

	// Loop to retry the request, up to the context deadline.
	var pause time.Duration
	bo := backoff()

	for {
		select {
		case <-ctx.Done():
			// If we got an error, and the context has been canceled,
			// the context's error is probably more useful.
			if err == nil {
				err = ctx.Err()
			}
			return resp, err
		case <-time.After(pause):
		}

		resp, err = client.Do(req.WithContext(ctx))

		var status int
		if resp != nil {
			status = resp.StatusCode
		}

		// Check if we can retry the request. A retry can only be done if the error
		// is retryable and the request body can be re-created using GetBody (this
		// will not be possible if the body was unbuffered).
		if req.GetBody == nil || !shouldRetry(status, err) {
			break
		}
		var errBody error
		req.Body, errBody = req.GetBody()
		if errBody != nil {
			break
		}

		pause = bo.Pause()
		if resp != nil && resp.Body != nil {
			resp.Body.Close()
		}
	}
	return resp, err
}

// DecodeResponse decodes the body of res into target. If there is no body,
// target is unchanged.
func DecodeResponse(target interface{}, res *http.Response) error {
	if res.StatusCode == http.StatusNoContent {
		return nil
	}
	return json.NewDecoder(res.Body).Decode(target)
}
