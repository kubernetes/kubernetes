// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/googleapis/gax-go/v2"
)

// Use this error type to return an error which allows introspection of both
// the context error and the error from the service.
type wrappedCallErr struct {
	ctxErr     error
	wrappedErr error
}

func (e wrappedCallErr) Error() string {
	return fmt.Sprintf("retry failed with %v; last error: %v", e.ctxErr, e.wrappedErr)
}

func (e wrappedCallErr) Unwrap() error {
	return e.wrappedErr
}

// Is allows errors.Is to match the error from the call as well as context
// sentinel errors.
func (e wrappedCallErr) Is(target error) bool {
	return errors.Is(e.ctxErr, target) || errors.Is(e.wrappedErr, target)
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
	return send(ctx, client, req)
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
func SendRequestWithRetry(ctx context.Context, client *http.Client, req *http.Request, retry *RetryConfig) (*http.Response, error) {
	// Disallow Accept-Encoding because it interferes with the automatic gzip handling
	// done by the default http.Transport. See https://github.com/google/google-api-go-client/issues/219.
	if _, ok := req.Header["Accept-Encoding"]; ok {
		return nil, errors.New("google api: custom Accept-Encoding headers not allowed")
	}
	if ctx == nil {
		return client.Do(req)
	}
	return sendAndRetry(ctx, client, req, retry)
}

func sendAndRetry(ctx context.Context, client *http.Client, req *http.Request, retry *RetryConfig) (*http.Response, error) {
	if client == nil {
		client = http.DefaultClient
	}

	var resp *http.Response
	var err error
	attempts := 1
	invocationID := uuid.New().String()
	baseXGoogHeader := req.Header.Get("X-Goog-Api-Client")

	// Loop to retry the request, up to the context deadline.
	var pause time.Duration
	var bo Backoff
	if retry != nil && retry.Backoff != nil {
		bo = &gax.Backoff{
			Initial:    retry.Backoff.Initial,
			Max:        retry.Backoff.Max,
			Multiplier: retry.Backoff.Multiplier,
		}
	} else {
		bo = backoff()
	}

	var errorFunc = retry.errorFunc()

	for {
		select {
		case <-ctx.Done():
			// If we got an error and the context has been canceled, return an error acknowledging
			// both the context cancelation and the service error.
			if err != nil {
				return resp, wrappedCallErr{ctx.Err(), err}
			}
			return resp, ctx.Err()
		case <-time.After(pause):
		}

		if ctx.Err() != nil {
			// Check for context cancellation once more. If more than one case in a
			// select is satisfied at the same time, Go will choose one arbitrarily.
			// That can cause an operation to go through even if the context was
			// canceled before.
			if err != nil {
				return resp, wrappedCallErr{ctx.Err(), err}
			}
			return resp, ctx.Err()
		}
		invocationHeader := fmt.Sprintf("gccl-invocation-id/%s gccl-attempt-count/%d", invocationID, attempts)
		xGoogHeader := strings.Join([]string{invocationHeader, baseXGoogHeader}, " ")
		req.Header.Set("X-Goog-Api-Client", xGoogHeader)

		resp, err = client.Do(req.WithContext(ctx))

		var status int
		if resp != nil {
			status = resp.StatusCode
		}

		// Check if we can retry the request. A retry can only be done if the error
		// is retryable and the request body can be re-created using GetBody (this
		// will not be possible if the body was unbuffered).
		if req.GetBody == nil || !errorFunc(status, err) {
			break
		}
		attempts++
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
