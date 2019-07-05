// Copyright 2017 Google LLC
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

package gensupport

import (
	"context"
	"io"
	"net"
	"net/http"
	"time"
)

// Retry invokes the given function, retrying it multiple times if the connection failed or
// the HTTP status response indicates the request should be attempted again. ctx may be nil.
func Retry(ctx context.Context, f func() (*http.Response, error), backoff BackoffStrategy) (*http.Response, error) {
	for {
		resp, err := f()

		var status int
		if resp != nil {
			status = resp.StatusCode
		}

		// Return if we shouldn't retry.
		pause, retry := backoff.Pause()
		if !shouldRetry(status, err) || !retry {
			return resp, err
		}

		// Ensure the response body is closed, if any.
		if resp != nil && resp.Body != nil {
			resp.Body.Close()
		}

		// Pause, but still listen to ctx.Done if context is not nil.
		var done <-chan struct{}
		if ctx != nil {
			done = ctx.Done()
		}
		select {
		case <-done:
			return nil, ctx.Err()
		case <-time.After(pause):
		}
	}
}

// DefaultBackoffStrategy returns a default strategy to use for retrying failed upload requests.
func DefaultBackoffStrategy() BackoffStrategy {
	return &ExponentialBackoff{
		Base: 250 * time.Millisecond,
		Max:  16 * time.Second,
	}
}

// shouldRetry returns true if the HTTP response / error indicates that the
// request should be attempted again.
func shouldRetry(status int, err error) bool {
	if 500 <= status && status <= 599 {
		return true
	}
	if status == statusTooManyRequests {
		return true
	}
	if err == io.ErrUnexpectedEOF {
		return true
	}
	if err, ok := err.(net.Error); ok {
		return err.Temporary()
	}
	return false
}
