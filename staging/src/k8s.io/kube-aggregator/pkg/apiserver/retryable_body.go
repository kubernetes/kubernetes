/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package apiserver

import (
	"bytes"
	"errors"
	"io"
)

const (
	// defaultRetryableBodyLimit is the max bytes buffered for HTTP/2 GOAWAY retries
	defaultRetryableBodyLimit = 32 * 1024 // 32KiB
	// defaultMaxRetryAttempts is the max GetBody calls allowed.
	// Follows the same heuristic from http2/transport.go:
	// https://go.googlesource.com/net/+/master/http2/transport.go#611
	defaultMaxRetryAttempts = 6
)

var (
	errRequestBodyTooLarge   = errors.New("http/2.0 request body too large for retry")
	errRetryAlreadyAttempted = errors.New("http/2.0 max retries attempted")
)

// retryableBodyConfig configures retry behavior
type retryableBodyConfig struct {
	limit       int
	maxAttempts int
}

// defaultRetryableBodyConfig returns defaults
func defaultRetryableBodyConfig() retryableBodyConfig {
	return retryableBodyConfig{
		limit:       defaultRetryableBodyLimit,
		maxAttempts: defaultMaxRetryAttempts,
	}
}

// WrapBodyForRetry wraps a request body to support HTTP/2 GOAWAY retries.
// Returns the wrapped body and a GetBody function.
func wrapBodyForRetry(originalBody io.ReadCloser, config retryableBodyConfig) (io.ReadCloser, func() (io.ReadCloser, error)) {
	if originalBody == nil {
		return nil, nil
	}

	var buf bytes.Buffer
	var exceeded bool
	var attempts int

	lw := &limitedWriter{buf: &buf, limit: config.limit, exceeded: &exceeded}

	wrappedBody := &retryableBody{
		reader: io.TeeReader(originalBody, lw),
		closer: originalBody,
	}

	getBody := func() (io.ReadCloser, error) {
		if exceeded {
			return nil, errRequestBodyTooLarge
		}
		if attempts >= config.maxAttempts {
			return nil, errRetryAlreadyAttempted
		}
		attempts++
		return io.NopCloser(io.MultiReader(bytes.NewReader(buf.Bytes()), originalBody)), nil
	}

	return wrappedBody, getBody
}

// limitedWriter buffers up to limit bytes; once exceeded it discards further writes.
type limitedWriter struct {
	buf      *bytes.Buffer
	limit    int
	exceeded *bool
}

func (lw *limitedWriter) Write(p []byte) (n int, err error) {
	if *lw.exceeded {
		return len(p), nil
	}
	if lw.buf.Len()+len(p) > lw.limit {
		*lw.exceeded = true
		lw.buf.Reset()
		return len(p), nil
	}
	return lw.buf.Write(p)
}

// retryableBody wraps a reader with a no-op Close
type retryableBody struct {
	reader io.Reader
	closer io.Closer
}

func (r *retryableBody) Read(p []byte) (n int, err error) {
	return r.reader.Read(p)
}

func (r *retryableBody) Close() error {
	// No-op. We delegate closing of the underlying body to "finishRequest".
	// https://go.googlesource.com/go/+/master/src/net/http/server.go#1680
	return nil
}
