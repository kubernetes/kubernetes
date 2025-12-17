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
	"sync"
	"sync/atomic"
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
	errRequestBodyTooLarge   = errors.New("request body too large for retry")
	errRetryAlreadyAttempted = errors.New("GetBody max retries attempted")
)

// retryableBodyConfig configures retry behavior
type retryableBodyConfig struct {
	maxRetryBytes int
	maxAttempts   int
}

// defaultRetryableBodyConfig returns defaults
func defaultRetryableBodyConfig() retryableBodyConfig {
	return retryableBodyConfig{
		maxRetryBytes: defaultRetryableBodyLimit,
		maxAttempts:   defaultMaxRetryAttempts,
	}
}

// WrapBodyForRetry wraps a request body to support HTTP/2 GOAWAY retries.
// Returns the wrapped body and a GetBody function.
func wrapBodyForRetry(originalBody io.ReadCloser, config retryableBodyConfig) (io.ReadCloser, func() (io.ReadCloser, error)) {
	if originalBody == nil {
		return nil, nil
	}

	var buf bytes.Buffer
	var exceeded atomic.Bool
	var attempts atomic.Int32

	lw := &limitedWriter{buf: &buf, limit: config.maxRetryBytes, exceeded: &exceeded}

	wrappedBody := io.NopCloser(io.TeeReader(originalBody, lw))

	getBody := func() (io.ReadCloser, error) {
		if exceeded.Load() {
			return nil, errRequestBodyTooLarge
		}
		if int(attempts.Add(1)) > config.maxAttempts {
			return nil, errRetryAlreadyAttempted
		}
		return io.NopCloser(io.MultiReader(bytes.NewReader(lw.bytes()), originalBody)), nil
	}

	return wrappedBody, getBody
}

// limitedWriter buffers up to limit bytes; once exceeded it discards further writes.
type limitedWriter struct {
	mu       sync.Mutex
	buf      *bytes.Buffer
	limit    int
	exceeded *atomic.Bool
}

func (lw *limitedWriter) Write(p []byte) (n int, err error) {
	lw.mu.Lock()
	defer lw.mu.Unlock()

	if lw.exceeded.Load() {
		return len(p), nil
	}

	if lw.buf.Len()+len(p) > lw.limit {
		lw.exceeded.Store(true)
		lw.buf = nil
		return len(p), nil
	}
	return lw.buf.Write(p)
}

// bytes returns a copy of the buffered data, safe for concurrent use
func (lw *limitedWriter) bytes() []byte {
	lw.mu.Lock()
	defer lw.mu.Unlock()
	if lw.buf == nil {
		return nil
	}
	return lw.buf.Bytes()
}
