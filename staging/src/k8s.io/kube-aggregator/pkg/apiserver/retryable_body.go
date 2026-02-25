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

// wrapBodyForRetry wraps a request body to support HTTP/2 GOAWAY retries.
// Returns the wrapped body and a GetBody function.
func wrapBodyForRetry(originalBody io.ReadCloser, config retryableBodyConfig) (io.ReadCloser, func() (io.ReadCloser, error)) {
	if originalBody == nil {
		return nil, nil
	}

	lw := &limitedWriter{buf: &bytes.Buffer{}, limit: config.maxRetryBytes}

	wrappedBody := io.NopCloser(io.TeeReader(originalBody, lw))

	getBody := func() (io.ReadCloser, error) {
		buffered, err := lw.snapshotForRetry(config.maxAttempts)
		if err != nil {
			return nil, err
		}
		return io.NopCloser(io.MultiReader(bytes.NewReader(buffered), io.TeeReader(originalBody, lw))), nil
	}

	return wrappedBody, getBody
}

// limitedWriter buffers up to limit bytes; once exceeded it discards further
// writes. All methods are safe for concurrent use.
type limitedWriter struct {
	mu       sync.Mutex
	buf      *bytes.Buffer
	limit    int
	exceeded bool
	attempts int
}

func (lw *limitedWriter) Write(p []byte) (n int, err error) {
	lw.mu.Lock()
	defer lw.mu.Unlock()

	if lw.exceeded {
		return len(p), nil
	}

	if lw.buf.Len()+len(p) > lw.limit {
		lw.exceeded = true
		lw.buf = nil
		return len(p), nil
	}
	return lw.buf.Write(p)
}

// snapshotForRetry checks the exceeded flag, increments the retry
// counter, and returns a copy of the buffered data. Combining all three under
// a single lock eliminates TOCTOU races between concurrent Write and GetBody
// calls.
func (lw *limitedWriter) snapshotForRetry(maxAttempts int) ([]byte, error) {
	lw.mu.Lock()
	defer lw.mu.Unlock()

	if lw.exceeded {
		return nil, errRequestBodyTooLarge
	}

	lw.attempts++
	if lw.attempts > maxAttempts {
		return nil, errRetryAlreadyAttempted
	}

	// Return a copy so the snapshot is decoupled from lw.buf's backing array,
	// which grows as TeeReader writes during retries.
	content := lw.buf.Bytes()
	contentCopy := make([]byte, len(content))
	copy(contentCopy, content)
	return contentCopy, nil
}
