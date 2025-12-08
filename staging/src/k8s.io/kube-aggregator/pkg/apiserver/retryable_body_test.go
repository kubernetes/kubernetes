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
	"testing"
)

func TestWrapBodyForRetry(t *testing.T) {
	tests := []struct {
		name           string
		bodyContent    string
		limit          int
		maxAttempts    int
		wantGetBodyErr error
	}{
		{
			name:        "small body within limit",
			bodyContent: "hello world",
			limit:       1024,
			maxAttempts: 2,
		},
		{
			name:           "body exceeds limit",
			bodyContent:    "this body is way too large for the limit we set",
			limit:          10,
			maxAttempts:    2,
			wantGetBodyErr: errRequestBodyTooLarge,
		},
		{
			name:        "exact limit boundary",
			bodyContent: "exactly10!",
			limit:       10,
			maxAttempts: 2,
		},
		{
			name:        "empty body",
			bodyContent: "",
			limit:       1024,
			maxAttempts: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			originalBody := io.NopCloser(bytes.NewBufferString(tc.bodyContent))
			config := retryableBodyConfig{limit: tc.limit, maxAttempts: tc.maxAttempts}

			wrappedBody, getBody := wrapBodyForRetry(originalBody, config)
			if wrappedBody == nil {
				t.Fatal("expected non-nil wrapped body")
			}

			// Read the entire body through the wrapper
			_, err := io.ReadAll(wrappedBody)
			if err != nil {
				t.Errorf("unexpected read error: %v", err)
			}

			// Close the wrapped body (should be no-op)
			if err := wrappedBody.Close(); err != nil {
				t.Errorf("unexpected close error: %v", err)
			}

			retryBody, err := getBody()
			if tc.wantGetBodyErr != nil {
				if err == nil {
					t.Errorf("expected GetBody error %v, got none", tc.wantGetBodyErr)
				} else if !errors.Is(err, tc.wantGetBodyErr) {
					t.Errorf("expected GetBody error %v, got %v", tc.wantGetBodyErr, err)
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected GetBody error: %v", err)
				return
			}

			// Read the retry body and verify content
			retryData, err := io.ReadAll(retryBody)
			if err != nil {
				t.Errorf("unexpected retry body read error: %v", err)
				return
			}
			if string(retryData) != tc.bodyContent {
				t.Errorf("expected retry body %q, got %q", tc.bodyContent, string(retryData))
			}
		})
	}
}

func TestWrapBodyForRetryMaxAttempts(t *testing.T) {
	tests := []struct {
		name        string
		maxAttempts int
		callCount   int
		wantLastErr error
	}{
		{
			name:        "single attempt allowed, first call succeeds",
			maxAttempts: 1,
			callCount:   1,
			wantLastErr: nil,
		},
		{
			name:        "single attempt allowed, second call fails",
			maxAttempts: 1,
			callCount:   2,
			wantLastErr: errRetryAlreadyAttempted,
		},
		{
			name:        "six attempts allowed, all succeed",
			maxAttempts: 6,
			callCount:   6,
			wantLastErr: nil,
		},
		{
			name:        "six attempts allowed, seventh fails",
			maxAttempts: 6,
			callCount:   7,
			wantLastErr: errRetryAlreadyAttempted,
		},
		{
			name:        "default max attempts exceeded",
			maxAttempts: defaultMaxRetryAttempts,
			callCount:   defaultMaxRetryAttempts + 1,
			wantLastErr: errRetryAlreadyAttempted,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bodyContent := "test body content"
			originalBody := io.NopCloser(bytes.NewBufferString(bodyContent))
			config := retryableBodyConfig{limit: 1024, maxAttempts: tc.maxAttempts}

			wrappedBody, getBody := wrapBodyForRetry(originalBody, config)

			if _, err := io.ReadAll(wrappedBody); err != nil {
				t.Fatalf("unexpected read error: %v", err)
			}

			// Call GetBody multiple times
			var lastErr error
			for i := 0; i < tc.callCount; i++ {
				body, err := getBody()
				lastErr = err
				if err == nil && body != nil {
					body.Close()
				}
			}

			if tc.wantLastErr != nil {
				if lastErr == nil {
					t.Errorf("expected error %v on call %d, got none", tc.wantLastErr, tc.callCount)
				} else if !errors.Is(lastErr, tc.wantLastErr) {
					t.Errorf("expected error %v, got %v", tc.wantLastErr, lastErr)
				}
			} else {
				if lastErr != nil {
					t.Errorf("unexpected error on call %d: %v", tc.callCount, lastErr)
				}
			}
		})
	}
}

func TestLimitedWriterStopsBuffering(t *testing.T) {
	tests := []struct {
		name         string
		limit        int
		writes       []string
		wantExceeded bool
		wantBufLen   int
	}{
		{
			name:         "within limit",
			limit:        100,
			writes:       []string{"hello", " ", "world"},
			wantExceeded: false,
			wantBufLen:   11,
		},
		{
			name:         "exactly at limit",
			limit:        11,
			writes:       []string{"hello", " ", "world"},
			wantExceeded: false,
			wantBufLen:   11,
		},
		{
			name:         "exceeds on first write",
			limit:        3,
			writes:       []string{"hello"},
			wantExceeded: true,
		},
		{
			name:         "exceeds on later write",
			limit:        8,
			writes:       []string{"hello", " ", "world"},
			wantExceeded: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			exceeded := false
			lw := &limitedWriter{buf: &buf, limit: tc.limit, exceeded: &exceeded}

			for _, w := range tc.writes {
				n, err := lw.Write([]byte(w))
				if err != nil {
					t.Errorf("unexpected write error: %v", err)
				}
				if n != len(w) {
					t.Errorf("expected write length %d, got %d", len(w), n)
				}
			}

			if exceeded != tc.wantExceeded {
				t.Errorf("expected exceeded=%v, got %v", tc.wantExceeded, exceeded)
			}

			// When exceeded, buffer is nil; otherwise check length
			if tc.wantExceeded {
				if lw.buf != nil {
					t.Errorf("expected buffer to be nil after exceeding limit")
				}
			} else if lw.buf.Len() != tc.wantBufLen {
				t.Errorf("expected buffer length %d, got %d", tc.wantBufLen, lw.buf.Len())
			}
		})
	}
}
