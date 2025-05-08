/*
Copyright 2025 The Kubernetes Authors.

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

package apitesting

import (
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"

	"github.com/google/go-cmp/cmp" //nolint:depguard // Test library
)

// errReadOnClosedResBody is returned by methods in the "http" package, when
// reading from a response body after it's been closed.
// Detecting this error is required because read is not cancellable.
// From https://github.com/golang/go/blob/go1.20/src/net/http/transport.go#L2779
var errReadOnClosedResBody = errors.New("http: read on closed response body")

// errCloseOnClosedWebSocket is returned by methods in the "k8s.io/utils/net"
// package, when accepting or closing a websocket multiListener that is already
// closed.
var errCloseOnClosedWebSocket = fmt.Errorf("use of closed network connection")

// AssertBodyClosed fails the test if the response Body is NOT closed.
// If not already closed, the response body will be drained and closed.
//
// Defer when your test is expected to close the response body before ending.
func AssertBodyClosed(t TestingT, body io.ReadCloser) {
	t.Helper()
	assertEqual(t, errReadOnClosedResBody, DrainAndCloseBody(body))
}

// AssertWebSocketClosed fails the test if the WebSocket is NOT closed.
// If not already closed, the response body will be drained and closed.
//
// Defer when your test is expected to close the WebSocket before ending.
func AssertWebSocketClosed(t TestingT, ws io.ReadCloser) {
	t.Helper()
	// The expected error is a errors.Join of two net.OpError instances, a read
	// and a write. But we don't know the source or destination, so we can't
	// match the exact error.
	AssertWebSocketClosedError(t, DrainAndCloseBody(ws))
}

// AssertWebSocketClosedError fails the test if the WebSocket error is NOT
// errCloseOnClosedWebSocket or wrapping errCloseOnClosedWebSocket.
//
// Use in your test when a WebSocket operation is expected to error due to
// having already been closed.
func AssertWebSocketClosedError(t TestingT, err error) {
	t.Helper()
	// The expected error is a net.OpError instance, but we don't know the
	// operation, source, or destination, so we can't match the exact error.
	assertErrorContains(t, err, errCloseOnClosedWebSocket.Error())
}

// Close closes the closer and fails the test if close errors.
//
// Defer when your test does not need to fully read or drain the response body
// before ending.
func Close(t TestingT, body io.Closer) {
	t.Helper()
	assertNoError(t, body.Close())
}

// DrainAndCloseBody reads from the response body until EOF, discarding the
// content, and closes the response body when finished or on error.
// Returns an error when either Read or Close error. If both error, the errors
// are joined and returned.
//
// In a defer from a test, use with t.Error or assert.NoError, NOT t.Fatal or
// require.NoError, unless the defer also captures panics, otherwise the test
// may not fail.
func DrainAndCloseBody(body io.ReadCloser) error {
	errCh := make(chan error)
	go func() {
		// Close after done reading
		defer func() {
			defer close(errCh)
			if err := body.Close(); err != nil {
				errCh <- err
			}
		}()
		// Read until EOF and discard
		if _, err := io.Copy(io.Discard, body); err != nil {
			errCh <- err
		}
	}()

	// Wait until Read and Close are both done.
	// Combine errors, if multiple.
	var multiErr error
	for err := range errCh {
		if multiErr != nil {
			multiErr = errors.Join(multiErr, err)
		} else {
			multiErr = err
		}
	}
	return multiErr
}

// ReadAndCloseBody reads from the response body until EOF and then
// closing the body, returning the content and any errors.
// Returns an error when either Read or Close error. If both error, the errors
// are joined and returned.
func ReadAndCloseBody(body io.ReadCloser) ([]byte, error) {
	errCh := make(chan error)
	bodyCh := make(chan []byte)
	go func() {
		// Close after done reading
		defer func() {
			defer close(errCh)
			if err := body.Close(); err != nil {
				errCh <- err
			}
		}()
		defer close(bodyCh)
		// Read until EOF and discard
		bodyBytes, err := io.ReadAll(body)
		if err != nil {
			errCh <- err
		}
		bodyCh <- bodyBytes
	}()

	// Wait until Read and Close are both done.
	// Combine errors, if multiple.
	var bodyBytes []byte
	var multiErr error
	var errClosed, bodyClosed bool
	for {
		select {
		case err, ok := <-errCh:
			if !ok {
				if bodyClosed {
					return bodyBytes, multiErr
				}
				errClosed = true
				continue
			}
			if multiErr != nil {
				multiErr = errors.Join(multiErr, err)
			} else {
				multiErr = err
			}
		case b, ok := <-bodyCh:
			if !ok {
				if errClosed {
					return bodyBytes, multiErr
				}
				bodyClosed = true
				continue
			}
			bodyBytes = b
		}
	}
}

// TestingT simulates assert.TestingT and assert.tHelper without requiring an
// extra non-test dependency.
type TestingT interface {
	Errorf(format string, args ...interface{})
	Helper()
}

// assertEqual simulates assert.Equal without requiring an extra non-test
// dependency. Use github.com/stretchr/testify/assert for tests.
func assertEqual[T any](t TestingT, expected, actual T) {
	t.Helper()
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Not equal: \n"+
			"expected: %s\n"+
			"actual  : %s%s",
			expected, actual, cmp.Diff(expected, actual))
	}
}

// assertErrorContains simulates assert.ErrorContains without requiring an extra
// non-test dependency. Use github.com/stretchr/testify/assert for tests.
func assertErrorContains(t TestingT, err error, substr string) {
	t.Helper()
	if err == nil {
		t.Errorf("An error is expected but got nil.")
	} else if !strings.Contains(err.Error(), substr) {
		t.Errorf("Error %#v does not contain %#v", err, substr)
	}
}

// assertNoError simulates assert.NoError without requiring an extra non-test
// dependency. Use github.com/stretchr/testify/assert for tests.
func assertNoError(t TestingT, err error) {
	t.Helper()
	if err != nil {
		t.Errorf("Received unexpected error:\n%+v", err)
	}
}
