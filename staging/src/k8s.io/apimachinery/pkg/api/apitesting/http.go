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
	"strings"
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

// AssertReadOnClosedBodyError fails the test if the error indicates that the
// HTTP response body was read from after being closed.
func AssertReadOnClosedBodyError(t TestingT, err error) {
	t.Helper()
	assertEqualError(t, err, errReadOnClosedResBody.Error())
}

// AssertWebSocketClosedError fails the test if the error indicates that the
// WebSocket was read from or written to after being closed.
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

// TestingT simulates assert.TestingT and assert.tHelper without requiring an
// extra non-test dependency.
type TestingT interface {
	Errorf(format string, args ...interface{})
	Helper()
}

// assertEqualError simulates assert.EqualError without requiring an extra
// non-test dependency. Use github.com/stretchr/testify/assert for tests.
func assertEqualError(t TestingT, err error, errString string) {
	t.Helper()
	if err == nil {
		t.Errorf("An error is expected but got nil.")
		return
	}
	expected := errString
	actual := err.Error()
	if expected != actual {
		t.Errorf("Error message not equal:\n"+
			"expected: %q\n"+
			"actual  : %q", expected, actual)
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
