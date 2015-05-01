/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package remotecommand

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/httpstream"
)

type fakeUpgrader struct {
	conn *fakeUpgradeConnection
	err  error
}

func (u *fakeUpgrader) upgrade(req *client.Request, config *client.Config) (httpstream.Connection, error) {
	return u.conn, u.err
}

type fakeUpgradeConnection struct {
	closeCalled bool
	lock        sync.Mutex

	stdin                   *fakeUpgradeStream
	stdout                  *fakeUpgradeStream
	stdoutData              string
	stderr                  *fakeUpgradeStream
	stderrData              string
	errorStream             *fakeUpgradeStream
	errorData               string
	unexpectedStreamCreated bool
}

func newFakeUpgradeConnection() *fakeUpgradeConnection {
	return &fakeUpgradeConnection{}
}

func (c *fakeUpgradeConnection) CreateStream(headers http.Header) (httpstream.Stream, error) {
	c.lock.Lock()
	defer c.lock.Unlock()

	stream := &fakeUpgradeStream{}
	switch headers.Get(api.StreamType) {
	case api.StreamTypeStdin:
		c.stdin = stream
	case api.StreamTypeStdout:
		c.stdout = stream
		stream.data = c.stdoutData
	case api.StreamTypeStderr:
		c.stderr = stream
		stream.data = c.stderrData
	case api.StreamTypeError:
		c.errorStream = stream
		stream.data = c.errorData
	default:
		c.unexpectedStreamCreated = true
	}

	return stream, nil
}

func (c *fakeUpgradeConnection) Close() error {
	c.lock.Lock()
	defer c.lock.Unlock()

	c.closeCalled = true
	return nil
}

func (c *fakeUpgradeConnection) CloseChan() <-chan bool {
	return make(chan bool)
}

func (c *fakeUpgradeConnection) SetIdleTimeout(timeout time.Duration) {
}

type fakeUpgradeStream struct {
	readCalled  bool
	writeCalled bool
	dataWritten []byte
	closeCalled bool
	resetCalled bool
	data        string
	lock        sync.Mutex
}

func (s *fakeUpgradeStream) Read(p []byte) (int, error) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.readCalled = true
	b := []byte(s.data)
	n := copy(p, b)
	return n, io.EOF
}

func (s *fakeUpgradeStream) Write(p []byte) (int, error) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.writeCalled = true
	s.dataWritten = make([]byte, len(p))
	copy(s.dataWritten, p)
	return len(p), io.EOF
}

func (s *fakeUpgradeStream) Close() error {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.closeCalled = true
	return nil
}

func (s *fakeUpgradeStream) Reset() error {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.resetCalled = true
	return nil
}

func (s *fakeUpgradeStream) Headers() http.Header {
	s.lock.Lock()
	defer s.lock.Unlock()
	return http.Header{}
}

func TestRequestExecuteRemoteCommand(t *testing.T) {
	testCases := []struct {
		Upgrader    *fakeUpgrader
		Stdin       string
		Stdout      string
		Stderr      string
		Error       string
		Tty         bool
		ShouldError bool
	}{
		{
			Upgrader:    &fakeUpgrader{err: errors.New("bail")},
			ShouldError: true,
		},
		{
			Upgrader:    &fakeUpgrader{conn: newFakeUpgradeConnection()},
			Stdin:       "a",
			Stdout:      "b",
			Stderr:      "c",
			Error:       "bail",
			ShouldError: true,
		},
		{
			Upgrader: &fakeUpgrader{conn: newFakeUpgradeConnection()},
			Stdin:    "a",
			Stdout:   "b",
			Stderr:   "c",
		},
		{
			Upgrader: &fakeUpgrader{conn: newFakeUpgradeConnection()},
			Stdin:    "a",
			Stdout:   "b",
			Stderr:   "c",
			Tty:      true,
		},
	}

	for i, testCase := range testCases {
		if testCase.Error != "" {
			testCase.Upgrader.conn.errorData = testCase.Error
		}
		if testCase.Stdout != "" {
			testCase.Upgrader.conn.stdoutData = testCase.Stdout
		}
		if testCase.Stderr != "" {
			testCase.Upgrader.conn.stderrData = testCase.Stderr
		}
		var localOut, localErr *bytes.Buffer
		if testCase.Stdout != "" {
			localOut = &bytes.Buffer{}
		}
		if testCase.Stderr != "" {
			localErr = &bytes.Buffer{}
		}
		e := New(&client.Request{}, &client.Config{}, []string{"ls", "/"}, strings.NewReader(testCase.Stdin), localOut, localErr, testCase.Tty)
		e.upgrader = testCase.Upgrader
		err := e.Execute()
		hasErr := err != nil
		if hasErr != testCase.ShouldError {
			t.Fatalf("%d: expected %t, got %t: %v", i, testCase.ShouldError, hasErr, err)
		}

		conn := testCase.Upgrader.conn
		if testCase.Error != "" {
			if conn.errorStream == nil {
				t.Fatalf("%d: expected error stream creation", i)
			}
			if !conn.errorStream.readCalled {
				t.Fatalf("%d: expected error stream read", i)
			}
			if e, a := testCase.Error, err.Error(); !strings.Contains(a, e) {
				t.Fatalf("%d: expected error stream read '%v', got '%v'", i, e, a)
			}
			if !conn.errorStream.resetCalled {
				t.Fatalf("%d: expected error reset", i)
			}
		}

		if testCase.ShouldError {
			continue
		}

		if testCase.Stdin != "" {
			if conn.stdin == nil {
				t.Fatalf("%d: expected stdin stream creation", i)
			}
			if !conn.stdin.writeCalled {
				t.Fatalf("%d: expected stdin stream write", i)
			}
			if e, a := testCase.Stdin, string(conn.stdin.dataWritten); e != a {
				t.Fatalf("%d: expected stdin write %v, got %v", i, e, a)
			}
			if !conn.stdin.resetCalled {
				t.Fatalf("%d: expected stdin reset", i)
			}
		}

		if testCase.Stdout != "" {
			if conn.stdout == nil {
				t.Fatalf("%d: expected stdout stream creation", i)
			}
			if !conn.stdout.readCalled {
				t.Fatalf("%d: expected stdout stream read", i)
			}
			if e, a := testCase.Stdout, localOut; e != a.String() {
				t.Fatalf("%d: expected stdout data '%s', got '%s'", i, e, a)
			}
			if !conn.stdout.resetCalled {
				t.Fatalf("%d: expected stdout reset", i)
			}
		}

		if testCase.Stderr != "" {
			if testCase.Tty {
				if conn.stderr != nil {
					t.Fatalf("%d: unexpected stderr stream creation", i)
				}
				if localErr.String() != "" {
					t.Fatalf("%d: unexpected stderr data '%s'", i, localErr)
				}
			} else {
				if conn.stderr == nil {
					t.Fatalf("%d: expected stderr stream creation", i)
				}
				if !conn.stderr.readCalled {
					t.Fatalf("%d: expected stderr stream read", i)
				}
				if e, a := testCase.Stderr, localErr; e != a.String() {
					t.Fatalf("%d: expected stderr data '%s', got '%s'", i, e, a)
				}
				if !conn.stderr.resetCalled {
					t.Fatalf("%d: expected stderr reset", i)
				}
			}
		}

		if !conn.closeCalled {
			t.Fatalf("%d: expected upgraded connection to get closed", i)
		}
	}
}
