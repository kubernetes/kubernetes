/*
Copyright 2016 The Kubernetes Authors.

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

package portforward

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"syscall"
	"testing"
	"time"

	"k8s.io/streaming/pkg/httpstream"
)

func TestHTTPStreamReceived(t *testing.T) {
	tests := map[string]struct {
		port          string
		streamType    string
		expectedError string
	}{
		"missing port": {
			expectedError: `"port" header is required`,
		},
		"unable to parse port": {
			port:          "abc",
			expectedError: `unable to parse "abc" as a port: strconv.ParseUint: parsing "abc": invalid syntax`,
		},
		"negative port": {
			port:          "-1",
			expectedError: `unable to parse "-1" as a port: strconv.ParseUint: parsing "-1": invalid syntax`,
		},
		"missing stream type": {
			port:          "80",
			expectedError: `"streamType" header is required`,
		},
		"valid port with error stream": {
			port:       "80",
			streamType: "error",
		},
		"valid port with data stream": {
			port:       "80",
			streamType: "data",
		},
		"invalid stream type": {
			port:          "80",
			streamType:    "foo",
			expectedError: `invalid stream type "foo"`,
		},
	}
	for name, test := range tests {
		streams := make(chan httpstream.Stream, 1)
		f := httpStreamReceived(streams)
		stream := newFakeHTTPStream()
		if len(test.port) > 0 {
			stream.headers.Set("port", test.port)
		}
		if len(test.streamType) > 0 {
			stream.headers.Set("streamType", test.streamType)
		}
		replySent := make(chan struct{})
		err := f(stream, replySent)
		close(replySent)
		if len(test.expectedError) > 0 {
			if err == nil {
				t.Errorf("%s: expected err=%q, but it was nil", name, test.expectedError)
			}
			if e, a := test.expectedError, err.Error(); e != a {
				t.Errorf("%s: expected err=%q, got %q", name, e, a)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: unexpected error %v", name, err)
			continue
		}
		if s := <-streams; s != stream {
			t.Errorf("%s: expected stream %#v, got %#v", name, stream, s)
		}
	}
}

type fakeConn struct {
	removeStreamsCalled bool
}

func (*fakeConn) CreateStream(headers http.Header) (httpstream.Stream, error) { return nil, nil }
func (*fakeConn) Close() error                                                { return nil }
func (*fakeConn) CloseChan() <-chan bool                                      { return nil }
func (*fakeConn) SetIdleTimeout(timeout time.Duration)                        {}
func (f *fakeConn) RemoveStreams(streams ...httpstream.Stream)                { f.removeStreamsCalled = true }

func TestGetStreamPair(t *testing.T) {
	timeout := make(chan time.Time)

	conn := &fakeConn{}
	h := &httpStreamHandler{
		streamPairs: make(map[string]*httpStreamPair),
		conn:        conn,
	}

	// test adding a new entry
	p, created := h.getStreamPair("1")
	if p == nil {
		t.Fatalf("unexpected nil pair")
	}
	if !created {
		t.Fatal("expected created=true")
	}
	if p.dataStream != nil {
		t.Errorf("unexpected non-nil data stream")
	}
	if p.errorStream != nil {
		t.Errorf("unexpected non-nil error stream")
	}

	// start the monitor for this pair
	monitorDone := make(chan struct{})
	go func() {
		h.monitorStreamPair(p, timeout)
		close(monitorDone)
	}()

	if !h.hasStreamPair("1") {
		t.Fatal("This should still be true")
	}

	// make sure we can retrieve an existing entry
	p2, created := h.getStreamPair("1")
	if created {
		t.Fatal("expected created=false")
	}
	if p != p2 {
		t.Fatalf("retrieving an existing pair: expected %#v, got %#v", p, p2)
	}

	// removed via complete
	dataStream := newFakeHTTPStream()
	dataStream.headers.Set(StreamType, StreamTypeData)
	complete, err := p.add(dataStream)
	if err != nil {
		t.Fatalf("unexpected error adding data stream to pair: %v", err)
	}
	if complete {
		t.Fatalf("unexpected complete")
	}

	errorStream := newFakeHTTPStream()
	errorStream.headers.Set(StreamType, StreamTypeError)
	complete, err = p.add(errorStream)
	if err != nil {
		t.Fatalf("unexpected error adding error stream to pair: %v", err)
	}
	if !complete {
		t.Fatal("unexpected incomplete")
	}

	// make sure monitorStreamPair completed
	<-monitorDone

	if !conn.removeStreamsCalled {
		t.Fatalf("connection remove stream not called")
	}
	conn.removeStreamsCalled = false

	// make sure the pair was removed
	if h.hasStreamPair("1") {
		t.Fatal("expected removal of pair after both data and error streams received")
	}

	// removed via timeout
	p, created = h.getStreamPair("2")
	if !created {
		t.Fatal("expected created=true")
	}
	if p == nil {
		t.Fatal("expected p not to be nil")
	}

	monitorDone = make(chan struct{})
	go func() {
		h.monitorStreamPair(p, timeout)
		close(monitorDone)
	}()
	// cause the timeout
	close(timeout)
	// make sure monitorStreamPair completed
	<-monitorDone
	if h.hasStreamPair("2") {
		t.Fatal("expected stream pair to be removed")
	}
	if !conn.removeStreamsCalled {
		t.Fatalf("connection remove stream not called")
	}
}

func TestRequestID(t *testing.T) {
	h := &httpStreamHandler{}

	s := newFakeHTTPStream()
	s.headers.Set(StreamType, StreamTypeError)
	s.id = 1
	if e, a := "1", h.requestID(s); e != a {
		t.Errorf("expected %q, got %q", e, a)
	}

	s.headers.Set(StreamType, StreamTypeData)
	s.id = 3
	if e, a := "1", h.requestID(s); e != a {
		t.Errorf("expected %q, got %q", e, a)
	}

	s.id = 7
	s.headers.Set(PortForwardRequestIDHeader, "2")
	if e, a := "2", h.requestID(s); e != a {
		t.Errorf("expected %q, got %q", e, a)
	}
}

type fakePortForwarder struct {
	err error
}

func (f *fakePortForwarder) PortForward(_ context.Context, _ string, _ string, _ int32, _ io.ReadWriteCloser) error {
	return f.err
}

func TestPortForwardErrorHandling(t *testing.T) {
	tests := map[string]struct {
		err              error
		expectErrMessage bool
		expectReset      bool
	}{
		"success": {},
		"connection reset by target": {
			err:         fmt.Errorf("failed to execute portforward in network namespace %q: %w", "/var/run/netns/test", &os.SyscallError{Syscall: "read", Err: syscall.ECONNRESET}),
			expectReset: true,
		},
		"broken pipe to target": {
			err:         fmt.Errorf("failed to execute portforward in network namespace %q: %w", "/var/run/netns/test", syscall.EPIPE),
			expectReset: true,
		},
		"other error": {
			err:              errors.New("network namespace for sandbox \"abc\" is closed"),
			expectErrMessage: true,
			expectReset:      true,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			dataStream := newFakeHTTPStream()
			dataStream.headers.Set(PortHeader, "5432")
			errorStream := newFakeHTTPStream()
			h := &httpStreamHandler{
				pod:       "pod",
				uid:       "uid",
				forwarder: &fakePortForwarder{err: test.err},
			}

			h.portForward(&httpStreamPair{requestID: "1", dataStream: dataStream, errorStream: errorStream})

			if msg := string(errorStream.written); test.expectErrMessage != (len(msg) > 0) {
				t.Errorf("expected error message written: %v, got %q", test.expectErrMessage, msg)
			} else if test.expectErrMessage && !strings.Contains(msg, test.err.Error()) {
				t.Errorf("expected error message to contain %q, got %q", test.err.Error(), msg)
			}
			if dataStream.reset != test.expectReset || errorStream.reset != test.expectReset {
				t.Errorf("expected streams reset: %v, got data=%v error=%v", test.expectReset, dataStream.reset, errorStream.reset)
			}
		})
	}
}

type fakeHTTPStream struct {
	headers http.Header
	id      uint32
	written []byte
	reset   bool
}

func newFakeHTTPStream() *fakeHTTPStream {
	return &fakeHTTPStream{
		headers: make(http.Header),
	}
}

var _ httpstream.Stream = &fakeHTTPStream{}

func (s *fakeHTTPStream) Read(data []byte) (int, error) {
	return 0, nil
}

func (s *fakeHTTPStream) Write(data []byte) (int, error) {
	s.written = append(s.written, data...)
	return len(data), nil
}

func (s *fakeHTTPStream) Close() error {
	return nil
}

func (s *fakeHTTPStream) Reset() error {
	s.reset = true
	return nil
}

func (s *fakeHTTPStream) Headers() http.Header {
	return s.headers
}

func (s *fakeHTTPStream) Identifier() uint32 {
	return s.id
}
