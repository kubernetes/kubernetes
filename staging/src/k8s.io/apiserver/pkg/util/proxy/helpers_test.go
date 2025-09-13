/*
Copyright 2024 The Kubernetes Authors.

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

package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	rcconstants "k8s.io/apimachinery/pkg/util/remotecommand"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/client-go/transport"
)

// This file contains common helper functions, structs, and mocks used across the
// proxy package's test files.

// fakeTransport returns a new http.Transport for tests.
func fakeTransport() (*http.Transport, error) {
	cfg := &transport.Config{
		TLS: transport.TLSConfig{
			Insecure: true,
			CAFile:   "",
		},
	}
	rt, err := transport.New(cfg)
	if err != nil {
		return nil, err
	}
	t, ok := rt.(*http.Transport)
	if !ok {
		return nil, fmt.Errorf("unknown transport type: %T", rt)
	}
	return t, nil
}

// streamContext encapsulates the structures necessary to communicate through
// a SPDY connection, including the Reader/Writer streams.
type streamContext struct {
	conn         io.Closer
	stdinStream  io.ReadCloser
	stdoutStream io.WriteCloser
	stderrStream io.WriteCloser
	resizeStream io.ReadCloser
	resizeChan   chan remotecommand.TerminalSize
	writeStatus  func(status *apierrors.StatusError) error
}

type streamAndReply struct {
	httpstream.Stream
	replySent <-chan struct{}
}

// createSPDYServerStreams upgrades the passed HTTP request to a SPDY bi-directional streaming
// connection with remote command streams defined in passed options. Returns a streamContext
// structure containing the Reader/Writer streams to communicate through the SDPY connection.
// Returns an error if unable to upgrade the HTTP connection to a SPDY connection.
func createSPDYServerStreams(w http.ResponseWriter, req *http.Request, opts Options) (*streamContext, error) {
	_, err := httpstream.Handshake(req, w, []string{rcconstants.StreamProtocolV4Name})
	if err != nil {
		return nil, err
	}

	upgrader := spdy.NewResponseUpgrader()
	streamCh := make(chan streamAndReply)
	conn := upgrader.UpgradeResponse(w, req, func(stream httpstream.Stream, replySent <-chan struct{}) error {
		streamCh <- streamAndReply{Stream: stream, replySent: replySent}
		return nil
	})
	ctx := &streamContext{
		conn: conn,
	}

	// wait for stream
	replyChan := make(chan struct{}, 5)
	defer close(replyChan)
	receivedStreams := 0
	expectedStreams := 1 // expect at least the error stream
	if opts.Stdout {
		expectedStreams++
	}
	if opts.Stdin {
		expectedStreams++
	}
	if opts.Stderr {
		expectedStreams++
	}
	if opts.Tty {
		expectedStreams++
	}
WaitForStreams:
	for {
		select {
		case stream := <-streamCh:
			streamType := stream.Headers().Get(v1.StreamType)
			streamHandled := false
			switch streamType {
			case v1.StreamTypeError:
				ctx.writeStatus = v4WriteStatusFunc(stream)
				streamHandled = true
			case v1.StreamTypeStdout:
				if opts.Stdout {
					ctx.stdoutStream = stream
					streamHandled = true
				}
			case v1.StreamTypeStdin:
				if opts.Stdin {
					ctx.stdinStream = stream
					streamHandled = true
				}
			case v1.StreamTypeStderr:
				if opts.Stderr {
					ctx.stderrStream = stream
					streamHandled = true
				}
			case v1.StreamTypeResize:
				if opts.Tty {
					ctx.resizeStream = stream
					streamHandled = true
				}
			}

			if streamHandled {
				replyChan <- struct{}{}
			} else {
				// This is a known but unexpected stream type, or an unknown one.
				// We must reset it to signal the client we won't be using it.
				stream.Reset() //nolint:errcheck
			}
		case <-replyChan:
			receivedStreams++
			if receivedStreams == expectedStreams {
				break WaitForStreams
			}
		}
	}

	if ctx.resizeStream != nil {
		ctx.resizeChan = make(chan remotecommand.TerminalSize)
		go handleResizeEvents(req.Context(), ctx.resizeStream, ctx.resizeChan)
	}

	return ctx, nil
}

func v4WriteStatusFunc(stream io.Writer) func(status *apierrors.StatusError) error {
	return func(status *apierrors.StatusError) error {
		bs, err := json.Marshal(status.Status())
		if err != nil {
			return err
		}
		_, err = stream.Write(bs)
		return err
	}
}

// justQueueStream skips the usual stream validation before
// queueing the stream on the stream channel.
func justQueueStream(streams chan httpstream.Stream) func(httpstream.Stream, <-chan struct{}) error {
	return func(stream httpstream.Stream, replySent <-chan struct{}) error {
		streams <- stream
		return nil
	}
}

// mockConn implements "net.Conn" interface.
var _ net.Conn = &mockConn{}

type mockConn struct {
	written       []byte
	localAddr     *net.TCPAddr
	remoteAddr    *net.TCPAddr
	readDeadline  time.Time
	writeDeadline time.Time
	deadlineErr   error
}

func (mc *mockConn) Write(p []byte) (int, error) {
	mc.written = append(mc.written, p...)
	return len(p), nil
}

func (mc *mockConn) Read(p []byte) (int, error) { return 0, nil }
func (mc *mockConn) Close() error               { return nil }
func (mc *mockConn) LocalAddr() net.Addr        { return mc.localAddr }
func (mc *mockConn) RemoteAddr() net.Addr       { return mc.remoteAddr }
func (mc *mockConn) SetDeadline(t time.Time) error {
	mc.SetReadDeadline(t)  //nolint:errcheck
	mc.SetWriteDeadline(t) // nolint:errcheck
	return mc.deadlineErr
}
func (mc *mockConn) SetReadDeadline(t time.Time) error  { mc.readDeadline = t; return mc.deadlineErr }
func (mc *mockConn) SetWriteDeadline(t time.Time) error { mc.writeDeadline = t; return mc.deadlineErr }

// mockResponseWriter implements "http.ResponseWriter" interface
type mockResponseWriter struct {
	header     http.Header
	written    []byte
	statusCode int
}

func (mrw *mockResponseWriter) Header() http.Header { return mrw.header }
func (mrw *mockResponseWriter) Write(p []byte) (int, error) {
	mrw.written = append(mrw.written, p...)
	return len(p), nil
}
func (mrw *mockResponseWriter) WriteHeader(statusCode int) { mrw.statusCode = statusCode }

// fakeResponder implements "rest.Responder" interface.
var _ rest.Responder = &fakeResponder{}

type fakeResponder struct{}

func (fr *fakeResponder) Object(statusCode int, obj runtime.Object) {}
func (fr *fakeResponder) Error(err error)                           {}
