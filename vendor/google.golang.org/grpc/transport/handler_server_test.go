/*
 * Copyright 2016, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package transport

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

func TestHandlerTransport_NewServerHandlerTransport(t *testing.T) {
	type testCase struct {
		name    string
		req     *http.Request
		wantErr string
		modrw   func(http.ResponseWriter) http.ResponseWriter
		check   func(*serverHandlerTransport, *testCase) error
	}
	tests := []testCase{
		{
			name: "http/1.1",
			req: &http.Request{
				ProtoMajor: 1,
				ProtoMinor: 1,
			},
			wantErr: "gRPC requires HTTP/2",
		},
		{
			name: "bad method",
			req: &http.Request{
				ProtoMajor: 2,
				Method:     "GET",
				Header:     http.Header{},
				RequestURI: "/",
			},
			wantErr: "invalid gRPC request method",
		},
		{
			name: "bad content type",
			req: &http.Request{
				ProtoMajor: 2,
				Method:     "POST",
				Header: http.Header{
					"Content-Type": {"application/foo"},
				},
				RequestURI: "/service/foo.bar",
			},
			wantErr: "invalid gRPC request content-type",
		},
		{
			name: "not flusher",
			req: &http.Request{
				ProtoMajor: 2,
				Method:     "POST",
				Header: http.Header{
					"Content-Type": {"application/grpc"},
				},
				RequestURI: "/service/foo.bar",
			},
			modrw: func(w http.ResponseWriter) http.ResponseWriter {
				// Return w without its Flush method
				type onlyCloseNotifier interface {
					http.ResponseWriter
					http.CloseNotifier
				}
				return struct{ onlyCloseNotifier }{w.(onlyCloseNotifier)}
			},
			wantErr: "gRPC requires a ResponseWriter supporting http.Flusher",
		},
		{
			name: "not closenotifier",
			req: &http.Request{
				ProtoMajor: 2,
				Method:     "POST",
				Header: http.Header{
					"Content-Type": {"application/grpc"},
				},
				RequestURI: "/service/foo.bar",
			},
			modrw: func(w http.ResponseWriter) http.ResponseWriter {
				// Return w without its CloseNotify method
				type onlyFlusher interface {
					http.ResponseWriter
					http.Flusher
				}
				return struct{ onlyFlusher }{w.(onlyFlusher)}
			},
			wantErr: "gRPC requires a ResponseWriter supporting http.CloseNotifier",
		},
		{
			name: "valid",
			req: &http.Request{
				ProtoMajor: 2,
				Method:     "POST",
				Header: http.Header{
					"Content-Type": {"application/grpc"},
				},
				URL: &url.URL{
					Path: "/service/foo.bar",
				},
				RequestURI: "/service/foo.bar",
			},
			check: func(t *serverHandlerTransport, tt *testCase) error {
				if t.req != tt.req {
					return fmt.Errorf("t.req = %p; want %p", t.req, tt.req)
				}
				if t.rw == nil {
					return errors.New("t.rw = nil; want non-nil")
				}
				return nil
			},
		},
		{
			name: "with timeout",
			req: &http.Request{
				ProtoMajor: 2,
				Method:     "POST",
				Header: http.Header{
					"Content-Type": []string{"application/grpc"},
					"Grpc-Timeout": {"200m"},
				},
				URL: &url.URL{
					Path: "/service/foo.bar",
				},
				RequestURI: "/service/foo.bar",
			},
			check: func(t *serverHandlerTransport, tt *testCase) error {
				if !t.timeoutSet {
					return errors.New("timeout not set")
				}
				if want := 200 * time.Millisecond; t.timeout != want {
					return fmt.Errorf("timeout = %v; want %v", t.timeout, want)
				}
				return nil
			},
		},
		{
			name: "with bad timeout",
			req: &http.Request{
				ProtoMajor: 2,
				Method:     "POST",
				Header: http.Header{
					"Content-Type": []string{"application/grpc"},
					"Grpc-Timeout": {"tomorrow"},
				},
				URL: &url.URL{
					Path: "/service/foo.bar",
				},
				RequestURI: "/service/foo.bar",
			},
			wantErr: `stream error: code = Internal desc = "malformed time-out: transport: timeout unit is not recognized: \"tomorrow\""`,
		},
		{
			name: "with metadata",
			req: &http.Request{
				ProtoMajor: 2,
				Method:     "POST",
				Header: http.Header{
					"Content-Type": []string{"application/grpc"},
					"meta-foo":     {"foo-val"},
					"meta-bar":     {"bar-val1", "bar-val2"},
					"user-agent":   {"x/y a/b"},
				},
				URL: &url.URL{
					Path: "/service/foo.bar",
				},
				RequestURI: "/service/foo.bar",
			},
			check: func(ht *serverHandlerTransport, tt *testCase) error {
				want := metadata.MD{
					"meta-bar":   {"bar-val1", "bar-val2"},
					"user-agent": {"x/y"},
					"meta-foo":   {"foo-val"},
				}
				if !reflect.DeepEqual(ht.headerMD, want) {
					return fmt.Errorf("metdata = %#v; want %#v", ht.headerMD, want)
				}
				return nil
			},
		},
	}

	for _, tt := range tests {
		rw := newTestHandlerResponseWriter()
		if tt.modrw != nil {
			rw = tt.modrw(rw)
		}
		got, gotErr := NewServerHandlerTransport(rw, tt.req)
		if (gotErr != nil) != (tt.wantErr != "") || (gotErr != nil && gotErr.Error() != tt.wantErr) {
			t.Errorf("%s: error = %v; want %q", tt.name, gotErr, tt.wantErr)
			continue
		}
		if gotErr != nil {
			continue
		}
		if tt.check != nil {
			if err := tt.check(got.(*serverHandlerTransport), &tt); err != nil {
				t.Errorf("%s: %v", tt.name, err)
			}
		}
	}
}

type testHandlerResponseWriter struct {
	*httptest.ResponseRecorder
	closeNotify chan bool
}

func (w testHandlerResponseWriter) CloseNotify() <-chan bool { return w.closeNotify }
func (w testHandlerResponseWriter) Flush()                   {}

func newTestHandlerResponseWriter() http.ResponseWriter {
	return testHandlerResponseWriter{
		ResponseRecorder: httptest.NewRecorder(),
		closeNotify:      make(chan bool, 1),
	}
}

type handleStreamTest struct {
	t     *testing.T
	bodyw *io.PipeWriter
	req   *http.Request
	rw    testHandlerResponseWriter
	ht    *serverHandlerTransport
}

func newHandleStreamTest(t *testing.T) *handleStreamTest {
	bodyr, bodyw := io.Pipe()
	req := &http.Request{
		ProtoMajor: 2,
		Method:     "POST",
		Header: http.Header{
			"Content-Type": {"application/grpc"},
		},
		URL: &url.URL{
			Path: "/service/foo.bar",
		},
		RequestURI: "/service/foo.bar",
		Body:       bodyr,
	}
	rw := newTestHandlerResponseWriter().(testHandlerResponseWriter)
	ht, err := NewServerHandlerTransport(rw, req)
	if err != nil {
		t.Fatal(err)
	}
	return &handleStreamTest{
		t:     t,
		bodyw: bodyw,
		ht:    ht.(*serverHandlerTransport),
		rw:    rw,
	}
}

func TestHandlerTransport_HandleStreams(t *testing.T) {
	st := newHandleStreamTest(t)
	handleStream := func(s *Stream) {
		if want := "/service/foo.bar"; s.method != want {
			t.Errorf("stream method = %q; want %q", s.method, want)
		}
		st.bodyw.Close() // no body
		st.ht.WriteStatus(s, status.New(codes.OK, ""))
	}
	st.ht.HandleStreams(
		func(s *Stream) { go handleStream(s) },
		func(ctx context.Context, method string) context.Context { return ctx },
	)
	wantHeader := http.Header{
		"Date":         nil,
		"Content-Type": {"application/grpc"},
		"Trailer":      {"Grpc-Status", "Grpc-Message"},
		"Grpc-Status":  {"0"},
	}
	if !reflect.DeepEqual(st.rw.HeaderMap, wantHeader) {
		t.Errorf("Header+Trailer Map: %#v; want %#v", st.rw.HeaderMap, wantHeader)
	}
}

// Tests that codes.Unimplemented will close the body, per comment in handler_server.go.
func TestHandlerTransport_HandleStreams_Unimplemented(t *testing.T) {
	handleStreamCloseBodyTest(t, codes.Unimplemented, "thingy is unimplemented")
}

// Tests that codes.InvalidArgument will close the body, per comment in handler_server.go.
func TestHandlerTransport_HandleStreams_InvalidArgument(t *testing.T) {
	handleStreamCloseBodyTest(t, codes.InvalidArgument, "bad arg")
}

func handleStreamCloseBodyTest(t *testing.T, statusCode codes.Code, msg string) {
	st := newHandleStreamTest(t)
	handleStream := func(s *Stream) {
		st.ht.WriteStatus(s, status.New(statusCode, msg))
	}
	st.ht.HandleStreams(
		func(s *Stream) { go handleStream(s) },
		func(ctx context.Context, method string) context.Context { return ctx },
	)
	wantHeader := http.Header{
		"Date":         nil,
		"Content-Type": {"application/grpc"},
		"Trailer":      {"Grpc-Status", "Grpc-Message"},
		"Grpc-Status":  {fmt.Sprint(uint32(statusCode))},
		"Grpc-Message": {encodeGrpcMessage(msg)},
	}
	if !reflect.DeepEqual(st.rw.HeaderMap, wantHeader) {
		t.Errorf("Header+Trailer mismatch.\n got: %#v\nwant: %#v", st.rw.HeaderMap, wantHeader)
	}
}

func TestHandlerTransport_HandleStreams_Timeout(t *testing.T) {
	bodyr, bodyw := io.Pipe()
	req := &http.Request{
		ProtoMajor: 2,
		Method:     "POST",
		Header: http.Header{
			"Content-Type": {"application/grpc"},
			"Grpc-Timeout": {"200m"},
		},
		URL: &url.URL{
			Path: "/service/foo.bar",
		},
		RequestURI: "/service/foo.bar",
		Body:       bodyr,
	}
	rw := newTestHandlerResponseWriter().(testHandlerResponseWriter)
	ht, err := NewServerHandlerTransport(rw, req)
	if err != nil {
		t.Fatal(err)
	}
	runStream := func(s *Stream) {
		defer bodyw.Close()
		select {
		case <-s.ctx.Done():
		case <-time.After(5 * time.Second):
			t.Errorf("timeout waiting for ctx.Done")
			return
		}
		err := s.ctx.Err()
		if err != context.DeadlineExceeded {
			t.Errorf("ctx.Err = %v; want %v", err, context.DeadlineExceeded)
			return
		}
		ht.WriteStatus(s, status.New(codes.DeadlineExceeded, "too slow"))
	}
	ht.HandleStreams(
		func(s *Stream) { go runStream(s) },
		func(ctx context.Context, method string) context.Context { return ctx },
	)
	wantHeader := http.Header{
		"Date":         nil,
		"Content-Type": {"application/grpc"},
		"Trailer":      {"Grpc-Status", "Grpc-Message"},
		"Grpc-Status":  {"4"},
		"Grpc-Message": {encodeGrpcMessage("too slow")},
	}
	if !reflect.DeepEqual(rw.HeaderMap, wantHeader) {
		t.Errorf("Header+Trailer Map mismatch.\n got: %#v\nwant: %#v", rw.HeaderMap, wantHeader)
	}
}
