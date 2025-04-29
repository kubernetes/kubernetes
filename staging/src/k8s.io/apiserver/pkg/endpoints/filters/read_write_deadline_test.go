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

package filters

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"os"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
)

// NOTE: The following tests assert on low-level error returned from net/http, if
// a new version of Go changes the error type returned, then some of these tests
// might fail during GoLang update, we can fix the broken test(s) by changing the
// wanted error in the test setup to the new observed error.

func TestPerRequestWithWriteEventuallyReturnsDeadlineError(t *testing.T) {
	// This test documents the behavior of the per request write deadline
	// using a standard net http server.
	//
	// scenario: we want to observe the circumstances under which the Write
	// method of the ResponseWriter object returns an "i/o timeout" error:
	//  - a) write deadline is set to 100ms
	//  - b) the handler waits until the write deadline exceeds
	//  - c) after timeout occurs, the handler keeps writing 1kB of data at a time
	// we expect the following:
	//  - d) client: expected to receive an error from the server
	//  - e) server: the Write method of the ResponseWriter object should
	//       return an "i/o timeout" error once its internal buffer is full
	//
	// The size of the internal buffer is 4kB bytes:
	//
	// http/1x: the net.Conn is written to by a buffered Writer
	// (*bufio.Writer) of default size of 4kB bytes:
	//  a) https://github.com/golang/go/blob/ffb3e574012ce9d3d5193d7b8df135189b8a6671/src/net/http/server.go#L1650
	//  b) https://github.com/golang/go/blob/ffb3e574012ce9d3d5193d7b8df135189b8a6671/src/net/http/server.go#L2014
	//
	// http/2.0: a buffered writer (4kB bytes) writes to the unerlying net.Conn
	//  a) https://github.com/golang/go/blob/ffb3e574012ce9d3d5193d7b8df135189b8a6671/src/net/http/h2_bundle.go#L3568
	//  b) https://github.com/golang/go/blob/ffb3e574012ce9d3d5193d7b8df135189b8a6671/src/net/http/h2_bundle.go#L4279
	t.Parallel()

	const deadline = 100 * time.Millisecond
	tests := []struct {
		protoMajor         int
		waiter             waiter
		clientErrVerifier  verifier
		handlerErrVerifier verifier
	}{
		{
			protoMajor: 1, // http/1x
			// NOTE: we can't use channel based waiiter for http/1x,
			// since the request handler is executed in the same
			// goroutine as the connection serving goroutine
			// https://github.com/kubernetes/enhancements/tree/master/keps/sig-api-machinery/4460-per-request-deadline#client-hanging-indefinitely
			// write timeout is set to 100ms, a wait of 5s should be
			// enough to withstand flakes in CI.
			waiter:             &waitWithDuration{after: 5 * time.Second},
			clientErrVerifier:  wantContains{contains: "local error: tls: bad record MAC"},
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
		{
			protoMajor:         2, // http/2.0
			waiter:             &waitForClose{after: make(chan time.Time)},
			clientErrVerifier:  wantContains{"stream error: stream ID 1; INTERNAL_ERROR; received from peer"},
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("HTTP/%d", test.protoMajor), func(t *testing.T) {
			handlerErrCh := make(chanErr, 1)
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				defer close(handlerErrCh)
				if req.ProtoMajor != test.protoMajor {
					t.Errorf("server: expected ProtoMajor: %d, but got: %s", test.protoMajor, req.Proto)
					return
				}

				// a) set write deadline for this request
				ctrl := http.NewResponseController(w)
				if err := ctrl.SetWriteDeadline(time.Now().Add(deadline)); err != nil {
					t.Errorf("server: unexpected error from SetWriteDeadline: %v", err)
					return
				}

				// b) wait until the write deadline exceeds
				<-test.waiter.wait()

				// c) keep writing 1kB of data at a time, Write
				// will eventually return an i/o timeout error
				func() {
					now := time.Now()
					written := 0
					defer func() {
						t.Logf("server: after timeout, Write wrote %dkB, duration: %s", written/1000, time.Since(now))
					}()
					for {
						n, err := w.Write(bytes.Repeat([]byte("a"), 1000))
						if err != nil {
							handlerErrCh <- err
							break
						}
						written += n
					}
				}()
			}))

			defer server.Close()
			if test.protoMajor == 2 {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			func() {
				defer test.waiter.close()
				client := server.Client()
				client.Timeout = wait.ForeverTestTimeout

				_, err := client.Get(server.URL)
				// d) verify that the client receives the appropriate error
				// the client should not see a response body since the timeout
				// exceeded before the handler returned, and the handler
				// never invoked flush before the timeout.
				test.clientErrVerifier.verify(t, err)
			}()

			// e) verify that the handler returns the right error
			err := handlerErrCh.wait(t, time.After(wait.ForeverTestTimeout))
			test.handlerErrVerifier.verify(t, err)
		})
	}
}

func TestPerRequestWithFlushReturnsErrorAfterDeadline(t *testing.T) {
	// This test documents the behavior of the per request write deadline
	// using a standard net http server.
	//
	// scenario: the handler writes to and flushes the ResponseWriter object
	// after write timeout occurs:
	//  - a) write deadline is set to 100ms
	//  - b) the handler waits until the write deadline exceeds
	//  - c) the handler writes a few bytes so there is some content to be flushed
	//  - d) the handler invokes FlushError
	// timeout occurs before the handler writes to the ResponseWriter object,
	// so we expect the following:
	//    e) client: expected to receive an error from the server
	//    f) server: FlushError method of the ResponseWriter object should
	//       return an "i/o timeout" error immediately.
	t.Parallel()

	const deadline = 100 * time.Millisecond
	tests := []struct {
		protoMajor         int
		waiter             waiter
		clientErrVerifier  verifier
		handlerErrVerifier verifier
	}{
		{
			protoMajor: 1,
			// write timeout is set to 100ms, a wait of 5s should
			// be enough to withstand flakes in CI.
			waiter:             &waitWithDuration{after: 5 * time.Second},
			clientErrVerifier:  wantContains{contains: "local error: tls: bad record MAC"},
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
		{
			protoMajor:         2,
			waiter:             &waitForClose{after: make(chan time.Time)},
			clientErrVerifier:  wantContains{"stream error: stream ID 1; INTERNAL_ERROR; received from peer"},
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("HTTP/%d", test.protoMajor), func(t *testing.T) {
			handlerErrCh := make(chanErr, 1)
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				defer close(handlerErrCh)
				if req.ProtoMajor != test.protoMajor {
					t.Errorf("server: expected ProtoMajor: %d, but got: %s", test.protoMajor, req.Proto)
					return
				}
				flusher, ok := w.(interface{ FlushError() error })
				if !ok {
					t.Errorf("server: expected ResponseWriter object to implement FlushError")
					return
				}

				// a) set write deadline for this request
				ctrl := http.NewResponseController(w)
				if err := ctrl.SetWriteDeadline(time.Now().Add(deadline)); err != nil {
					t.Errorf("server: unexpected error from SetWriteDeadline: %v", err)
					return
				}

				// b) wait until the write deadline exceeds
				<-test.waiter.wait()

				// c) write a few bytes so there is something in the buffer to
				// flush, the client should never see these bytes in the response.
				// NOTE: since the intenal buffered Writer has a size of 4kB, this
				// Write operation should not cause an actual write to the net.Conn
				if _, err := w.Write([]byte("hello")); err != nil {
					t.Errorf("server: unexpected error from Write after timeout: %v", err)
					return
				}
				// d) flush the payload in the buffer
				// NOTE: FlushError, on the other hand, is expected
				// to return a timeout error immediately.
				if err := flusher.FlushError(); err != nil {
					handlerErrCh <- err
				}

				// NOTE: the handler goroutine is expected to terminate as soon
				// as it receives an error from either Write or FlushError. This
				// is how we prevent a request handler from running indefinitely.
			}))

			defer server.Close()
			if test.protoMajor == 2 {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			func() {
				defer test.waiter.close()

				_, err := server.Client().Get(server.URL)
				// e) verify that the client receives the appropriate error
				test.clientErrVerifier.verify(t, err)
			}()

			// f) verify that the handler returns the right error
			err := handlerErrCh.wait(t, time.After(wait.ForeverTestTimeout))
			test.handlerErrVerifier.verify(t, err)
		})
	}
}

func TestPerRequestWithClientNeverReceivesContentFlushedAfterDeadline(t *testing.T) {
	// This test documents the behavior of the per handler write
	// deadline using a standard net http server.
	//
	// scenario: the handler writes to, but does not flush the ResponseWriter object
	// before write timeout occurs:
	//  - a) the handler writes a few bytes so there is some content to be flushed later
	//  - b) write deadline is set to 100ms
	//  - c) wait until the write deadline elapses
	//  - d) the handler invokes FlushError
	// timeout occurs after the handler writes to but has not flushed the
	// ResponseWriter object yet, we expect the following:
	//  - e) client: expected to receive an error from the server
	/// - f) server: FlushError method of the ResponseWriter object should
	//       return an "i/o timeout" error immediately
	t.Parallel()

	const deadline = 100 * time.Millisecond
	tests := []struct {
		protoMajor         int
		waiter             waiter
		clientErrVerifier  verifier
		handlerErrVerifier verifier
	}{
		{
			protoMajor: 1,
			// write timeout is set to 100ms, a wait of 5s should
			// be enough to withstand flakes in CI.
			waiter:             &waitWithDuration{after: 5 * time.Second},
			clientErrVerifier:  wantContains{contains: "local error: tls: bad record MAC"},
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
		{
			protoMajor:         2,
			waiter:             &waitForClose{after: make(chan time.Time)},
			clientErrVerifier:  wantContains{"stream error: stream ID 1; INTERNAL_ERROR; received from peer"},
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("HTTP/%d", test.protoMajor), func(t *testing.T) {
			handlerErrCh := make(chanErr, 1)
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				defer close(handlerErrCh)
				if req.ProtoMajor != test.protoMajor {
					t.Errorf("server: expected ProtoMajor: %d, but got: %s", test.protoMajor, req.Proto)
					return
				}
				flusher, ok := w.(interface{ FlushError() error })
				if !ok {
					t.Errorf("server: expected ResponseWriter object to implement FlushError")
					return
				}

				// a) write a few bytes so the internal buffer
				// is not full, don't flush yet
				if _, err := w.Write([]byte("hello")); err != nil {
					t.Errorf("server: unexpected error from Write: %v", err)
					return
				}

				// b) set write deadline for this request
				ctrl := http.NewResponseController(w)
				if err := ctrl.SetWriteDeadline(time.Now().Add(deadline)); err != nil {
					t.Errorf("server: unexpected error from SetWriteDeadline: %v", err)
					return
				}

				// c) wait until the write deadline exceeds
				<-test.waiter.wait()

				// d) flush the payload in the buffer
				if err := flusher.FlushError(); err != nil {
					handlerErrCh <- err
				}
			}))

			defer server.Close()
			if test.protoMajor == 2 {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			func() {
				defer test.waiter.close()

				_, err := server.Client().Get(server.URL)
				// f) verify that the client receives the appropriate error
				// NOTE: due to 'a' (the handler wrote but did not flush), and
				// since the handler did not return before the timeout, the
				// client will not receive an http.Response with the data written.
				test.clientErrVerifier.verify(t, err)
			}()

			// f) verify that the handler returns the right error
			err := handlerErrCh.wait(t, time.After(wait.ForeverTestTimeout))
			test.handlerErrVerifier.verify(t, err)
		})
	}
}

func TestPerRequestWithClientReceivesContentFlushedBeforeDeadline(t *testing.T) {
	// This test documents the behavior of the per handler write
	// deadline using a standard net http server.
	//
	// scenario: the handler writes to and flushes the ResponseWriter object
	// before the write deadline elapses:
	//  - a) the handler writes message 1 ("hello") to the ResponseWriter object
	//  - b) the handler invokes FlushError
	//  - c) write deadline is set to 100ms
	//  - d) wait until the write deadline elapses
	//  - e) the handler writes message 2 ("world") to the ResponseWriter object
	//  - f) the handler invokes FlushError
	// timeout occurs after the handler writes to and flushes the
	// ResponseWriter object, so we expect the following:
	//  - g) client: expected to receive a response (200 OK) from the server,
	//       while reading the Body of the response, the client will read
	//       message 1 from the stream, and then it will encounter an error.
	//       the client should never see message 2.
	//  - h) server: FlushError method of the ResponseWriter object should
	//       return an "i/o timeout" error immediately after the write timeout
	t.Parallel()

	const deadline = 100 * time.Millisecond
	tests := []struct {
		protoMajor         int
		waiter             waiter
		clientErrVerifier  verifier
		handlerErrVerifier verifier
	}{
		{
			protoMajor: 1,
			// write timeout is set to 100ms, a wait of 5s should
			// be enough to withstand flakes in CI.
			waiter:             &waitWithDuration{after: 5 * time.Second},
			clientErrVerifier:  wantContains{contains: "local error: tls: bad record MAC"},
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
		{
			protoMajor:         2,
			waiter:             &waitForClose{after: make(chan time.Time)},
			clientErrVerifier:  wantContains{"stream error: stream ID 1; INTERNAL_ERROR; received from peer"},
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("HTTP/%d", test.protoMajor), func(t *testing.T) {
			msg1, msg2 := "hello", "world"
			handlerErrCh := make(chanErr, 1)
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				defer close(handlerErrCh)
				if req.ProtoMajor != test.protoMajor {
					t.Errorf("server: expected ProtoMajor: %d, but got: %s", test.protoMajor, req.Proto)
					return
				}
				flusher, ok := w.(interface{ FlushError() error })
				if !ok {
					t.Errorf("server: expected ResponseWriter object to implement FlushError")
					return
				}

				// a) write message 1
				if _, err := w.Write([]byte(msg1)); err != nil {
					t.Errorf("server: unexpected error from Write: %v", err)
					return
				}
				// b) flush the payload that has been written to the buffer
				if err := flusher.FlushError(); err != nil {
					t.Errorf("server: unexpected error from FlushError: %v", err)
					return
				}

				// c) set write deadline for this request
				ctrl := http.NewResponseController(w)
				if err := ctrl.SetWriteDeadline(time.Now().Add(deadline)); err != nil {
					t.Errorf("server: unexpected error from SetWriteDeadline: %v", err)
					return
				}

				// d) wait until the write deadline exceeds
				<-test.waiter.wait()

				// e) write message 2 (the client should never
				// see the following message, and the test
				// verifies that in step g)
				if _, err := w.Write([]byte(msg2)); err != nil {
					t.Errorf("server: unexpected error from Write after timeout: %v", err)
				}
				// f) flush the payload in the buffer
				if err := flusher.FlushError(); err != nil {
					handlerErrCh <- err
				}
			}))

			defer server.Close()
			if test.protoMajor == 2 {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			func() {
				defer test.waiter.close()
				resp, err := server.Client().Get(server.URL)
				if err != nil {
					t.Errorf("unexpected error from client.Get: %v", err)
					return
				}
				// g) the client expects a response from the server since content
				// has been written and flushed before the write deadline elapsed
				if resp.StatusCode != http.StatusOK {
					t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
				}
				defer closeResponseBody(t, resp)

				// g) the client expects to read the paylaod from the Body of
				// the response that was flushed before the write deadline elapsed
				b := make([]byte, len(msg1))
				if _, err := io.ReadFull(resp.Body, b); err != nil || msg1 != string(b) {
					t.Errorf("expected to read the initial write from the server, but got: %v", err)
				}

				// g) at this point, any further attempt to read
				// the Body of the response should yield an error
				_, err = io.ReadAll(resp.Body)
				test.clientErrVerifier.verify(t, err)
			}()

			// h) verify that the handler returns the right error
			err := handlerErrCh.wait(t, time.After(wait.ForeverTestTimeout))
			test.handlerErrVerifier.verify(t, err)
		})
	}
}

func TestPerRequestWithHandlerShouldAbortWriteAfterDeadline(t *testing.T) {
	// This test documents the behavior of the per handler write
	// deadline using a standard net http server.
	//
	// scenario: the handler keeps writing to the ResponseWriter object indefinitely:
	//  - a) write deadline is set to 300ms
	//  - b) the handler keeps writing to the ResponseWriter object
	//       indefinitely, no explicit FlushError invocation
	// expectation:
	//  - c) client: expected to receive a response (200 OK) from the server since
	//       some content has been written to and flushed by the server.
	//  - d) server: the Write method is expected to return an "i/o timeout"
	//       error as soon as write deadline returns, and the handler is
	//       expected to terminate as a result
	t.Parallel()

	const deadline = 100 * time.Millisecond
	tests := []struct {
		protoMajor         int
		clientErrVerifier  verifier
		handlerErrVerifier verifier
	}{
		{
			protoMajor: 1,
			clientErrVerifier: wantEitherOr{
				err:      io.ErrUnexpectedEOF,
				contains: "local error: tls: bad record MA",
			},
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
		{
			protoMajor:         2,
			clientErrVerifier:  wantContains{"stream error: stream ID 1; INTERNAL_ERROR; received from peer"},
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("HTTP/%d", test.protoMajor), func(t *testing.T) {
			handlerErrCh := make(chanErr, 1)
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				defer close(handlerErrCh)
				if req.ProtoMajor != test.protoMajor {
					t.Errorf("server: expected ProtoMajor: %d, but got: %s", test.protoMajor, req.Proto)
					return
				}

				// a) set write deadline for this request
				ctrl := http.NewResponseController(w)
				if err := ctrl.SetWriteDeadline(time.Now().Add(deadline)); err != nil {
					t.Errorf("server: unexpected error from SetWriteDeadline: %v", err)
					return
				}

				// b) keep writing to the ResponseWriter indefinitely
				if _, err := io.Copy(w, neverEnding('a')); err != nil {
					handlerErrCh <- err
				}
			}))

			defer server.Close()
			if test.protoMajor == 2 {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			func() {
				client := server.Client()
				client.Timeout = wait.ForeverTestTimeout
				resp, err := client.Get(server.URL)
				// c) under normal conditions the client expects a
				// response from the server, but under stress the
				// server may return an error as well:
				//   go test -race k8s.io/apiserver/pkg/endpoints/filters -c
				//   stress -p=256 ./filters.test -test.run=TestPerRequestWriteDeadlineWithHandlerWritingIndefinitely
				// the above stress test causes the server to return error
				if err != nil {
					test.clientErrVerifier.verify(t, err)
					return
				}
				defer closeResponseBody(t, resp)
				_, err = io.ReadAll(resp.Body)
				// c) reading the Body of the response should yield an error
				test.clientErrVerifier.verify(t, err)
			}()

			// d) verify that the handler returns the right error
			err := handlerErrCh.wait(t, time.After(wait.ForeverTestTimeout))
			test.handlerErrVerifier.verify(t, err)
		})
	}
}

func TestPerRequestWithBodyReadShouldTimeoutAfterDeadline(t *testing.T) {
	// This test documents the behavior of the per handler read
	// deadline using a standard net http server.
	//
	// scenario: the client sets up a request body, but never sends any content:
	//  - a) client sets up a request body, keeps the stream open but does not send any content
	//  - b) read deadline is set to 500ms
	//  - c) the handler tries to read the Body of the request
	//  - d) read timeout elapses
	//  - e) the handler writes to and flushes the ResponseWriter object
	// after read read timeout occurs, we expect the following:
	//  - f) client: expected to see a response from the server since the
	//       the handler, in the case of this test completed normally
	//  - g) server: reading the Body of the request should yield
	//       an "i/o timeout" error
	t.Parallel()

	const deadline = 100 * time.Millisecond
	tests := []struct {
		protoMajor         int
		handlerErrVerifier verifier
	}{
		{
			protoMajor:         1,
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
		{
			protoMajor:         2,
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("HTTP/%d", test.protoMajor), func(t *testing.T) {
			msg := "hello"
			handlerErrCh := make(chanErr, 1)
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				defer close(handlerErrCh)
				if req.ProtoMajor != test.protoMajor {
					t.Errorf("server: expected ProtoMajor: %d, but got: %s", test.protoMajor, req.Proto)
					return
				}
				flusher, ok := w.(interface{ FlushError() error })
				if !ok {
					t.Errorf("server: expected ResponseWriter object to implement FlushError")
					return
				}

				// b) set read deadline for this request
				ctrl := http.NewResponseController(w)
				if err := ctrl.SetReadDeadline(time.Now().Add(deadline)); err != nil {
					t.Errorf("server: unexpected error from SetReadDeadline: %v", err)
					return
				}

				// c) read the Body of the request
				if _, err := io.Copy(io.Discard, req.Body); err != nil {
					// d) read timeout elapses here since the client
					// never sent any data before the read deadline elapsed
					handlerErrCh <- err
				}

				// e) the handler writes to and flushes the
				// ResponseWriter object after read timeout occurs
				if _, err := w.Write([]byte(msg)); err != nil {
					t.Errorf("server: unexpected error from Write after read timeout: %v", err)
					return
				}
				if err := flusher.FlushError(); err != nil {
					t.Errorf("server: unexpected error from FlushError after read timeout: %v", err)
					return
				}
			}))

			defer server.Close()
			if test.protoMajor == 2 {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			reader, writer := io.Pipe()
			defer func() {
				if err := writer.Close(); err != nil {
					t.Errorf("expected no error from Close, but got: %v", err)
				}
			}()

			func() {
				// a) the client sets up the request with a Body
				// but never writes to it.
				resp, err := server.Client().Post(server.URL, "text/foo", reader)
				if err != nil {
					t.Errorf("expected no error from Post, but got: %v", err)
					return
				}
				defer closeResponseBody(t, resp)

				// f) sine the handler wrote a response, we expect
				// the client to read it successfully.
				if resp.StatusCode != http.StatusOK {
					t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
				}
				if got, err := io.ReadAll(resp.Body); err != nil || string(got) != msg {
					t.Errorf("expected the client to read the response: want: %q, got: %q, error: %v", msg, string(got), err)
				}
			}()

			// g) verify that the handler returns the right error
			err := handlerErrCh.wait(t, time.After(wait.ForeverTestTimeout))
			test.handlerErrVerifier.verify(t, err)
		})
	}
}

func TestPerRequestWithBodyReadShouldYieldPartialContentBeforeDeadline(t *testing.T) {
	// This test documents the behavior of the per request read
	// deadline using a standard net http server.
	//
	// scenario: read deadline elapses after the handler partially
	// reads the Body of the request:
	//  - a) client sets up a request with a Body
	//  - b) client writes (message 1 + message 2) to the Body, and then keeps the stream open
	//  - c) the handler reads exactly N bytes, N=len(message 1) from the Body of the request
	//  - d) read deadline is set to 300ms
	//  - e) the handler tries to read the remaining bytes off of request body
	//  - f) read timeout occurs after 300ms (since the stream is still open)
	//  - g) the handler terminates
	//  - h) client closes the stream after the handler terminates
	// expectation:
	//  - i) client: since the handler in the case of this test returns normally,
	//       the client is expected to see a response from the server.
	//  - j) server: the handler should see an "i/o timeout" error while reading
	//       the Body of the request
	t.Parallel()

	const deadline = 100 * time.Millisecond
	tests := []struct {
		protoMajor         int
		handlerErrVerifier verifier
	}{
		{
			protoMajor:         1,
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
		{
			protoMajor:         2,
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("HTTP/%d", test.protoMajor), func(t *testing.T) {
			msg1, msg2 := "hello", "world"
			handlerErrCh, handlerDoneCh := make(chanErr, 1), make(chan struct{})
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				defer close(handlerErrCh)
				defer close(handlerDoneCh)
				if req.ProtoMajor != test.protoMajor {
					t.Errorf("server: expected an ProtoMajor: %d, but got: %s", test.protoMajor, req.Proto)
					return
				}

				// c) read message 1, N = len(message 1) from the Body of the request
				b := make([]byte, len(msg1))
				n, err := io.ReadFull(req.Body, b)
				if err != nil || string(b[:n]) != msg1 {
					t.Errorf("server: expected content in request body, want: %s, got: %s, error: %v", msg1, string(b[:n]), err)
					return
				}

				// d) set read deadline for this request
				ctrl := http.NewResponseController(w)
				if err := ctrl.SetReadDeadline(time.Now().Add(deadline)); err != nil {
					t.Errorf("server: unexpected error from SetReadDeadline: %v", err)
					return
				}

				// e) the handler reads the remaining bytes off
				// of the Body and then eventually times out.
				// NOTE: at this point, there are some bytes left
				// in the Body of the request, and the EOF will
				// not be reached since the client is keeping the
				// stream open until after the handler terminates
				got, err := io.ReadAll(req.Body)
				// f) read timeout has occurred
				if msg2 != string(got) {
					t.Errorf("server: expected %q, but got: %q", msg2, string(got))
					return
				}
				if err != nil {
					handlerErrCh <- err
				}

				// g) the handler terminates
			}))

			defer server.Close()
			if test.protoMajor == 2 {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			reader, writer := io.Pipe()
			clientWriterDoneCh := make(chan error, 1)
			go func() {
				defer close(clientWriterDoneCh)
				// b) write (message 1 + message 2) to the Body,
				// but don't close the stream yet
				if _, err := writer.Write([]byte(msg1 + msg2)); err != nil {
					t.Errorf("unexpected error from Write to Body of the request: %v", err)
				}

				// h) wait until the handler terminates before
				// closing the stream
				<-handlerDoneCh
				if err := writer.Close(); err != nil {
					t.Errorf("unexpected error from Close: %v", err)
				}
			}()

			client := server.Client()

			// a) sets up a request with a Body
			resp, err := client.Post(server.URL, "text/foo", reader)
			if err != nil {
				t.Errorf("expected no error, but got: %v", err)
				return
			}

			select {
			case <-clientWriterDoneCh:
			case <-time.After(wait.ForeverTestTimeout):
				t.Errorf("expected the client writer to have terminated")
			}

			// i) the client receives a response since the handler terminates normally
			if resp.StatusCode != http.StatusOK {
				t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
			}
			defer closeResponseBody(t, resp)

			// j) verify that the handler returns the right error
			err = handlerErrCh.wait(t, time.After(wait.ForeverTestTimeout))
			test.handlerErrVerifier.verify(t, err)
		})
	}
}

func TestPerRequestWithReadingEmptyBodyShouldNotYieldErrorAfterDeadline(t *testing.T) {
	// This test documents the behavior of the per request read
	// deadline using a standard net http server.
	//
	// scenario: does the read deadline have any impact if the request body is empty?
	//  - a) client sets up a request with an empty body
	//  - b) read deadline is set to 100ms
	//  - c) wait for the read deadline to elapse
	//  - d) the handler reads the Body of the request
	// expectation:
	//  - e) server: read timeout has no effect on the handler
	t.Parallel()

	const deadline = 100 * time.Millisecond
	tests := []struct {
		protoMajor         int
		waiter             waiter
		handlerErrVerifier verifier
	}{
		{
			protoMajor:         1,
			waiter:             &waitWithDuration{after: 5 * time.Second},
			handlerErrVerifier: wantNoError{},
		},
		{
			protoMajor:         2,
			waiter:             &waitWithDuration{after: 5 * time.Second},
			handlerErrVerifier: wantNoError{},
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("HTTP/%d", test.protoMajor), func(t *testing.T) {
			handlerErrCh := make(chanErr, 1)
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				defer close(handlerErrCh)
				if req.ProtoMajor != test.protoMajor {
					t.Errorf("server: expected ProtoMajor: %d, but got: %s", test.protoMajor, req.Proto)
					return
				}

				// b) set read deadline for this request
				ctrl := http.NewResponseController(w)
				if err := ctrl.SetReadDeadline(time.Now().Add(deadline)); err != nil {
					t.Errorf("server: unexpected error SetReadDeadline: %v", err)
					return
				}

				// c) wait for the read deadline to elapse
				<-test.waiter.wait()

				// we expect no timeout reading from an empty body since
				// for server requests, the Request Body is always non-nil
				// but will return EOF immediately when no body is present.
				t.Logf("request Body type: %T", req.Body)
				// d) the handler reads the Body of the request,
				// this should return immediately without any error.
				if _, err := io.Copy(io.Discard, req.Body); err != nil {
					handlerErrCh <- err
				}
			}))

			defer server.Close()
			if test.protoMajor == 2 {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			func() {
				defer test.waiter.close()
				// a) client sets up a request without any body
				resp, err := server.Client().Get(server.URL)
				if err != nil {
					t.Errorf("expected no error from Post, but got: %v", err)
					return
				}
				defer closeResponseBody(t, resp)
				if resp.StatusCode != http.StatusOK {
					t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
				}
			}()

			// e) verify that the handler did not return an error
			err := handlerErrCh.wait(t, time.After(wait.ForeverTestTimeout))
			test.handlerErrVerifier.verify(t, err)
		})
	}

}

func TestPerRequestWithHijackedConnectionShouldResetDeadline(t *testing.T) {
	// This test documents the behavior of the per handler read/write
	// deadline using a standard net http server.
	//
	// scenario: does the read/write deadline have any adverse impact on
	// a hijacked connection?
	//  - a) both read and write deadline are set for the request
	//  - b) the handler hijacks the connection
	//  - c) let read and write deadline elapse
	//  - d) the handler uses the hijacked connection to send a message to the client
	//  - e) the client receieves the message
	// we expect the read/write deadline set by the handler to have no
	// impact on a hijacked connection.
	t.Parallel()

	const deadline = 100 * time.Millisecond
	msg := "hello"
	handlerDoneCh := make(chanErr, 1)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(handlerDoneCh)
		hijacker, ok := w.(http.Hijacker)
		if !ok {
			t.Errorf("server: expected ResponseWriter object to implement Hijacker")
			return
		}

		// a) set both read and write deadline for this request
		ctrl := http.NewResponseController(w)
		if err := ctrl.SetWriteDeadline(time.Now().Add(deadline)); err != nil {
			t.Errorf("server: unexpected error from SetWriteDeadline: %v", err)
			return
		}
		if err := ctrl.SetReadDeadline(time.Now().Add(deadline)); err != nil {
			t.Errorf("server: unexpected error from SetReadDeadline: %v", err)
			return
		}

		// b) hijack the connection
		conn, bufrw, err := hijacker.Hijack()
		if err != nil {
			t.Errorf("server: unexpected error from Hijack: %v", err)
			return
		}

		// c) let both read and write deadline to elapse, 5s should be
		// enough to withstand any flakes in CI
		<-time.After(5 * time.Second)

		// d) use the raw connection to send a message to the client,
		// the connection should be usable after read/write timeout
		payload := fmt.Sprintf("HTTP/1.0 200 OK\r\nContent-Length: %d\r\n\r\n%s", len(msg), msg)
		if _, err := bufrw.Write([]byte(payload)); err != nil {
			t.Errorf("server: unexpected error from Write: %v", err)
			return
		}
		if err := bufrw.Flush(); err != nil {
			t.Errorf("server: unexpected error from Flush: %v", err)
			return
		}
		if err := conn.Close(); err != nil {
			t.Errorf("unexpected error from Close: %v", err)
		}
	}))

	defer server.Close()
	server.StartTLS()

	func() {
		resp, err := server.Client().Get(server.URL)
		if err != nil {
			t.Errorf("expected no error from client.Get")
			return
		}
		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
		}
		defer closeResponseBody(t, resp)
		// e) verify that the client receives the message as expected
		if got, err := io.ReadAll(resp.Body); err != nil || msg != string(got) {
			t.Errorf("expected to read what the hijacked connection wrote, want: %q, got: %q, err: %v", msg, got, err)
		}
	}()

	err := handlerDoneCh.wait(t, time.After(wait.ForeverTestTimeout))
	wantNoError{}.verify(t, err)
}

func TestPerRequestWithConnectionIsReused(t *testing.T) {
	// This test documents the behavior of the per request write deadline
	// using a standard net http server.
	//
	// scenario: we want to observe whether a connection on the client is
	// reused after a previous request riding on it fails with write timeout,
	// we send two requests, "/1", and "/2" serially:
	//  - a) the client sends the first request, at "/1"
	//  - b) the handler writes to and flushes some content for the first request
	//  - c) the handler sets write deadline to 100ms for the first request
	//  - d) the handler waits until the write deadline exceeds
	//  - e) the handler writes to and flushes the ResponseWriter object after timeout
	//  - f) the client receives a response for the first request
	//  - g) the client sends the second request, at "/2"
	//  - h) the server sends 200 OK to the client
	t.Parallel()

	const deadline = 100 * time.Millisecond
	tests := []struct {
		protoMajor         int
		waiter             waiter
		clientErrVerifier  verifier
		handlerErrVerifier verifier
		connReuseFn        func(t *testing.T, ci httptrace.GotConnInfo)
	}{
		{
			protoMajor: 1, // http/1x
			// write timeout is set to 100ms, a wait of 5s should be
			// enough to withstand flakes in CI.
			waiter:             &waitWithDuration{after: 5 * time.Second},
			clientErrVerifier:  wantContains{contains: "local error: tls: bad record MAC"},
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
			connReuseFn:        shouldUseNewConnection,
		},
		{
			protoMajor:         2, // http/2.0
			waiter:             &waitForClose{after: make(chan time.Time)},
			clientErrVerifier:  wantContains{"stream error: stream ID 1; INTERNAL_ERROR; received from peer"},
			handlerErrVerifier: wantError{err: os.ErrDeadlineExceeded},
			connReuseFn:        shouldUseExistingConnection,
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("HTTP/%d", test.protoMajor), func(t *testing.T) {
			handlerErrCh := make(chanErr, 1)
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				switch {
				case req.URL.Path == "/1":
					func() {
						defer close(handlerErrCh)
						if req.ProtoMajor != test.protoMajor {
							t.Errorf("server: expected ProtoMajor: %d, but got: %s", test.protoMajor, req.Proto)
							return
						}
						flusher, ok := w.(interface{ FlushError() error })
						if !ok {
							t.Errorf("server: expected ResponseWriter object to implement FlushError")
							return
						}

						// b) write to and flush the writer so client sees some content
						if _, err := w.Write([]byte("hello")); err != nil {
							t.Errorf("server: unexpected error from Write: %v", err)
							return
						}
						if err := flusher.FlushError(); err != nil {
							t.Errorf("server: unexpected error from FlushError: %v", err)
							return
						}

						// c) set write deadline for the first request
						ctrl := http.NewResponseController(w)
						if err := ctrl.SetWriteDeadline(time.Now().Add(deadline)); err != nil {
							t.Errorf("server: unexpected error from SetWriteDeadline: %v", err)
							return
						}

						// d) wait until the write deadline exceeds
						<-test.waiter.wait()

						// e) write to and flush the writer to get the timeout error
						if _, err := w.Write([]byte("world")); err != nil {
							t.Errorf("server: unexpected error from Write after timeout: %v", err)
							return
						}
						if err := flusher.FlushError(); err != nil {
							handlerErrCh <- err
							return
						}
					}()
				case req.URL.Path == "/2":
					// h) return here, it sends OK 200 to the client
				}
			}))

			defer server.Close()
			if test.protoMajor == 2 {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			client := server.Client()
			func() {
				defer test.waiter.close()
				// a) send the first request, it should always be on a new request
				resp, err := sendWithTrace(t, client, server.URL+"/1", shouldUseNewConnection)
				if err != nil {
					t.Errorf("unexpected error from client.Get for the first request: %v", err)
					return
				}

				if resp.StatusCode != http.StatusOK {
					t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
				}
				defer closeResponseBody(t, resp)

				_, err = io.ReadAll(resp.Body)
				// f) verify that the client receives the appropriate error
				test.clientErrVerifier.verify(t, err)
			}()

			err := handlerErrCh.wait(t, time.After(wait.ForeverTestTimeout))
			test.handlerErrVerifier.verify(t, err)

			// g) send the second request using the same underlying transport object
			// verify that the connection from the first request is reused
			func() {
				resp, err := sendWithTrace(t, client, server.URL+"/2", test.connReuseFn)
				if err != nil {
					t.Errorf("unexpected error from client.Get: %v", err)
					return
				}
				defer closeResponseBody(t, resp)
				if resp.StatusCode != http.StatusOK {
					t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
				}
			}()
		})
	}
}

func TestPerRequestWithSlowReader(t *testing.T) {
	// This test documents the behavior of the per handler write
	// deadline using a standard net http server.
	//
	// scenario: simulate a slow network by adding delays while the client
	// reads the response from the server:
	// - a) the handler assigns a context to the request with a deadline of 5ms
	// - b) the handler sets write deadline for the rquest to 50ms
	// - c) the handler writes to and flushes the ResponseWriter object with
	//   1kB of data at a time, until it writes 10MB of data or write deadline
	//   elapses, whichever happens first
	// - d) the client receives a response from the server
	// - e) the client reads 125 bytes at a time with a delay of 1ms
	t.Skip("This test is experimental, to simulate a client that reads the response body slowly")

	const (
		// set it to false to reproduce an indefinitely hanging request
		// NOTE: setting this to false will cause a test timeout in CI
		enableWriteTimeout = true

		ctxDeadline   = 5 * time.Millisecond
		writeDeadline = 50 * time.Millisecond
	)

	handlerErrCh := make(chanErr, 1)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		now := time.Now()
		var written int
		defer func() {
			t.Logf("server: the handler completed, written %dkB of payload, took: %s", written/1000, time.Since(now))
			close(handlerErrCh)
		}()

		// a) assign a context with a deadline of 5ms
		// NOTE: the context deadline has no bearing on the ResponseWriter object
		ctx, cancel := context.WithTimeout(context.Background(), ctxDeadline)
		defer cancel()
		req = req.WithContext(ctx)

		if req.ProtoMajor != 2 {
			t.Errorf("server: expected an HTTP/2.0 request, but got: %s", req.Proto)
			return
		}

		flusher, ok := w.(interface{ FlushError() error })
		if !ok {
			t.Errorf("server: expected ResponseWriter object to implement FlushError")
			return
		}
		if enableWriteTimeout {
			// b) set write deadline for the request
			ctrl := http.NewResponseController(w)
			if err := ctrl.SetWriteDeadline(time.Now().Add(writeDeadline)); err != nil {
				t.Errorf("unexpected error from SetWriteDeadline: %v", err)
				return
			}
		}

		// c) the handler writes 1kB of data at a time, and it will
		// write at most 10MB (10*1000*1000) bytes, it will abort as
		// soon as Write returns an error.
		for i := 0; i < 10*1000; i++ {
			n, err := w.Write(bytes.Repeat([]byte("a"), 1000))
			if err != nil {
				handlerErrCh <- err
				return
			}
			written += n
			if err := flusher.FlushError(); err != nil {
				handlerErrCh <- err
				return
			}
		}
	}))

	defer server.Close()
	server.EnableHTTP2 = true
	server.StartTLS()

	clientErrCh := make(chanErr, 1)
	go func() {
		defer close(clientErrCh)

		// we don't set up a request context with deadline on the client-side
		req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, server.URL+"/foo", nil)
		if err != nil {
			t.Errorf("failed to create a new request: %v", err)
			return
		}

		resp, err := server.Client().Do(req)
		if err != nil {
			t.Errorf("unexpected error from client.Do: %v", err)
			return
		}

		// d) client expects a response from the server
		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected staus code: %d, but got: %d", http.StatusOK, resp.StatusCode)
		}
		defer closeResponseBody(t, resp)

		// e) we read 125 bytes at a time, and then sleep for 1ms,
		// it will add 80 seconds (10*1000*1000*/125 milliseconds)
		// of delay to the entire read of the response, this simulates
		// a slow reader or a slow network that exceeds
		// wait.ForeverTestTimeout (30s), so we expect the test to
		// timeout and fail after ForeverTestTimeout elapses.
		//
		// without write timeout, this is what we observe:
		// server:
		//  - the handler successfully writes the entire payload of 10MB,
		//    it takes about 1m for the handler to finish.
		//  - FlushError completes without error
		//  - the handler returns after 1m
		// note that having a request context with a deadline of 5ms has no
		// impact on the Write/Flush operations.
		//
		// client:
		//  - the client continues to read from the response body, and
		//    it successfully finishes reading the body if the test
		//    does not preemptively end after ForeverTestTimeout elapses.
		//  - the connection stays active since the Body of the
		//    response is still in use
		//
		// when we enable write timeout on the server we see the following:
		// server:
		//  - write timeout happens after 50ms, the handler writes the
		//    payload partially only.
		//  - Write or FlushError return an 'i/o timeout' error
		//  - the handler returns immediately after the write timeout
		//
		// client:
		//  - client will finish reading the partial payload from the
		//    response body, and then it will see a stream reset error
		b := make([]byte, 125)
		now := time.Now()
		var read, iteration int
		for {
			iteration++
			n, err := io.ReadFull(resp.Body, b)
			if err != nil {
				clientErrCh <- err
				break
			}
			read += n
			time.Sleep(time.Millisecond)
			if iteration%1000 == 0 {
				t.Logf("elapsed=%s, read=%dkB", time.Since(now), read/1000)
			}
		}
		t.Logf("elapsed=%s, read=%dKB", time.Since(now), read/1000)
	}()

	handlerWait, clientWait := time.After(wait.ForeverTestTimeout), time.After(wait.ForeverTestTimeout)
	err := handlerErrCh.wait(t, handlerWait)
	wantError{err: os.ErrDeadlineExceeded}.verify(t, err)

	err = clientErrCh.wait(t, clientWait)
	wantContains{contains: "stream error: stream ID 1; INTERNAL_ERROR; received from peer"}.verify(t, err)
}

func sendWithTrace(t *testing.T, client *http.Client, url string, f func(*testing.T, httptrace.GotConnInfo)) (*http.Response, error) {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		t.Fatalf("failed to create new request: %v", err)
	}

	trace := &httptrace.ClientTrace{
		GotConn: func(ci httptrace.GotConnInfo) {
			t.Logf("GotConnInfo: %+v", ci)
			f(t, ci)
		},
	}
	req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))
	return client.Do(req)
}

func shouldUseExistingConnection(t *testing.T, ci httptrace.GotConnInfo) {
	t.Helper()
	if !ci.Reused {
		t.Errorf("expected an existing TCP connection to be reused, but got: %+v", ci)
	}
}

func shouldUseNewConnection(t *testing.T, ci httptrace.GotConnInfo) {
	t.Helper()
	if ci.Reused {
		t.Errorf("expected a new connection, but got: %+v", ci)
	}
}

type neverEnding byte

func (b neverEnding) Read(p []byte) (n int, err error) {
	for i := range p {
		p[i] = byte(b)
	}
	return len(p), nil
}

func closeResponseBody(t *testing.T, resp *http.Response) {
	t.Helper()
	if err := resp.Body.Close(); err != nil {
		t.Errorf("unexpected error while closing the Body of the Response object: %v", err)
	}
}

type waiter interface {
	wait() <-chan time.Time
	close()
}

// sleep based waiter implementation, the request handler sleeps for certain
// duration before it returns, we need to choose the sleep duration wisely
// in order to avoid flakes in CI.
type waitWithDuration struct {
	after time.Duration
}

func (w waitWithDuration) wait() <-chan time.Time { return time.After(w.after) }
func (w waitWithDuration) close()                 {}

// channel based waiter implementation, the request handler waits on a channel
// to close, these are the steps:
// a) the client sends a request to the http/2.0 server
// a) the request handler sets per-request write timeout, and then
// b) the request handler blocks indefinitely on this channel to close
// c) write timeout elapses, and http/2.0 server asynchronously resets the stream
// d) the client receives a stream reset error immediately
// after the write timeout occurs.
// e) the client then closes this channel
// f) the request handler unblocks and terminates
//
// This waiter can be used for http/2.0 only, since the request handler executes
// on a separate goroutine than the tcp connection serving gorutine, this allows
// the connection serving loop to asynchronously reset the http2 stream. On the
// other hand, http/1x executes the request handler in the same goroutine as the
// connection serving goroutine, this forces the connection serving goroutine to
// wait for he handler to return.
// See https://github.com/golang/go/blob/b8ac61e6e64c92f23d8cf868a92a70d13e20a124/src/net/http/server.go#L3285
type waitForClose struct {
	after chan time.Time
}

func (w waitForClose) wait() <-chan time.Time {
	return w.after
}

func (w waitForClose) close() { close(w.after) }

type chanErr chan error

func (ch chanErr) wait(t *testing.T, timeoutCh <-chan time.Time) error {
	t.Helper()

	var (
		err error
		ok  bool
	)

	select {
	case err, ok = <-ch:
		if !ok {
			return err
		}
	case <-timeoutCh:
		t.Errorf("timed out while waiting to receive from the channel")
	}

	// let's wait for the channel to be closed, this permits the test
	// goroutine to wait for other goroutines to finish before.
	select {
	case <-ch:
	case <-timeoutCh:
		t.Errorf("timed out while waiting to receive from the channel")
	}

	return err
}

type verifier interface {
	verify(t *testing.T, got error)
}

type wantNoError struct{}

func (v wantNoError) verify(t *testing.T, got error) {
	t.Helper()
	if got != nil {
		t.Errorf("unexpected error: %v", got)
	}
}

type wantContains struct {
	contains string
}

func (v wantContains) verify(t *testing.T, got error) {
	t.Helper()

	switch {
	case got != nil:
		if !strings.Contains(got.Error(), v.contains) {
			t.Errorf("expected the error to contain: %q, but got: %v", v.contains, got)
		}
	default:
		t.Errorf("expected an error that contains %q, but got none", v.contains)
	}
}

type wantError struct {
	err error
}

func (v wantError) verify(t *testing.T, got error) {
	t.Helper()

	switch {
	case got != nil:
		if !errors.Is(got, v.err) {
			t.Errorf("expected error: %v, but got: %v", v.err, got)
		}
	default:
		t.Errorf("expected an error %v, but got none", v.err)
	}
}

type wantEitherOr struct {
	err      error
	contains string
}

func (v wantEitherOr) verify(t *testing.T, got error) {
	t.Helper()

	switch {
	case got != nil:
		if !(errors.Is(got, v.err) || strings.Contains(got.Error(), v.contains)) {
			t.Errorf("expected the error to contain: %q or be: %v, but got: %v", v.contains, v.err, got)
		}
	default:
		t.Errorf("expected an error to contain: %q or be: %v, but got none", v.contains, v.err)
	}
}
