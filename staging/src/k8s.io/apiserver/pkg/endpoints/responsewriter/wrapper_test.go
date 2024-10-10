/*
Copyright 2021 The Kubernetes Authors.

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

package responsewriter_test

import (
	"bufio"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/endpoints/responsewriter"
	responsewritertesting "k8s.io/apiserver/pkg/endpoints/responsewriter/testing"
)

func TestWrapForHTTP1Or2(t *testing.T) {
	for _, proto := range []string{"HTTP/1.1", "HTTP/2.0"} {
		t.Run(proto, func(t *testing.T) {
			// this is the first handler in the chain, it executes first:
			//  - it keeps track of the original ResponseWriter
			//  object passed to the request handler by net/http
			var originalResponseWriterWant http.ResponseWriter
			first := func(h http.Handler) http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.Proto != proto {
						t.Errorf("expected protocol: %q, but got: %q", proto, r.Proto)
					}

					originalResponseWriterWant = w
					if want, got := originalResponseWriterWant, responsewriter.GetOriginal(w); want != got {
						t.Errorf("expected GetOriginal to return the original ResponseWriter object")
					}

					h.ServeHTTP(w, r)
				})
			}

			// this is the handler in the middle:
			//  - it wraps the given ResponseWriter object with a
			//  fake decorator, in addition,
			//  - it asserts that the outer and inner ResponseWriter
			//  objects interface compatible.
			counterGot := &counter{}
			decorator := func(h http.Handler) http.Handler {
				return http.HandlerFunc(func(inner http.ResponseWriter, r *http.Request) {
					responsewritertesting.AssertResponseWriterImplementsExtendedInterfaces(t, inner, r)

					// wrap the inner ResponseWriter object
					middle := &fakeResponseWriterDecorator{
						ResponseWriter: inner,
						counter:        counterGot,
					}
					outer := responsewriter.WrapForHTTP1Or2(middle)

					// the wrapped ResponseWriter must satisfy the same interface compatibility
					responsewritertesting.AssertResponseWriterImplementsExtendedInterfaces(t, outer, r)
					responsewritertesting.AssertResponseWriterInterfaceCompatibility(t, inner, outer)

					if want, got := originalResponseWriterWant, responsewriter.GetOriginal(inner); want != got {
						t.Errorf("expected GetOriginal to return the original ResponseWriter object")
					}

					h.ServeHTTP(outer, r)
				})
			}

			// this is the last handler in the chain, it executes last:
			// a) it gets passed the wrapped ResponseWriter
			// b) it invokes CloseNotify, Flush, and Hijack on the wrapped ResponseWriter
			// c) it verifies that the decorated methods have been invoked as expected
			handlerDoneCh := make(chan struct{})
			last := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(handlerDoneCh)
				// the server will execute the following handler chain:
				//   first -> decorator(1) -> decorator(2) -> decorator(3) -> last
				counterWant := &counter{
					FlushInvoked:       3,
					CloseNotifyInvoked: 1, // the wrapper short circuits this call
					HijackInvoked:      0,
				}
				if r.Proto == "HTTP/1.1" {
					counterWant.HijackInvoked = 3
				}
				defer counterGot.assert(t, counterWant)

				//nolint:staticcheck // SA1019
				w.(http.CloseNotifier).CloseNotify()
				w.(http.Flusher).Flush()

				if r.Proto == "HTTP/1.1" {
					conn, bufrw, err := w.(http.Hijacker).Hijack()
					if err != nil {
						t.Errorf("Expected Hijack to succeed, but got error: %v", err)
						return
					}
					msg := "hello"
					payload := fmt.Sprintf("HTTP/1.0 200 OK\r\nContent-Length: %d\r\n\r\n%s", len(msg), msg)
					if _, err := bufrw.Write([]byte(payload)); err != nil {
						t.Errorf("unexpected error from Write: %v", err)
						return
					}
					if err := bufrw.Flush(); err != nil {
						t.Errorf("unexpected error from Flush: %v", err)
						return
					}
					if err := conn.Close(); err != nil {
						t.Errorf("unexpected error while closing the hijacked connection: %v", err)
					}
				}
			})

			// the server will execute the following handler chain:
			//   first -> decorator(1) -> decorator(2) -> decorator(3) -> last
			handler := decorator(last)
			handler = decorator(handler)
			handler = decorator(handler)
			handler = first(handler)

			server := httptest.NewUnstartedServer(handler)
			defer server.Close()
			if proto == "HTTP/2.0" {
				server.EnableHTTP2 = true
			}
			server.StartTLS()

			client := server.Client()
			client.Timeout = wait.ForeverTestTimeout
			resp, err := client.Get(server.URL)
			if err != nil {
				t.Errorf("unexpected error from client.Get: %v", err)
				return
			}
			if resp.StatusCode != http.StatusOK {
				t.Errorf("Expected an OK response from the server, but got: %v", resp)
			}

			select {
			case <-handlerDoneCh:
			case <-time.After(wait.ForeverTestTimeout):
				t.Errorf("expected the request handler to have terminated")
			}
		})
	}
}

func TestGetOriginal(t *testing.T) {
	tests := []struct {
		name          string
		wrap          func() (http.ResponseWriter, http.ResponseWriter)
		panicExpected bool
	}{
		{
			name: "not wrapped",
			wrap: func() (http.ResponseWriter, http.ResponseWriter) {
				original := &responsewritertesting.FakeResponseWriter{}
				return original, original
			},
		},
		{
			name: "wrapped once",
			wrap: func() (http.ResponseWriter, http.ResponseWriter) {
				original := &responsewritertesting.FakeResponseWriter{}
				return original, &fakeResponseWriterDecorator{
					ResponseWriter: original,
				}
			},
		},
		{
			name: "wrapped multiple times",
			wrap: func() (http.ResponseWriter, http.ResponseWriter) {
				original := &responsewritertesting.FakeResponseWriter{}
				return original, &fakeResponseWriterDecorator{
					ResponseWriter: &fakeResponseWriterDecorator{
						ResponseWriter: &fakeResponseWriterDecorator{
							ResponseWriter: original,
						},
					},
				}
			},
		},
		{
			name: "wraps itself",
			wrap: func() (http.ResponseWriter, http.ResponseWriter) {
				faulty := &fakeResponseWriterDecorator{}
				faulty.ResponseWriter = faulty
				return faulty, &fakeResponseWriterDecorator{
					ResponseWriter: faulty,
				}
			},
			panicExpected: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			originalExpected, wrapped := test.wrap()

			func() {
				defer func() {
					err := recover()
					switch {
					case err != nil:
						if !test.panicExpected {
							t.Errorf("Expected no panic, but got: %v", err)
						}
					default:
						if test.panicExpected {
							t.Errorf("Expected a panic")
						}
					}
				}()

				originalGot := responsewriter.GetOriginal(wrapped)
				if originalExpected != originalGot {
					t.Errorf("Expected to get tehe original http.ResponseWriter object")
				}
			}()
		})
	}
}

func TestResponseWriterDecorator(t *testing.T) {
	tests := []struct {
		name      string
		http1x    bool
		decorator func(inner http.ResponseWriter) responsewriter.UserProvidedDecorator
	}{
		{
			name: "decorator overrides Write only",
			decorator: func(inner http.ResponseWriter) responsewriter.UserProvidedDecorator {
				return &fakeWriteOnly{ResponseWriter: inner}
			},
		},
		{
			name: "decorator overrides Hijack only",
			decorator: func(inner http.ResponseWriter) responsewriter.UserProvidedDecorator {
				return &fakeHijackOnly{ResponseWriter: inner}
			},
		},
		{
			name: "decorator overrides no method",
			decorator: func(inner http.ResponseWriter) responsewriter.UserProvidedDecorator {
				return &fakeDecoratorNone{ResponseWriter: inner}
			},
		},
	}

	for _, test := range tests {
		for _, proto := range []string{"HTTP/1.1", "HTTP/2.0"} {
			t.Run(fmt.Sprintf("%s/%s", test.name, proto), func(t *testing.T) {
				// this handler wraps the ResponseWriter object
				decorator := func(d http.Handler) http.Handler {
					return http.HandlerFunc(func(inner http.ResponseWriter, r *http.Request) {
						if r.Proto != proto {
							t.Errorf("expected protocol: %q, but got: %q", proto, r.Proto)
						}
						responsewritertesting.AssertResponseWriterImplementsExtendedInterfaces(t, inner, r)

						middle := test.decorator(inner)
						outer := responsewriter.WrapForHTTP1Or2(middle)

						responsewritertesting.AssertResponseWriterInterfaceCompatibility(t, inner, outer)

						d.ServeHTTP(outer, r)
					})
				}

				// this is the inner handler that gets passed the wrapped ResponseWriter object
				doneCh := make(chan struct{}, 1)
				handler := decorator(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					defer close(doneCh)
					responsewritertesting.AssertResponseWriterImplementsExtendedInterfaces(t, w, r)
				}))

				server := httptest.NewUnstartedServer(handler)
				defer server.Close()
				if proto == "HTTP/2.0" {
					server.EnableHTTP2 = true
				}
				server.StartTLS()

				client := server.Client()
				client.Timeout = wait.ForeverTestTimeout
				resp, err := client.Get(server.URL)
				if err != nil {
					t.Errorf("unexpected error from client.Get: %v", err)
					return
				}
				if resp.StatusCode != http.StatusOK {
					t.Errorf("Expected an OK response from the server, but got: %v", resp)
				}

				select {
				case <-doneCh:
				case <-time.After(wait.ForeverTestTimeout):
					t.Errorf("expected the request handler to have terminated")
				}
			})
		}
	}
}

type counter struct {
	FlushInvoked       int
	HijackInvoked      int
	CloseNotifyInvoked int
}

func (c *counter) assert(t *testing.T, expected *counter) {
	if expected.FlushInvoked != c.FlushInvoked {
		t.Errorf("Expected Flush() count to match, wanted: %d, but got: %d", expected.FlushInvoked, c.FlushInvoked)
	}
	if expected.CloseNotifyInvoked != c.CloseNotifyInvoked {
		t.Errorf("Expected CloseNotify() count to match, wanted: %d, but got: %d", expected.CloseNotifyInvoked, c.CloseNotifyInvoked)
	}
	if expected.HijackInvoked != c.HijackInvoked {
		t.Errorf("Expected Hijack() count to match, wanted: %d, but got: %d", expected.HijackInvoked, c.HijackInvoked)
	}
}

type fakeResponseWriterDecorator struct {
	http.ResponseWriter
	counter *counter
}

func (fw *fakeResponseWriterDecorator) Unwrap() http.ResponseWriter { return fw.ResponseWriter }
func (fw *fakeResponseWriterDecorator) Flush() {
	fw.counter.FlushInvoked++
	fw.ResponseWriter.(http.Flusher).Flush()
}
func (fw *fakeResponseWriterDecorator) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	fw.counter.HijackInvoked++
	return fw.ResponseWriter.(http.Hijacker).Hijack()
}
func (fw *fakeResponseWriterDecorator) CloseNotify() <-chan bool {
	fw.counter.CloseNotifyInvoked++
	// we short circuit the call here
	return nil
}

type fakeWriteOnly struct {
	http.ResponseWriter
}

func (fw *fakeWriteOnly) Unwrap() http.ResponseWriter { return fw.ResponseWriter }
func (fw *fakeWriteOnly) Write(b []byte) (int, error) { return fw.ResponseWriter.Write(b) }

type fakeHijackOnly struct {
	http.ResponseWriter
}

func (fw *fakeHijackOnly) Unwrap() http.ResponseWriter { return fw.ResponseWriter }
func (fw *fakeHijackOnly) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return fw.ResponseWriter.(http.Hijacker).Hijack()
}

type fakeDecoratorNone struct {
	http.ResponseWriter
}

func (fw *fakeDecoratorNone) Unwrap() http.ResponseWriter { return fw.ResponseWriter }
