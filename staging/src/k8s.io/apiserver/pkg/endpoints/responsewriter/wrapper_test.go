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

package responsewriter

import (
	"bufio"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	responsewritertesting "k8s.io/apiserver/pkg/endpoints/responsewriter/testing"
)

func TestWrapForHTTP1Or2WithServer(t *testing.T) {
	tests := []struct {
		name   string
		http1x bool
	}{
		{
			name:   "http/1.x",
			http1x: true,
		},
		{
			name: "http2",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// keep track of the first ResponseWriter object handed
			// to the handler by net/http, this is the first handler
			// we want to execute.
			var originalWant http.ResponseWriter
			first := func(d http.Handler) http.Handler {
				return http.HandlerFunc(func(inner http.ResponseWriter, r *http.Request) {
					originalWant = inner
					if originalGot := GetOriginal(inner); originalWant != originalGot {
						t.Errorf("Expected GetOriginal to return the original ResponseWriter object")
						return
					}

					d.ServeHTTP(inner, r)
				})
			}

			counterGot := &counter{}
			// this handler wraps the given ResponseWriter object
			// with a decorator, multiple times. this handler
			// asserts that the outer and inner are compatible.
			intermediary := func(d http.Handler) http.Handler {
				return http.HandlerFunc(func(inner http.ResponseWriter, r *http.Request) {
					assertCloseNotifierFlusher(t, inner)
					if test.http1x {
						assertHijacker(t, inner)
					}

					middle := &fakeResponseWriterDecorator{
						ResponseWriter: inner,
						counter:        counterGot,
					}
					outer := WrapForHTTP1Or2(middle)

					assertCloseNotifierFlusher(t, outer)
					if test.http1x {
						assertHijacker(t, inner)
					}

					if originalGot := GetOriginal(outer); originalWant != originalGot {
						t.Errorf("Expected GetOriginal to return the original ResponseWriter object")
						return
					}

					d.ServeHTTP(outer, r)
				})
			}

			// this handler executes last
			last := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// the original ResponseWriter object should be
				// wrapped three times, and the test decorator
				// short circuits CloseNotify method,
				counterWant := &counter{FlushInvoked: 3, CloseNotifyInvoked: 1, HijackInvoked: 0}
				if test.http1x {
					counterWant.HijackInvoked = 3
				}
				defer counterGot.assert(t, counterWant)

				//nolint:staticcheck // SA1019
				w.(http.CloseNotifier).CloseNotify()
				w.(FlusherError).FlushError()

				// using Flush is expected to result in a panic
				func() {
					expected := fmt.Errorf("Flush not allowed, use FlushError function instead")
					defer func() {
						r := recover()
						if r == nil {
							t.Errorf("Expected a panic")
							return
						}
						if err, ok := r.(error); !ok || expected.Error() != err.Error() {
							t.Errorf("Expected error: %v, but got: %v", expected, err)
						}
					}()
					w.(http.Flusher).Flush()
				}()

				if test.http1x {
					conn, _, err := w.(http.Hijacker).Hijack()
					if err != nil {
						t.Errorf("Expected Hijack to succeed, but got error: %v", err)
						return
					}
					conn.Close()
				}
			})

			// we want the following handler chain:
			//  first -> intermediary(1) -> intermediary(2) -> intermediary(3) -> last
			handler := intermediary(last)
			handler = intermediary(handler)
			handler = intermediary(handler)
			handler = first(handler)

			server := newServer(t, handler, false)
			defer server.Close()

			sendRequest(t, server)
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

				originalGot := GetOriginal(wrapped)
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
		decorator func(inner http.ResponseWriter) UserProvidedDecorator
	}{
		{
			name:   "http/1.x, decorator overrides Write only",
			http1x: true,
			decorator: func(inner http.ResponseWriter) UserProvidedDecorator {
				return &fakeWriteOnly{ResponseWriter: inner}
			},
		},
		{
			name: "http2, decorator overrides Write only",
			decorator: func(inner http.ResponseWriter) UserProvidedDecorator {
				return &fakeWriteOnly{ResponseWriter: inner}
			},
		},
		{
			name:   "http/1.x, decorator overrides Hijack only",
			http1x: true,
			decorator: func(inner http.ResponseWriter) UserProvidedDecorator {
				return &fakeHijackOnly{ResponseWriter: inner}
			},
		},
		{
			name: "http2, decorator overrides Hijack only",
			decorator: func(inner http.ResponseWriter) UserProvidedDecorator {
				return &fakeHijackOnly{ResponseWriter: inner}
			},
		},
		{
			name:   "http/1.x, decorator overrides no method",
			http1x: true,
			decorator: func(inner http.ResponseWriter) UserProvidedDecorator {
				return &fakeDecoratorNone{ResponseWriter: inner}
			},
		},
		{
			name: "http2, decorator overrides no method",
			decorator: func(inner http.ResponseWriter) UserProvidedDecorator {
				return &fakeDecoratorNone{ResponseWriter: inner}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			chainer := func(d http.Handler) http.Handler {
				return http.HandlerFunc(func(inner http.ResponseWriter, r *http.Request) {
					assertCloseNotifierFlusher(t, inner)
					if test.http1x {
						assertHijacker(t, inner)
					}

					middle := test.decorator(inner)
					outer := WrapForHTTP1Or2(middle)

					d.ServeHTTP(outer, r)
				})
			}

			invokedCh := make(chan struct{}, 1)
			handler := chainer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(invokedCh)

				assertCloseNotifierFlusher(t, w)
				if test.http1x {
					assertHijacker(t, w)
				}
			}))

			server := newServer(t, handler, !test.http1x)
			defer server.Close()
			sendRequest(t, server)

			select {
			case <-invokedCh:
			case <-time.After(wait.ForeverTestTimeout):
				t.Errorf("Expected the handler to be invoked")
			}
		})
	}
}

func TestDecoratorShouldNotUseFlush(t *testing.T) {
	tests := []struct {
		name      string
		http1x    bool
		decorator func(inner http.ResponseWriter) UserProvidedDecorator
	}{
		{
			name:   "http/1.x",
			http1x: true,
			decorator: func(inner http.ResponseWriter) UserProvidedDecorator {
				return &shoundNotBeUsedflusher{ResponseWriter: inner}
			},
		},
		{
			name: "http2",
			decorator: func(inner http.ResponseWriter) UserProvidedDecorator {
				return &shoundNotBeUsedflusher{ResponseWriter: inner}
			},
		},
	}

	errWant := fmt.Errorf("Flush not allowed, use FlushError function instead")
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			invokedCh := make(chan struct{}, 1)
			handler := http.HandlerFunc(func(inner http.ResponseWriter, r *http.Request) {
				assertCloseNotifierFlusher(t, inner)

				func() {
					defer func() {
						close(invokedCh)
						r := recover()
						if r == nil {
							t.Errorf("Expected a panic")
							return
						}
						if errGot, ok := r.(error); !ok || errWant.Error() != errGot.Error() {
							t.Errorf("Expected error: %v, but got: %v", errWant, errGot)
						}
					}()
					middle := test.decorator(inner)
					WrapForHTTP1Or2(middle)
				}()
			})

			server := newServer(t, handler, !test.http1x)
			defer server.Close()
			sendRequest(t, server)

			select {
			case <-invokedCh:
			case <-time.After(wait.ForeverTestTimeout):
				t.Errorf("Expected the handler to be invoked")
			}
		})
	}
}

func TestHTTP1xResponseWriterInvariant(t *testing.T) {
	doneCh := make(chan struct{})
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(doneCh)

		if r.ProtoMajor != 1 {
			t.Errorf("expected http/1x")
		}
		if _, ok := w.(http.Flusher); !ok {
			t.Errorf("expected the ResponseWriter object to implement http.Flusher")
		}
		if _, ok := w.(interface{ FlushError() error }); !ok {
			t.Errorf("expected the ResponseWriter object to implement Flusher with error")
		}
		if _, ok := w.(http.CloseNotifier); !ok {
			t.Errorf("expected the http.ResponseWriter object to implement http.CloseNotifier")
		}
		if _, ok := w.(http.Hijacker); !ok {
			t.Errorf("expected the http.ResponseWriter object to implement http.Hijacker")
		}
	})

	server := httptest.NewUnstartedServer(handler)
	defer server.Close()
	server.StartTLS()

	if _, err := server.Client().Get(server.URL + "/ping"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	select {
	case <-doneCh:
	default:
		t.Errorf("expected the request handler to be invoked")
	}
}

func TestHTTP2ResponseWriterInvariant(t *testing.T) {
	doneCh := make(chan struct{})
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(doneCh)

		if r.ProtoMajor != 2 {
			t.Errorf("expected http/2.0")
		}
		if _, ok := w.(http.Flusher); !ok {
			t.Errorf("expected the ResponseWriter object to implement http.Flusher")
		}

		if _, ok := w.(interface{ FlushError() error }); !ok {
			t.Errorf("expected the ResponseWriter object to implement Flusher with error")
		}
		if _, ok := w.(http.CloseNotifier); !ok {
			t.Errorf("expected the http.ResponseWriter object to implement http.CloseNotifier")
		}
	})

	server := httptest.NewUnstartedServer(handler)
	server.EnableHTTP2 = true
	defer server.Close()
	server.StartTLS()

	if _, err := server.Client().Get(server.URL + "/ping"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	select {
	case <-doneCh:
	default:
		t.Errorf("expected the request handler to be invoked")
	}
}

func newServer(t *testing.T, h http.Handler, http2 bool) *httptest.Server {
	server := httptest.NewUnstartedServer(h)
	if http2 {
		server.EnableHTTP2 = true
		server.StartTLS()
	} else {
		server.Start()
	}
	_, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("Expected the server to have a valid URL, but got: %s", server.URL)
	}
	return server
}

func sendRequest(t *testing.T, server *httptest.Server) {
	req, err := http.NewRequest("GET", server.URL, nil)
	if err != nil {
		t.Fatalf("error creating request: %v", err)
	}

	client := server.Client()
	client.Timeout = wait.ForeverTestTimeout
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Unexpected non-nil err from client.Do: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected an OK response from the server, but got: %v", resp)
	}
}

func assertCloseNotifierFlusher(t *testing.T, w http.ResponseWriter) {
	t.Helper()

	// the http.ResponseWriter object for both http/1.x and http2
	// implement http.Flusher, FlusherError, and http.CloseNotifier
	if _, ok := w.(http.Flusher); !ok {
		t.Errorf("Expected the http.ResponseWriter object to implement http.Flusher")
	}
	if _, ok := w.(FlusherError); !ok {
		t.Errorf("Expected the http.ResponseWriter object to implement FlusherError")
	}

	//nolint:staticcheck // SA1019
	if _, ok := w.(http.CloseNotifier); !ok {
		t.Errorf("Expected the http.ResponseWriter object to implement http.CloseNotifier")
	}
}

func assertHijacker(t *testing.T, w http.ResponseWriter) {
	t.Helper()

	// http/1.x implements http.Hijacker, not http2
	if _, ok := w.(http.Hijacker); !ok {
		t.Errorf("Expected the http.ResponseWriter object to implement http.Hijacker")
	}
}

type counter struct {
	FlushInvoked       int
	HijackInvoked      int
	CloseNotifyInvoked int
}

func (c *counter) assert(t *testing.T, expected *counter) {
	if expected.FlushInvoked != c.FlushInvoked {
		t.Errorf("Expected FlushError() count to match, wanted: %d, but got: %d", expected.FlushInvoked, c.FlushInvoked)
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
func (fw *fakeResponseWriterDecorator) FlushError() error {
	fw.counter.FlushInvoked++
	return fw.ResponseWriter.(FlusherError).FlushError()
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

type shoundNotBeUsedflusher struct {
	http.ResponseWriter
}

func (d shoundNotBeUsedflusher) Unwrap() http.ResponseWriter { return d.ResponseWriter }
func (shoundNotBeUsedflusher) Flush()                        {}
