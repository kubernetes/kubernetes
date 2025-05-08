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
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestWithHTTP1(t *testing.T) {
	var originalWant http.ResponseWriter
	counterGot := &counter{}
	chain := func(h http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if originalWant == nil {
				originalWant = w
			}

			assertCloseNotifierFlusherHijacker(t, true, w)

			decorator := &fakeResponseWriterDecorator{
				ResponseWriter: w,
				counter:        counterGot,
			}
			wrapped := WrapForHTTP1Or2(decorator)

			assertCloseNotifierFlusherHijacker(t, true, wrapped)

			originalGot := GetOriginal(wrapped)
			if originalWant != originalGot {
				t.Errorf("Expected GetOriginal to return the original ResponseWriter object")
				return
			}

			h.ServeHTTP(wrapped, r)
		})
	}

	// wrap the original http.ResponseWriter multiple times
	handler := chain(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// at this point, the original ResponseWriter object has been wrapped three times
		// so each decorator is expected to tick the count by one for each method.
		defer counterGot.assert(t, &counter{FlushInvoked: 3, CloseNotifyInvoked: 3, HijackInvoked: 3})

		//nolint:staticcheck // SA1019
		w.(http.CloseNotifier).CloseNotify()
		w.(http.Flusher).Flush()

		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Errorf("Expected Hijack to succeed, but got error: %v", err)
			return
		}
		conn.Close()
	}))
	handler = chain(handler)
	handler = chain(handler)

	server := newServer(t, handler, false)
	defer server.Close()

	sendRequest(t, server)
}

func TestWithHTTP2(t *testing.T) {
	var originalWant http.ResponseWriter
	counterGot := &counter{}
	chain := func(h http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if originalWant == nil {
				originalWant = w
			}

			assertCloseNotifierFlusherHijacker(t, false, w)

			decorator := &fakeResponseWriterDecorator{
				ResponseWriter: w,
				counter:        counterGot,
			}
			wrapped := WrapForHTTP1Or2(decorator)

			assertCloseNotifierFlusherHijacker(t, false, wrapped)

			originalGot := GetOriginal(wrapped)
			if originalWant != originalGot {
				t.Errorf("Expected GetOriginal to return the original ResponseWriter object")
				return
			}

			h.ServeHTTP(wrapped, r)
		})
	}

	// wrap the original http.ResponseWriter multiple times
	handler := chain(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// at this point, the original ResponseWriter object has been wrapped three times
		// so each decorator is expected to tick the count by one for each method.
		defer counterGot.assert(t, &counter{FlushInvoked: 3, CloseNotifyInvoked: 3, HijackInvoked: 0})

		//nolint:staticcheck // SA1019
		w.(http.CloseNotifier).CloseNotify()
		w.(http.Flusher).Flush()

	}))
	handler = chain(handler)
	handler = chain(handler)

	server := newServer(t, handler, true)
	defer server.Close()

	sendRequest(t, server)
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
				original := &FakeResponseWriter{}
				return original, original
			},
		},
		{
			name: "wrapped once",
			wrap: func() (http.ResponseWriter, http.ResponseWriter) {
				original := &FakeResponseWriter{}
				return original, &fakeResponseWriterDecorator{
					ResponseWriter: original,
				}
			},
		},
		{
			name: "wrapped multiple times",
			wrap: func() (http.ResponseWriter, http.ResponseWriter) {
				original := &FakeResponseWriter{}
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
	req, err := http.NewRequest(request.MethodGet, server.URL, nil)
	if err != nil {
		t.Fatalf("error creating request: %v", err)
	}

	client := server.Client()
	client.Timeout = 30 * time.Second
	_, err = client.Do(req)
	if err != nil {
		t.Fatalf("Unexpected non-nil err from client.Do: %v", err)
	}
}

func assertCloseNotifierFlusherHijacker(t *testing.T, hijackableExpected bool, w http.ResponseWriter) {
	// the http.ResponseWriter object for both http/1.x and http2
	// implement http.Flusher and http.CloseNotifier
	if _, ok := w.(http.Flusher); !ok {
		t.Errorf("Expected the http.ResponseWriter object to implement http.Flusher")
	}

	//nolint:staticcheck // SA1019
	if _, ok := w.(http.CloseNotifier); !ok {
		t.Errorf("Expected the http.ResponseWriter object to implement http.CloseNotifier")
	}

	// http/1.x implements http.Hijacker, not http2
	if _, ok := w.(http.Hijacker); ok != hijackableExpected {
		t.Errorf("Unexpected http.Hijacker implementation, expected: %t, but got: %t", hijackableExpected, ok)
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
	if fw.counter != nil {
		fw.counter.FlushInvoked++
	}
	fw.ResponseWriter.(http.Flusher).Flush()
}
func (fw *fakeResponseWriterDecorator) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	if fw.counter != nil {
		fw.counter.HijackInvoked++
	}
	return fw.ResponseWriter.(http.Hijacker).Hijack()
}
func (fw *fakeResponseWriterDecorator) CloseNotify() <-chan bool {
	if fw.counter != nil {
		fw.counter.CloseNotifyInvoked++
	}
	//nolint:staticcheck // SA1019
	return fw.ResponseWriter.(http.CloseNotifier).CloseNotify()
}
