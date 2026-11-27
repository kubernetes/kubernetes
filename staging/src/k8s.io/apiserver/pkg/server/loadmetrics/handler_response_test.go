/*
Copyright The Kubernetes Authors.

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

package loadmetrics

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"net/textproto"
	"net/url"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	utilproxy "k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/apiserver/pkg/server/filters"
)

func TestWithEndpointLoadMetrics_ResponseCommitPaths(t *testing.T) {
	setBackendHeaders := func(w http.ResponseWriter) {
		w.Header().Set(ResponseHeader, "TEXT cpu_utilization=0.9900, rps_fractional=999.0000")
		w.Header().Set(ResponseHeaderBin, "backend-binary-value")
	}
	earlyHints := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for range 2 {
			setBackendHeaders(w)
			w.WriteHeader(http.StatusEarlyHints)
		}
		setBackendHeaders(w)
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "ok")
	})
	backend := httptest.NewServer(earlyHints)
	t.Cleanup(backend.Close)
	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name      string
		handler   http.Handler
		wantBody  string
		wantHints int
	}{
		{
			name: "implicit 200",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				setBackendHeaders(w)
			}),
		},
		{
			name: "write implies 200",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				setBackendHeaders(w)
				_, _ = io.WriteString(w, "ok")
			}),
			wantBody: "ok",
		},
		{
			name: "flush before write",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				setBackendHeaders(w)
				w.(http.Flusher).Flush()
				_, _ = io.WriteString(w, "ok")
			}),
			wantBody: "ok",
		},
		{
			name: "response controller flush",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				setBackendHeaders(w)
				if err := http.NewResponseController(w).Flush(); err != nil {
					t.Error(err)
				}
			}),
		},
		{
			name: "early hints then implicit 200",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				setBackendHeaders(w)
				w.WriteHeader(http.StatusEarlyHints)
				setBackendHeaders(w)
			}),
			wantHints: 1,
		},
		{
			name:      "early hints then explicit 200",
			handler:   earlyHints,
			wantBody:  "ok",
			wantHints: 2,
		},
		{
			name:      "proxy early hints then 200",
			handler:   utilproxy.NewUpgradeAwareHandler(backendURL, http.DefaultTransport, false, false, nil),
			wantBody:  "ok",
			wantHints: 2,
		},
	} {
		for _, http2 := range []bool{false, true} {
			for _, optedIn := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/http2=%t/optedIn=%t", tc.name, http2, optedIn), func(t *testing.T) {
					t0 := time.Unix(1700000000, 0)
					sampler, reporter := newTestReporter(t, loadSample{timestamp: t0})
					sampler.emitSample(t0.Add(100*time.Millisecond), 0)
					sampler.emitSample(t0.Add(200*time.Millisecond), 0)
					wantHeader := ""
					if optedIn {
						wantHeader = formatORCATextReport(*reporter.LoadReport())
					}
					server := httptest.NewUnstartedServer(WithEndpointLoadMetrics(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						if _, ok := w.(http.Hijacker); ok == http2 {
							t.Errorf("Hijacker supported = %t, want %t", ok, !http2)
						}
						tc.handler.ServeHTTP(w, r)
					}), reporter))
					server.EnableHTTP2 = http2
					server.StartTLS()
					t.Cleanup(server.Close)
					var hints []http.Header
					trace := &httptrace.ClientTrace{
						Got1xxResponse: func(code int, header textproto.MIMEHeader) error {
							if code != http.StatusEarlyHints {
								return fmt.Errorf("informational status = %d, want 103", code)
							}
							hints = append(hints, http.Header(header).Clone())
							return nil
						},
					}
					req, err := http.NewRequestWithContext(httptrace.WithClientTrace(context.Background(), trace), http.MethodGet, server.URL, nil)
					if err != nil {
						t.Fatal(err)
					}
					if optedIn {
						req.Header.Set(RequestHeader, FormatText)
					}
					response, err := server.Client().Do(req)
					if err != nil {
						t.Fatal(err)
					}
					defer response.Body.Close()
					if (response.ProtoMajor == 2) != http2 {
						t.Fatalf("protocol = %s, want HTTP/2 = %t", response.Proto, http2)
					}
					if response.StatusCode != http.StatusOK {
						t.Errorf("status = %d, want 200", response.StatusCode)
					}
					body, err := io.ReadAll(response.Body)
					if err != nil {
						t.Fatal(err)
					}
					if string(body) != tc.wantBody {
						t.Errorf("body = %q, want %q", body, tc.wantBody)
					}
					if len(hints) != tc.wantHints {
						t.Errorf("early hints = %d, want %d", len(hints), tc.wantHints)
					}
					for i, header := range append(hints, response.Header) {
						if got := header.Get(ResponseHeaderBin); got != "" {
							t.Errorf("response %d: binary ORCA header = %q, want absent", i, got)
						}
						if got := header.Get(ResponseHeader); got != wantHeader {
							t.Errorf("response %d: text ORCA header = %q, want %q", i, got, wantHeader)
						}
					}
				})
			}
		}
	}
}

func TestWithEndpointLoadMetrics_ResponseOwnership(t *testing.T) {
	for _, mode := range []string{"final response", "switching protocols", "hijack", "failed hijack", "panic", "abort"} {
		t.Run(mode, func(t *testing.T) {
			_, reporter := newTestReporter(t, loadSample{timestamp: time.Unix(1700000000, 0)})
			writer := &headerGuardResponseWriter{ResponseRecorder: httptest.NewRecorder()}
			hijackErr := errors.New("test hijack failure")
			panicValue := any(errors.New("test handler panic"))
			if mode == "abort" {
				panicValue = http.ErrAbortHandler
			}
			handler := WithEndpointLoadMetrics(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set(ResponseHeader, "backend-text")
				w.Header().Set(ResponseHeaderBin, "backend-binary")
				switch mode {
				case "final response":
					w.WriteHeader(http.StatusOK)
				case "switching protocols":
					w.WriteHeader(http.StatusSwitchingProtocols)
				case "hijack":
					if _, _, err := w.(http.Hijacker).Hijack(); err != nil {
						t.Error(err)
					}
				case "failed hijack":
					writer.hijackErr = hijackErr
					if _, _, err := w.(http.Hijacker).Hijack(); err != hijackErr {
						t.Errorf("Hijack error = %v, want %v", err, hijackErr)
					}
					return
				case "panic", "abort":
					writer.forbidHeader = true
					panic(panicValue)
				}
				writer.forbidHeader = true
			}), reporter)
			var recovered any
			func() {
				defer func() { recovered = recover() }()
				handler.ServeHTTP(writer, httptest.NewRequest(http.MethodGet, "/", nil))
			}()
			if mode == "panic" || mode == "abort" {
				if recovered != panicValue {
					t.Fatalf("panic = %v, want %v", recovered, panicValue)
				}
			} else if recovered != nil {
				t.Fatalf("unexpected panic: %v", recovered)
			}
			if mode == "failed hijack" {
				for _, name := range []string{ResponseHeader, ResponseHeaderBin} {
					if got := writer.Result().Header.Get(name); got != "" {
						t.Errorf("%s = %q after failed hijack, want absent", name, got)
					}
				}
			}
		})
	}
}

func TestWithEndpointLoadMetrics_Timeout(t *testing.T) {
	t0 := time.Unix(1700000000, 0)
	sampler, reporter := newTestReporter(t, loadSample{timestamp: t0})
	sampler.emitSample(t0.Add(100*time.Millisecond), 0)
	sampler.emitSample(t0.Add(200*time.Millisecond), 0)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	releaseHandler := make(chan struct{})
	writeErr := make(chan error, 1)
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cancel()
		<-releaseHandler
		w.Header().Set(ResponseHeaderBin, "late-backend-header")
		_, err := io.WriteString(w, "late response")
		writeErr <- err
	})
	timeoutHandler := filters.WithTimeout(inner, func(r *http.Request) (*http.Request, bool, func(), *apierrors.StatusError) {
		return r, false, func() {}, apierrors.NewTimeoutError("test request timeout", 0)
	})
	handler := WithEndpointLoadMetrics(timeoutHandler, reporter)
	writer := &headerGuardResponseWriter{ResponseRecorder: httptest.NewRecorder()}
	req := httptest.NewRequest(http.MethodGet, "/", nil).WithContext(ctx)
	req.Header.Set(RequestHeader, FormatText)
	handler.ServeHTTP(writer, req)
	writer.forbidHeader = true
	close(releaseHandler)
	if err := <-writeErr; !errors.Is(err, http.ErrHandlerTimeout) {
		t.Errorf("late Write error = %v, want %v", err, http.ErrHandlerTimeout)
	}
	response := writer.Result()
	defer response.Body.Close()
	if response.StatusCode != http.StatusGatewayTimeout {
		t.Errorf("status = %d, want 504", response.StatusCode)
	}
	if got := response.Header.Get(ResponseHeader); got != formatORCATextReport(*reporter.LoadReport()) {
		t.Errorf("timeout report = %q, want current apiserver report", got)
	}
	if got := response.Header.Get(ResponseHeaderBin); got != "" {
		t.Errorf("binary ORCA header = %q, want absent", got)
	}
}

type headerGuardResponseWriter struct {
	*httptest.ResponseRecorder
	forbidHeader bool
	hijackErr    error
}

func (w *headerGuardResponseWriter) Header() http.Header {
	if w.forbidHeader {
		panic("response headers accessed after response ownership ended")
	}
	return w.ResponseRecorder.Header()
}

func (w *headerGuardResponseWriter) CloseNotify() <-chan bool {
	return nil
}

func (w *headerGuardResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return nil, nil, w.hijackErr
}
