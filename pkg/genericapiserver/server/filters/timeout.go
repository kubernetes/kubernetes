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

package filters

import (
	"bufio"
	"errors"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/genericapiserver/endpoints/handlers/responsewriters"
)

const globalTimeout = time.Minute

var errConnKilled = fmt.Errorf("kill connection/stream")

// WithTimeoutForNonLongRunningRequests times out non-long-running requests after the time given by globalTimeout.
// TODO unify this with apiserver.MaxInFlightLimit
func WithTimeoutForNonLongRunningRequests(handler http.Handler, requestContextMapper apirequest.RequestContextMapper, s runtime.NegotiatedSerializer, longRunning LongRunningRequestCheck) http.Handler {
	if longRunning == nil {
		return handler
	}
	handlerWithTimeout := WithTimeout(handler, globalTimeout, requestContextMapper, s)
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, ok := requestContextMapper.Get(req)
		if !ok {
			responsewriters.InternalError(w, req, errors.New("no context found for request"))
		}

		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if !ok {
			responsewriters.InternalError(w, req, errors.New("no request info found for request"))
		}

		if longRunning(req, requestInfo) {
			handler.ServeHTTP(w, req)
			return
		}
		handlerWithTimeout.ServeHTTP(w, req)
	})
}

// WithTimeout returns an http.Handler that runs handler with a timeout
// The new http.Handler calls h.ServeHTTP to handle
// each request, but if a call runs for longer than its time limit, the
// handler responds with a 504 StatusGatewayTimeout error.
func WithTimeout(handler http.Handler, timeout time.Duration, requestContextMapper apirequest.RequestContextMapper, s runtime.NegotiatedSerializer) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		timedout := time.After(timeout)
		done := make(chan struct{})
		tw := newTimeoutWriter(w, func() {
			ctx, ok := requestContextMapper.Get(r)
			if !ok {
				responsewriters.InternalError(w, r, errors.New("no context found for request"))
			}

			requestInfo, ok := apirequest.RequestInfoFrom(ctx)
			if !ok {
				responsewriters.InternalError(w, r, errors.New("no request info found for request"))
			}

			gvr := schema.GroupVersionResource{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion, Resource: requestInfo.Resource}
			if gvr.Version == "" {
				gvr.Version = "v1"
			}
			status := apierrors.NewServerTimeout(gvr.GroupResource(), requestInfo.Verb, 0)
			status.ErrStatus.Code = http.StatusGatewayTimeout
			responsewriters.ErrorNegotiated(status, s, gvr.GroupVersion(), w, r)
		})
		go func() {
			handler.ServeHTTP(tw, r)
			close(done)
		}()

		select {
		case <-done:
			return
		case <-timedout:
			tw.timeout()
		}
	})
}

type timeoutWriter interface {
	http.ResponseWriter
	timeout()
}

func newTimeoutWriter(w http.ResponseWriter, writeError func()) timeoutWriter {
	base := &baseTimeoutWriter{w: w, writeError: writeError}

	_, notifiable := w.(http.CloseNotifier)
	_, hijackable := w.(http.Hijacker)

	switch {
	case notifiable && hijackable:
		return &closeHijackTimeoutWriter{base}
	case notifiable:
		return &closeTimeoutWriter{base}
	case hijackable:
		return &hijackTimeoutWriter{base}
	default:
		return base
	}
}

type baseTimeoutWriter struct {
	w http.ResponseWriter

	// called to actually write the error
	writeError func()

	mu sync.Mutex
	// if the timeout handler has timedout
	timedOut bool
	// if this timeout writer has wrote header
	wroteHeader bool
	// if this timeout writer has been hijacked
	hijacked bool
}

func (tw *baseTimeoutWriter) Header() http.Header {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	if tw.timedOut {
		return http.Header{}
	}

	return tw.w.Header()
}

func (tw *baseTimeoutWriter) Write(p []byte) (int, error) {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	if tw.timedOut {
		return 0, http.ErrHandlerTimeout
	}
	if tw.hijacked {
		return 0, http.ErrHijacked
	}

	tw.wroteHeader = true
	return tw.w.Write(p)
}

func (tw *baseTimeoutWriter) Flush() {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	if tw.timedOut {
		return
	}

	if flusher, ok := tw.w.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (tw *baseTimeoutWriter) WriteHeader(code int) {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	if tw.timedOut || tw.wroteHeader || tw.hijacked {
		return
	}

	tw.wroteHeader = true
	tw.w.WriteHeader(code)
}

func (tw *baseTimeoutWriter) timeout() {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	tw.timedOut = true

	// The timeout writer has not been used by the inner handler.
	// We can safely timeout the HTTP request by sending by a timeout
	// handler
	if !tw.wroteHeader && !tw.hijacked {
		tw.writeError()
	} else {
		// The timeout writer has been used by the inner handler. There is
		// no way to timeout the HTTP request at the point. We have to shutdown
		// the connection for HTTP1 or reset stream for HTTP2.
		//
		// Note from: Brad Fitzpatrick
		// if the ServeHTTP goroutine panics, that will do the best possible thing for both
		// HTTP/1 and HTTP/2. In HTTP/1, assuming you're replying with at least HTTP/1.1 and
		// you've already flushed the headers so it's using HTTP chunking, it'll kill the TCP
		// connection immediately without a proper 0-byte EOF chunk, so the peer will recognize
		// the response as bogus. In HTTP/2 the server will just RST_STREAM the stream, leaving
		// the TCP connection open, but resetting the stream to the peer so it'll have an error,
		// like the HTTP/1 case.
		panic(errConnKilled)
	}
}

func (tw *baseTimeoutWriter) closeNotify() <-chan bool {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	if tw.timedOut {
		done := make(chan bool)
		close(done)
		return done
	}

	return tw.w.(http.CloseNotifier).CloseNotify()
}

func (tw *baseTimeoutWriter) hijack() (net.Conn, *bufio.ReadWriter, error) {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	if tw.timedOut {
		return nil, nil, http.ErrHandlerTimeout
	}
	conn, rw, err := tw.w.(http.Hijacker).Hijack()
	if err == nil {
		tw.hijacked = true
	}
	return conn, rw, err
}

type closeTimeoutWriter struct {
	*baseTimeoutWriter
}

func (tw *closeTimeoutWriter) CloseNotify() <-chan bool {
	return tw.closeNotify()
}

type hijackTimeoutWriter struct {
	*baseTimeoutWriter
}

func (tw *hijackTimeoutWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return tw.hijack()
}

type closeHijackTimeoutWriter struct {
	*baseTimeoutWriter
}

func (tw *closeHijackTimeoutWriter) CloseNotify() <-chan bool {
	return tw.closeNotify()
}

func (tw *closeHijackTimeoutWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return tw.hijack()
}
