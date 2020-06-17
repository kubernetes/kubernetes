/*
Copyright 2020 The Kubernetes Authors.

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

package apiserver

import (
	"bufio"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"syscall"

	knet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/proxy"
)

// newResponseWriterInterceptor wraps http.ResponseWriter for detecting Hijacked connections
func newResponseWriterInterceptor(w http.ResponseWriter) http.ResponseWriter {
	_, supportsCloseNotifier := w.(http.CloseNotifier)
	_, supportsFlusher := w.(http.Flusher)
	if supportsCloseNotifier && supportsFlusher {
		w = newExtendedResponseWriterInterceptor(newSimpleResponseWriterInterceptor(w))
	} else {
		w = newSimpleResponseWriterInterceptor(w)
	}
	return w
}

type responseWriterInterceptor interface {
	WasHijacked() bool
	StatusCode() int
}

type responseWriterExtended struct {
	*responseWriter
}

func (w *responseWriterExtended) CloseNotify() <-chan bool {
	return w.ResponseWriter.(http.CloseNotifier).CloseNotify()
}

func (w *responseWriterExtended) Flush() {
	w.ResponseWriter.(http.Flusher).Flush()
}

// newExtendedResponseWriterInterceptor extends responseWriter
// primarily to satisfy metrics.InstrumentRouteFunc/InstrumentHandlerFunc
//
// it turns out that not all ResponseWrites support CloseNotify and Flush methods
func newExtendedResponseWriterInterceptor(rw *responseWriter) *responseWriterExtended {
	return &responseWriterExtended{rw}
}

// responseWriter wraps http.ResponseWriter for detecting Hijacked connections
type responseWriter struct {
	http.ResponseWriter

	statusCode int
	wasHijacked bool
}

// newSimpleResponseWriterInterceptor wraps the given ResponseWrite and intercept WriteHeader() and Hijack() methods
func newSimpleResponseWriterInterceptor(w http.ResponseWriter) *responseWriter {
	return &responseWriter{w,  0, false}
}

func (w *responseWriter) WriteHeader(code int) {
	w.statusCode = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *responseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	requestHijacker, ok := w.ResponseWriter.(http.Hijacker)
	if !ok {
		return nil, nil, fmt.Errorf("unable to hijack response writer: %T", w.ResponseWriter)
	}

	w.wasHijacked = true
	return requestHijacker.Hijack()
}

func (w *responseWriter) WasHijacked() bool {
	return w.wasHijacked
}

func (w *responseWriter) StatusCode() int {
	return w.statusCode
}

type retriable interface {
	shouldRetry() bool
	reset()
}

type retryDecorator struct {
	delegate retriable
	rw responseWriterInterceptor
}

// newRetryDecorator wraps delegate for detecting Hijacked connections
func newRetryDecorator(rw responseWriterInterceptor, delegate retriable, retry int) *retryDecorator {
	return &retryDecorator{withMaxRetries(delegate, retry), rw}
}

// RetryIfNeeded returns true if the request failed and can be safely retried otherwise it returns false
func (p *retryDecorator) RetryIfNeeded() bool {
	if p.rw.WasHijacked() {
		return false
	}

	//TODO: for env with only a single server we should back off before retrying
	if p.delegate.shouldRetry() {
		p.delegate.reset()
		return true
	}
	return false
}

type maxRetries struct {
	retriable
	counter int
	max int
}

var _ retriable = &maxRetries{}

func withMaxRetries(delegate retriable, max int) retriable {
	return &maxRetries{retriable:delegate, max:max}
}

func (r *maxRetries) shouldRetry() bool {
	r.counter++
	if r.counter > r.max {
		return false
	}

	return r.retriable.shouldRetry()
}

// hijackErrorResponder wraps proxy.ErrorResponder and prevents errors from being written to the client if they can be retried
type hijackErrorResponder struct {
	// delegate knows how to write errors to the client
	delegate proxy.ErrorResponder
	req *http.Request
	retry  bool
	lastKnownError error
}

var _ proxy.ErrorResponder = &hijackErrorResponder{}
var _ retriable = &hijackErrorResponder{}

// newHijackErrorResponder creates a new ErrorResponder that wraps the delegate for supporting reties
func newHijackErrorResponder(delegate proxy.ErrorResponder, req *http.Request) *hijackErrorResponder {
	return &hijackErrorResponder{delegate: delegate, req: req}
}

// Error reports the err to the client or suppress it if it's not retriable
func (hr *hijackErrorResponder) Error(w http.ResponseWriter, r *http.Request, err error) {
	// if we can retry the request do not send a response to the client
	hr.lastKnownError = err
	if !hr.canRetry(err) {
		hr.retry = false
		hr.delegate.Error(w, r, err)
		return
	}
	hr.retry = true
}

func (hr *hijackErrorResponder) reset() {
	hr.retry = false
	hr.lastKnownError = nil
}

func (hr *hijackErrorResponder) shouldRetry() bool {
	return hr.retry
}

func (hr *hijackErrorResponder) canRetry(err error) bool {
	if isHTTPVerbRetriable(hr.req) && (knet.IsConnectionReset(err) || knet.IsConnectionRefused(err) || knet.IsProbableEOF(err) ||  isExperimental(err)) {
		return true
	}
	return false
}

func isHTTPVerbRetriable(req *http.Request) bool {
	return req.Method == "GET" || req.Method == "DELETE"
}

func isExperimental(err error) bool {
	var osErr *os.SyscallError
	if errors.As(err, &osErr) {
		err = osErr.Err
	}

	// blocking the network traffic to a node gives: dial tcp 10.129.0.31:8443: connect: no route to host
	// no rsp has been sent to the client so it's okay to retry and pick up a different EP
	if errno, ok := err.(syscall.Errno); ok && errno == syscall.EHOSTUNREACH {
		return true
	}
	return false
}
