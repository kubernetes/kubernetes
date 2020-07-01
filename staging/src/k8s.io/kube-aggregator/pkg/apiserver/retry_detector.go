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
	"fmt"
	"net"
	"net/http"
	"time"

	knet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/client-go/util/flowcontrol"
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
	StatusCode() (hijacked bool, code int)
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

	statusCode  int
	wasHijacked bool
}

// newSimpleResponseWriterInterceptor wraps the given ResponseWrite and intercept WriteHeader() and Hijack() methods
func newSimpleResponseWriterInterceptor(w http.ResponseWriter) *responseWriter {
	return &responseWriter{w, 0, false}
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

func (w *responseWriter) StatusCode() (hijacked bool, code int) {
	return w.wasHijacked, w.statusCode
}

type retriable interface {
	retry() bool
	reset()
}

type retryDecorator struct {
	delegate retriable
	rw       responseWriterInterceptor
}

// newRetryDecorator wraps delegate for detecting Hijacked connections
func newRetryDecorator(rw responseWriterInterceptor, errRsp *retriableHijackErrorResponder, singleEndpoint bool, retry int) *retryDecorator {
	var delegate retriable

	if singleEndpoint {
		errRsp.isRetriable = func(req *http.Request, err error) bool {
			return false
		}
		delegate = withBackOff(errRsp)
	} else {
		delegate = withHijackErrorResponderForMultipleEndpoints(errRsp)
	}

	return &retryDecorator{withMaxRetries(delegate, retry), rw}
}

// RetryIfNeeded returns true if the request failed and can be safely retried otherwise it returns false
func (p *retryDecorator) RetryIfNeeded() bool {
	// do not retry if the request has been hijacked or a response has already been sent to a client
	if wasHijacked, code := p.rw.StatusCode(); wasHijacked || code != 0 {
		return false
	}

	if p.delegate.retry() {
		p.delegate.reset()
		return true
	}
	return false
}

type maxRetries struct {
	retriable
	counter int
	max     int
}

var _ retriable = &maxRetries{}

func withMaxRetries(delegate retriable, max int) retriable {
	return &maxRetries{retriable: delegate, max: max}
}

func (r *maxRetries) retry() bool {
	r.counter++
	if r.counter > r.max {
		return false
	}

	return r.retriable.retry()
}

type backOff struct {
	key     string
	manager *flowcontrol.Backoff
	retriable
}

var _ retriable = &backOff{}

func withBackOff(delegate retriable) retriable {
	return &backOff{"static-key-for-single-host", flowcontrol.NewBackOff(4*time.Second, 30*time.Second), delegate}
}

func (b *backOff) retry() bool {
	if b.retriable.retry() {
		b.manager.Next(b.key, b.manager.Clock.Now())
		b.manager.Clock.Sleep(b.manager.Get(b.key))
		return true
	}
	return false
}

// retriableHijackErrorResponder wraps proxy.ErrorResponder and prevents errors from being written to the client if they can be retried
type retriableHijackErrorResponder struct {
	// delegate knows how to write errors to the client
	delegate       proxy.ErrorResponder
	req            *http.Request
	lastKnownError error
	isRetriable    func(req *http.Request, err error) bool
}

var _ proxy.ErrorResponder = &retriableHijackErrorResponder{}
var _ retriable = &retriableHijackErrorResponder{}

// newHijackErrorResponder creates a new ErrorResponder that wraps the delegate for supporting reties
func newHijackErrorResponder(delegate proxy.ErrorResponder, req *http.Request) *retriableHijackErrorResponder {
	return &retriableHijackErrorResponder{delegate: delegate, req: req}
}

// Error reports the err to the client or suppress it if it's not retriable
func (hr *retriableHijackErrorResponder) Error(w http.ResponseWriter, r *http.Request, err error) {
	// if we can retry the request do not send a response to the client
	hr.lastKnownError = err
	hr.req = r
	if !hr.isRetriable(hr.req, err) {
		hr.delegate.Error(w, r, err) // this might send a response to a client
	}
}

func (hr *retriableHijackErrorResponder) reset() {
	hr.lastKnownError = nil
}

func (hr *retriableHijackErrorResponder) retry() bool {
	return hr.isRetriable(hr.req, hr.lastKnownError)
}

// withHijackErrorResponderForMultipleEndpoints is used by the ErrorResponder to determine
// if the given error is safe to retry when more than one endpoint is available
func withHijackErrorResponderForMultipleEndpoints(hr *retriableHijackErrorResponder) retriable {
	hr.isRetriable = func(req *http.Request, err error) bool {
		// we always want to retry connection refused errors as the aggregator will pick up a different host on the next try.
		// the error is of particular interest during graceful shutdown of the backend server:
		// 	net/http2 library automatically retries requests/streams after receiving a GOAWAY frame.
		// 	this is true for StreamsWithID > LastStreamID (present in the GOAWAY frame) which represents the last stream identifier the server has processed or is aware of.
		// 	StreamsWithID <= LastStreamID might be fully processed because the server waits until all current streams are done or the timeout expires.
		// 	in case of the timeout, we will get "http2: server sent GOAWAY and closed the connection" error which is handled by IsProbableEOF() and it is safe to retry only for particular verbs (check isHTTPVerbRetriable method)
		// 	on the next try a new connection will be opened to the same host and will fail with "connection refused" error because the remote host stopped listening on the port.
		if knet.IsConnectionRefused(err) || knet.IsNoRouteToHost(err) {
			return true
		}
		return false
	}
	return hr
}
