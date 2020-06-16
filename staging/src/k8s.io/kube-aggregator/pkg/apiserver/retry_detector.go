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
	"os"
	"syscall"
	"errors"

	knet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/proxy"
)

type retriable interface {
	ShouldRetry() bool
	Reset()
	LastKnownError() error
}

type retryDetector struct {
	delegates []retriable
}

var _ retriable = &retryDetector{}

func newRetryDetector(delegates ...retriable) *retryDetector {
	return &retryDetector{delegates: delegates}
}

func (d *retryDetector) ShouldRetry() bool {
	for _, delegate := range d.delegates {
		if delegate.ShouldRetry() {
			return true
		}
	}
	return false
}

func (d *retryDetector) Reset() {
	for _, delegate := range d.delegates {
		delegate.Reset()
	}
}

func (d *retryDetector) LastKnownError() error {
	for _, delegate := range d.delegates {
		err := delegate.LastKnownError()
		if err != nil {
			return err
		}
	}

	return nil
}

type statusResponseWriter struct {
	http.ResponseWriter

	req *http.Request
	statusCode int
	wasHijacked bool
}


func newStatusResponseWriter(w http.ResponseWriter, req *http.Request) *statusResponseWriter {
	return &statusResponseWriter{w, req, 0, false}
}

func (w *statusResponseWriter) WriteHeader(code int) {
	w.statusCode = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	requestHijacker, ok := w.ResponseWriter.(http.Hijacker)
	if !ok {
		return nil, nil, fmt.Errorf("unable to hijack response writer: %T", w.ResponseWriter)
	}

	w.wasHijacked = true
	return requestHijacker.Hijack()
}

type hijackProtector struct {
	delegate retriable
	rw *statusResponseWriter
}

var _ retriable = &hijackProtector{}

func newHijackProtector(rw *statusResponseWriter, delegate retriable) *hijackProtector {
	return &hijackProtector{delegate, rw}
}

func (p *hijackProtector) ShouldRetry() bool {
	if p.rw.wasHijacked {
		return false
	}
	return p.delegate.ShouldRetry()
}

func (p *hijackProtector) Reset() {
	p.delegate.Reset()
}

func (p *hijackProtector) LastKnownError() error {
	return p.delegate.LastKnownError()
}

type maxRetries struct {
	delegate retriable
	counter int
	max int
}

var _ retriable = &maxRetries{}

func newMaxRetries(delegate retriable, max int) *maxRetries {
	return &maxRetries{delegate:delegate, max:max}
}

func (r *maxRetries) Reset() {
	r.delegate.Reset()
}

func (r *maxRetries) ShouldRetry() bool {
	r.counter++
	if r.counter > r.max {
		return false
	}

	return r.delegate.ShouldRetry()
}

func (r *maxRetries) LastKnownError() error {
	return r.delegate.LastKnownError()
}

type hijackResponder struct {
	delegate proxy.ErrorResponder
	req *http.Request
	retry  bool
	lastKnownError error
}

var _ proxy.ErrorResponder = &hijackResponder{}
var _ retriable = &hijackResponder{}

func newHijackResponder(delegate proxy.ErrorResponder, req *http.Request) *hijackResponder {
	return &hijackResponder{delegate: delegate, req: req}
}

func (hr *hijackResponder) Error(w http.ResponseWriter, r *http.Request, err error) {
	// if we can retry the request do not send a response to the client
	hr.lastKnownError = err
	if !hr.canRetry(err) {
		hr.delegate.Error(w, r, err)
		return
	}
	hr.retry = true
}

func (hr *hijackResponder) Reset() {
	hr.retry = false
	hr.lastKnownError = nil
}

func (hr *hijackResponder) ShouldRetry() bool {
	return hr.retry
}

func (hr *hijackResponder) LastKnownError() error {
	return hr.lastKnownError
}

func (hr *hijackResponder) canRetry(err error) bool {
	if isHTTPVerbRetriable(hr.req) && (knet.IsConnectionReset(err) || knet.IsConnectionRefused(err) ||  isExperimental(err)) {
		return true
	}
	return false
}

func isHTTPVerbRetriable(req *http.Request) bool {
	return req.Method == "GET"
}

func isExperimental(err error) bool {
	var osErr *os.SyscallError
	if errors.As(err, &osErr) {
		err = osErr.Err
	}

	// blocking the network traffic to a node gives: dial tcp 10.129.0.31:8443: connect: no route to host
	// no rsp has been sent to the client so it's okay to retry and can pick up a different EP
	if errno, ok := err.(syscall.Errno); ok && errno == syscall.EHOSTUNREACH {
		return true
	}
	return false
}
