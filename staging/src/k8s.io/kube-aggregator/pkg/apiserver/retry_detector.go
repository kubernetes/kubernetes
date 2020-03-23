package apiserver

import (
	"bufio"
	"fmt"
	"net"
	"net/http"

	knet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/proxy"
)

type retriable interface {
	ShouldRetry() bool
	Reset()
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
	// no-op
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
	// no-op
}

func (r *maxRetries) ShouldRetry() bool {
	r.counter++
	if r.counter > r.max {
		return false
	}

	return r.delegate.ShouldRetry()
}

type hijackResponder struct {
	delegate proxy.ErrorResponder
	req *http.Request
	retry  bool
}

var _ proxy.ErrorResponder = &hijackResponder{}
var _ retriable = &hijackResponder{}

func newHijackResponder(delegate proxy.ErrorResponder, req *http.Request) *hijackResponder {
	return &hijackResponder{delegate: delegate, req: req}
}

func (hr *hijackResponder) Error(w http.ResponseWriter, r *http.Request, err error) {
	// if we can retry the request do not send a response to the client
	if !hr.canRetry(err) {
		hr.delegate.Error(w, r, err)
		return
	}
	hr.retry = true
}

func (hr *hijackResponder) Reset() {
	hr.retry = false
}

func (hr *hijackResponder) ShouldRetry() bool {
	return hr.retry
}

func (hr *hijackResponder) canRetry(err error) bool {
	if isHTTPVerbRetriable(hr.req) && (knet.IsConnectionReset(err) || knet.IsConnectionRefused(err)) {
		return true
	}
	return false
}

func isHTTPVerbRetriable(req *http.Request) bool {
  return req.Method == "GET"
}