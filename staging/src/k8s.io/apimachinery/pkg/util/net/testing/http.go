/*
Copyright 2023 The Kubernetes Authors.

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

// Package nettesting contains utilities for testing networking functionality.
// Don't use these utilities in production code. They have not been security
// reviewed.
package nettesting

import (
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"sync"
	"testing"
)

// NewHTTPProxyHandler returns a new HTTPProxyHandler. It accepts an optional
// hook which is called early in the handler to export request state. If the
// hook returns false, the handler returns immediately with a server error.
// Ensure that this is only used in tests. This code has not been security
// reviewed.
func NewHTTPProxyHandler(t testing.TB, hook func(req *http.Request) bool) *HTTPProxyHandler {
	h := &HTTPProxyHandler{
		hook: hook,
		httpProxy: httputil.ReverseProxy{
			Director: func(req *http.Request) {
				req.URL.Scheme = "http"
				req.URL.Host = req.Host
			},
		},
		t: t,
	}
	return h
}

// HTTPProxyHandler implements a simple handler for http_proxy and https_proxy
// requests for use in testing.
type HTTPProxyHandler struct {
	handlerDone sync.WaitGroup
	hook        func(r *http.Request) bool
	// httpProxy is the reverse proxy we use for standard http proxy requests.
	httpProxy httputil.ReverseProxy
	t         testing.TB
}

// ServeHTTP handles an HTTP proxy request.
func (h *HTTPProxyHandler) ServeHTTP(rw http.ResponseWriter, req *http.Request) {
	h.handlerDone.Add(1)
	defer h.handlerDone.Done()

	if h.hook != nil {
		if ok := h.hook(req); !ok {
			rw.WriteHeader(http.StatusInternalServerError)
			return
		}
	}

	b, err := httputil.DumpRequest(req, false)
	if err != nil {
		h.t.Logf("Failed to dump request, host=%s: %v", req.Host, err)
	} else {
		h.t.Logf("Proxy Request: %s", string(b))
	}

	if req.Method != http.MethodConnect {
		h.httpProxy.ServeHTTP(rw, req)
		return
	}

	// CONNECT proxy

	sconn, err := net.Dial("tcp", req.Host)
	if err != nil {
		h.t.Logf("Failed to dial proxy backend, host=%s: %v", req.Host, err)
		rw.WriteHeader(http.StatusInternalServerError)
		return
	}
	defer sconn.Close()

	hj, ok := rw.(http.Hijacker)
	if !ok {
		h.t.Logf("Can't switch protocols using non-Hijacker ResponseWriter: type=%T, host=%s", rw, req.Host)
		rw.WriteHeader(http.StatusInternalServerError)
		return
	}

	rw.WriteHeader(http.StatusOK)

	conn, brw, err := hj.Hijack()
	if err != nil {
		h.t.Logf("Failed to hijack client connection, host=%s: %v", req.Host, err)
		return
	}
	defer conn.Close()

	if err := brw.Flush(); err != nil {
		h.t.Logf("Failed to flush pending writes to client, host=%s: %v", req.Host, err)
		return
	}
	if _, err := io.Copy(sconn, io.LimitReader(brw, int64(brw.Reader.Buffered()))); err != nil {
		h.t.Logf("Failed to flush buffered reads to server, host=%s: %v", req.Host, err)
		return
	}

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		defer h.t.Logf("Server read close, host=%s", req.Host)
		io.Copy(conn, sconn)
	}()
	go func() {
		defer wg.Done()
		defer h.t.Logf("Server write close, host=%s", req.Host)
		io.Copy(sconn, conn)
	}()

	wg.Wait()
	h.t.Logf("Done handling CONNECT request, host=%s", req.Host)
}

func (h *HTTPProxyHandler) Wait() {
	h.handlerDone.Wait()
}
