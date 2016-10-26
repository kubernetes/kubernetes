/*
Copyright 2014 The Kubernetes Authors.

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

package rest

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api/errors"
	utilconfig "k8s.io/kubernetes/pkg/util/config"
	"k8s.io/kubernetes/pkg/util/httpstream"
	"k8s.io/kubernetes/pkg/util/proxy"

	"github.com/golang/glog"
	"github.com/mxk/go-flowrate/flowrate"
)

// UpgradeAwareProxyHandler is a handler for proxy requests that may require an upgrade
type UpgradeAwareProxyHandler struct {
	UpgradeRequired bool
	Location        *url.URL
	// Transport provides an optional round tripper to use to proxy. If nil, the default proxy transport is used
	Transport http.RoundTripper
	// WrapTransport indicates whether the provided Transport should be wrapped with default proxy transport behavior (URL rewriting, X-Forwarded-* header setting)
	WrapTransport bool
	// InterceptRedirects determines whether the proxy should sniff backend responses for redirects,
	// following them as necessary.
	InterceptRedirects bool
	FlushInterval      time.Duration
	MaxBytesPerSec     int64
	Responder          ErrorResponder
}

const defaultFlushInterval = 200 * time.Millisecond

// ErrorResponder abstracts error reporting to the proxy handler to remove the need to hardcode a particular
// error format.
type ErrorResponder interface {
	Error(err error)
}

// NewUpgradeAwareProxyHandler creates a new proxy handler with a default flush interval. Responder is required for returning
// errors to the caller.
func NewUpgradeAwareProxyHandler(location *url.URL, transport http.RoundTripper, wrapTransport, upgradeRequired bool, responder ErrorResponder) *UpgradeAwareProxyHandler {
	return &UpgradeAwareProxyHandler{
		Location:        location,
		Transport:       transport,
		WrapTransport:   wrapTransport,
		UpgradeRequired: upgradeRequired,
		FlushInterval:   defaultFlushInterval,
		Responder:       responder,
	}
}

// ServeHTTP handles the proxy request
func (h *UpgradeAwareProxyHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if len(h.Location.Scheme) == 0 {
		h.Location.Scheme = "http"
	}
	if h.tryUpgrade(w, req) {
		return
	}
	if h.UpgradeRequired {
		h.Responder.Error(errors.NewBadRequest("Upgrade request required"))
		return
	}

	loc := *h.Location
	loc.RawQuery = req.URL.RawQuery

	// If original request URL ended in '/', append a '/' at the end of the
	// of the proxy URL
	if !strings.HasSuffix(loc.Path, "/") && strings.HasSuffix(req.URL.Path, "/") {
		loc.Path += "/"
	}

	// From pkg/apiserver/proxy.go#ServeHTTP:
	// Redirect requests with an empty path to a location that ends with a '/'
	// This is essentially a hack for http://issue.k8s.io/4958.
	// Note: Keep this code after tryUpgrade to not break that flow.
	if len(loc.Path) == 0 {
		var queryPart string
		if len(req.URL.RawQuery) > 0 {
			queryPart = "?" + req.URL.RawQuery
		}
		w.Header().Set("Location", req.URL.Path+"/"+queryPart)
		w.WriteHeader(http.StatusMovedPermanently)
		return
	}

	if h.Transport == nil || h.WrapTransport {
		h.Transport = h.defaultProxyTransport(req.URL, h.Transport)
	}

	newReq, err := http.NewRequest(req.Method, loc.String(), req.Body)
	if err != nil {
		h.Responder.Error(err)
		return
	}
	newReq.Header = req.Header
	newReq.ContentLength = req.ContentLength
	// Copy the TransferEncoding is for future-proofing. Currently Go only supports "chunked" and
	// it can determine the TransferEncoding based on ContentLength and the Body.
	newReq.TransferEncoding = req.TransferEncoding

	proxy := httputil.NewSingleHostReverseProxy(&url.URL{Scheme: h.Location.Scheme, Host: h.Location.Host})
	proxy.Transport = h.Transport
	proxy.FlushInterval = h.FlushInterval
	proxy.ServeHTTP(w, newReq)
}

// tryUpgrade returns true if the request was handled.
func (h *UpgradeAwareProxyHandler) tryUpgrade(w http.ResponseWriter, req *http.Request) bool {
	if !httpstream.IsUpgradeRequest(req) {
		return false
	}
	if err := h.handleUpgrade(w, req); err != nil {
		h.Responder.Error(err)
	}
	return true
}

func (h *UpgradeAwareProxyHandler) handleUpgrade(w http.ResponseWriter, req *http.Request) error {
	requestHijacker, ok := w.(http.Hijacker)
	if !ok {
		return fmt.Errorf("request connection cannot be hijacked: %T", w)
	}
	requestHijackedConn, _, err := requestHijacker.Hijack()
	if err != nil {
		return fmt.Errorf("error hijacking request connection: %v", err)
	}
	defer requestHijackedConn.Close()

	var backendConn net.Conn
	if h.InterceptRedirects && utilconfig.DefaultFeatureGate.StreamingProxyRedirects() {
		backendConn, err = h.connectBackendWithRedirects(req, requestHijackedConn)
	} else {
		backendConn, err = h.connectBackend(req, h.Location)
	}
	if err != nil {
		return fmt.Errorf("error dialing backend: %v", err)
	}
	defer backendConn.Close()

	// Proxy the connection.
	wg := &sync.WaitGroup{}
	wg.Add(2)

	go func() {
		var writer io.WriteCloser
		if h.MaxBytesPerSec > 0 {
			writer = flowrate.NewWriter(backendConn, h.MaxBytesPerSec)
		} else {
			writer = backendConn
		}
		_, err := io.Copy(writer, requestHijackedConn)
		if err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
			glog.Errorf("Error proxying data from client to backend: %v", err)
		}
		wg.Done()
	}()

	go func() {
		var reader io.ReadCloser
		if h.MaxBytesPerSec > 0 {
			reader = flowrate.NewReader(backendConn, h.MaxBytesPerSec)
		} else {
			reader = backendConn
		}
		_, err := io.Copy(requestHijackedConn, reader)
		if err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
			glog.Errorf("Error proxying data from backend to client: %v", err)
		}
		wg.Done()
	}()

	wg.Wait()
	return nil
}

// connectBackend dials the backend at location and forwards a copy of the client request.
func (h *UpgradeAwareProxyHandler) connectBackend(req *http.Request, location *url.URL) (conn net.Conn, err error) {
	defer func() {
		if err != nil && conn != nil {
			conn.Close()
		}
	}()

	beReq, err := http.NewRequest(req.Method, location.String(), req.Body)
	if err != nil {
		return nil, err
	}
	beReq.Header = req.Header

	conn, err = proxy.DialURL(location, h.Transport)
	if err != nil {
		return conn, fmt.Errorf("error dialing backend: %v", err)
	}

	if err = beReq.Write(conn); err != nil {
		return conn, fmt.Errorf("error sending request: %v", err)
	}

	return conn, err
}

//
func (h *UpgradeAwareProxyHandler) connectBackendWithRedirects(req *http.Request, clientConn net.Conn) (net.Conn, error) {
	const (
		maxRedirects    = 10
		maxResponseSize = 1024
	)
	var (
		rawResponse      = bytes.NewBuffer(make([]byte, 0, 256))
		location         = h.Location
		intermediateConn net.Conn
		err              error
	)
	defer func() {
		if intermediateConn != nil {
			intermediateConn.Close()
		}
	}()

redirectLoop:
	for redirects := 0; ; redirects++ {
		if redirects == maxRedirects {
			return nil, fmt.Errorf("too many redirects (%d)", redirects)
		}

		intermediateConn, err = h.connectBackend(req, location)
		if err != nil {
			return nil, err
		}

		// Peek at the backend response.
		rawResponse.Reset()
		respReader := bufio.NewReader(io.TeeReader(
			io.LimitReader(intermediateConn, maxResponseSize), // Don't read more than maxResponseSize bytes.
			rawResponse)) // Save the raw response.
		resp, err := http.ReadResponse(respReader, req)
		if err != nil {
			// Unable to read the backend response; let the client handle it.
			glog.Warningf("Error reading backend response: %v", err)
			break redirectLoop
		}
		resp.Body.Close() // Unused.

		switch resp.StatusCode {
		case http.StatusFound, http.StatusSeeOther:
			// Redirect, continue.
		default:
			// Don't redirect.
			break redirectLoop
		}

		// Reset the connection.
		intermediateConn.Close()
		intermediateConn = nil

		// Prepare to follow the redirect.
		redirectStr := resp.Header.Get("Location")
		if redirectStr == "" {
			return nil, fmt.Errorf("%d response missing Location header", resp.StatusCode)
		}
		location, err = h.Location.Parse(redirectStr)
		if err != nil {
			return nil, fmt.Errorf("malformed Location header: %v", err)
		}
	}

	// Forward raw response bytes back to client.
	_, err = io.CopyN(clientConn, rawResponse, int64(rawResponse.Len()))
	if err != nil {
		return nil, fmt.Errorf("forwarding response: %v", err)
	}

	backendConn := intermediateConn
	intermediateConn = nil // Don't close the connection when we return it.
	return backendConn, nil
}

func (h *UpgradeAwareProxyHandler) defaultProxyTransport(url *url.URL, internalTransport http.RoundTripper) http.RoundTripper {
	scheme := url.Scheme
	host := url.Host
	suffix := h.Location.Path
	if strings.HasSuffix(url.Path, "/") && !strings.HasSuffix(suffix, "/") {
		suffix += "/"
	}
	pathPrepend := strings.TrimSuffix(url.Path, suffix)
	rewritingTransport := &proxy.Transport{
		Scheme:       scheme,
		Host:         host,
		PathPrepend:  pathPrepend,
		RoundTripper: internalTransport,
	}
	return &corsRemovingTransport{
		RoundTripper: rewritingTransport,
	}
}

// corsRemovingTransport is a wrapper for an internal transport. It removes CORS headers
// from the internal response.
// Implements pkg/util/net.RoundTripperWrapper
type corsRemovingTransport struct {
	http.RoundTripper
}

func (rt *corsRemovingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := rt.RoundTripper.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	removeCORSHeaders(resp)
	return resp, nil
}

func (rt *corsRemovingTransport) WrappedRoundTripper() http.RoundTripper {
	return rt.RoundTripper
}

// removeCORSHeaders strip CORS headers sent from the backend
// This should be called on all responses before returning
func removeCORSHeaders(resp *http.Response) {
	resp.Header.Del("Access-Control-Allow-Credentials")
	resp.Header.Del("Access-Control-Allow-Headers")
	resp.Header.Del("Access-Control-Allow-Methods")
	resp.Header.Del("Access-Control-Allow-Origin")
}
