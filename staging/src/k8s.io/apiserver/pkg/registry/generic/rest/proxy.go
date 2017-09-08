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
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/httpstream"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/proxy"

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

// normalizeLocation returns the result of parsing the full URL, with scheme set to http if missing
func normalizeLocation(location *url.URL) *url.URL {
	normalized, _ := url.Parse(location.String())
	if len(normalized.Scheme) == 0 {
		normalized.Scheme = "http"
	}
	return normalized
}

// NewUpgradeAwareProxyHandler creates a new proxy handler with a default flush interval. Responder is required for returning
// errors to the caller.
func NewUpgradeAwareProxyHandler(location *url.URL, transport http.RoundTripper, wrapTransport, upgradeRequired bool, responder ErrorResponder) *UpgradeAwareProxyHandler {
	return &UpgradeAwareProxyHandler{
		Location:        normalizeLocation(location),
		Transport:       transport,
		WrapTransport:   wrapTransport,
		UpgradeRequired: upgradeRequired,
		FlushInterval:   defaultFlushInterval,
		Responder:       responder,
	}
}

// ServeHTTP handles the proxy request
func (h *UpgradeAwareProxyHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
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

	// From pkg/genericapiserver/endpoints/handlers/proxy.go#ServeHTTP:
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

	// WithContext creates a shallow clone of the request with the new context.
	newReq := req.WithContext(context.Background())
	newReq.Header = utilnet.CloneHeader(req.Header)
	newReq.URL = &loc

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

	var (
		backendConn net.Conn
		rawResponse []byte
		err         error
	)

	clone := utilnet.CloneRequest(req)
	// Only append X-Forwarded-For in the upgrade path, since httputil.NewSingleHostReverseProxy
	// handles this in the non-upgrade path.
	utilnet.AppendForwardedForHeader(clone)
	if h.InterceptRedirects && utilfeature.DefaultFeatureGate.Enabled(genericfeatures.StreamingProxyRedirects) {
		backendConn, rawResponse, err = utilnet.ConnectWithRedirects(req.Method, h.Location, clone.Header, req.Body, h)
	} else {
		clone.URL = h.Location
		backendConn, err = h.Dial(clone)
	}
	if err != nil {
		h.Responder.Error(err)
		return true
	}
	defer backendConn.Close()

	// Once the connection is hijacked, the ErrorResponder will no longer work, so
	// hijacking should be the last step in the upgrade.
	requestHijacker, ok := w.(http.Hijacker)
	if !ok {
		h.Responder.Error(fmt.Errorf("request connection cannot be hijacked: %T", w))
		return true
	}
	requestHijackedConn, _, err := requestHijacker.Hijack()
	if err != nil {
		h.Responder.Error(fmt.Errorf("error hijacking request connection: %v", err))
		return true
	}
	defer requestHijackedConn.Close()

	// Forward raw response bytes back to client.
	if len(rawResponse) > 0 {
		if _, err = requestHijackedConn.Write(rawResponse); err != nil {
			utilruntime.HandleError(fmt.Errorf("Error proxying response from backend to client: %v", err))
		}
	}

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
	return true
}

// Dial dials the backend at req.URL and writes req to it.
func (h *UpgradeAwareProxyHandler) Dial(req *http.Request) (net.Conn, error) {
	conn, err := proxy.DialURL(req.URL, h.Transport)
	if err != nil {
		return nil, fmt.Errorf("error dialing backend: %v", err)
	}

	if err = req.Write(conn); err != nil {
		conn.Close()
		return nil, fmt.Errorf("error sending request: %v", err)
	}

	return conn, err
}

var _ utilnet.Dialer = &UpgradeAwareProxyHandler{}

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

var _ = utilnet.RoundTripperWrapper(&corsRemovingTransport{})

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
