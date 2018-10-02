/*
Copyright 2017 The Kubernetes Authors.

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

package proxy

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/httpstream"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"

	"github.com/golang/glog"
	"github.com/mxk/go-flowrate/flowrate"
)

// UpgradeRequestRoundTripper provides an additional method to decorate a request
// with any authentication or other protocol level information prior to performing
// an upgrade on the server. Any response will be handled by the intercepting
// proxy.
type UpgradeRequestRoundTripper interface {
	http.RoundTripper
	// WrapRequest takes a valid HTTP request and returns a suitably altered version
	// of request with any HTTP level values required to complete the request half of
	// an upgrade on the server. It does not get a chance to see the response and
	// should bypass any request side logic that expects to see the response.
	WrapRequest(*http.Request) (*http.Request, error)
}

// UpgradeAwareHandler is a handler for proxy requests that may require an upgrade
type UpgradeAwareHandler struct {
	// UpgradeRequired will reject non-upgrade connections if true.
	UpgradeRequired bool
	// Location is the location of the upstream proxy. It is used as the location to Dial on the upstream server
	// for upgrade requests unless UseRequestLocationOnUpgrade is true.
	Location *url.URL
	// Transport provides an optional round tripper to use to proxy. If nil, the default proxy transport is used
	Transport http.RoundTripper
	// UpgradeTransport, if specified, will be used as the backend transport when upgrade requests are provided.
	// This allows clients to disable HTTP/2.
	UpgradeTransport UpgradeRequestRoundTripper
	// WrapTransport indicates whether the provided Transport should be wrapped with default proxy transport behavior (URL rewriting, X-Forwarded-* header setting)
	WrapTransport bool
	// InterceptRedirects determines whether the proxy should sniff backend responses for redirects,
	// following them as necessary.
	InterceptRedirects bool
	// RequireSameHostRedirects only allows redirects to the same host. It is only used if InterceptRedirects=true.
	RequireSameHostRedirects bool
	// UseRequestLocation will use the incoming request URL when talking to the backend server.
	UseRequestLocation bool
	// FlushInterval controls how often the standard HTTP proxy will flush content from the upstream.
	FlushInterval time.Duration
	// MaxBytesPerSec controls the maximum rate for an upstream connection. No rate is imposed if the value is zero.
	MaxBytesPerSec int64
	// Responder is passed errors that occur while setting up proxying.
	Responder ErrorResponder
}

const defaultFlushInterval = 200 * time.Millisecond

// ErrorResponder abstracts error reporting to the proxy handler to remove the need to hardcode a particular
// error format.
type ErrorResponder interface {
	Error(w http.ResponseWriter, req *http.Request, err error)
}

// SimpleErrorResponder is the legacy implementation of ErrorResponder for callers that only
// service a single request/response per proxy.
type SimpleErrorResponder interface {
	Error(err error)
}

func NewErrorResponder(r SimpleErrorResponder) ErrorResponder {
	return simpleResponder{r}
}

type simpleResponder struct {
	responder SimpleErrorResponder
}

func (r simpleResponder) Error(w http.ResponseWriter, req *http.Request, err error) {
	r.responder.Error(err)
}

// upgradeRequestRoundTripper implements proxy.UpgradeRequestRoundTripper.
type upgradeRequestRoundTripper struct {
	http.RoundTripper
	upgrader http.RoundTripper
}

var (
	_ UpgradeRequestRoundTripper  = &upgradeRequestRoundTripper{}
	_ utilnet.RoundTripperWrapper = &upgradeRequestRoundTripper{}
)

// WrappedRoundTripper returns the round tripper that a caller would use.
func (rt *upgradeRequestRoundTripper) WrappedRoundTripper() http.RoundTripper {
	return rt.RoundTripper
}

// WriteToRequest calls the nested upgrader and then copies the returned request
// fields onto the passed request.
func (rt *upgradeRequestRoundTripper) WrapRequest(req *http.Request) (*http.Request, error) {
	resp, err := rt.upgrader.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	return resp.Request, nil
}

// onewayRoundTripper captures the provided request - which is assumed to have
// been modified by other round trippers - and then returns a fake response.
type onewayRoundTripper struct{}

// RoundTrip returns a simple 200 OK response that captures the provided request.
func (onewayRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		Status:     "200 OK",
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(&bytes.Buffer{}),
		Request:    req,
	}, nil
}

// MirrorRequest is a round tripper that can be called to get back the calling request as
// the core round tripper in a chain.
var MirrorRequest http.RoundTripper = onewayRoundTripper{}

// NewUpgradeRequestRoundTripper takes two round trippers - one for the underlying TCP connection, and
// one that is able to write headers to an HTTP request. The request rt is used to set the request headers
// and that is written to the underlying connection rt.
func NewUpgradeRequestRoundTripper(connection, request http.RoundTripper) UpgradeRequestRoundTripper {
	return &upgradeRequestRoundTripper{
		RoundTripper: connection,
		upgrader:     request,
	}
}

// normalizeLocation returns the result of parsing the full URL, with scheme set to http if missing
func normalizeLocation(location *url.URL) *url.URL {
	normalized, _ := url.Parse(location.String())
	if len(normalized.Scheme) == 0 {
		normalized.Scheme = "http"
	}
	return normalized
}

// NewUpgradeAwareHandler creates a new proxy handler with a default flush interval. Responder is required for returning
// errors to the caller.
func NewUpgradeAwareHandler(location *url.URL, transport http.RoundTripper, wrapTransport, upgradeRequired bool, responder ErrorResponder) *UpgradeAwareHandler {
	return &UpgradeAwareHandler{
		Location:        normalizeLocation(location),
		Transport:       transport,
		WrapTransport:   wrapTransport,
		UpgradeRequired: upgradeRequired,
		FlushInterval:   defaultFlushInterval,
		Responder:       responder,
	}
}

// ServeHTTP handles the proxy request
func (h *UpgradeAwareHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if h.tryUpgrade(w, req) {
		return
	}
	if h.UpgradeRequired {
		h.Responder.Error(w, req, errors.NewBadRequest("Upgrade request required"))
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
	if !h.UseRequestLocation {
		newReq.URL = &loc
	}

	proxy := httputil.NewSingleHostReverseProxy(&url.URL{Scheme: h.Location.Scheme, Host: h.Location.Host})
	proxy.Transport = h.Transport
	proxy.FlushInterval = h.FlushInterval
	proxy.ServeHTTP(w, newReq)
}

// tryUpgrade returns true if the request was handled.
func (h *UpgradeAwareHandler) tryUpgrade(w http.ResponseWriter, req *http.Request) bool {
	if !httpstream.IsUpgradeRequest(req) {
		glog.V(6).Infof("Request was not an upgrade")
		return false
	}

	var (
		backendConn net.Conn
		rawResponse []byte
		err         error
	)

	location := *h.Location
	if h.UseRequestLocation {
		location = *req.URL
		location.Scheme = h.Location.Scheme
		location.Host = h.Location.Host
	}

	clone := utilnet.CloneRequest(req)
	// Only append X-Forwarded-For in the upgrade path, since httputil.NewSingleHostReverseProxy
	// handles this in the non-upgrade path.
	utilnet.AppendForwardedForHeader(clone)
	if h.InterceptRedirects {
		glog.V(6).Infof("Connecting to backend proxy (intercepting redirects) %s\n  Headers: %v", &location, clone.Header)
		backendConn, rawResponse, err = utilnet.ConnectWithRedirects(req.Method, &location, clone.Header, req.Body, utilnet.DialerFunc(h.DialForUpgrade), h.RequireSameHostRedirects)
	} else {
		glog.V(6).Infof("Connecting to backend proxy (direct dial) %s\n  Headers: %v", &location, clone.Header)
		clone.URL = &location
		backendConn, err = h.DialForUpgrade(clone)
	}
	if err != nil {
		glog.V(6).Infof("Proxy connection error: %v", err)
		h.Responder.Error(w, req, err)
		return true
	}
	defer backendConn.Close()

	// Once the connection is hijacked, the ErrorResponder will no longer work, so
	// hijacking should be the last step in the upgrade.
	requestHijacker, ok := w.(http.Hijacker)
	if !ok {
		glog.V(6).Infof("Unable to hijack response writer: %T", w)
		h.Responder.Error(w, req, fmt.Errorf("request connection cannot be hijacked: %T", w))
		return true
	}
	requestHijackedConn, _, err := requestHijacker.Hijack()
	if err != nil {
		glog.V(6).Infof("Unable to hijack response: %v", err)
		h.Responder.Error(w, req, fmt.Errorf("error hijacking connection: %v", err))
		return true
	}
	defer requestHijackedConn.Close()

	// Forward raw response bytes back to client.
	if len(rawResponse) > 0 {
		glog.V(6).Infof("Writing %d bytes to hijacked connection", len(rawResponse))
		if _, err = requestHijackedConn.Write(rawResponse); err != nil {
			utilruntime.HandleError(fmt.Errorf("Error proxying response from backend to client: %v", err))
		}
	}

	// Proxy the connection. This is bidirectional, so we need a goroutine
	// to copy in each direction. Once one side of the connection exits, we
	// exit the function which performs cleanup and in the process closes
	// the other half of the connection in the defer.
	writerComplete := make(chan struct{})
	readerComplete := make(chan struct{})

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
		close(writerComplete)
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
		close(readerComplete)
	}()

	// Wait for one half the connection to exit. Once it does the defer will
	// clean up the other half of the connection.
	select {
	case <-writerComplete:
	case <-readerComplete:
	}
	glog.V(6).Infof("Disconnecting from backend proxy %s\n  Headers: %v", &location, clone.Header)

	return true
}

func (h *UpgradeAwareHandler) Dial(req *http.Request) (net.Conn, error) {
	return dial(req, h.Transport)
}

func (h *UpgradeAwareHandler) DialForUpgrade(req *http.Request) (net.Conn, error) {
	if h.UpgradeTransport == nil {
		return dial(req, h.Transport)
	}
	updatedReq, err := h.UpgradeTransport.WrapRequest(req)
	if err != nil {
		return nil, err
	}
	return dial(updatedReq, h.UpgradeTransport)
}

// dial dials the backend at req.URL and writes req to it.
func dial(req *http.Request, transport http.RoundTripper) (net.Conn, error) {
	conn, err := DialURL(req.Context(), req.URL, transport)
	if err != nil {
		return nil, fmt.Errorf("error dialing backend: %v", err)
	}

	if err = req.Write(conn); err != nil {
		conn.Close()
		return nil, fmt.Errorf("error sending request: %v", err)
	}

	return conn, err
}

var _ utilnet.Dialer = &UpgradeAwareHandler{}

func (h *UpgradeAwareHandler) defaultProxyTransport(url *url.URL, internalTransport http.RoundTripper) http.RoundTripper {
	scheme := url.Scheme
	host := url.Host
	suffix := h.Location.Path
	if strings.HasSuffix(url.Path, "/") && !strings.HasSuffix(suffix, "/") {
		suffix += "/"
	}
	pathPrepend := strings.TrimSuffix(url.Path, suffix)
	rewritingTransport := &Transport{
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
