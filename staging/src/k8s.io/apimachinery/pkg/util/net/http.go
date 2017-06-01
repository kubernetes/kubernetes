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

package net

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"golang.org/x/net/http2"
)

// IsProbableEOF returns true if the given error resembles a connection termination
// scenario that would justify assuming that the watch is empty.
// These errors are what the Go http stack returns back to us which are general
// connection closure errors (strongly correlated) and callers that need to
// differentiate probable errors in connection behavior between normal "this is
// disconnected" should use the method.
func IsProbableEOF(err error) bool {
	if uerr, ok := err.(*url.Error); ok {
		err = uerr.Err
	}
	switch {
	case err == io.EOF:
		return true
	case err.Error() == "http: can't write HTTP request on broken connection":
		return true
	case strings.Contains(err.Error(), "connection reset by peer"):
		return true
	case strings.Contains(strings.ToLower(err.Error()), "use of closed network connection"):
		return true
	}
	return false
}

var defaultTransport = http.DefaultTransport.(*http.Transport)

// SetOldTransportDefaults applies the defaults from http.DefaultTransport
// for the Proxy, Dial, and TLSHandshakeTimeout fields if unset
func SetOldTransportDefaults(t *http.Transport) *http.Transport {
	if t.Proxy == nil || isDefault(t.Proxy) {
		// http.ProxyFromEnvironment doesn't respect CIDRs and that makes it impossible to exclude things like pod and service IPs from proxy settings
		// ProxierWithNoProxyCIDR allows CIDR rules in NO_PROXY
		t.Proxy = NewProxierWithNoProxyCIDR(http.ProxyFromEnvironment)
	}
	if t.Dial == nil {
		t.Dial = defaultTransport.Dial
	}
	if t.TLSHandshakeTimeout == 0 {
		t.TLSHandshakeTimeout = defaultTransport.TLSHandshakeTimeout
	}
	return t
}

// SetTransportDefaults applies the defaults from http.DefaultTransport
// for the Proxy, Dial, and TLSHandshakeTimeout fields if unset
func SetTransportDefaults(t *http.Transport) *http.Transport {
	t = SetOldTransportDefaults(t)
	// Allow clients to disable http2 if needed.
	if s := os.Getenv("DISABLE_HTTP2"); len(s) > 0 {
		glog.Infof("HTTP2 has been explicitly disabled")
	} else {
		if err := http2.ConfigureTransport(t); err != nil {
			glog.Warningf("Transport failed http2 configuration: %v", err)
		}
	}
	return t
}

type RoundTripperWrapper interface {
	http.RoundTripper
	WrappedRoundTripper() http.RoundTripper
}

type DialFunc func(net, addr string) (net.Conn, error)

func DialerFor(transport http.RoundTripper) (DialFunc, error) {
	if transport == nil {
		return nil, nil
	}

	switch transport := transport.(type) {
	case *http.Transport:
		return transport.Dial, nil
	case RoundTripperWrapper:
		return DialerFor(transport.WrappedRoundTripper())
	default:
		return nil, fmt.Errorf("unknown transport type: %v", transport)
	}
}

type TLSClientConfigHolder interface {
	TLSClientConfig() *tls.Config
}

func TLSClientConfig(transport http.RoundTripper) (*tls.Config, error) {
	if transport == nil {
		return nil, nil
	}

	switch transport := transport.(type) {
	case *http.Transport:
		return transport.TLSClientConfig, nil
	case TLSClientConfigHolder:
		return transport.TLSClientConfig(), nil
	case RoundTripperWrapper:
		return TLSClientConfig(transport.WrappedRoundTripper())
	default:
		return nil, fmt.Errorf("unknown transport type: %v", transport)
	}
}

func FormatURL(scheme string, host string, port int, path string) *url.URL {
	return &url.URL{
		Scheme: scheme,
		Host:   net.JoinHostPort(host, strconv.Itoa(port)),
		Path:   path,
	}
}

func GetHTTPClient(req *http.Request) string {
	if userAgent, ok := req.Header["User-Agent"]; ok {
		if len(userAgent) > 0 {
			return userAgent[0]
		}
	}
	return "unknown"
}

// Extracts and returns the clients IP from the given request.
// Looks at X-Forwarded-For header, X-Real-Ip header and request.RemoteAddr in that order.
// Returns nil if none of them are set or is set to an invalid value.
func GetClientIP(req *http.Request) net.IP {
	hdr := req.Header
	// First check the X-Forwarded-For header for requests via proxy.
	hdrForwardedFor := hdr.Get("X-Forwarded-For")
	if hdrForwardedFor != "" {
		// X-Forwarded-For can be a csv of IPs in case of multiple proxies.
		// Use the first valid one.
		parts := strings.Split(hdrForwardedFor, ",")
		for _, part := range parts {
			ip := net.ParseIP(strings.TrimSpace(part))
			if ip != nil {
				return ip
			}
		}
	}

	// Try the X-Real-Ip header.
	hdrRealIp := hdr.Get("X-Real-Ip")
	if hdrRealIp != "" {
		ip := net.ParseIP(hdrRealIp)
		if ip != nil {
			return ip
		}
	}

	// Fallback to Remote Address in request, which will give the correct client IP when there is no proxy.
	// Remote Address in Go's HTTP server is in the form host:port so we need to split that first.
	host, _, err := net.SplitHostPort(req.RemoteAddr)
	if err == nil {
		return net.ParseIP(host)
	}

	// Fallback if Remote Address was just IP.
	return net.ParseIP(req.RemoteAddr)
}

var defaultProxyFuncPointer = fmt.Sprintf("%p", http.ProxyFromEnvironment)

// isDefault checks to see if the transportProxierFunc is pointing to the default one
func isDefault(transportProxier func(*http.Request) (*url.URL, error)) bool {
	transportProxierPointer := fmt.Sprintf("%p", transportProxier)
	return transportProxierPointer == defaultProxyFuncPointer
}

// NewProxierWithNoProxyCIDR constructs a Proxier function that respects CIDRs in NO_PROXY and delegates if
// no matching CIDRs are found
func NewProxierWithNoProxyCIDR(delegate func(req *http.Request) (*url.URL, error)) func(req *http.Request) (*url.URL, error) {
	// we wrap the default method, so we only need to perform our check if the NO_PROXY envvar has a CIDR in it
	noProxyEnv := os.Getenv("NO_PROXY")
	noProxyRules := strings.Split(noProxyEnv, ",")

	cidrs := []*net.IPNet{}
	for _, noProxyRule := range noProxyRules {
		_, cidr, _ := net.ParseCIDR(noProxyRule)
		if cidr != nil {
			cidrs = append(cidrs, cidr)
		}
	}

	if len(cidrs) == 0 {
		return delegate
	}

	return func(req *http.Request) (*url.URL, error) {
		host := req.URL.Host
		// for some urls, the Host is already the host, not the host:port
		if net.ParseIP(host) == nil {
			var err error
			host, _, err = net.SplitHostPort(req.URL.Host)
			if err != nil {
				return delegate(req)
			}
		}

		ip := net.ParseIP(host)
		if ip == nil {
			return delegate(req)
		}

		for _, cidr := range cidrs {
			if cidr.Contains(ip) {
				return nil, nil
			}
		}

		return delegate(req)
	}
}

// Dialer dials a host and writes a request to it.
type Dialer interface {
	// Dial connects to the host specified by req's URL, writes the request to the connection, and
	// returns the opened net.Conn.
	Dial(req *http.Request) (net.Conn, error)
}

// ConnectWithRedirects uses dialer to send req, following up to 10 redirects (relative to
// originalLocation). It returns the opened net.Conn and the raw response bytes.
func ConnectWithRedirects(originalMethod string, originalLocation *url.URL, header http.Header, originalBody io.Reader, dialer Dialer) (net.Conn, []byte, error) {
	const (
		maxRedirects    = 10
		maxResponseSize = 16384 // play it safe to allow the potential for lots of / large headers
	)

	var (
		location         = originalLocation
		method           = originalMethod
		intermediateConn net.Conn
		rawResponse      = bytes.NewBuffer(make([]byte, 0, 256))
		body             = originalBody
	)

	defer func() {
		if intermediateConn != nil {
			intermediateConn.Close()
		}
	}()

redirectLoop:
	for redirects := 0; ; redirects++ {
		if redirects > maxRedirects {
			return nil, nil, fmt.Errorf("too many redirects (%d)", redirects)
		}

		req, err := http.NewRequest(method, location.String(), body)
		if err != nil {
			return nil, nil, err
		}

		req.Header = header

		intermediateConn, err = dialer.Dial(req)
		if err != nil {
			return nil, nil, err
		}

		// Peek at the backend response.
		rawResponse.Reset()
		respReader := bufio.NewReader(io.TeeReader(
			io.LimitReader(intermediateConn, maxResponseSize), // Don't read more than maxResponseSize bytes.
			rawResponse)) // Save the raw response.
		resp, err := http.ReadResponse(respReader, nil)
		if err != nil {
			// Unable to read the backend response; let the client handle it.
			glog.Warningf("Error reading backend response: %v", err)
			break redirectLoop
		}

		switch resp.StatusCode {
		case http.StatusFound:
			// Redirect, continue.
		default:
			// Don't redirect.
			break redirectLoop
		}

		// Redirected requests switch to "GET" according to the HTTP spec:
		// https://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html#sec10.3
		method = "GET"
		// don't send a body when following redirects
		body = nil

		resp.Body.Close() // not used

		// Reset the connection.
		intermediateConn.Close()
		intermediateConn = nil

		// Prepare to follow the redirect.
		redirectStr := resp.Header.Get("Location")
		if redirectStr == "" {
			return nil, nil, fmt.Errorf("%d response missing Location header", resp.StatusCode)
		}
		// We have to parse relative to the current location, NOT originalLocation. For example,
		// if we request http://foo.com/a and get back "http://bar.com/b", the result should be
		// http://bar.com/b. If we then make that request and get back a redirect to "/c", the result
		// should be http://bar.com/c, not http://foo.com/c.
		location, err = location.Parse(redirectStr)
		if err != nil {
			return nil, nil, fmt.Errorf("malformed Location header: %v", err)
		}
	}

	connToReturn := intermediateConn
	intermediateConn = nil // Don't close the connection when we return it.
	return connToReturn, rawResponse.Bytes(), nil
}

// CloneRequest creates a shallow copy of the request along with a deep copy of the Headers.
func CloneRequest(req *http.Request) *http.Request {
	r := new(http.Request)

	// shallow clone
	*r = *req

	// deep copy headers
	r.Header = CloneHeader(req.Header)

	return r
}

// CloneHeader creates a deep copy of an http.Header.
func CloneHeader(in http.Header) http.Header {
	out := make(http.Header, len(in))
	for key, values := range in {
		newValues := make([]string, len(values))
		copy(newValues, values)
		out[key] = newValues
	}
	return out
}
