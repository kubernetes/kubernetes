/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
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

// SetTransportDefaults applies the defaults from http.DefaultTransport
// for the Proxy, Dial, and TLSHandshakeTimeout fields if unset
func SetTransportDefaults(t *http.Transport) *http.Transport {
	if t.Proxy == nil {
		t.Proxy = defaultTransport.Proxy
	}
	if t.Dial == nil {
		t.Dial = defaultTransport.Dial
	}
	if t.TLSHandshakeTimeout == 0 {
		t.TLSHandshakeTimeout = defaultTransport.TLSHandshakeTimeout
	}
	return t
}

type RoundTripperWrapper interface {
	http.RoundTripper
	WrappedRoundTripper() http.RoundTripper
}

type DialFunc func(net, addr string) (net.Conn, error)

func Dialer(transport http.RoundTripper) (DialFunc, error) {
	if transport == nil {
		return nil, nil
	}

	switch transport := transport.(type) {
	case *http.Transport:
		return transport.Dial, nil
	case RoundTripperWrapper:
		return Dialer(transport.WrappedRoundTripper())
	default:
		return nil, fmt.Errorf("unknown transport type: %v", transport)
	}
}

func TLSClientConfig(transport http.RoundTripper) (*tls.Config, error) {
	if transport == nil {
		return nil, nil
	}

	switch transport := transport.(type) {
	case *http.Transport:
		return transport.TLSClientConfig, nil
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
	ip := net.ParseIP(req.RemoteAddr)
	return ip
}
