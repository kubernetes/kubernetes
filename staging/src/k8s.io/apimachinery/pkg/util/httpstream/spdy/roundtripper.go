/*
Copyright 2015 The Kubernetes Authors.

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

package spdy

import (
	"bufio"
	"crypto/tls"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/third_party/forked/golang/netutil"
)

// SpdyRoundTripper knows how to upgrade an HTTP request to one that supports
// multiplexed streams. After RoundTrip() is invoked, Conn will be set
// and usable. SpdyRoundTripper implements the UpgradeRoundTripper interface.
type SpdyRoundTripper struct {
	//tlsConfig holds the TLS configuration settings to use when connecting
	//to the remote server.
	tlsConfig *tls.Config

	/* TODO according to http://golang.org/pkg/net/http/#RoundTripper, a RoundTripper
	   must be safe for use by multiple concurrent goroutines. If this is absolutely
	   necessary, we could keep a map from http.Request to net.Conn. In practice,
	   a client will create an http.Client, set the transport to a new insteace of
	   SpdyRoundTripper, and use it a single time, so this hopefully won't be an issue.
	*/
	// conn is the underlying network connection to the remote server.
	conn net.Conn

	// Dialer is the dialer used to connect.  Used if non-nil.
	Dialer *net.Dialer

	// proxier knows which proxy to use given a request, defaults to http.ProxyFromEnvironment
	// Used primarily for mocking the proxy discovery in tests.
	proxier func(req *http.Request) (*url.URL, error)
}

// NewRoundTripper creates a new SpdyRoundTripper that will use
// the specified tlsConfig.
func NewRoundTripper(tlsConfig *tls.Config) httpstream.UpgradeRoundTripper {
	return NewSpdyRoundTripper(tlsConfig)
}

// NewSpdyRoundTripper creates a new SpdyRoundTripper that will use
// the specified tlsConfig. This function is mostly meant for unit tests.
func NewSpdyRoundTripper(tlsConfig *tls.Config) *SpdyRoundTripper {
	return &SpdyRoundTripper{tlsConfig: tlsConfig}
}

// implements pkg/util/net.TLSClientConfigHolder for proper TLS checking during proxying with a spdy roundtripper
func (s *SpdyRoundTripper) TLSClientConfig() *tls.Config {
	return s.tlsConfig
}

// dial dials the host specified by req, using TLS if appropriate, optionally
// using a proxy server if one is configured via environment variables.
func (s *SpdyRoundTripper) dial(req *http.Request) (net.Conn, error) {
	proxier := s.proxier
	if proxier == nil {
		proxier = http.ProxyFromEnvironment
	}
	proxyURL, err := proxier(req)
	if err != nil {
		return nil, err
	}

	if proxyURL == nil {
		return s.dialWithoutProxy(req.URL)
	}

	// ensure we use a canonical host with proxyReq
	targetHost := netutil.CanonicalAddr(req.URL)

	// proxying logic adapted from http://blog.h6t.eu/post/74098062923/golang-websocket-with-http-proxy-support
	proxyReq := http.Request{
		Method: "CONNECT",
		URL:    &url.URL{},
		Host:   targetHost,
	}

	if pa := s.proxyAuth(proxyURL); pa != "" {
		proxyReq.Header = http.Header{}
		proxyReq.Header.Set("Proxy-Authorization", pa)
	}

	proxyDialConn, err := s.dialWithoutProxy(proxyURL)
	if err != nil {
		return nil, err
	}

	proxyClientConn := httputil.NewProxyClientConn(proxyDialConn, nil)
	_, err = proxyClientConn.Do(&proxyReq)
	if err != nil && err != httputil.ErrPersistEOF {
		return nil, err
	}

	rwc, _ := proxyClientConn.Hijack()

	if req.URL.Scheme != "https" {
		return rwc, nil
	}

	host, _, err := net.SplitHostPort(targetHost)
	if err != nil {
		return nil, err
	}

	if s.tlsConfig == nil {
		s.tlsConfig = &tls.Config{}
	}

	if len(s.tlsConfig.ServerName) == 0 {
		s.tlsConfig.ServerName = host
	}

	tlsConn := tls.Client(rwc, s.tlsConfig)

	// need to manually call Handshake() so we can call VerifyHostname() below
	if err := tlsConn.Handshake(); err != nil {
		return nil, err
	}

	// Return if we were configured to skip validation
	if s.tlsConfig != nil && s.tlsConfig.InsecureSkipVerify {
		return tlsConn, nil
	}

	if err := tlsConn.VerifyHostname(host); err != nil {
		return nil, err
	}

	return tlsConn, nil
}

// dialWithoutProxy dials the host specified by url, using TLS if appropriate.
func (s *SpdyRoundTripper) dialWithoutProxy(url *url.URL) (net.Conn, error) {
	dialAddr := netutil.CanonicalAddr(url)

	if url.Scheme == "http" {
		if s.Dialer == nil {
			return net.Dial("tcp", dialAddr)
		} else {
			return s.Dialer.Dial("tcp", dialAddr)
		}
	}

	// TODO validate the TLSClientConfig is set up?
	var conn *tls.Conn
	var err error
	if s.Dialer == nil {
		conn, err = tls.Dial("tcp", dialAddr, s.tlsConfig)
	} else {
		conn, err = tls.DialWithDialer(s.Dialer, "tcp", dialAddr, s.tlsConfig)
	}
	if err != nil {
		return nil, err
	}

	// Return if we were configured to skip validation
	if s.tlsConfig != nil && s.tlsConfig.InsecureSkipVerify {
		return conn, nil
	}

	host, _, err := net.SplitHostPort(dialAddr)
	if err != nil {
		return nil, err
	}
	err = conn.VerifyHostname(host)
	if err != nil {
		return nil, err
	}

	return conn, nil
}

// proxyAuth returns, for a given proxy URL, the value to be used for the Proxy-Authorization header
func (s *SpdyRoundTripper) proxyAuth(proxyURL *url.URL) string {
	if proxyURL == nil || proxyURL.User == nil {
		return ""
	}
	credentials := proxyURL.User.String()
	encodedAuth := base64.StdEncoding.EncodeToString([]byte(credentials))
	return fmt.Sprintf("Basic %s", encodedAuth)
}

// RoundTrip executes the Request and upgrades it. After a successful upgrade,
// clients may call SpdyRoundTripper.Connection() to retrieve the upgraded
// connection.
func (s *SpdyRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	const (
		maxRedirects = 10
	)

	var (
		location         = req.URL
		method           = req.Method
		intermediateConn net.Conn
		resp             *http.Response
	)

	defer func() {
		if intermediateConn != nil {
			intermediateConn.Close()
		}
	}()

	r := new(http.Request)
	// shallow clone
	*r = *req
	// deep copy headers
	header := make(http.Header, len(req.Header))
	for key, values := range req.Header {
		newValues := make([]string, len(values))
		copy(newValues, values)
		header[key] = newValues
	}

	header.Add(httpstream.HeaderConnection, httpstream.HeaderUpgrade)
	header.Add(httpstream.HeaderUpgrade, HeaderSpdy31)

redirectLoop:
	for redirects := 0; ; redirects++ {
		if redirects > maxRedirects {
			return nil, fmt.Errorf("too many redirects (%d)", redirects)
		}
		if redirects != 0 {
			// Redirected requests switch to "GET" according to the HTTP spec:
			// https://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html#sec10.3
			method = "GET"
		}

		r, err := http.NewRequest(method, location.String(), req.Body)
		if err != nil {
			return nil, err
		}
		r.Header = header

		intermediateConn, err = s.dial(r)
		if err != nil {
			return nil, err
		}

		if err := r.Write(intermediateConn); err != nil {
			return nil, err
		}

		resp, err = http.ReadResponse(bufio.NewReader(intermediateConn), r)
		if err != nil {
			// Unable to read the backend response; let the client handle it.
			break redirectLoop
		}

		switch resp.StatusCode {
		case http.StatusFound:
			// Redirect, continue.
		default:
			// Don't redirect.
			break redirectLoop
		}

		resp.Body.Close() // not used

		// Reset the connection.
		intermediateConn.Close()
		intermediateConn = nil

		// Prepare to follow the redirect.
		redirectStr := resp.Header.Get("Location")
		if redirectStr == "" {
			return nil, fmt.Errorf("%d response missing Location header", resp.StatusCode)
		}
		location, err = r.URL.Parse(redirectStr)
		if err != nil {
			return nil, fmt.Errorf("malformed Location header: %v", err)
		}
	}

	s.conn = intermediateConn
	intermediateConn = nil // Don't close the connection when we return it.
	return resp, nil
}

// NewConnection validates the upgrade response, creating and returning a new
// httpstream.Connection if there were no errors.
func (s *SpdyRoundTripper) NewConnection(resp *http.Response) (httpstream.Connection, error) {
	connectionHeader := strings.ToLower(resp.Header.Get(httpstream.HeaderConnection))
	upgradeHeader := strings.ToLower(resp.Header.Get(httpstream.HeaderUpgrade))
	if (resp.StatusCode != http.StatusSwitchingProtocols) || !strings.Contains(connectionHeader, strings.ToLower(httpstream.HeaderUpgrade)) || !strings.Contains(upgradeHeader, strings.ToLower(HeaderSpdy31)) {
		defer resp.Body.Close()
		responseError := ""
		responseErrorBytes, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			responseError = "unable to read error from server response"
		} else {
			// TODO: I don't belong here, I should be abstracted from this class
			if obj, _, err := statusCodecs.UniversalDecoder().Decode(responseErrorBytes, nil, &metav1.Status{}); err == nil {
				if status, ok := obj.(*metav1.Status); ok {
					return nil, &apierrors.StatusError{ErrStatus: *status}
				}
			}
			responseError = string(responseErrorBytes)
			responseError = strings.TrimSpace(responseError)
		}

		return nil, fmt.Errorf("unable to upgrade connection: %s", responseError)
	}

	return NewClientConnection(s.conn)
}

// statusScheme is private scheme for the decoding here until someone fixes the TODO in NewConnection
var statusScheme = runtime.NewScheme()

// ParameterCodec knows about query parameters used with the meta v1 API spec.
var statusCodecs = serializer.NewCodecFactory(statusScheme)

func init() {
	statusScheme.AddUnversionedTypes(metav1.SchemeGroupVersion,
		&metav1.Status{},
	)
}
