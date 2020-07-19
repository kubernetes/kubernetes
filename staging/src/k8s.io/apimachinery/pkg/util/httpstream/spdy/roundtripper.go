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
	"bytes"
	"context"
	"crypto/tls"
	"encoding/base64"
	"fmt"
	"io"
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
	utilnet "k8s.io/apimachinery/pkg/util/net"
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

	// followRedirects indicates if the round tripper should examine responses for redirects and
	// follow them.
	followRedirects bool
	// requireSameHostRedirects restricts redirect following to only follow redirects to the same host
	// as the original request.
	requireSameHostRedirects bool
}

var _ utilnet.TLSClientConfigHolder = &SpdyRoundTripper{}
var _ httpstream.UpgradeRoundTripper = &SpdyRoundTripper{}
var _ utilnet.Dialer = &SpdyRoundTripper{}

// NewRoundTripper creates a new SpdyRoundTripper that will use the specified
// tlsConfig.
func NewRoundTripper(tlsConfig *tls.Config, followRedirects, requireSameHostRedirects bool) *SpdyRoundTripper {
	return NewRoundTripperWithProxy(tlsConfig, followRedirects, requireSameHostRedirects, utilnet.NewProxierWithNoProxyCIDR(http.ProxyFromEnvironment))
}

// NewRoundTripperWithProxy creates a new SpdyRoundTripper that will use the
// specified tlsConfig and proxy func.
func NewRoundTripperWithProxy(tlsConfig *tls.Config, followRedirects, requireSameHostRedirects bool, proxier func(*http.Request) (*url.URL, error)) *SpdyRoundTripper {
	return &SpdyRoundTripper{
		tlsConfig:                tlsConfig,
		followRedirects:          followRedirects,
		requireSameHostRedirects: requireSameHostRedirects,
		proxier:                  proxier,
	}
}

// TLSClientConfig implements pkg/util/net.TLSClientConfigHolder for proper TLS checking during
// proxying with a spdy roundtripper.
func (s *SpdyRoundTripper) TLSClientConfig() *tls.Config {
	return s.tlsConfig
}

// Dial implements k8s.io/apimachinery/pkg/util/net.Dialer.
func (s *SpdyRoundTripper) Dial(req *http.Request) (net.Conn, error) {
	conn, err := s.dial(req)
	if err != nil {
		return nil, err
	}

	if err := req.Write(conn); err != nil {
		conn.Close()
		return nil, err
	}

	return conn, nil
}

// dial dials the host specified by req, using TLS if appropriate, optionally
// using a proxy server if one is configured via environment variables.
func (s *SpdyRoundTripper) dial(req *http.Request) (net.Conn, error) {
	proxyURL, err := s.proxier(req)
	if err != nil {
		return nil, err
	}

	if proxyURL == nil {
		return s.dialWithoutProxy(req.Context(), req.URL)
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

	proxyDialConn, err := s.dialWithoutProxy(req.Context(), proxyURL)
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

	tlsConfig := s.tlsConfig
	switch {
	case tlsConfig == nil:
		tlsConfig = &tls.Config{ServerName: host}
	case len(tlsConfig.ServerName) == 0:
		tlsConfig = tlsConfig.Clone()
		tlsConfig.ServerName = host
	}

	tlsConn := tls.Client(rwc, tlsConfig)

	// need to manually call Handshake() so we can call VerifyHostname() below
	if err := tlsConn.Handshake(); err != nil {
		return nil, err
	}

	// Return if we were configured to skip validation
	if tlsConfig.InsecureSkipVerify {
		return tlsConn, nil
	}

	if err := tlsConn.VerifyHostname(tlsConfig.ServerName); err != nil {
		return nil, err
	}

	return tlsConn, nil
}

// dialWithoutProxy dials the host specified by url, using TLS if appropriate.
func (s *SpdyRoundTripper) dialWithoutProxy(ctx context.Context, url *url.URL) (net.Conn, error) {
	dialAddr := netutil.CanonicalAddr(url)

	if url.Scheme == "http" {
		if s.Dialer == nil {
			var d net.Dialer
			return d.DialContext(ctx, "tcp", dialAddr)
		} else {
			return s.Dialer.DialContext(ctx, "tcp", dialAddr)
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
	if s.tlsConfig != nil && len(s.tlsConfig.ServerName) > 0 {
		host = s.tlsConfig.ServerName
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
	header := utilnet.CloneHeader(req.Header)
	header.Add(httpstream.HeaderConnection, httpstream.HeaderUpgrade)
	header.Add(httpstream.HeaderUpgrade, HeaderSpdy31)

	var (
		conn        net.Conn
		rawResponse []byte
		err         error
	)

	if s.followRedirects {
		conn, rawResponse, err = utilnet.ConnectWithRedirects(req.Method, req.URL, header, req.Body, s, s.requireSameHostRedirects)
	} else {
		clone := utilnet.CloneRequest(req)
		clone.Header = header
		conn, err = s.Dial(clone)
	}
	if err != nil {
		return nil, err
	}

	responseReader := bufio.NewReader(
		io.MultiReader(
			bytes.NewBuffer(rawResponse),
			conn,
		),
	)

	resp, err := http.ReadResponse(responseReader, nil)
	if err != nil {
		if conn != nil {
			conn.Close()
		}
		return nil, err
	}

	s.conn = conn

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
