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
	"context"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"

	"golang.org/x/net/proxy"
	"k8s.io/streaming/pkg/httpstream"
	utilnet "k8s.io/utils/net"
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

	// proxier knows which proxy to use given a request, defaults to a proxier that
	// preserves NO_PROXY CIDR behavior while delegating to http.ProxyFromEnvironment.
	// Used primarily for mocking the proxy discovery in tests.
	proxier func(req *http.Request) (*url.URL, error)

	// pingPeriod is a period for sending Ping frames over established
	// connections.
	pingPeriod time.Duration

	// upgradeTransport is an optional substitute for dialing if present. This field is
	// mutually exclusive with the "tlsConfig", "Dialer", and "proxier".
	upgradeTransport http.RoundTripper
}

type tlsClientConfigHolder interface {
	TLSClientConfig() *tls.Config
}

type roundTripperWrapper interface {
	http.RoundTripper
	WrappedRoundTripper() http.RoundTripper
}

type dialFunc func(ctx context.Context, network, addr string) (net.Conn, error)

var _ tlsClientConfigHolder = &SpdyRoundTripper{}
var _ httpstream.UpgradeRoundTripper = &SpdyRoundTripper{}

// NewRoundTripper creates a new SpdyRoundTripper that will use the specified
// tlsConfig.
func NewRoundTripper(tlsConfig *tls.Config) (*SpdyRoundTripper, error) {
	return NewRoundTripperWithConfig(RoundTripperConfig{
		TLS:              tlsConfig,
		UpgradeTransport: nil,
	})
}

// NewRoundTripperWithProxy creates a new SpdyRoundTripper that will use the
// specified tlsConfig and proxy func.
func NewRoundTripperWithProxy(tlsConfig *tls.Config, proxier func(*http.Request) (*url.URL, error)) (*SpdyRoundTripper, error) {
	return NewRoundTripperWithConfig(RoundTripperConfig{
		TLS:              tlsConfig,
		Proxier:          proxier,
		UpgradeTransport: nil,
	})
}

// NewRoundTripperWithConfig creates a new SpdyRoundTripper with the specified
// configuration. Returns an error if the SpdyRoundTripper is misconfigured.
func NewRoundTripperWithConfig(cfg RoundTripperConfig) (*SpdyRoundTripper, error) {
	// Process UpgradeTransport, which is mutually exclusive to TLSConfig and Proxier.
	if cfg.UpgradeTransport != nil {
		if cfg.TLS != nil || cfg.Proxier != nil {
			return nil, fmt.Errorf("SpdyRoundTripper: UpgradeTransport is mutually exclusive to TLSConfig or Proxier")
		}
		tlsConfig, err := tlsConfigForTransport(cfg.UpgradeTransport)
		if err != nil {
			return nil, fmt.Errorf("SpdyRoundTripper: unable to retrieve TLS config from UpgradeTransport: %w", err)
		}
		cfg.TLS = tlsConfig
	}
	if cfg.Proxier == nil {
		cfg.Proxier = newProxierWithNoProxyCIDR(http.ProxyFromEnvironment)
	}
	return &SpdyRoundTripper{
		tlsConfig:        cfg.TLS,
		proxier:          cfg.Proxier,
		pingPeriod:       cfg.PingPeriod,
		upgradeTransport: cfg.UpgradeTransport,
	}, nil
}

// newProxierWithNoProxyCIDR preserves CIDR matching in NO_PROXY/no_proxy while
// delegating all other behavior to the supplied proxy function.
func newProxierWithNoProxyCIDR(delegate func(req *http.Request) (*url.URL, error)) func(req *http.Request) (*url.URL, error) {
	noProxyEnv := os.Getenv("NO_PROXY")
	if noProxyEnv == "" {
		noProxyEnv = os.Getenv("no_proxy")
	}
	noProxyRules := strings.Split(noProxyEnv, ",")

	cidrs := make([]*net.IPNet, 0, len(noProxyRules))
	for _, noProxyRule := range noProxyRules {
		noProxyRule = strings.TrimSpace(noProxyRule)
		if noProxyRule == "" {
			continue
		}
		_, cidr, err := utilnet.ParseCIDRSloppy(noProxyRule)
		if err == nil {
			cidrs = append(cidrs, cidr)
		}
	}

	if len(cidrs) == 0 {
		return delegate
	}

	return func(req *http.Request) (*url.URL, error) {
		ip := utilnet.ParseIPSloppy(req.URL.Hostname())
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

// RoundTripperConfig is a set of options for an SpdyRoundTripper.
type RoundTripperConfig struct {
	// TLS configuration used by the round tripper if UpgradeTransport not present.
	TLS *tls.Config
	// Proxier is a proxy function invoked on each request. Optional.
	Proxier func(*http.Request) (*url.URL, error)
	// PingPeriod is a period for sending SPDY Pings on the connection.
	// Optional.
	PingPeriod time.Duration
	// UpgradeTransport is a subtitute transport used for dialing. If set,
	// this field will be used instead of "TLS" and "Proxier" for connection creation.
	// Optional.
	UpgradeTransport http.RoundTripper
}

// TLSClientConfig implements pkg/util/net.TLSClientConfigHolder for proper TLS checking during
// proxying with a spdy roundtripper.
func (s *SpdyRoundTripper) TLSClientConfig() *tls.Config {
	return s.tlsConfig
}

// Dial opens a network connection for an upgrade request.
func (s *SpdyRoundTripper) Dial(req *http.Request) (net.Conn, error) {
	var conn net.Conn
	var err error
	if s.upgradeTransport != nil {
		conn, err = dialURLWithTransport(req.Context(), req.URL, s.upgradeTransport)
	} else {
		conn, err = s.dial(req)
	}
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

	switch proxyURL.Scheme {
	case "socks5":
		return s.dialWithSocks5Proxy(req, proxyURL)
	case "https", "http", "":
		return s.dialWithHttpProxy(req, proxyURL)
	}

	return nil, fmt.Errorf("proxy URL scheme not supported: %s", proxyURL.Scheme)
}

// dialWithHttpProxy dials the host specified by url through an http or an https proxy.
func (s *SpdyRoundTripper) dialWithHttpProxy(req *http.Request, proxyURL *url.URL) (net.Conn, error) {
	// ensure we use a canonical host with proxyReq
	targetHost := canonicalAddr(req.URL)

	// proxying logic adapted from http://blog.h6t.eu/post/74098062923/golang-websocket-with-http-proxy-support
	proxyReq := http.Request{
		Method: http.MethodConnect,
		URL:    &url.URL{},
		Host:   targetHost,
	}

	proxyReq = *proxyReq.WithContext(req.Context())

	if pa := s.proxyAuth(proxyURL); pa != "" {
		proxyReq.Header = http.Header{}
		proxyReq.Header.Set("Proxy-Authorization", pa)
	}

	proxyDialConn, err := s.dialWithoutProxy(proxyReq.Context(), proxyURL)
	if err != nil {
		return nil, err
	}

	//nolint:staticcheck // SA1019 ignore deprecated httputil.NewProxyClientConn
	proxyClientConn := httputil.NewProxyClientConn(proxyDialConn, nil)
	response, err := proxyClientConn.Do(&proxyReq)
	//nolint:staticcheck // SA1019 ignore deprecated httputil.ErrPersistEOF: it might be
	// returned from the invocation of proxyClientConn.Do
	if err != nil && err != httputil.ErrPersistEOF {
		return nil, err
	}
	if response != nil && response.StatusCode >= 300 || response.StatusCode < 200 {
		return nil, fmt.Errorf("CONNECT request to %s returned response: %s", proxyURL.Redacted(), response.Status)
	}

	rwc, _ := proxyClientConn.Hijack()

	if req.URL.Scheme == "https" {
		return s.tlsConn(proxyReq.Context(), rwc, targetHost)
	}
	return rwc, nil
}

// dialWithSocks5Proxy dials the host specified by url through a socks5 proxy.
func (s *SpdyRoundTripper) dialWithSocks5Proxy(req *http.Request, proxyURL *url.URL) (net.Conn, error) {
	// ensure we use a canonical host with proxyReq
	targetHost := canonicalAddr(req.URL)
	proxyDialAddr := canonicalAddr(proxyURL)

	var auth *proxy.Auth
	if proxyURL.User != nil {
		pass, _ := proxyURL.User.Password()
		auth = &proxy.Auth{
			User:     proxyURL.User.Username(),
			Password: pass,
		}
	}

	dialer := s.Dialer
	if dialer == nil {
		dialer = &net.Dialer{
			Timeout: 30 * time.Second,
		}
	}

	proxyDialer, err := proxy.SOCKS5("tcp", proxyDialAddr, auth, dialer)
	if err != nil {
		return nil, err
	}

	// According to the implementation of proxy.SOCKS5, the type assertion will always succeed
	contextDialer, ok := proxyDialer.(proxy.ContextDialer)
	if !ok {
		return nil, errors.New("SOCKS5 Dialer must implement ContextDialer")
	}

	proxyDialConn, err := contextDialer.DialContext(req.Context(), "tcp", targetHost)
	if err != nil {
		return nil, err
	}

	if req.URL.Scheme == "https" {
		return s.tlsConn(req.Context(), proxyDialConn, targetHost)
	}
	return proxyDialConn, nil
}

// tlsConn returns a TLS client side connection using rwc as the underlying transport.
func (s *SpdyRoundTripper) tlsConn(ctx context.Context, rwc net.Conn, targetHost string) (net.Conn, error) {

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

	if err := tlsConn.HandshakeContext(ctx); err != nil {
		tlsConn.Close()
		return nil, err
	}

	return tlsConn, nil
}

// dialWithoutProxy dials the host specified by url, using TLS if appropriate.
func (s *SpdyRoundTripper) dialWithoutProxy(ctx context.Context, url *url.URL) (net.Conn, error) {
	dialAddr := canonicalAddr(url)
	dialer := s.Dialer
	if dialer == nil {
		dialer = &net.Dialer{}
	}

	if url.Scheme == "http" {
		return dialer.DialContext(ctx, "tcp", dialAddr)
	}

	tlsDialer := tls.Dialer{
		NetDialer: dialer,
		Config:    s.tlsConfig,
	}
	return tlsDialer.DialContext(ctx, "tcp", dialAddr)
}

// proxyAuth returns, for a given proxy URL, the value to be used for the Proxy-Authorization header
func (s *SpdyRoundTripper) proxyAuth(proxyURL *url.URL) string {
	if proxyURL == nil || proxyURL.User == nil {
		return ""
	}
	username := proxyURL.User.Username()
	password, _ := proxyURL.User.Password()
	auth := username + ":" + password
	return "Basic " + base64.StdEncoding.EncodeToString([]byte(auth))
}

// RoundTrip executes the Request and upgrades it. After a successful upgrade,
// clients may call SpdyRoundTripper.Connection() to retrieve the upgraded
// connection.
func (s *SpdyRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.Header = req.Header.Clone()
	req.Header.Add(httpstream.HeaderConnection, httpstream.HeaderUpgrade)
	req.Header.Add(httpstream.HeaderUpgrade, HeaderSpdy31)

	conn, err := s.Dial(req)
	if err != nil {
		return nil, err
	}

	responseReader := bufio.NewReader(conn)

	resp, err := http.ReadResponse(responseReader, nil)
	if err != nil {
		conn.Close()
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
		responseErrorBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("unable to upgrade connection: unable to read error from server response")
		}
		return nil, fmt.Errorf("unable to upgrade connection: %s", upgradeErrorMessage(responseErrorBytes))
	}

	return NewClientConnectionWithPings(s.conn, s.pingPeriod)
}

func tlsConfigForTransport(transport http.RoundTripper) (*tls.Config, error) {
	if transport == nil {
		return nil, nil
	}
	switch transport := transport.(type) {
	case *http.Transport:
		return transport.TLSClientConfig, nil
	case tlsClientConfigHolder:
		return transport.TLSClientConfig(), nil
	case roundTripperWrapper:
		return tlsConfigForTransport(transport.WrappedRoundTripper())
	default:
		return nil, fmt.Errorf("transport %T does not expose TLS client config", transport)
	}
}

func canonicalAddr(url *url.URL) string {
	host := url.Hostname()
	port := url.Port()
	if len(port) == 0 {
		switch strings.ToLower(url.Scheme) {
		case "http", "ws":
			port = "80"
		case "https", "wss":
			port = "443"
		}
	}
	return net.JoinHostPort(host, port)
}

func upgradeErrorMessage(responseErrorBytes []byte) string {
	type statusLike struct {
		Message string `json:"message"`
		Error   string `json:"error"`
	}

	var status statusLike
	if err := json.Unmarshal(responseErrorBytes, &status); err == nil {
		if msg := strings.TrimSpace(status.Message); msg != "" {
			return msg
		}
		if msg := strings.TrimSpace(status.Error); msg != "" {
			return msg
		}
	}

	msg := strings.TrimSpace(string(responseErrorBytes))
	if msg == "" {
		return "empty server response"
	}
	return msg
}

func dialURLWithTransport(ctx context.Context, url *url.URL, transport http.RoundTripper) (net.Conn, error) {
	dialAddr := canonicalAddr(url)

	dialer, err := dialerFor(transport)
	if err != nil {
		dialer = nil
	}

	switch url.Scheme {
	case "http":
		if dialer != nil {
			return dialer(ctx, "tcp", dialAddr)
		}
		var d net.Dialer
		return d.DialContext(ctx, "tcp", dialAddr)
	case "https":
		tlsConfig, err := tlsConfigForTransport(transport)
		if err != nil {
			tlsConfig = nil
		}

		if dialer != nil {
			netConn, err := dialer(ctx, "tcp", dialAddr)
			if err != nil {
				return nil, err
			}

			if tlsConfig == nil {
				tlsConfig = &tls.Config{InsecureSkipVerify: true}
			} else if len(tlsConfig.ServerName) == 0 && !tlsConfig.InsecureSkipVerify {
				inferredHost := dialAddr
				if host, _, err := net.SplitHostPort(dialAddr); err == nil {
					inferredHost = host
				}
				tlsConfigCopy := tlsConfig.Clone()
				tlsConfigCopy.ServerName = inferredHost
				tlsConfig = tlsConfigCopy
			}

			if supportsHTTP11(tlsConfig.NextProtos) {
				tlsConfig = tlsConfig.Clone()
				tlsConfig.NextProtos = []string{"http/1.1"}
			}

			tlsConn := tls.Client(netConn, tlsConfig)
			if err := tlsConn.HandshakeContext(ctx); err != nil {
				netConn.Close()
				return nil, err
			}
			return tlsConn, nil
		}

		tlsDialer := tls.Dialer{Config: tlsConfig}
		return tlsDialer.DialContext(ctx, "tcp", dialAddr)
	default:
		return nil, fmt.Errorf("unknown scheme: %s", url.Scheme)
	}
}

func dialerFor(transport http.RoundTripper) (dialFunc, error) {
	if transport == nil {
		return nil, nil
	}

	switch transport := transport.(type) {
	case *http.Transport:
		if transport.DialContext != nil {
			return transport.DialContext, nil
		}
		if transport.Dial != nil {
			return func(ctx context.Context, network, addr string) (net.Conn, error) {
				return transport.Dial(network, addr)
			}, nil
		}
		return nil, nil
	case roundTripperWrapper:
		return dialerFor(transport.WrappedRoundTripper())
	default:
		return nil, fmt.Errorf("unknown transport type: %T", transport)
	}
}

func supportsHTTP11(nextProtos []string) bool {
	if len(nextProtos) == 0 {
		return true
	}
	for _, proto := range nextProtos {
		if proto == "http/1.1" {
			return true
		}
	}
	return false
}
