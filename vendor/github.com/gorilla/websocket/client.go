// Copyright 2013 The Gorilla WebSocket Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log"

	"net"
	"net/http"
	"net/http/httptrace"
	"net/url"
	"strings"
	"time"

	"golang.org/x/net/proxy"
)

// ErrBadHandshake is returned when the server response to opening handshake is
// invalid.
var ErrBadHandshake = errors.New("websocket: bad handshake")

var errInvalidCompression = errors.New("websocket: invalid compression negotiation")

// NewClient creates a new client connection using the given net connection.
// The URL u specifies the host and request URI. Use requestHeader to specify
// the origin (Origin), subprotocols (Sec-WebSocket-Protocol) and cookies
// (Cookie). Use the response.Header to get the selected subprotocol
// (Sec-WebSocket-Protocol) and cookies (Set-Cookie).
//
// If the WebSocket handshake fails, ErrBadHandshake is returned along with a
// non-nil *http.Response so that callers can handle redirects, authentication,
// etc.
//
// Deprecated: Use Dialer instead.
func NewClient(netConn net.Conn, u *url.URL, requestHeader http.Header, readBufSize, writeBufSize int) (c *Conn, response *http.Response, err error) {
	d := Dialer{
		ReadBufferSize:  readBufSize,
		WriteBufferSize: writeBufSize,
		NetDial: func(net, addr string) (net.Conn, error) {
			return netConn, nil
		},
	}
	return d.Dial(u.String(), requestHeader)
}

// A Dialer contains options for connecting to WebSocket server.
//
// It is safe to call Dialer's methods concurrently.
type Dialer struct {
	// NetDial specifies the dial function for creating TCP connections. If
	// NetDial is nil, net.Dial is used.
	NetDial func(network, addr string) (net.Conn, error)

	// NetDialContext specifies the dial function for creating TCP connections. If
	// NetDialContext is nil, NetDial is used.
	NetDialContext func(ctx context.Context, network, addr string) (net.Conn, error)

	// NetDialTLSContext specifies the dial function for creating TLS/TCP connections. If
	// NetDialTLSContext is nil, NetDialContext is used.
	// If NetDialTLSContext is set, Dial assumes the TLS handshake is done there and
	// TLSClientConfig is ignored.
	NetDialTLSContext func(ctx context.Context, network, addr string) (net.Conn, error)

	// Proxy specifies a function to return a proxy for a given
	// Request. If the function returns a non-nil error, the
	// request is aborted with the provided error.
	// If Proxy is nil or returns a nil *URL, no proxy is used.
	Proxy func(*http.Request) (*url.URL, error)

	// TLSClientConfig specifies the TLS configuration to use with tls.Client.
	// If nil, the default configuration is used.
	// If either NetDialTLS or NetDialTLSContext are set, Dial assumes the TLS handshake
	// is done there and TLSClientConfig is ignored.
	TLSClientConfig *tls.Config

	// HandshakeTimeout specifies the duration for the handshake to complete.
	HandshakeTimeout time.Duration

	// ReadBufferSize and WriteBufferSize specify I/O buffer sizes in bytes. If a buffer
	// size is zero, then a useful default size is used. The I/O buffer sizes
	// do not limit the size of the messages that can be sent or received.
	ReadBufferSize, WriteBufferSize int

	// WriteBufferPool is a pool of buffers for write operations. If the value
	// is not set, then write buffers are allocated to the connection for the
	// lifetime of the connection.
	//
	// A pool is most useful when the application has a modest volume of writes
	// across a large number of connections.
	//
	// Applications should use a single pool for each unique value of
	// WriteBufferSize.
	WriteBufferPool BufferPool

	// Subprotocols specifies the client's requested subprotocols.
	Subprotocols []string

	// EnableCompression specifies if the client should attempt to negotiate
	// per message compression (RFC 7692). Setting this value to true does not
	// guarantee that compression will be supported. Currently only "no context
	// takeover" modes are supported.
	EnableCompression bool

	// Jar specifies the cookie jar.
	// If Jar is nil, cookies are not sent in requests and ignored
	// in responses.
	Jar http.CookieJar
}

// Dial creates a new client connection by calling DialContext with a background context.
func (d *Dialer) Dial(urlStr string, requestHeader http.Header) (*Conn, *http.Response, error) {
	return d.DialContext(context.Background(), urlStr, requestHeader)
}

var errMalformedURL = errors.New("malformed ws or wss URL")

func hostPortNoPort(u *url.URL) (hostPort, hostNoPort string) {
	hostPort = u.Host
	hostNoPort = u.Host
	if i := strings.LastIndex(u.Host, ":"); i > strings.LastIndex(u.Host, "]") {
		hostNoPort = hostNoPort[:i]
	} else {
		switch u.Scheme {
		case "wss":
			hostPort += ":443"
		case "https":
			hostPort += ":443"
		default:
			hostPort += ":80"
		}
	}
	return hostPort, hostNoPort
}

// DefaultDialer is a dialer with all fields set to the default values.
var DefaultDialer = &Dialer{
	Proxy:            http.ProxyFromEnvironment,
	HandshakeTimeout: 45 * time.Second,
}

// nilDialer is dialer to use when receiver is nil.
var nilDialer = *DefaultDialer

// DialContext creates a new client connection. Use requestHeader to specify the
// origin (Origin), subprotocols (Sec-WebSocket-Protocol) and cookies (Cookie).
// Use the response.Header to get the selected subprotocol
// (Sec-WebSocket-Protocol) and cookies (Set-Cookie).
//
// The context will be used in the request and in the Dialer.
//
// If the WebSocket handshake fails, ErrBadHandshake is returned along with a
// non-nil *http.Response so that callers can handle redirects, authentication,
// etcetera. The response body may not contain the entire response and does not
// need to be closed by the application.
func (d *Dialer) DialContext(ctx context.Context, urlStr string, requestHeader http.Header) (*Conn, *http.Response, error) {
	if d == nil {
		d = &nilDialer
	}

	challengeKey, err := generateChallengeKey()
	if err != nil {
		return nil, nil, err
	}

	u, err := url.Parse(urlStr)
	if err != nil {
		return nil, nil, err
	}

	switch u.Scheme {
	case "ws":
		u.Scheme = "http"
	case "wss":
		u.Scheme = "https"
	default:
		return nil, nil, errMalformedURL
	}

	if u.User != nil {
		// User name and password are not allowed in websocket URIs.
		return nil, nil, errMalformedURL
	}

	req := &http.Request{
		Method:     http.MethodGet,
		URL:        u,
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     make(http.Header),
		Host:       u.Host,
	}
	req = req.WithContext(ctx)

	// Set the cookies present in the cookie jar of the dialer
	if d.Jar != nil {
		for _, cookie := range d.Jar.Cookies(u) {
			req.AddCookie(cookie)
		}
	}

	// Set the request headers using the capitalization for names and values in
	// RFC examples. Although the capitalization shouldn't matter, there are
	// servers that depend on it. The Header.Set method is not used because the
	// method canonicalizes the header names.
	req.Header["Upgrade"] = []string{"websocket"}
	req.Header["Connection"] = []string{"Upgrade"}
	req.Header["Sec-WebSocket-Key"] = []string{challengeKey}
	req.Header["Sec-WebSocket-Version"] = []string{"13"}
	if len(d.Subprotocols) > 0 {
		req.Header["Sec-WebSocket-Protocol"] = []string{strings.Join(d.Subprotocols, ", ")}
	}
	for k, vs := range requestHeader {
		switch {
		case k == "Host":
			if len(vs) > 0 {
				req.Host = vs[0]
			}
		case k == "Upgrade" ||
			k == "Connection" ||
			k == "Sec-Websocket-Key" ||
			k == "Sec-Websocket-Version" ||
			//#nosec G101 (CWE-798): Potential HTTP request smuggling via parameter pollution
			k == "Sec-Websocket-Extensions" ||
			(k == "Sec-Websocket-Protocol" && len(d.Subprotocols) > 0):
			return nil, nil, errors.New("websocket: duplicate header not allowed: " + k)
		case k == "Sec-Websocket-Protocol":
			req.Header["Sec-WebSocket-Protocol"] = vs
		default:
			req.Header[k] = vs
		}
	}

	if d.EnableCompression {
		req.Header["Sec-WebSocket-Extensions"] = []string{"permessage-deflate; server_no_context_takeover; client_no_context_takeover"}
	}

	if d.HandshakeTimeout != 0 {
		var cancel func()
		ctx, cancel = context.WithTimeout(ctx, d.HandshakeTimeout)
		defer cancel()
	}

	// Get network dial function.
	var netDial func(network, add string) (net.Conn, error)

	switch u.Scheme {
	case "http":
		if d.NetDialContext != nil {
			netDial = func(network, addr string) (net.Conn, error) {
				return d.NetDialContext(ctx, network, addr)
			}
		} else if d.NetDial != nil {
			netDial = d.NetDial
		}
	case "https":
		if d.NetDialTLSContext != nil {
			netDial = func(network, addr string) (net.Conn, error) {
				return d.NetDialTLSContext(ctx, network, addr)
			}
		} else if d.NetDialContext != nil {
			netDial = func(network, addr string) (net.Conn, error) {
				return d.NetDialContext(ctx, network, addr)
			}
		} else if d.NetDial != nil {
			netDial = d.NetDial
		}
	default:
		return nil, nil, errMalformedURL
	}

	if netDial == nil {
		netDialer := &net.Dialer{}
		netDial = func(network, addr string) (net.Conn, error) {
			return netDialer.DialContext(ctx, network, addr)
		}
	}

	// If needed, wrap the dial function to set the connection deadline.
	if deadline, ok := ctx.Deadline(); ok {
		forwardDial := netDial
		netDial = func(network, addr string) (net.Conn, error) {
			c, err := forwardDial(network, addr)
			if err != nil {
				return nil, err
			}
			err = c.SetDeadline(deadline)
			if err != nil {
				if err := c.Close(); err != nil {
					log.Printf("websocket: failed to close network connection: %v", err)
				}
				return nil, err
			}
			return c, nil
		}
	}

	// If needed, wrap the dial function to connect through a proxy.
	if d.Proxy != nil {
		proxyURL, err := d.Proxy(req)
		if err != nil {
			return nil, nil, err
		}
		if proxyURL != nil {
			dialer, err := proxy.FromURL(proxyURL, netDialerFunc(netDial))
			if err != nil {
				return nil, nil, err
			}
			netDial = dialer.Dial
		}
	}

	hostPort, hostNoPort := hostPortNoPort(u)
	trace := httptrace.ContextClientTrace(ctx)
	if trace != nil && trace.GetConn != nil {
		trace.GetConn(hostPort)
	}

	netConn, err := netDial("tcp", hostPort)
	if err != nil {
		return nil, nil, err
	}
	if trace != nil && trace.GotConn != nil {
		trace.GotConn(httptrace.GotConnInfo{
			Conn: netConn,
		})
	}

	defer func() {
		if netConn != nil {
			if err := netConn.Close(); err != nil {
				log.Printf("websocket: failed to close network connection: %v", err)
			}
		}
	}()

	if u.Scheme == "https" && d.NetDialTLSContext == nil {
		// If NetDialTLSContext is set, assume that the TLS handshake has already been done

		cfg := cloneTLSConfig(d.TLSClientConfig)
		if cfg.ServerName == "" {
			cfg.ServerName = hostNoPort
		}
		tlsConn := tls.Client(netConn, cfg)
		netConn = tlsConn

		if trace != nil && trace.TLSHandshakeStart != nil {
			trace.TLSHandshakeStart()
		}
		err := doHandshake(ctx, tlsConn, cfg)
		if trace != nil && trace.TLSHandshakeDone != nil {
			trace.TLSHandshakeDone(tlsConn.ConnectionState(), err)
		}

		if err != nil {
			return nil, nil, err
		}
	}

	conn := newConn(netConn, false, d.ReadBufferSize, d.WriteBufferSize, d.WriteBufferPool, nil, nil)

	if err := req.Write(netConn); err != nil {
		return nil, nil, err
	}

	if trace != nil && trace.GotFirstResponseByte != nil {
		if peek, err := conn.br.Peek(1); err == nil && len(peek) == 1 {
			trace.GotFirstResponseByte()
		}
	}

	resp, err := http.ReadResponse(conn.br, req)
	if err != nil {
		if d.TLSClientConfig != nil {
			for _, proto := range d.TLSClientConfig.NextProtos {
				if proto != "http/1.1" {
					return nil, nil, fmt.Errorf(
						"websocket: protocol %q was given but is not supported;"+
							"sharing tls.Config with net/http Transport can cause this error: %w",
						proto, err,
					)
				}
			}
		}
		return nil, nil, err
	}

	if d.Jar != nil {
		if rc := resp.Cookies(); len(rc) > 0 {
			d.Jar.SetCookies(u, rc)
		}
	}

	if resp.StatusCode != 101 ||
		!tokenListContainsValue(resp.Header, "Upgrade", "websocket") ||
		!tokenListContainsValue(resp.Header, "Connection", "upgrade") ||
		resp.Header.Get("Sec-Websocket-Accept") != computeAcceptKey(challengeKey) {
		// Before closing the network connection on return from this
		// function, slurp up some of the response to aid application
		// debugging.
		buf := make([]byte, 1024)
		n, _ := io.ReadFull(resp.Body, buf)
		resp.Body = io.NopCloser(bytes.NewReader(buf[:n]))
		return nil, resp, ErrBadHandshake
	}

	for _, ext := range parseExtensions(resp.Header) {
		if ext[""] != "permessage-deflate" {
			continue
		}
		_, snct := ext["server_no_context_takeover"]
		_, cnct := ext["client_no_context_takeover"]
		if !snct || !cnct {
			return nil, resp, errInvalidCompression
		}
		conn.newCompressionWriter = compressNoContextTakeover
		conn.newDecompressionReader = decompressNoContextTakeover
		break
	}

	resp.Body = io.NopCloser(bytes.NewReader([]byte{}))
	conn.subprotocol = resp.Header.Get("Sec-Websocket-Protocol")

	if err := netConn.SetDeadline(time.Time{}); err != nil {
		return nil, nil, err
	}
	netConn = nil // to avoid close in defer.
	return conn, resp, nil
}

func cloneTLSConfig(cfg *tls.Config) *tls.Config {
	if cfg == nil {
		return &tls.Config{MinVersion: tls.VersionTLS12}
	}
	return cfg.Clone()
}
