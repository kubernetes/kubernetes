// Copyright 2017 The Gorilla WebSocket Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"net"
	"net/http"
	"net/url"
	"strings"

	"golang.org/x/net/proxy"
)

type netDialerFunc func(ctx context.Context, network, addr string) (net.Conn, error)

func (fn netDialerFunc) Dial(network, addr string) (net.Conn, error) {
	return fn(context.Background(), network, addr)
}

func (fn netDialerFunc) DialContext(ctx context.Context, network, addr string) (net.Conn, error) {
	return fn(ctx, network, addr)
}

func proxyFromURL(proxyURL *url.URL, forwardDial netDialerFunc) (netDialerFunc, error) {
	if proxyURL.Scheme == "http" || proxyURL.Scheme == "https" {
		return (&httpProxyDialer{proxyURL: proxyURL, forwardDial: forwardDial}).DialContext, nil
	}
	dialer, err := proxy.FromURL(proxyURL, forwardDial)
	if err != nil {
		return nil, err
	}
	if d, ok := dialer.(proxy.ContextDialer); ok {
		return d.DialContext, nil
	}
	return func(ctx context.Context, net, addr string) (net.Conn, error) {
		return dialer.Dial(net, addr)
	}, nil
}

type httpProxyDialer struct {
	proxyURL    *url.URL
	forwardDial netDialerFunc
}

func (hpd *httpProxyDialer) DialContext(ctx context.Context, network string, addr string) (net.Conn, error) {
	hostPort, _ := hostPortNoPort(hpd.proxyURL)
	conn, err := hpd.forwardDial(ctx, network, hostPort)
	if err != nil {
		return nil, err
	}

	connectHeader := make(http.Header)
	if user := hpd.proxyURL.User; user != nil {
		proxyUser := user.Username()
		if proxyPassword, passwordSet := user.Password(); passwordSet {
			credential := base64.StdEncoding.EncodeToString([]byte(proxyUser + ":" + proxyPassword))
			connectHeader.Set("Proxy-Authorization", "Basic "+credential)
		}
	}
	connectReq := &http.Request{
		Method: http.MethodConnect,
		URL:    &url.URL{Opaque: addr},
		Host:   addr,
		Header: connectHeader,
	}

	if err := connectReq.Write(conn); err != nil {
		conn.Close()
		return nil, err
	}

	// Read response. It's OK to use and discard buffered reader here because
	// the remote server does not speak until spoken to.
	br := bufio.NewReader(conn)
	resp, err := http.ReadResponse(br, connectReq)
	if err != nil {
		conn.Close()
		return nil, err
	}

	// Close the response body to silence false positives from linters. Reset
	// the buffered reader first to ensure that Close() does not read from
	// conn.
	// Note: Applications must call resp.Body.Close() on a response returned
	// http.ReadResponse to inspect trailers or read another response from the
	// buffered reader. The call to resp.Body.Close() does not release
	// resources.
	br.Reset(bytes.NewReader(nil))
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		_ = conn.Close()
		f := strings.SplitN(resp.Status, " ", 2)
		return nil, errors.New(f[1])
	}
	return conn, nil
}
