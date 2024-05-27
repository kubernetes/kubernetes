// Copyright 2017 The Gorilla WebSocket Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bufio"
	"encoding/base64"
	"errors"
	"log"
	"net"
	"net/http"
	"net/url"
	"strings"

	"golang.org/x/net/proxy"
)

type netDialerFunc func(network, addr string) (net.Conn, error)

func (fn netDialerFunc) Dial(network, addr string) (net.Conn, error) {
	return fn(network, addr)
}

func init() {
	proxy.RegisterDialerType("http", func(proxyURL *url.URL, forwardDialer proxy.Dialer) (proxy.Dialer, error) {
		return &httpProxyDialer{proxyURL: proxyURL, forwardDial: forwardDialer.Dial}, nil
	})
}

type httpProxyDialer struct {
	proxyURL    *url.URL
	forwardDial func(network, addr string) (net.Conn, error)
}

func (hpd *httpProxyDialer) Dial(network string, addr string) (net.Conn, error) {
	hostPort, _ := hostPortNoPort(hpd.proxyURL)
	conn, err := hpd.forwardDial(network, hostPort)
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
		if err := conn.Close(); err != nil {
			log.Printf("httpProxyDialer: failed to close connection: %v", err)
		}
		return nil, err
	}

	// Read response. It's OK to use and discard buffered reader here becaue
	// the remote server does not speak until spoken to.
	br := bufio.NewReader(conn)
	resp, err := http.ReadResponse(br, connectReq)
	if err != nil {
		if err := conn.Close(); err != nil {
			log.Printf("httpProxyDialer: failed to close connection: %v", err)
		}
		return nil, err
	}

	if resp.StatusCode != 200 {
		if err := conn.Close(); err != nil {
			log.Printf("httpProxyDialer: failed to close connection: %v", err)
		}
		f := strings.SplitN(resp.Status, " ", 2)
		return nil, errors.New(f[1])
	}
	return conn, nil
}
