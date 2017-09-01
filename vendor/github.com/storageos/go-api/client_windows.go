// +build windows
// Copyright 2016 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package storageos

import (
	"net"
	"net/http"
	"time"

	"github.com/Microsoft/go-winio"
)

const namedPipeConnectTimeout = 2 * time.Second

type pipeDialer struct {
	dialFunc func(network, addr string) (net.Conn, error)
}

func (p pipeDialer) Dial(network, address string) (net.Conn, error) {
	return p.dialFunc(network, address)
}

// initializeNativeClient initializes the native Named Pipe client for Windows
func (c *Client) initializeNativeClient() {
	if c.endpointURL.Scheme != namedPipeProtocol {
		return
	}
	namedPipePath := c.endpointURL.Path
	dialFunc := func(network, addr string) (net.Conn, error) {
		timeout := namedPipeConnectTimeout
		return winio.DialPipe(namedPipePath, &timeout)
	}
	tr := defaultTransport()
	tr.Dial = dialFunc
	c.Dialer = &pipeDialer{dialFunc}
	c.nativeHTTPClient = &http.Client{Transport: tr}
}
