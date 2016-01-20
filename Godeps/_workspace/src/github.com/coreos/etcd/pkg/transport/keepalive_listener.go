// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package transport

import (
	"crypto/tls"
	"fmt"
	"net"
	"time"
)

// NewKeepAliveListener returns a listener that listens on the given address.
// http://tldp.org/HOWTO/TCP-Keepalive-HOWTO/overview.html
func NewKeepAliveListener(addr string, scheme string, info TLSInfo) (net.Listener, error) {
	l, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}

	if scheme == "https" {
		if info.Empty() {
			return nil, fmt.Errorf("cannot listen on TLS for %s: KeyFile and CertFile are not presented", scheme+"://"+addr)
		}
		cfg, err := info.ServerConfig()
		if err != nil {
			return nil, err
		}

		return newTLSKeepaliveListener(l, cfg), nil
	}

	return &keepaliveListener{
		Listener: l,
	}, nil
}

type keepaliveListener struct{ net.Listener }

func (kln *keepaliveListener) Accept() (net.Conn, error) {
	c, err := kln.Listener.Accept()
	if err != nil {
		return nil, err
	}
	tcpc := c.(*net.TCPConn)
	// detection time: tcp_keepalive_time + tcp_keepalive_probes + tcp_keepalive_intvl
	// default on linux:  30 + 8 * 30
	// default on osx:    30 + 8 * 75
	tcpc.SetKeepAlive(true)
	tcpc.SetKeepAlivePeriod(30 * time.Second)
	return tcpc, nil
}

// A tlsKeepaliveListener implements a network listener (net.Listener) for TLS connections.
type tlsKeepaliveListener struct {
	net.Listener
	config *tls.Config
}

// Accept waits for and returns the next incoming TLS connection.
// The returned connection c is a *tls.Conn.
func (l *tlsKeepaliveListener) Accept() (c net.Conn, err error) {
	c, err = l.Listener.Accept()
	if err != nil {
		return
	}
	tcpc := c.(*net.TCPConn)
	// detection time: tcp_keepalive_time + tcp_keepalive_probes + tcp_keepalive_intvl
	// default on linux:  30 + 8 * 30
	// default on osx:    30 + 8 * 75
	tcpc.SetKeepAlive(true)
	tcpc.SetKeepAlivePeriod(30 * time.Second)
	c = tls.Server(c, l.config)
	return
}

// NewListener creates a Listener which accepts connections from an inner
// Listener and wraps each connection with Server.
// The configuration config must be non-nil and must have
// at least one certificate.
func newTLSKeepaliveListener(inner net.Listener, config *tls.Config) net.Listener {
	l := &tlsKeepaliveListener{}
	l.Listener = inner
	l.config = config
	return l
}
