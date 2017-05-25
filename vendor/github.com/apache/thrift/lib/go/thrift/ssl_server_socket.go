/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package thrift

import (
	"crypto/tls"
	"net"
	"time"
)

type TSSLServerSocket struct {
	listener      net.Listener
	addr          net.Addr
	clientTimeout time.Duration
	interrupted   bool
	cfg           *tls.Config
}

func NewTSSLServerSocket(listenAddr string, cfg *tls.Config) (*TSSLServerSocket, error) {
	return NewTSSLServerSocketTimeout(listenAddr, cfg, 0)
}

func NewTSSLServerSocketTimeout(listenAddr string, cfg *tls.Config, clientTimeout time.Duration) (*TSSLServerSocket, error) {
	if cfg.MinVersion == 0 {
		cfg.MinVersion = tls.VersionTLS10
	}
	addr, err := net.ResolveTCPAddr("tcp", listenAddr)
	if err != nil {
		return nil, err
	}
	return &TSSLServerSocket{addr: addr, clientTimeout: clientTimeout, cfg: cfg}, nil
}

func (p *TSSLServerSocket) Listen() error {
	if p.IsListening() {
		return nil
	}
	l, err := tls.Listen(p.addr.Network(), p.addr.String(), p.cfg)
	if err != nil {
		return err
	}
	p.listener = l
	return nil
}

func (p *TSSLServerSocket) Accept() (TTransport, error) {
	if p.interrupted {
		return nil, errTransportInterrupted
	}
	if p.listener == nil {
		return nil, NewTTransportException(NOT_OPEN, "No underlying server socket")
	}
	conn, err := p.listener.Accept()
	if err != nil {
		return nil, NewTTransportExceptionFromError(err)
	}
	return NewTSSLSocketFromConnTimeout(conn, p.cfg, p.clientTimeout), nil
}

// Checks whether the socket is listening.
func (p *TSSLServerSocket) IsListening() bool {
	return p.listener != nil
}

// Connects the socket, creating a new socket object if necessary.
func (p *TSSLServerSocket) Open() error {
	if p.IsListening() {
		return NewTTransportException(ALREADY_OPEN, "Server socket already open")
	}
	if l, err := tls.Listen(p.addr.Network(), p.addr.String(), p.cfg); err != nil {
		return err
	} else {
		p.listener = l
	}
	return nil
}

func (p *TSSLServerSocket) Addr() net.Addr {
	return p.addr
}

func (p *TSSLServerSocket) Close() error {
	defer func() {
		p.listener = nil
	}()
	if p.IsListening() {
		return p.listener.Close()
	}
	return nil
}

func (p *TSSLServerSocket) Interrupt() error {
	p.interrupted = true
	return nil
}
