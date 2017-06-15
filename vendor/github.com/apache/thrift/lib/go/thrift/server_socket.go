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
	"net"
	"sync"
	"time"
)

type TServerSocket struct {
	listener      net.Listener
	addr          net.Addr
	clientTimeout time.Duration

	// Protects the interrupted value to make it thread safe.
	mu          sync.RWMutex
	interrupted bool
}

func NewTServerSocket(listenAddr string) (*TServerSocket, error) {
	return NewTServerSocketTimeout(listenAddr, 0)
}

func NewTServerSocketTimeout(listenAddr string, clientTimeout time.Duration) (*TServerSocket, error) {
	addr, err := net.ResolveTCPAddr("tcp", listenAddr)
	if err != nil {
		return nil, err
	}
	return &TServerSocket{addr: addr, clientTimeout: clientTimeout}, nil
}

func (p *TServerSocket) Listen() error {
	if p.IsListening() {
		return nil
	}
	l, err := net.Listen(p.addr.Network(), p.addr.String())
	if err != nil {
		return err
	}
	p.listener = l
	return nil
}

func (p *TServerSocket) Accept() (TTransport, error) {
	p.mu.RLock()
	interrupted := p.interrupted
	p.mu.RUnlock()

	if interrupted {
		return nil, errTransportInterrupted
	}
	if p.listener == nil {
		return nil, NewTTransportException(NOT_OPEN, "No underlying server socket")
	}
	conn, err := p.listener.Accept()
	if err != nil {
		return nil, NewTTransportExceptionFromError(err)
	}
	return NewTSocketFromConnTimeout(conn, p.clientTimeout), nil
}

// Checks whether the socket is listening.
func (p *TServerSocket) IsListening() bool {
	return p.listener != nil
}

// Connects the socket, creating a new socket object if necessary.
func (p *TServerSocket) Open() error {
	if p.IsListening() {
		return NewTTransportException(ALREADY_OPEN, "Server socket already open")
	}
	if l, err := net.Listen(p.addr.Network(), p.addr.String()); err != nil {
		return err
	} else {
		p.listener = l
	}
	return nil
}

func (p *TServerSocket) Addr() net.Addr {
	if p.listener != nil {
		return p.listener.Addr()
	}
	return p.addr
}

func (p *TServerSocket) Close() error {
	defer func() {
		p.listener = nil
	}()
	if p.IsListening() {
		return p.listener.Close()
	}
	return nil
}

func (p *TServerSocket) Interrupt() error {
	p.mu.Lock()
	p.interrupted = true
	p.Close()
	p.mu.Unlock()

	return nil
}
