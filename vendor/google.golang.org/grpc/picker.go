/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package grpc

import (
	"container/list"
	"fmt"
	"sync"

	"golang.org/x/net/context"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/naming"
	"google.golang.org/grpc/transport"
)

// Picker picks a Conn for RPC requests.
// This is EXPERIMENTAL and please do not implement your own Picker for now.
type Picker interface {
	// Init does initial processing for the Picker, e.g., initiate some connections.
	Init(cc *ClientConn) error
	// Pick blocks until either a transport.ClientTransport is ready for the upcoming RPC
	// or some error happens.
	Pick(ctx context.Context) (transport.ClientTransport, error)
	// PickAddr picks a peer address for connecting. This will be called repeated for
	// connecting/reconnecting.
	PickAddr() (string, error)
	// State returns the connectivity state of the underlying connections.
	State() (ConnectivityState, error)
	// WaitForStateChange blocks until the state changes to something other than
	// the sourceState. It returns the new state or error.
	WaitForStateChange(ctx context.Context, sourceState ConnectivityState) (ConnectivityState, error)
	// Close closes all the Conn's owned by this Picker.
	Close() error
}

// unicastPicker is the default Picker which is used when there is no custom Picker
// specified by users. It always picks the same Conn.
type unicastPicker struct {
	target string
	conn   *Conn
}

func (p *unicastPicker) Init(cc *ClientConn) error {
	c, err := NewConn(cc)
	if err != nil {
		return err
	}
	p.conn = c
	return nil
}

func (p *unicastPicker) Pick(ctx context.Context) (transport.ClientTransport, error) {
	return p.conn.Wait(ctx)
}

func (p *unicastPicker) PickAddr() (string, error) {
	return p.target, nil
}

func (p *unicastPicker) State() (ConnectivityState, error) {
	return p.conn.State(), nil
}

func (p *unicastPicker) WaitForStateChange(ctx context.Context, sourceState ConnectivityState) (ConnectivityState, error) {
	return p.conn.WaitForStateChange(ctx, sourceState)
}

func (p *unicastPicker) Close() error {
	if p.conn != nil {
		return p.conn.Close()
	}
	return nil
}

// unicastNamingPicker picks an address from a name resolver to set up the connection.
type unicastNamingPicker struct {
	cc       *ClientConn
	resolver naming.Resolver
	watcher  naming.Watcher
	mu       sync.Mutex
	// The list of the addresses are obtained from watcher.
	addrs *list.List
	// It tracks the current picked addr by PickAddr(). The next PickAddr may
	// push it forward on addrs.
	pickedAddr *list.Element
	conn       *Conn
}

// NewUnicastNamingPicker creates a Picker to pick addresses from a name resolver
// to connect.
func NewUnicastNamingPicker(r naming.Resolver) Picker {
	return &unicastNamingPicker{
		resolver: r,
		addrs:    list.New(),
	}
}

type addrInfo struct {
	addr string
	// Set to true if this addrInfo needs to be deleted in the next PickAddrr() call.
	deleting bool
}

// processUpdates calls Watcher.Next() once and processes the obtained updates.
func (p *unicastNamingPicker) processUpdates() error {
	updates, err := p.watcher.Next()
	if err != nil {
		return err
	}
	for _, update := range updates {
		switch update.Op {
		case naming.Add:
			p.mu.Lock()
			p.addrs.PushBack(&addrInfo{
				addr: update.Addr,
			})
			p.mu.Unlock()
			// Initial connection setup
			if p.conn == nil {
				conn, err := NewConn(p.cc)
				if err != nil {
					return err
				}
				p.conn = conn
			}
		case naming.Delete:
			p.mu.Lock()
			for e := p.addrs.Front(); e != nil; e = e.Next() {
				if update.Addr == e.Value.(*addrInfo).addr {
					if e == p.pickedAddr {
						// Do not remove the element now if it is the current picked
						// one. We leave the deletion to the next PickAddr() call.
						e.Value.(*addrInfo).deleting = true
						// Notify Conn to close it. All the live RPCs on this connection
						// will be aborted.
						p.conn.NotifyReset()
					} else {
						p.addrs.Remove(e)
					}
				}
			}
			p.mu.Unlock()
		default:
			grpclog.Println("Unknown update.Op ", update.Op)
		}
	}
	return nil
}

// monitor runs in a standalone goroutine to keep watching name resolution updates until the watcher
// is closed.
func (p *unicastNamingPicker) monitor() {
	for {
		if err := p.processUpdates(); err != nil {
			return
		}
	}
}

func (p *unicastNamingPicker) Init(cc *ClientConn) error {
	w, err := p.resolver.Resolve(cc.target)
	if err != nil {
		return err
	}
	p.watcher = w
	p.cc = cc
	// Get the initial name resolution.
	if err := p.processUpdates(); err != nil {
		return err
	}
	go p.monitor()
	return nil
}

func (p *unicastNamingPicker) Pick(ctx context.Context) (transport.ClientTransport, error) {
	return p.conn.Wait(ctx)
}

func (p *unicastNamingPicker) PickAddr() (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.pickedAddr == nil {
		p.pickedAddr = p.addrs.Front()
	} else {
		pa := p.pickedAddr
		p.pickedAddr = pa.Next()
		if pa.Value.(*addrInfo).deleting {
			p.addrs.Remove(pa)
		}
		if p.pickedAddr == nil {
			p.pickedAddr = p.addrs.Front()
		}
	}
	if p.pickedAddr == nil {
		return "", fmt.Errorf("there is no address available to pick")
	}
	return p.pickedAddr.Value.(*addrInfo).addr, nil
}

func (p *unicastNamingPicker) State() (ConnectivityState, error) {
	return 0, fmt.Errorf("State() is not supported for unicastNamingPicker")
}

func (p *unicastNamingPicker) WaitForStateChange(ctx context.Context, sourceState ConnectivityState) (ConnectivityState, error) {
	return 0, fmt.Errorf("WaitForStateChange is not supported for unicastNamingPciker")
}

func (p *unicastNamingPicker) Close() error {
	p.watcher.Close()
	p.conn.Close()
	return nil
}
