/*
Copyright 2018 The Kubernetes Authors.

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

// Package connrotation implements a connection dialer that tracks and can close
// all created connections.
//
// This is used for credential rotation of long-lived connections, when there's
// no way to re-authenticate on a live connection.
package connrotation

import (
	"context"
	"net"
	"sync"
)

// DialFunc is a shorthand for signature of net.DialContext.
type DialFunc func(ctx context.Context, network, address string) (net.Conn, error)

// Dialer opens connections through Dial and tracks them.
type Dialer struct {
	dial DialFunc

	mu    sync.Mutex
	conns map[*closableConn]struct{}
}

// NewDialer creates a new Dialer instance.
//
// If dial is not nil, it will be used to create new underlying connections.
// Otherwise net.DialContext is used.
func NewDialer(dial DialFunc) *Dialer {
	return &Dialer{
		dial:  dial,
		conns: make(map[*closableConn]struct{}),
	}
}

// CloseAll forcibly closes all tracked connections.
//
// Note: new connections may get created before CloseAll returns.
func (d *Dialer) CloseAll() {
	d.mu.Lock()
	conns := d.conns
	d.conns = make(map[*closableConn]struct{})
	d.mu.Unlock()

	for conn := range conns {
		conn.Close()
	}
}

// Dial creates a new tracked connection.
func (d *Dialer) Dial(network, address string) (net.Conn, error) {
	return d.DialContext(context.Background(), network, address)
}

// DialContext creates a new tracked connection.
func (d *Dialer) DialContext(ctx context.Context, network, address string) (net.Conn, error) {
	conn, err := d.dial(ctx, network, address)
	if err != nil {
		return nil, err
	}

	closable := &closableConn{Conn: conn}

	// When the connection is closed, remove it from the map. This will
	// be no-op if the connection isn't in the map, e.g. if CloseAll()
	// is called.
	closable.onClose = func() {
		d.mu.Lock()
		delete(d.conns, closable)
		d.mu.Unlock()
	}

	// Start tracking the connection
	d.mu.Lock()
	d.conns[closable] = struct{}{}
	d.mu.Unlock()

	return closable, nil
}

type closableConn struct {
	onClose func()
	net.Conn
}

func (c *closableConn) Close() error {
	go c.onClose()
	return c.Conn.Close()
}
