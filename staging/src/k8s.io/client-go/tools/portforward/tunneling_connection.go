/*
Copyright 2023 The Kubernetes Authors.

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

package portforward

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"github.com/coder/websocket"

	"k8s.io/klog/v2"
)

var _ net.Conn = &TunnelingConnection{}

// TunnelingConnection implements the "net.Conn" interface, wrapping
// a websocket connection that tunnels SPDY.
type TunnelingConnection struct {
	name              string
	conn              *websocket.Conn
	ctx               context.Context
	cancel            context.CancelFunc
	inProgressMessage io.Reader
	closeOnce         sync.Once
	readDeadline      time.Time
	writeDeadline     time.Time
	deadlineMu        sync.RWMutex

	// Addresses captured during connection establishment
	// (coder/websocket doesn't expose NetConn directly)
	localAddr  net.Addr
	remoteAddr net.Addr
}

// NewTunnelingConnection wraps the passed coder/websocket connection
// with the TunnelingConnection struct (implementing net.Conn).
func NewTunnelingConnection(name string, conn *websocket.Conn) *TunnelingConnection {
	return NewTunnelingConnectionWithAddrs(name, conn, nil, nil)
}

// NewTunnelingConnectionWithAddrs wraps the passed coder/websocket connection
// with the TunnelingConnection struct (implementing net.Conn), including
// pre-captured local and remote addresses.
func NewTunnelingConnectionWithAddrs(name string, conn *websocket.Conn, localAddr, remoteAddr net.Addr) *TunnelingConnection {
	ctx, cancel := context.WithCancel(context.Background())
	return &TunnelingConnection{
		name:       name,
		conn:       conn,
		ctx:        ctx,
		cancel:     cancel,
		localAddr:  localAddr,
		remoteAddr: remoteAddr,
	}
}

// withDeadline returns a context that will be cancelled when the deadline expires.
// Uses context.AfterFunc to schedule cancellation at the deadline time.
// Returns the context and a stop function that should be called when the operation completes.
func withDeadline(parent context.Context, deadline time.Time) (context.Context, context.CancelFunc, func() bool) {
	if deadline.IsZero() {
		return parent, func() {}, func() bool { return true }
	}

	ctx, cancel := context.WithCancel(parent)

	d := time.Until(deadline)
	if d <= 0 {
		// Deadline already passed
		cancel()
		return ctx, cancel, func() bool { return false }
	}

	// Schedule cancellation at the deadline
	timer := time.AfterFunc(d, cancel)
	stop := func() bool {
		return timer.Stop()
	}

	return ctx, cancel, stop
}

// Read implements "io.Reader" interface, reading from the stored connection
// into the passed buffer "p". Returns the number of bytes read and an error.
// Can keep track of the "inProgress" messsage from the tunneled connection.
func (c *TunnelingConnection) Read(p []byte) (int, error) {
	klog.V(7).Infof("%s: tunneling connection read...", c.name)
	defer klog.V(7).Infof("%s: tunneling connection read...complete", c.name)
	for {
		if c.inProgressMessage == nil {
			klog.V(8).Infof("%s: tunneling connection read before Reader()...", c.name)

			// Get current deadline
			c.deadlineMu.RLock()
			deadline := c.readDeadline
			c.deadlineMu.RUnlock()

			// Create context with deadline using AfterFunc pattern
			ctx, cancel, stop := withDeadline(c.ctx, deadline)
			messageType, nextReader, err := c.conn.Reader(ctx)

			// Stop the deadline timer if it hasn't fired
			if !stop() {
				// Timer already fired, deadline exceeded
				cancel()
				return 0, context.DeadlineExceeded
			}
			cancel()

			if err != nil {
				// Check for normal closure
				if websocket.CloseStatus(err) == websocket.StatusNormalClosure {
					return 0, io.EOF
				}
				// Check for context cancellation (which may indicate deadline exceeded)
				if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
					return 0, err
				}
				klog.V(4).Infof("%s: tunneling connection Reader() error: %v", c.name, err)
				return 0, err
			}
			if messageType != websocket.MessageBinary {
				return 0, fmt.Errorf("invalid message type received")
			}
			c.inProgressMessage = nextReader
		}
		klog.V(8).Infof("%s: tunneling connection read in progress message...", c.name)
		i, err := c.inProgressMessage.Read(p)
		if i == 0 && err == io.EOF {
			c.inProgressMessage = nil
		} else {
			klog.V(8).Infof("%s: read %d bytes, error=%v, bytes=% X", c.name, i, err, p[:i])
			return i, err
		}
	}
}

// Write implements "io.Writer" interface, copying the data in the passed
// byte array "p" into the stored tunneled connection. Returns the number
// of bytes written and an error.
func (c *TunnelingConnection) Write(p []byte) (n int, err error) {
	klog.V(7).Infof("%s: write: %d bytes, bytes=% X", c.name, len(p), p)
	defer klog.V(7).Infof("%s: tunneling connection write...complete", c.name)

	// Get current deadline
	c.deadlineMu.RLock()
	deadline := c.writeDeadline
	c.deadlineMu.RUnlock()

	// Create context with deadline using AfterFunc pattern
	ctx, cancel, stop := withDeadline(c.ctx, deadline)
	defer cancel()

	err = c.conn.Write(ctx, websocket.MessageBinary, p)

	// Stop the deadline timer if it hasn't fired
	if !stop() {
		// Timer already fired, deadline exceeded
		return 0, context.DeadlineExceeded
	}

	if err != nil {
		return 0, err
	}
	return len(p), nil
}

// Close implements "io.Closer" interface, signaling the other tunneled connection
// endpoint, and closing the tunneled connection only once.
func (c *TunnelingConnection) Close() error {
	var err error
	c.closeOnce.Do(func() {
		klog.V(7).Infof("%s: tunneling connection Close()...", c.name)
		// Cancel the context to abort any ongoing operations
		c.cancel()
		// Close the websocket connection with normal closure status
		err = c.conn.Close(websocket.StatusNormalClosure, "")
	})
	return err
}

// LocalAddr implements part of the "net.Conn" interface, returning the local
// endpoint network address of the tunneled connection.
// Note: coder/websocket doesn't expose the underlying net.Conn directly,
// so this returns the address captured during connection establishment.
func (c *TunnelingConnection) LocalAddr() net.Addr {
	return c.localAddr
}

// RemoteAddr implements part of the "net.Conn" interface, returning the remote
// endpoint network address of the tunneled connection.
// Note: coder/websocket doesn't expose the underlying net.Conn directly,
// so this returns the address captured during connection establishment.
func (c *TunnelingConnection) RemoteAddr() net.Addr {
	return c.remoteAddr
}

// SetDeadline sets the *absolute* time in the future for both
// read and write deadlines. Returns an error if one occurs.
func (c *TunnelingConnection) SetDeadline(t time.Time) error {
	rerr := c.SetReadDeadline(t)
	werr := c.SetWriteDeadline(t)
	return errors.Join(rerr, werr)
}

// SetReadDeadline sets the *absolute* time in the future for the
// read deadlines. Returns an error if one occurs.
func (c *TunnelingConnection) SetReadDeadline(t time.Time) error {
	c.deadlineMu.Lock()
	c.readDeadline = t
	c.deadlineMu.Unlock()
	return nil
}

// SetWriteDeadline sets the *absolute* time in the future for the
// write deadlines. Returns an error if one occurs.
func (c *TunnelingConnection) SetWriteDeadline(t time.Time) error {
	c.deadlineMu.Lock()
	c.writeDeadline = t
	c.deadlineMu.Unlock()
	return nil
}
