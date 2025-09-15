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
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	gwebsocket "github.com/gorilla/websocket"

	"k8s.io/klog/v2"
)

var _ net.Conn = &TunnelingConnection{}

// TunnelingConnection implements the "httpstream.Connection" interface, wrapping
// a websocket connection that tunnels SPDY.
type TunnelingConnection struct {
	name              string
	conn              *gwebsocket.Conn
	inProgressMessage io.Reader
	closeOnce         sync.Once
}

// NewTunnelingConnection wraps the passed gorilla/websockets connection
// with the TunnelingConnection struct (implementing net.Conn).
func NewTunnelingConnection(name string, conn *gwebsocket.Conn) *TunnelingConnection {
	return &TunnelingConnection{
		name: name,
		conn: conn,
	}
}

// Read implements "io.Reader" interface, reading from the stored connection
// into the passed buffer "p". Returns the number of bytes read and an error.
// Can keep track of the "inProgress" messsage from the tunneled connection.
func (c *TunnelingConnection) Read(p []byte) (int, error) {
	klog.V(7).Infof("%s: tunneling connection read...", c.name)
	defer klog.V(7).Infof("%s: tunneling connection read...complete", c.name)
	for {
		if c.inProgressMessage == nil {
			klog.V(8).Infof("%s: tunneling connection read before NextReader()...", c.name)
			messageType, nextReader, err := c.conn.NextReader()
			if err != nil {
				closeError := &gwebsocket.CloseError{}
				if errors.As(err, &closeError) && closeError.Code == gwebsocket.CloseNormalClosure {
					return 0, io.EOF
				}
				klog.V(4).Infof("%s:tunneling connection NextReader() error: %v", c.name, err)
				return 0, err
			}
			if messageType != gwebsocket.BinaryMessage {
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
	w, err := c.conn.NextWriter(gwebsocket.BinaryMessage)
	if err != nil {
		return 0, err
	}
	defer func() {
		// close, which flushes the message
		closeErr := w.Close()
		if closeErr != nil && err == nil {
			// if closing/flushing errored and we weren't already returning an error, return the close error
			err = closeErr
		}
	}()

	n, err = w.Write(p)
	return
}

// Close implements "io.Closer" interface, signaling the other tunneled connection
// endpoint, and closing the tunneled connection only once.
func (c *TunnelingConnection) Close() error {
	var err error
	c.closeOnce.Do(func() {
		klog.V(7).Infof("%s: tunneling connection Close()...", c.name)
		// Signal other endpoint that websocket connection is closing; ignore error.
		normalCloseMsg := gwebsocket.FormatCloseMessage(gwebsocket.CloseNormalClosure, "")
		writeControlErr := c.conn.WriteControl(gwebsocket.CloseMessage, normalCloseMsg, time.Now().Add(time.Second))
		closeErr := c.conn.Close()
		if closeErr != nil {
			err = closeErr
		} else if writeControlErr != nil {
			err = writeControlErr
		}
	})
	return err
}

// LocalAddr implements part of the "net.Conn" interface, returning the local
// endpoint network address of the tunneled connection.
func (c *TunnelingConnection) LocalAddr() net.Addr {
	return c.conn.LocalAddr()
}

// LocalAddr implements part of the "net.Conn" interface, returning the remote
// endpoint network address of the tunneled connection.
func (c *TunnelingConnection) RemoteAddr() net.Addr {
	return c.conn.RemoteAddr()
}

// SetDeadline sets the *absolute* time in the future for both
// read and write deadlines. Returns an error if one occurs.
func (c *TunnelingConnection) SetDeadline(t time.Time) error {
	rerr := c.SetReadDeadline(t)
	werr := c.SetWriteDeadline(t)
	return errors.Join(rerr, werr)
}

// SetDeadline sets the *absolute* time in the future for the
// read deadlines. Returns an error if one occurs.
func (c *TunnelingConnection) SetReadDeadline(t time.Time) error {
	return c.conn.SetReadDeadline(t)
}

// SetDeadline sets the *absolute* time in the future for the
// write deadlines. Returns an error if one occurs.
func (c *TunnelingConnection) SetWriteDeadline(t time.Time) error {
	return c.conn.SetWriteDeadline(t)
}
