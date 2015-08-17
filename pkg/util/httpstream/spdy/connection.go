/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package spdy

import (
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/docker/spdystream"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/httpstream"
)

// connection maintains state about a spdystream.Connection and its associated
// streams.
type connection struct {
	conn             *spdystream.Connection
	streams          []httpstream.Stream
	streamLock       sync.Mutex
	newStreamHandler httpstream.NewStreamHandler
}

// NewClientConnection creates a new SPDY client connection.
func NewClientConnection(conn net.Conn) (httpstream.Connection, error) {
	spdyConn, err := spdystream.NewConnection(conn, false)
	if err != nil {
		defer conn.Close()
		return nil, err
	}

	return newConnection(spdyConn, httpstream.NoOpNewStreamHandler), nil
}

// NewServerConnection creates a new SPDY server connection. newStreamHandler
// will be invoked when the server receives a newly created stream from the
// client.
func NewServerConnection(conn net.Conn, newStreamHandler httpstream.NewStreamHandler) (httpstream.Connection, error) {
	spdyConn, err := spdystream.NewConnection(conn, true)
	if err != nil {
		defer conn.Close()
		return nil, err
	}

	return newConnection(spdyConn, newStreamHandler), nil
}

// newConnection returns a new connection wrapping conn. newStreamHandler
// will be invoked when the server receives a newly created stream from the
// client.
func newConnection(conn *spdystream.Connection, newStreamHandler httpstream.NewStreamHandler) httpstream.Connection {
	c := &connection{conn: conn, newStreamHandler: newStreamHandler}
	go conn.Serve(c.newSpdyStream)
	return c
}

// createStreamResponseTimeout indicates how long to wait for the other side to
// acknowledge the new stream before timing out.
const createStreamResponseTimeout = 30 * time.Second

// Close first sends a reset for all of the connection's streams, and then
// closes the underlying spdystream.Connection.
func (c *connection) Close() error {
	c.streamLock.Lock()
	for _, s := range c.streams {
		s.Reset()
	}
	c.streams = make([]httpstream.Stream, 0)
	c.streamLock.Unlock()

	return c.conn.Close()
}

// CreateStream creates a new stream with the specified headers and registers
// it with the connection.
func (c *connection) CreateStream(headers http.Header) (httpstream.Stream, error) {
	stream, err := c.conn.CreateStream(headers, nil, false)
	if err != nil {
		return nil, err
	}
	if err = stream.WaitTimeout(createStreamResponseTimeout); err != nil {
		return nil, err
	}

	c.registerStream(stream)
	return stream, nil
}

// registerStream adds the stream s to the connection's list of streams that
// it owns.
func (c *connection) registerStream(s httpstream.Stream) {
	c.streamLock.Lock()
	c.streams = append(c.streams, s)
	c.streamLock.Unlock()
}

// CloseChan returns a channel that, when closed, indicates that the underlying
// spdystream.Connection has been closed.
func (c *connection) CloseChan() <-chan bool {
	return c.conn.CloseChan()
}

// newSpdyStream is the internal new stream handler used by spdystream.Connection.Serve.
// It calls connection's newStreamHandler, giving it the opportunity to accept or reject
// the stream. If newStreamHandler returns an error, the stream is rejected. If not, the
// stream is accepted and registered with the connection.
func (c *connection) newSpdyStream(stream *spdystream.Stream) {
	err := c.newStreamHandler(stream)
	rejectStream := (err != nil)
	if rejectStream {
		glog.Warningf("Stream rejected: %v", err)
		stream.Reset()
		return
	}

	c.registerStream(stream)
	stream.SendReply(http.Header{}, rejectStream)
}

// SetIdleTimeout sets the amount of time the connection may remain idle before
// it is automatically closed.
func (c *connection) SetIdleTimeout(timeout time.Duration) {
	c.conn.SetIdleTimeout(timeout)
}
