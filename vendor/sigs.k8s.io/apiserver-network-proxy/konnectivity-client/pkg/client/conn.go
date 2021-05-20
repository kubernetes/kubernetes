/*
Copyright 2019 The Kubernetes Authors.

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

package client

import (
	"errors"
	"io"
	"net"
	"time"

	"google.golang.org/grpc"
	"k8s.io/klog/v2"
	"sigs.k8s.io/apiserver-network-proxy/konnectivity-client/proto/client"
)

// CloseTimeout is the timeout to wait CLOSE_RSP packet after a
// successful delivery of CLOSE_REQ.
const CloseTimeout = 10 * time.Second

// conn is an implementation of net.Conn, where the data is transported
// over an established tunnel defined by a gRPC service ProxyService.
type conn struct {
	cc      *grpc.ClientConn
	stream  client.ProxyService_ProxyClient
	connID  int64
	readCh  chan []byte
	closeCh chan string
	rdata   []byte
}

var _ net.Conn = &conn{}

// Write sends the data thru the connection over proxy service
func (c *conn) Write(data []byte) (n int, err error) {
	req := &client.Packet{
		Type: client.PacketType_DATA,
		Payload: &client.Packet_Data{
			Data: &client.Data{
				ConnectID: c.connID,
				Data:      data,
			},
		},
	}

	klog.V(5).InfoS("[tracing] send req", "type", req.Type)

	err = c.stream.Send(req)
	if err != nil {
		return 0, err
	}
	return len(data), err
}

// Read receives data from the connection over proxy service
func (c *conn) Read(b []byte) (n int, err error) {
	var data []byte

	if c.rdata != nil {
		data = c.rdata
	} else {
		data = <-c.readCh
	}

	if data == nil {
		return 0, io.EOF
	}

	if len(data) > len(b) {
		copy(b, data[:len(b)])
		c.rdata = data[len(b):]
		return len(b), nil
	}

	c.rdata = nil
	copy(b, data)

	return len(data), nil
}

func (c *conn) LocalAddr() net.Addr {
	return nil
}

func (c *conn) RemoteAddr() net.Addr {
	return nil
}

func (c *conn) SetDeadline(t time.Time) error {
	return errors.New("not implemented")
}

func (c *conn) SetReadDeadline(t time.Time) error {
	return errors.New("not implemented")
}

func (c *conn) SetWriteDeadline(t time.Time) error {
	return errors.New("not implemented")
}

func (c *conn) cleanup() {
	close(c.readCh)
	close(c.closeCh)
	err := c.cc.Close()
	if err != nil && err == grpc.ErrClientConnClosing {
		klog.V(2).InfoS("grpc ClientConn is already closed", "err", err)
	}
	if err != nil && err != grpc.ErrClientConnClosing {
		klog.ErrorS(err, "error closing grpc ClientConn")
	}
}

// Close closes the connection. It also sends CLOSE_REQ packet over
// proxy service to notify remote to drop the connection.
func (c *conn) Close() error {
	klog.V(4).Infoln("closing connection")
	req := &client.Packet{
		Type: client.PacketType_CLOSE_REQ,
		Payload: &client.Packet_CloseRequest{
			CloseRequest: &client.CloseRequest{
				ConnectID: c.connID,
			},
		},
	}

	klog.V(5).InfoS("[tracing] send req", "type", req.Type)

	if err := c.stream.Send(req); err != nil {
		return err
	}

	select {
	case errMsg := <-c.closeCh:
		if errMsg != "" {
			return errors.New(errMsg)
		}
		return nil
	case <-time.After(CloseTimeout):
		// This means that the client either timed out receiving the CLOSE_RSP,
		// or is blocked in its serve() loop and can't deliver the CLOSE_RSP
		// over closeCh. We call the cleanup() to unblock the tunnels.
		c.cleanup()
	}

	return errors.New("close timeout")
}
