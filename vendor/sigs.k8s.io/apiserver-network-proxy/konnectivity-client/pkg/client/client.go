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
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"
	"k8s.io/klog/v2"
	"sigs.k8s.io/apiserver-network-proxy/konnectivity-client/proto/client"
)

// Tunnel provides ability to dial a connection through a tunnel.
type Tunnel interface {
	// Dial connects to the address on the named network, similar to
	// what net.Dial does. The only supported protocol is tcp.
	DialContext(ctx context.Context, protocol, address string) (net.Conn, error)
}

type dialResult struct {
	err    string
	connid int64
}

type pendingDial struct {
	// resultCh is the channel to send the dial result to
	resultCh chan<- dialResult
	// cancelCh is the channel closed when resultCh no longer has a receiver
	cancelCh <-chan struct{}
}

// grpcTunnel implements Tunnel
type grpcTunnel struct {
	stream          client.ProxyService_ProxyClient
	pendingDial     map[int64]pendingDial
	conns           map[int64]*conn
	pendingDialLock sync.RWMutex
	connsLock       sync.RWMutex

	// The tunnel will be closed if the caller fails to read via conn.Read()
	// more than readTimeoutSeconds after a packet has been received.
	readTimeoutSeconds int
}

type clientConn interface {
	Close() error
}

var _ clientConn = &grpc.ClientConn{}

// CreateSingleUseGrpcTunnel creates a Tunnel to dial to a remote server through a
// gRPC based proxy service.
// Currently, a single tunnel supports a single connection, and the tunnel is closed when the connection is terminated
// The Dial() method of the returned tunnel should only be called once
func CreateSingleUseGrpcTunnel(ctx context.Context, address string, opts ...grpc.DialOption) (Tunnel, error) {
	c, err := grpc.DialContext(ctx, address, opts...)
	if err != nil {
		return nil, err
	}

	grpcClient := client.NewProxyServiceClient(c)

	stream, err := grpcClient.Proxy(ctx)
	if err != nil {
		c.Close()
		return nil, err
	}

	tunnel := &grpcTunnel{
		stream:             stream,
		pendingDial:        make(map[int64]pendingDial),
		conns:              make(map[int64]*conn),
		readTimeoutSeconds: 10,
	}

	go tunnel.serve(c)

	return tunnel, nil
}

func (t *grpcTunnel) serve(c clientConn) {
	defer c.Close()

	for {
		pkt, err := t.stream.Recv()
		if err == io.EOF {
			return
		}
		if err != nil || pkt == nil {
			klog.ErrorS(err, "stream read failure")
			return
		}

		klog.V(5).InfoS("[tracing] recv packet", "type", pkt.Type)

		switch pkt.Type {
		case client.PacketType_DIAL_RSP:
			resp := pkt.GetDialResponse()
			t.pendingDialLock.RLock()
			pendingDial, ok := t.pendingDial[resp.Random]
			t.pendingDialLock.RUnlock()

			if !ok {
				klog.V(1).InfoS("DialResp not recognized; dropped", "connectionID", resp.ConnectID, "dialID", resp.Random)
				return
			} else {
				result := dialResult{
					err:    resp.Error,
					connid: resp.ConnectID,
				}
				select {
				// try to send to the result channel
				case pendingDial.resultCh <- result:
				// unblock if the cancel channel is closed
				case <-pendingDial.cancelCh:
					// If there are no readers of the pending dial channel above, it means one of two things:
					//   1. There was a second DIAL_RSP for the connection request (this is very unlikely but possible)
					//   2. grpcTunnel.DialContext() returned early due to a dial timeout or the client canceling the context
					//
					// In either scenario, we should return here as this tunnel is no longer needed.
					klog.V(1).InfoS("Pending dial has been cancelled; dropped", "connectionID", resp.ConnectID, "dialID", resp.Random)
					return
				}
			}

			if resp.Error != "" {
				// On dial error, avoid leaking serve goroutine.
				return
			}

		case client.PacketType_DATA:
			resp := pkt.GetData()
			// TODO: flow control
			t.connsLock.RLock()
			conn, ok := t.conns[resp.ConnectID]
			t.connsLock.RUnlock()

			if ok {
				timer := time.NewTimer((time.Duration)(t.readTimeoutSeconds) * time.Second)
				select {
				case conn.readCh <- resp.Data:
					timer.Stop()
				case <-timer.C:
					klog.ErrorS(fmt.Errorf("timeout"), "readTimeout has been reached, the grpc connection to the proxy server will be closed", "connectionID", conn.connID, "readTimeoutSeconds", t.readTimeoutSeconds)
					return
				}
			} else {
				klog.V(1).InfoS("connection not recognized", "connectionID", resp.ConnectID)
			}
		case client.PacketType_CLOSE_RSP:
			resp := pkt.GetCloseResponse()
			t.connsLock.RLock()
			conn, ok := t.conns[resp.ConnectID]
			t.connsLock.RUnlock()

			if ok {
				close(conn.readCh)
				conn.closeCh <- resp.Error
				close(conn.closeCh)
				t.connsLock.Lock()
				delete(t.conns, resp.ConnectID)
				t.connsLock.Unlock()
				return
			}
			klog.V(1).InfoS("connection not recognized", "connectionID", resp.ConnectID)
		}
	}
}

// Dial connects to the address on the named network, similar to
// what net.Dial does. The only supported protocol is tcp.
func (t *grpcTunnel) DialContext(ctx context.Context, protocol, address string) (net.Conn, error) {
	if protocol != "tcp" {
		return nil, errors.New("protocol not supported")
	}

	random := rand.Int63() /* #nosec G404 */

	// This channel is closed once we're returning and no longer waiting on resultCh
	cancelCh := make(chan struct{})
	defer close(cancelCh)

	// This channel MUST NOT be buffered. The sender needs to know when we are not receiving things, so they can abort.
	resCh := make(chan dialResult)

	t.pendingDialLock.Lock()
	t.pendingDial[random] = pendingDial{resultCh: resCh, cancelCh: cancelCh}
	t.pendingDialLock.Unlock()
	defer func() {
		t.pendingDialLock.Lock()
		delete(t.pendingDial, random)
		t.pendingDialLock.Unlock()
	}()

	req := &client.Packet{
		Type: client.PacketType_DIAL_REQ,
		Payload: &client.Packet_DialRequest{
			DialRequest: &client.DialRequest{
				Protocol: protocol,
				Address:  address,
				Random:   random,
			},
		},
	}
	klog.V(5).InfoS("[tracing] send packet", "type", req.Type)

	err := t.stream.Send(req)
	if err != nil {
		return nil, err
	}

	klog.V(5).Infoln("DIAL_REQ sent to proxy server")

	c := &conn{stream: t.stream, random: random}

	select {
	case res := <-resCh:
		if res.err != "" {
			return nil, errors.New(res.err)
		}
		c.connID = res.connid
		c.readCh = make(chan []byte, 10)
		c.closeCh = make(chan string, 1)
		t.connsLock.Lock()
		t.conns[res.connid] = c
		t.connsLock.Unlock()
	case <-time.After(30 * time.Second):
		klog.V(5).InfoS("Timed out waiting for DialResp", "dialID", random)
		return nil, errors.New("dial timeout, backstop")
	case <-ctx.Done():
		klog.V(5).InfoS("Context canceled waiting for DialResp", "ctxErr", ctx.Err(), "dialID", random)
		return nil, errors.New("dial timeout, context")
	}

	return c, nil
}
