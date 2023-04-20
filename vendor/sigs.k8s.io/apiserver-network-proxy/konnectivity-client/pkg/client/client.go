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
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"k8s.io/klog/v2"

	"sigs.k8s.io/apiserver-network-proxy/konnectivity-client/pkg/client/metrics"
	commonmetrics "sigs.k8s.io/apiserver-network-proxy/konnectivity-client/pkg/common/metrics"
	"sigs.k8s.io/apiserver-network-proxy/konnectivity-client/proto/client"
)

// Tunnel provides ability to dial a connection through a tunnel.
type Tunnel interface {
	// Dial connects to the address on the named network, similar to
	// what net.Dial does. The only supported protocol is tcp.
	DialContext(requestCtx context.Context, protocol, address string) (net.Conn, error)
	// Done returns a channel that is closed when the tunnel is no longer serving any connections,
	// and can no longer be used.
	Done() <-chan struct{}
}

type dialResult struct {
	err    *dialFailure
	connid int64
}

type pendingDial struct {
	// resultCh is the channel to send the dial result to
	resultCh chan<- dialResult
	// cancelCh is the channel closed when resultCh no longer has a receiver
	cancelCh <-chan struct{}
}

// TODO: Replace with a generic implementation once it is safe to assume the client is built with go1.18+
type pendingDialManager struct {
	pendingDials map[int64]pendingDial
	mutex        sync.RWMutex
}

func (p *pendingDialManager) add(dialID int64, pd pendingDial) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.pendingDials[dialID] = pd
}

func (p *pendingDialManager) remove(dialID int64) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	delete(p.pendingDials, dialID)
}

func (p *pendingDialManager) get(dialID int64) (pendingDial, bool) {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	pd, ok := p.pendingDials[dialID]
	return pd, ok
}

// TODO: Replace with a generic implementation once it is safe to assume the client is built with go1.18+
type connectionManager struct {
	conns map[int64]*conn
	mutex sync.RWMutex
}

func (cm *connectionManager) add(connID int64, c *conn) {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()
	cm.conns[connID] = c
}

func (cm *connectionManager) remove(connID int64) {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()
	delete(cm.conns, connID)
}

func (cm *connectionManager) get(connID int64) (*conn, bool) {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	c, ok := cm.conns[connID]
	return c, ok
}

func (cm *connectionManager) closeAll() {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()
	for _, conn := range cm.conns {
		close(conn.readCh)
	}
}

// grpcTunnel implements Tunnel
type grpcTunnel struct {
	stream      client.ProxyService_ProxyClient
	sendLock    sync.Mutex
	recvLock    sync.Mutex
	clientConn  clientConn
	pendingDial pendingDialManager
	conns       connectionManager

	// The tunnel will be closed if the caller fails to read via conn.Read()
	// more than readTimeoutSeconds after a packet has been received.
	readTimeoutSeconds int

	// The done channel is closed after the tunnel has cleaned up all connections and is no longer
	// serving.
	done chan struct{}

	// started is an atomic bool represented as a 0 or 1, and set to true when a single-use tunnel has been started (dialed).
	// started should only be accessed through atomic methods.
	// TODO: switch this to an atomic.Bool once the client is exclusively buit with go1.19+
	started uint32

	// closing is an atomic bool represented as a 0 or 1, and set to true when the tunnel is being closed.
	// closing should only be accessed through atomic methods.
	// TODO: switch this to an atomic.Bool once the client is exclusively buit with go1.19+
	closing uint32

	// Stores the current metrics.ClientConnectionStatus
	prevStatus atomic.Value
}

type clientConn interface {
	Close() error
}

var _ clientConn = &grpc.ClientConn{}

var (
	// Expose metrics for client to register.
	Metrics = metrics.Metrics
)

// CreateSingleUseGrpcTunnel creates a Tunnel to dial to a remote server through a
// gRPC based proxy service.
// Currently, a single tunnel supports a single connection, and the tunnel is closed when the connection is terminated
// The Dial() method of the returned tunnel should only be called once
// Deprecated 2022-06-07: use CreateSingleUseGrpcTunnelWithContext
func CreateSingleUseGrpcTunnel(tunnelCtx context.Context, address string, opts ...grpc.DialOption) (Tunnel, error) {
	return CreateSingleUseGrpcTunnelWithContext(context.TODO(), tunnelCtx, address, opts...)
}

// CreateSingleUseGrpcTunnelWithContext creates a Tunnel to dial to a remote server through a
// gRPC based proxy service.
// Currently, a single tunnel supports a single connection.
// The tunnel is normally closed when the connection is terminated.
// If createCtx is cancelled before tunnel creation, an error will be returned.
// If tunnelCtx is cancelled while the tunnel is still in use, the tunnel (and any in flight connections) will be closed.
// The Dial() method of the returned tunnel should only be called once
func CreateSingleUseGrpcTunnelWithContext(createCtx, tunnelCtx context.Context, address string, opts ...grpc.DialOption) (Tunnel, error) {
	c, err := grpc.DialContext(createCtx, address, opts...)
	if err != nil {
		return nil, err
	}

	grpcClient := client.NewProxyServiceClient(c)

	stream, err := grpcClient.Proxy(tunnelCtx)
	if err != nil {
		c.Close()
		return nil, err
	}

	tunnel := newUnstartedTunnel(stream, c)

	go tunnel.serve(tunnelCtx)

	return tunnel, nil
}

func newUnstartedTunnel(stream client.ProxyService_ProxyClient, c clientConn) *grpcTunnel {
	t := grpcTunnel{
		stream:             stream,
		clientConn:         c,
		pendingDial:        pendingDialManager{pendingDials: make(map[int64]pendingDial)},
		conns:              connectionManager{conns: make(map[int64]*conn)},
		readTimeoutSeconds: 10,
		done:               make(chan struct{}),
		started:            0,
	}
	s := metrics.ClientConnectionStatusCreated
	t.prevStatus.Store(s)
	metrics.Metrics.GetClientConnectionsMetric().WithLabelValues(string(s)).Inc()
	return &t
}

func (t *grpcTunnel) updateMetric(status metrics.ClientConnectionStatus) {
	select {
	case <-t.Done():
		return
	default:
	}

	prevStatus := t.prevStatus.Swap(status).(metrics.ClientConnectionStatus)

	m := metrics.Metrics.GetClientConnectionsMetric()
	m.WithLabelValues(string(prevStatus)).Dec()
	m.WithLabelValues(string(status)).Inc()
}

// closeMetric should be called exactly once to finalize client_connections metric.
func (t *grpcTunnel) closeMetric() {
	select {
	case <-t.Done():
		return
	default:
	}
	prevStatus := t.prevStatus.Load().(metrics.ClientConnectionStatus)

	metrics.Metrics.GetClientConnectionsMetric().WithLabelValues(string(prevStatus)).Dec()
}

func (t *grpcTunnel) serve(tunnelCtx context.Context) {
	defer func() {
		t.clientConn.Close()

		// A connection in t.conns after serve() returns means
		// we never received a CLOSE_RSP for it, so we need to
		// close any channels remaining for these connections.
		t.conns.closeAll()

		t.closeMetric()

		close(t.done)
	}()

	for {
		pkt, err := t.Recv()
		if err == io.EOF {
			return
		}
		isClosing := t.isClosing()
		if err != nil || pkt == nil {
			if !isClosing {
				klog.ErrorS(err, "stream read failure")
			}
			return
		}
		if isClosing {
			return
		}
		klog.V(5).InfoS("[tracing] recv packet", "type", pkt.Type)

		switch pkt.Type {
		case client.PacketType_DIAL_RSP:
			resp := pkt.GetDialResponse()
			pendingDial, ok := t.pendingDial.get(resp.Random)

			if !ok {
				// If the DIAL_RSP does not match a pending dial, it means one of two things:
				//   1. There was a second DIAL_RSP for the connection request (this is very unlikely but possible)
				//   2. grpcTunnel.DialContext() returned early due to a dial timeout or the client canceling the context
				//
				// In either scenario, we should return here and close the tunnel as it is no longer needed.
				kvs := []interface{}{"dialID", resp.Random, "connectID", resp.ConnectID}
				if resp.Error != "" {
					kvs = append(kvs, "error", resp.Error)
				}
				klog.V(1).InfoS("DialResp not recognized; dropped", kvs...)
				return
			}

			result := dialResult{connid: resp.ConnectID}
			if resp.Error != "" {
				result.err = &dialFailure{resp.Error, metrics.DialFailureEndpoint}
			} else {
				t.updateMetric(metrics.ClientConnectionStatusOk)
			}
			select {
			// try to send to the result channel
			case pendingDial.resultCh <- result:
			// unblock if the cancel channel is closed
			case <-pendingDial.cancelCh:
				// Note: this condition can only be hit by a race condition where the
				// DialContext() returns early (timeout) after the pendingDial is already
				// fetched here, but before the result is sent.
				klog.V(1).InfoS("Pending dial has been cancelled; dropped", "connectionID", resp.ConnectID, "dialID", resp.Random)
				return
			case <-tunnelCtx.Done():
				klog.V(1).InfoS("Tunnel has been closed; dropped", "connectionID", resp.ConnectID, "dialID", resp.Random)
				return
			}

			if resp.Error != "" {
				// On dial error, avoid leaking serve goroutine.
				return
			}

		case client.PacketType_DIAL_CLS:
			resp := pkt.GetCloseDial()
			pendingDial, ok := t.pendingDial.get(resp.Random)

			if !ok {
				// If the DIAL_CLS does not match a pending dial, it means one of two things:
				//   1. There was a DIAL_CLS receieved after a DIAL_RSP (unlikely but possible)
				//   2. grpcTunnel.DialContext() returned early due to a dial timeout or the client canceling the context
				//
				// In either scenario, we should return here and close the tunnel as it is no longer needed.
				klog.V(1).InfoS("DIAL_CLS after dial finished", "dialID", resp.Random)
			} else {
				result := dialResult{
					err: &dialFailure{"dial closed", metrics.DialFailureDialClosed},
				}
				select {
				case pendingDial.resultCh <- result:
				case <-pendingDial.cancelCh:
					// Note: this condition can only be hit by a race condition where the
					// DialContext() returns early (timeout) after the pendingDial is already
					// fetched here, but before the result is sent.
				case <-tunnelCtx.Done():
				}
			}
			return // Stop serving & close the tunnel.

		case client.PacketType_DATA:
			resp := pkt.GetData()
			if resp.ConnectID == 0 {
				klog.ErrorS(nil, "Received packet missing ConnectID", "packetType", "DATA")
				continue
			}
			// TODO: flow control
			conn, ok := t.conns.get(resp.ConnectID)

			if !ok {
				klog.ErrorS(nil, "Connection not recognized", "connectionID", resp.ConnectID, "packetType", "DATA")
				t.Send(&client.Packet{
					Type: client.PacketType_CLOSE_REQ,
					Payload: &client.Packet_CloseRequest{
						CloseRequest: &client.CloseRequest{
							ConnectID: resp.ConnectID,
						},
					},
				})
				continue
			}
			timer := time.NewTimer((time.Duration)(t.readTimeoutSeconds) * time.Second)
			select {
			case conn.readCh <- resp.Data:
				timer.Stop()
			case <-timer.C:
				klog.ErrorS(fmt.Errorf("timeout"), "readTimeout has been reached, the grpc connection to the proxy server will be closed", "connectionID", conn.connID, "readTimeoutSeconds", t.readTimeoutSeconds)
				return
			case <-tunnelCtx.Done():
				klog.V(1).InfoS("Tunnel has been closed, the grpc connection to the proxy server will be closed", "connectionID", conn.connID)
			}

		case client.PacketType_CLOSE_RSP:
			resp := pkt.GetCloseResponse()
			conn, ok := t.conns.get(resp.ConnectID)

			if !ok {
				klog.V(1).InfoS("Connection not recognized", "connectionID", resp.ConnectID, "packetType", "CLOSE_RSP")
				continue
			}
			close(conn.readCh)
			conn.closeCh <- resp.Error
			close(conn.closeCh)
			t.conns.remove(resp.ConnectID)
			return
		}
	}
}

// Dial connects to the address on the named network, similar to
// what net.Dial does. The only supported protocol is tcp.
func (t *grpcTunnel) DialContext(requestCtx context.Context, protocol, address string) (net.Conn, error) {
	conn, err := t.dialContext(requestCtx, protocol, address)
	if err != nil {
		_, reason := GetDialFailureReason(err)
		metrics.Metrics.ObserveDialFailure(reason)
	}
	return conn, err
}

func (t *grpcTunnel) dialContext(requestCtx context.Context, protocol, address string) (net.Conn, error) {
	prevStarted := atomic.SwapUint32(&t.started, 1)
	if prevStarted != 0 {
		return nil, &dialFailure{"single-use dialer already dialed", metrics.DialFailureAlreadyStarted}
	}

	select {
	case <-t.done:
		return nil, errors.New("tunnel is closed")
	default: // Tunnel is open, carry on.
	}

	if protocol != "tcp" {
		return nil, errors.New("protocol not supported")
	}

	t.updateMetric(metrics.ClientConnectionStatusDialing)

	random := rand.Int63() /* #nosec G404 */

	// This channel is closed once we're returning and no longer waiting on resultCh
	cancelCh := make(chan struct{})
	defer close(cancelCh)

	// This channel MUST NOT be buffered. The sender needs to know when we are not receiving things, so they can abort.
	resCh := make(chan dialResult)

	t.pendingDial.add(random, pendingDial{resultCh: resCh, cancelCh: cancelCh})
	defer t.pendingDial.remove(random)

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

	err := t.Send(req)
	if err != nil {
		return nil, err
	}

	klog.V(5).Infoln("DIAL_REQ sent to proxy server")

	c := &conn{
		tunnel:      t,
		random:      random,
		closeTunnel: t.closeTunnel,
	}

	select {
	case res := <-resCh:
		if res.err != nil {
			return nil, res.err
		}
		c.connID = res.connid
		c.readCh = make(chan []byte, 10)
		c.closeCh = make(chan string, 1)
		t.conns.add(res.connid, c)
	case <-time.After(30 * time.Second):
		klog.V(5).InfoS("Timed out waiting for DialResp", "dialID", random)
		go t.closeDial(random)
		return nil, &dialFailure{"dial timeout, backstop", metrics.DialFailureTimeout}
	case <-requestCtx.Done():
		klog.V(5).InfoS("Context canceled waiting for DialResp", "ctxErr", requestCtx.Err(), "dialID", random)
		go t.closeDial(random)
		return nil, &dialFailure{"dial timeout, context", metrics.DialFailureContext}
	case <-t.done:
		klog.V(5).InfoS("Tunnel closed while waiting for DialResp", "dialID", random)
		return nil, &dialFailure{"tunnel closed", metrics.DialFailureTunnelClosed}
	}

	return c, nil
}

func (t *grpcTunnel) Done() <-chan struct{} {
	return t.done
}

// Send a best-effort DIAL_CLS request for the given dial ID.
func (t *grpcTunnel) closeDial(dialID int64) {
	req := &client.Packet{
		Type: client.PacketType_DIAL_CLS,
		Payload: &client.Packet_CloseDial{
			CloseDial: &client.CloseDial{
				Random: dialID,
			},
		},
	}
	if err := t.Send(req); err != nil {
		klog.V(5).InfoS("Failed to send DIAL_CLS", "err", err, "dialID", dialID)
	}
	t.closeTunnel()
}

func (t *grpcTunnel) closeTunnel() {
	atomic.StoreUint32(&t.closing, 1)
	t.clientConn.Close()
}

func (t *grpcTunnel) isClosing() bool {
	return atomic.LoadUint32(&t.closing) != 0
}

func (t *grpcTunnel) Send(pkt *client.Packet) error {
	t.sendLock.Lock()
	defer t.sendLock.Unlock()

	const segment = commonmetrics.SegmentFromClient
	metrics.Metrics.ObservePacket(segment, pkt.Type)
	err := t.stream.Send(pkt)
	if err != nil && err != io.EOF {
		metrics.Metrics.ObserveStreamError(segment, err, pkt.Type)
	}
	return err
}

func (t *grpcTunnel) Recv() (*client.Packet, error) {
	t.recvLock.Lock()
	defer t.recvLock.Unlock()

	const segment = commonmetrics.SegmentToClient
	pkt, err := t.stream.Recv()
	if err != nil {
		if err != io.EOF {
			metrics.Metrics.ObserveStreamErrorNoPacket(segment, err)
		}
		return nil, err
	}
	metrics.Metrics.ObservePacket(segment, pkt.Type)
	return pkt, nil
}

func GetDialFailureReason(err error) (isDialFailure bool, reason metrics.DialFailureReason) {
	var df *dialFailure
	if errors.As(err, &df) {
		return true, df.reason
	}
	return false, metrics.DialFailureUnknown
}

type dialFailure struct {
	msg    string
	reason metrics.DialFailureReason
}

func (df *dialFailure) Error() string {
	return df.msg
}
