package raft

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"sync"
	"time"

	"github.com/hashicorp/go-msgpack/codec"
)

const (
	rpcAppendEntries uint8 = iota
	rpcRequestVote
	rpcInstallSnapshot

	// DefaultTimeoutScale is the default TimeoutScale in a NetworkTransport.
	DefaultTimeoutScale = 256 * 1024 // 256KB

	// rpcMaxPipeline controls the maximum number of outstanding
	// AppendEntries RPC calls.
	rpcMaxPipeline = 128
)

var (
	// ErrTransportShutdown is returned when operations on a transport are
	// invoked after it's been terminated.
	ErrTransportShutdown = errors.New("transport shutdown")

	// ErrPipelineShutdown is returned when the pipeline is closed.
	ErrPipelineShutdown = errors.New("append pipeline closed")
)

/*

NetworkTransport provides a network based transport that can be
used to communicate with Raft on remote machines. It requires
an underlying stream layer to provide a stream abstraction, which can
be simple TCP, TLS, etc.

This transport is very simple and lightweight. Each RPC request is
framed by sending a byte that indicates the message type, followed
by the MsgPack encoded request.

The response is an error string followed by the response object,
both are encoded using MsgPack.

InstallSnapshot is special, in that after the RPC request we stream
the entire state. That socket is not re-used as the connection state
is not known if there is an error.

*/
type NetworkTransport struct {
	connPool     map[ServerAddress][]*netConn
	connPoolLock sync.Mutex

	consumeCh chan RPC

	heartbeatFn     func(RPC)
	heartbeatFnLock sync.Mutex

	logger *log.Logger

	maxPool int

	shutdown     bool
	shutdownCh   chan struct{}
	shutdownLock sync.Mutex

	stream StreamLayer

	timeout      time.Duration
	TimeoutScale int
}

// StreamLayer is used with the NetworkTransport to provide
// the low level stream abstraction.
type StreamLayer interface {
	net.Listener

	// Dial is used to create a new outgoing connection
	Dial(address ServerAddress, timeout time.Duration) (net.Conn, error)
}

type netConn struct {
	target ServerAddress
	conn   net.Conn
	r      *bufio.Reader
	w      *bufio.Writer
	dec    *codec.Decoder
	enc    *codec.Encoder
}

func (n *netConn) Release() error {
	return n.conn.Close()
}

type netPipeline struct {
	conn  *netConn
	trans *NetworkTransport

	doneCh       chan AppendFuture
	inprogressCh chan *appendFuture

	shutdown     bool
	shutdownCh   chan struct{}
	shutdownLock sync.Mutex
}

// NewNetworkTransport creates a new network transport with the given dialer
// and listener. The maxPool controls how many connections we will pool. The
// timeout is used to apply I/O deadlines. For InstallSnapshot, we multiply
// the timeout by (SnapshotSize / TimeoutScale).
func NewNetworkTransport(
	stream StreamLayer,
	maxPool int,
	timeout time.Duration,
	logOutput io.Writer,
) *NetworkTransport {
	if logOutput == nil {
		logOutput = os.Stderr
	}
	return NewNetworkTransportWithLogger(stream, maxPool, timeout, log.New(logOutput, "", log.LstdFlags))
}

// NewNetworkTransportWithLogger creates a new network transport with the given dialer
// and listener. The maxPool controls how many connections we will pool. The
// timeout is used to apply I/O deadlines. For InstallSnapshot, we multiply
// the timeout by (SnapshotSize / TimeoutScale).
func NewNetworkTransportWithLogger(
	stream StreamLayer,
	maxPool int,
	timeout time.Duration,
	logger *log.Logger,
) *NetworkTransport {
	if logger == nil {
		logger = log.New(os.Stderr, "", log.LstdFlags)
	}
	trans := &NetworkTransport{
		connPool:     make(map[ServerAddress][]*netConn),
		consumeCh:    make(chan RPC),
		logger:       logger,
		maxPool:      maxPool,
		shutdownCh:   make(chan struct{}),
		stream:       stream,
		timeout:      timeout,
		TimeoutScale: DefaultTimeoutScale,
	}
	go trans.listen()
	return trans
}

// SetHeartbeatHandler is used to setup a heartbeat handler
// as a fast-pass. This is to avoid head-of-line blocking from
// disk IO.
func (n *NetworkTransport) SetHeartbeatHandler(cb func(rpc RPC)) {
	n.heartbeatFnLock.Lock()
	defer n.heartbeatFnLock.Unlock()
	n.heartbeatFn = cb
}

// Close is used to stop the network transport.
func (n *NetworkTransport) Close() error {
	n.shutdownLock.Lock()
	defer n.shutdownLock.Unlock()

	if !n.shutdown {
		close(n.shutdownCh)
		n.stream.Close()
		n.shutdown = true
	}
	return nil
}

// Consumer implements the Transport interface.
func (n *NetworkTransport) Consumer() <-chan RPC {
	return n.consumeCh
}

// LocalAddr implements the Transport interface.
func (n *NetworkTransport) LocalAddr() ServerAddress {
	return ServerAddress(n.stream.Addr().String())
}

// IsShutdown is used to check if the transport is shutdown.
func (n *NetworkTransport) IsShutdown() bool {
	select {
	case <-n.shutdownCh:
		return true
	default:
		return false
	}
}

// getExistingConn is used to grab a pooled connection.
func (n *NetworkTransport) getPooledConn(target ServerAddress) *netConn {
	n.connPoolLock.Lock()
	defer n.connPoolLock.Unlock()

	conns, ok := n.connPool[target]
	if !ok || len(conns) == 0 {
		return nil
	}

	var conn *netConn
	num := len(conns)
	conn, conns[num-1] = conns[num-1], nil
	n.connPool[target] = conns[:num-1]
	return conn
}

// getConn is used to get a connection from the pool.
func (n *NetworkTransport) getConn(target ServerAddress) (*netConn, error) {
	// Check for a pooled conn
	if conn := n.getPooledConn(target); conn != nil {
		return conn, nil
	}

	// Dial a new connection
	conn, err := n.stream.Dial(target, n.timeout)
	if err != nil {
		return nil, err
	}

	// Wrap the conn
	netConn := &netConn{
		target: target,
		conn:   conn,
		r:      bufio.NewReader(conn),
		w:      bufio.NewWriter(conn),
	}

	// Setup encoder/decoders
	netConn.dec = codec.NewDecoder(netConn.r, &codec.MsgpackHandle{})
	netConn.enc = codec.NewEncoder(netConn.w, &codec.MsgpackHandle{})

	// Done
	return netConn, nil
}

// returnConn returns a connection back to the pool.
func (n *NetworkTransport) returnConn(conn *netConn) {
	n.connPoolLock.Lock()
	defer n.connPoolLock.Unlock()

	key := conn.target
	conns, _ := n.connPool[key]

	if !n.IsShutdown() && len(conns) < n.maxPool {
		n.connPool[key] = append(conns, conn)
	} else {
		conn.Release()
	}
}

// AppendEntriesPipeline returns an interface that can be used to pipeline
// AppendEntries requests.
func (n *NetworkTransport) AppendEntriesPipeline(target ServerAddress) (AppendPipeline, error) {
	// Get a connection
	conn, err := n.getConn(target)
	if err != nil {
		return nil, err
	}

	// Create the pipeline
	return newNetPipeline(n, conn), nil
}

// AppendEntries implements the Transport interface.
func (n *NetworkTransport) AppendEntries(target ServerAddress, args *AppendEntriesRequest, resp *AppendEntriesResponse) error {
	return n.genericRPC(target, rpcAppendEntries, args, resp)
}

// RequestVote implements the Transport interface.
func (n *NetworkTransport) RequestVote(target ServerAddress, args *RequestVoteRequest, resp *RequestVoteResponse) error {
	return n.genericRPC(target, rpcRequestVote, args, resp)
}

// genericRPC handles a simple request/response RPC.
func (n *NetworkTransport) genericRPC(target ServerAddress, rpcType uint8, args interface{}, resp interface{}) error {
	// Get a conn
	conn, err := n.getConn(target)
	if err != nil {
		return err
	}

	// Set a deadline
	if n.timeout > 0 {
		conn.conn.SetDeadline(time.Now().Add(n.timeout))
	}

	// Send the RPC
	if err = sendRPC(conn, rpcType, args); err != nil {
		return err
	}

	// Decode the response
	canReturn, err := decodeResponse(conn, resp)
	if canReturn {
		n.returnConn(conn)
	}
	return err
}

// InstallSnapshot implements the Transport interface.
func (n *NetworkTransport) InstallSnapshot(target ServerAddress, args *InstallSnapshotRequest, resp *InstallSnapshotResponse, data io.Reader) error {
	// Get a conn, always close for InstallSnapshot
	conn, err := n.getConn(target)
	if err != nil {
		return err
	}
	defer conn.Release()

	// Set a deadline, scaled by request size
	if n.timeout > 0 {
		timeout := n.timeout * time.Duration(args.Size/int64(n.TimeoutScale))
		if timeout < n.timeout {
			timeout = n.timeout
		}
		conn.conn.SetDeadline(time.Now().Add(timeout))
	}

	// Send the RPC
	if err = sendRPC(conn, rpcInstallSnapshot, args); err != nil {
		return err
	}

	// Stream the state
	if _, err = io.Copy(conn.w, data); err != nil {
		return err
	}

	// Flush
	if err = conn.w.Flush(); err != nil {
		return err
	}

	// Decode the response, do not return conn
	_, err = decodeResponse(conn, resp)
	return err
}

// EncodePeer implements the Transport interface.
func (n *NetworkTransport) EncodePeer(p ServerAddress) []byte {
	return []byte(p)
}

// DecodePeer implements the Transport interface.
func (n *NetworkTransport) DecodePeer(buf []byte) ServerAddress {
	return ServerAddress(buf)
}

// listen is used to handling incoming connections.
func (n *NetworkTransport) listen() {
	for {
		// Accept incoming connections
		conn, err := n.stream.Accept()
		if err != nil {
			if n.IsShutdown() {
				return
			}
			n.logger.Printf("[ERR] raft-net: Failed to accept connection: %v", err)
			continue
		}
		n.logger.Printf("[DEBUG] raft-net: %v accepted connection from: %v", n.LocalAddr(), conn.RemoteAddr())

		// Handle the connection in dedicated routine
		go n.handleConn(conn)
	}
}

// handleConn is used to handle an inbound connection for its lifespan.
func (n *NetworkTransport) handleConn(conn net.Conn) {
	defer conn.Close()
	r := bufio.NewReader(conn)
	w := bufio.NewWriter(conn)
	dec := codec.NewDecoder(r, &codec.MsgpackHandle{})
	enc := codec.NewEncoder(w, &codec.MsgpackHandle{})

	for {
		if err := n.handleCommand(r, dec, enc); err != nil {
			if err != io.EOF {
				n.logger.Printf("[ERR] raft-net: Failed to decode incoming command: %v", err)
			}
			return
		}
		if err := w.Flush(); err != nil {
			n.logger.Printf("[ERR] raft-net: Failed to flush response: %v", err)
			return
		}
	}
}

// handleCommand is used to decode and dispatch a single command.
func (n *NetworkTransport) handleCommand(r *bufio.Reader, dec *codec.Decoder, enc *codec.Encoder) error {
	// Get the rpc type
	rpcType, err := r.ReadByte()
	if err != nil {
		return err
	}

	// Create the RPC object
	respCh := make(chan RPCResponse, 1)
	rpc := RPC{
		RespChan: respCh,
	}

	// Decode the command
	isHeartbeat := false
	switch rpcType {
	case rpcAppendEntries:
		var req AppendEntriesRequest
		if err := dec.Decode(&req); err != nil {
			return err
		}
		rpc.Command = &req

		// Check if this is a heartbeat
		if req.Term != 0 && req.Leader != nil &&
			req.PrevLogEntry == 0 && req.PrevLogTerm == 0 &&
			len(req.Entries) == 0 && req.LeaderCommitIndex == 0 {
			isHeartbeat = true
		}

	case rpcRequestVote:
		var req RequestVoteRequest
		if err := dec.Decode(&req); err != nil {
			return err
		}
		rpc.Command = &req

	case rpcInstallSnapshot:
		var req InstallSnapshotRequest
		if err := dec.Decode(&req); err != nil {
			return err
		}
		rpc.Command = &req
		rpc.Reader = io.LimitReader(r, req.Size)

	default:
		return fmt.Errorf("unknown rpc type %d", rpcType)
	}

	// Check for heartbeat fast-path
	if isHeartbeat {
		n.heartbeatFnLock.Lock()
		fn := n.heartbeatFn
		n.heartbeatFnLock.Unlock()
		if fn != nil {
			fn(rpc)
			goto RESP
		}
	}

	// Dispatch the RPC
	select {
	case n.consumeCh <- rpc:
	case <-n.shutdownCh:
		return ErrTransportShutdown
	}

	// Wait for response
RESP:
	select {
	case resp := <-respCh:
		// Send the error first
		respErr := ""
		if resp.Error != nil {
			respErr = resp.Error.Error()
		}
		if err := enc.Encode(respErr); err != nil {
			return err
		}

		// Send the response
		if err := enc.Encode(resp.Response); err != nil {
			return err
		}
	case <-n.shutdownCh:
		return ErrTransportShutdown
	}
	return nil
}

// decodeResponse is used to decode an RPC response and reports whether
// the connection can be reused.
func decodeResponse(conn *netConn, resp interface{}) (bool, error) {
	// Decode the error if any
	var rpcError string
	if err := conn.dec.Decode(&rpcError); err != nil {
		conn.Release()
		return false, err
	}

	// Decode the response
	if err := conn.dec.Decode(resp); err != nil {
		conn.Release()
		return false, err
	}

	// Format an error if any
	if rpcError != "" {
		return true, fmt.Errorf(rpcError)
	}
	return true, nil
}

// sendRPC is used to encode and send the RPC.
func sendRPC(conn *netConn, rpcType uint8, args interface{}) error {
	// Write the request type
	if err := conn.w.WriteByte(rpcType); err != nil {
		conn.Release()
		return err
	}

	// Send the request
	if err := conn.enc.Encode(args); err != nil {
		conn.Release()
		return err
	}

	// Flush
	if err := conn.w.Flush(); err != nil {
		conn.Release()
		return err
	}
	return nil
}

// newNetPipeline is used to construct a netPipeline from a given
// transport and connection.
func newNetPipeline(trans *NetworkTransport, conn *netConn) *netPipeline {
	n := &netPipeline{
		conn:         conn,
		trans:        trans,
		doneCh:       make(chan AppendFuture, rpcMaxPipeline),
		inprogressCh: make(chan *appendFuture, rpcMaxPipeline),
		shutdownCh:   make(chan struct{}),
	}
	go n.decodeResponses()
	return n
}

// decodeResponses is a long running routine that decodes the responses
// sent on the connection.
func (n *netPipeline) decodeResponses() {
	timeout := n.trans.timeout
	for {
		select {
		case future := <-n.inprogressCh:
			if timeout > 0 {
				n.conn.conn.SetReadDeadline(time.Now().Add(timeout))
			}

			_, err := decodeResponse(n.conn, future.resp)
			future.respond(err)
			select {
			case n.doneCh <- future:
			case <-n.shutdownCh:
				return
			}
		case <-n.shutdownCh:
			return
		}
	}
}

// AppendEntries is used to pipeline a new append entries request.
func (n *netPipeline) AppendEntries(args *AppendEntriesRequest, resp *AppendEntriesResponse) (AppendFuture, error) {
	// Create a new future
	future := &appendFuture{
		start: time.Now(),
		args:  args,
		resp:  resp,
	}
	future.init()

	// Add a send timeout
	if timeout := n.trans.timeout; timeout > 0 {
		n.conn.conn.SetWriteDeadline(time.Now().Add(timeout))
	}

	// Send the RPC
	if err := sendRPC(n.conn, rpcAppendEntries, future.args); err != nil {
		return nil, err
	}

	// Hand-off for decoding, this can also cause back-pressure
	// to prevent too many inflight requests
	select {
	case n.inprogressCh <- future:
		return future, nil
	case <-n.shutdownCh:
		return nil, ErrPipelineShutdown
	}
}

// Consumer returns a channel that can be used to consume complete futures.
func (n *netPipeline) Consumer() <-chan AppendFuture {
	return n.doneCh
}

// Closed is used to shutdown the pipeline connection.
func (n *netPipeline) Close() error {
	n.shutdownLock.Lock()
	defer n.shutdownLock.Unlock()
	if n.shutdown {
		return nil
	}

	// Release the connection
	n.conn.Release()

	n.shutdown = true
	close(n.shutdownCh)
	return nil
}
