package raft

import (
	"fmt"
	"io"
	"sync"
	"time"
)

// NewInmemAddr returns a new in-memory addr with
// a randomly generate UUID as the ID.
func NewInmemAddr() string {
	return generateUUID()
}

// inmemPipeline is used to pipeline requests for the in-mem transport.
type inmemPipeline struct {
	trans    *InmemTransport
	peer     *InmemTransport
	peerAddr string

	doneCh       chan AppendFuture
	inprogressCh chan *inmemPipelineInflight

	shutdown     bool
	shutdownCh   chan struct{}
	shutdownLock sync.Mutex
}

type inmemPipelineInflight struct {
	future *appendFuture
	respCh <-chan RPCResponse
}

// InmemTransport Implements the Transport interface, to allow Raft to be
// tested in-memory without going over a network.
type InmemTransport struct {
	sync.RWMutex
	consumerCh chan RPC
	localAddr  string
	peers      map[string]*InmemTransport
	pipelines  []*inmemPipeline
	timeout    time.Duration
}

// NewInmemTransport is used to initialize a new transport
// and generates a random local address if none is specified
func NewInmemTransport(addr string) (string, *InmemTransport) {
	if addr == "" {
		addr = NewInmemAddr()
	}
	trans := &InmemTransport{
		consumerCh: make(chan RPC, 16),
		localAddr:  addr,
		peers:      make(map[string]*InmemTransport),
		timeout:    50 * time.Millisecond,
	}
	return addr, trans
}

// SetHeartbeatHandler is used to set optional fast-path for
// heartbeats, not supported for this transport.
func (i *InmemTransport) SetHeartbeatHandler(cb func(RPC)) {
}

// Consumer implements the Transport interface.
func (i *InmemTransport) Consumer() <-chan RPC {
	return i.consumerCh
}

// LocalAddr implements the Transport interface.
func (i *InmemTransport) LocalAddr() string {
	return i.localAddr
}

// AppendEntriesPipeline returns an interface that can be used to pipeline
// AppendEntries requests.
func (i *InmemTransport) AppendEntriesPipeline(target string) (AppendPipeline, error) {
	i.RLock()
	peer, ok := i.peers[target]
	i.RUnlock()
	if !ok {
		return nil, fmt.Errorf("failed to connect to peer: %v", target)
	}
	pipeline := newInmemPipeline(i, peer, target)
	i.Lock()
	i.pipelines = append(i.pipelines, pipeline)
	i.Unlock()
	return pipeline, nil
}

// AppendEntries implements the Transport interface.
func (i *InmemTransport) AppendEntries(target string, args *AppendEntriesRequest, resp *AppendEntriesResponse) error {
	rpcResp, err := i.makeRPC(target, args, nil, i.timeout)
	if err != nil {
		return err
	}

	// Copy the result back
	out := rpcResp.Response.(*AppendEntriesResponse)
	*resp = *out
	return nil
}

// RequestVote implements the Transport interface.
func (i *InmemTransport) RequestVote(target string, args *RequestVoteRequest, resp *RequestVoteResponse) error {
	rpcResp, err := i.makeRPC(target, args, nil, i.timeout)
	if err != nil {
		return err
	}

	// Copy the result back
	out := rpcResp.Response.(*RequestVoteResponse)
	*resp = *out
	return nil
}

// InstallSnapshot implements the Transport interface.
func (i *InmemTransport) InstallSnapshot(target string, args *InstallSnapshotRequest, resp *InstallSnapshotResponse, data io.Reader) error {
	rpcResp, err := i.makeRPC(target, args, data, 10*i.timeout)
	if err != nil {
		return err
	}

	// Copy the result back
	out := rpcResp.Response.(*InstallSnapshotResponse)
	*resp = *out
	return nil
}

func (i *InmemTransport) makeRPC(target string, args interface{}, r io.Reader, timeout time.Duration) (rpcResp RPCResponse, err error) {
	i.RLock()
	peer, ok := i.peers[target]
	i.RUnlock()

	if !ok {
		err = fmt.Errorf("failed to connect to peer: %v", target)
		return
	}

	// Send the RPC over
	respCh := make(chan RPCResponse)
	peer.consumerCh <- RPC{
		Command:  args,
		Reader:   r,
		RespChan: respCh,
	}

	// Wait for a response
	select {
	case rpcResp = <-respCh:
		if rpcResp.Error != nil {
			err = rpcResp.Error
		}
	case <-time.After(timeout):
		err = fmt.Errorf("command timed out")
	}
	return
}

// EncodePeer implements the Transport interface. It uses the UUID as the
// address directly.
func (i *InmemTransport) EncodePeer(p string) []byte {
	return []byte(p)
}

// DecodePeer implements the Transport interface. It wraps the UUID in an
// InmemAddr.
func (i *InmemTransport) DecodePeer(buf []byte) string {
	return string(buf)
}

// Connect is used to connect this transport to another transport for
// a given peer name. This allows for local routing.
func (i *InmemTransport) Connect(peer string, t Transport) {
	trans := t.(*InmemTransport)
	i.Lock()
	defer i.Unlock()
	i.peers[peer] = trans
}

// Disconnect is used to remove the ability to route to a given peer.
func (i *InmemTransport) Disconnect(peer string) {
	i.Lock()
	defer i.Unlock()
	delete(i.peers, peer)

	// Disconnect any pipelines
	n := len(i.pipelines)
	for idx := 0; idx < n; idx++ {
		if i.pipelines[idx].peerAddr == peer {
			i.pipelines[idx].Close()
			i.pipelines[idx], i.pipelines[n-1] = i.pipelines[n-1], nil
			idx--
			n--
		}
	}
	i.pipelines = i.pipelines[:n]
}

// DisconnectAll is used to remove all routes to peers.
func (i *InmemTransport) DisconnectAll() {
	i.Lock()
	defer i.Unlock()
	i.peers = make(map[string]*InmemTransport)

	// Handle pipelines
	for _, pipeline := range i.pipelines {
		pipeline.Close()
	}
	i.pipelines = nil
}

// Close is used to permanently disable the transport
func (i *InmemTransport) Close() error {
	i.DisconnectAll()
	return nil
}

func newInmemPipeline(trans *InmemTransport, peer *InmemTransport, addr string) *inmemPipeline {
	i := &inmemPipeline{
		trans:        trans,
		peer:         peer,
		peerAddr:     addr,
		doneCh:       make(chan AppendFuture, 16),
		inprogressCh: make(chan *inmemPipelineInflight, 16),
		shutdownCh:   make(chan struct{}),
	}
	go i.decodeResponses()
	return i
}

func (i *inmemPipeline) decodeResponses() {
	timeout := i.trans.timeout
	for {
		select {
		case inp := <-i.inprogressCh:
			var timeoutCh <-chan time.Time
			if timeout > 0 {
				timeoutCh = time.After(timeout)
			}

			select {
			case rpcResp := <-inp.respCh:
				// Copy the result back
				*inp.future.resp = *rpcResp.Response.(*AppendEntriesResponse)
				inp.future.respond(rpcResp.Error)

				select {
				case i.doneCh <- inp.future:
				case <-i.shutdownCh:
					return
				}

			case <-timeoutCh:
				inp.future.respond(fmt.Errorf("command timed out"))
				select {
				case i.doneCh <- inp.future:
				case <-i.shutdownCh:
					return
				}

			case <-i.shutdownCh:
				return
			}
		case <-i.shutdownCh:
			return
		}
	}
}

func (i *inmemPipeline) AppendEntries(args *AppendEntriesRequest, resp *AppendEntriesResponse) (AppendFuture, error) {
	// Create a new future
	future := &appendFuture{
		start: time.Now(),
		args:  args,
		resp:  resp,
	}
	future.init()

	// Handle a timeout
	var timeout <-chan time.Time
	if i.trans.timeout > 0 {
		timeout = time.After(i.trans.timeout)
	}

	// Send the RPC over
	respCh := make(chan RPCResponse, 1)
	rpc := RPC{
		Command:  args,
		RespChan: respCh,
	}
	select {
	case i.peer.consumerCh <- rpc:
	case <-timeout:
		return nil, fmt.Errorf("command enqueue timeout")
	case <-i.shutdownCh:
		return nil, ErrPipelineShutdown
	}

	// Send to be decoded
	select {
	case i.inprogressCh <- &inmemPipelineInflight{future, respCh}:
		return future, nil
	case <-i.shutdownCh:
		return nil, ErrPipelineShutdown
	}
}

func (i *inmemPipeline) Consumer() <-chan AppendFuture {
	return i.doneCh
}

func (i *inmemPipeline) Close() error {
	i.shutdownLock.Lock()
	defer i.shutdownLock.Unlock()
	if i.shutdown {
		return nil
	}

	i.shutdown = true
	close(i.shutdownCh)
	return nil
}
