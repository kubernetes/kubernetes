// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package clientv3

import (
	"sync"
	"time"

	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

type (
	LeaseRevokeResponse pb.LeaseRevokeResponse
	LeaseID             int64
)

// LeaseGrantResponse is used to convert the protobuf grant response.
type LeaseGrantResponse struct {
	*pb.ResponseHeader
	ID    LeaseID
	TTL   int64
	Error string
}

// LeaseKeepAliveResponse is used to convert the protobuf keepalive response.
type LeaseKeepAliveResponse struct {
	*pb.ResponseHeader
	ID  LeaseID
	TTL int64
}

const (
	// defaultTTL is the assumed lease TTL used for the first keepalive
	// deadline before the actual TTL is known to the client.
	defaultTTL = 5 * time.Second
	// a small buffer to store unsent lease responses.
	leaseResponseChSize = 16
	// NoLease is a lease ID for the absence of a lease.
	NoLease LeaseID = 0
)

type Lease interface {
	// Grant creates a new lease.
	Grant(ctx context.Context, ttl int64) (*LeaseGrantResponse, error)

	// Revoke revokes the given lease.
	Revoke(ctx context.Context, id LeaseID) (*LeaseRevokeResponse, error)

	// KeepAlive keeps the given lease alive forever.
	KeepAlive(ctx context.Context, id LeaseID) (<-chan *LeaseKeepAliveResponse, error)

	// KeepAliveOnce renews the lease once. In most of the cases, Keepalive
	// should be used instead of KeepAliveOnce.
	KeepAliveOnce(ctx context.Context, id LeaseID) (*LeaseKeepAliveResponse, error)

	// Close releases all resources Lease keeps for efficient communication
	// with the etcd server.
	Close() error
}

type lessor struct {
	mu sync.Mutex // guards all fields

	// donec is closed when recvKeepAliveLoop stops
	donec chan struct{}

	remote pb.LeaseClient

	stream       pb.Lease_LeaseKeepAliveClient
	streamCancel context.CancelFunc

	stopCtx    context.Context
	stopCancel context.CancelFunc

	keepAlives map[LeaseID]*keepAlive

	// firstKeepAliveTimeout is the timeout for the first keepalive request
	// before the actual TTL is known to the lease client
	firstKeepAliveTimeout time.Duration
}

// keepAlive multiplexes a keepalive for a lease over multiple channels
type keepAlive struct {
	chs  []chan<- *LeaseKeepAliveResponse
	ctxs []context.Context
	// deadline is the time the keep alive channels close if no response
	deadline time.Time
	// nextKeepAlive is when to send the next keep alive message
	nextKeepAlive time.Time
	// donec is closed on lease revoke, expiration, or cancel.
	donec chan struct{}
}

func NewLease(c *Client) Lease {
	l := &lessor{
		donec:                 make(chan struct{}),
		keepAlives:            make(map[LeaseID]*keepAlive),
		remote:                RetryLeaseClient(c),
		firstKeepAliveTimeout: c.cfg.DialTimeout + time.Second,
	}
	if l.firstKeepAliveTimeout == time.Second {
		l.firstKeepAliveTimeout = defaultTTL
	}

	l.stopCtx, l.stopCancel = context.WithCancel(context.Background())
	go l.recvKeepAliveLoop()
	go l.deadlineLoop()
	return l
}

func (l *lessor) Grant(ctx context.Context, ttl int64) (*LeaseGrantResponse, error) {
	cctx, cancel := context.WithCancel(ctx)
	done := cancelWhenStop(cancel, l.stopCtx.Done())
	defer close(done)

	for {
		r := &pb.LeaseGrantRequest{TTL: ttl}
		resp, err := l.remote.LeaseGrant(cctx, r)
		if err == nil {
			gresp := &LeaseGrantResponse{
				ResponseHeader: resp.GetHeader(),
				ID:             LeaseID(resp.ID),
				TTL:            resp.TTL,
				Error:          resp.Error,
			}
			return gresp, nil
		}
		if isHaltErr(cctx, err) {
			return nil, toErr(ctx, err)
		}
		if nerr := l.newStream(); nerr != nil {
			return nil, nerr
		}
	}
}

func (l *lessor) Revoke(ctx context.Context, id LeaseID) (*LeaseRevokeResponse, error) {
	cctx, cancel := context.WithCancel(ctx)
	done := cancelWhenStop(cancel, l.stopCtx.Done())
	defer close(done)

	for {
		r := &pb.LeaseRevokeRequest{ID: int64(id)}
		resp, err := l.remote.LeaseRevoke(cctx, r)

		if err == nil {
			return (*LeaseRevokeResponse)(resp), nil
		}
		if isHaltErr(ctx, err) {
			return nil, toErr(ctx, err)
		}
		if nerr := l.newStream(); nerr != nil {
			return nil, nerr
		}
	}
}

func (l *lessor) KeepAlive(ctx context.Context, id LeaseID) (<-chan *LeaseKeepAliveResponse, error) {
	ch := make(chan *LeaseKeepAliveResponse, leaseResponseChSize)

	l.mu.Lock()
	ka, ok := l.keepAlives[id]
	if !ok {
		// create fresh keep alive
		ka = &keepAlive{
			chs:           []chan<- *LeaseKeepAliveResponse{ch},
			ctxs:          []context.Context{ctx},
			deadline:      time.Now().Add(l.firstKeepAliveTimeout),
			nextKeepAlive: time.Now(),
			donec:         make(chan struct{}),
		}
		l.keepAlives[id] = ka
	} else {
		// add channel and context to existing keep alive
		ka.ctxs = append(ka.ctxs, ctx)
		ka.chs = append(ka.chs, ch)
	}
	l.mu.Unlock()

	go l.keepAliveCtxCloser(id, ctx, ka.donec)

	return ch, nil
}

func (l *lessor) KeepAliveOnce(ctx context.Context, id LeaseID) (*LeaseKeepAliveResponse, error) {
	cctx, cancel := context.WithCancel(ctx)
	done := cancelWhenStop(cancel, l.stopCtx.Done())
	defer close(done)

	for {
		resp, err := l.keepAliveOnce(cctx, id)
		if err == nil {
			if resp.TTL == 0 {
				err = rpctypes.ErrLeaseNotFound
			}
			return resp, err
		}
		if isHaltErr(ctx, err) {
			return nil, toErr(ctx, err)
		}

		if nerr := l.newStream(); nerr != nil {
			return nil, nerr
		}
	}
}

func (l *lessor) Close() error {
	l.stopCancel()
	<-l.donec
	return nil
}

func (l *lessor) keepAliveCtxCloser(id LeaseID, ctx context.Context, donec <-chan struct{}) {
	select {
	case <-donec:
		return
	case <-l.donec:
		return
	case <-ctx.Done():
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	ka, ok := l.keepAlives[id]
	if !ok {
		return
	}

	// close channel and remove context if still associated with keep alive
	for i, c := range ka.ctxs {
		if c == ctx {
			close(ka.chs[i])
			ka.ctxs = append(ka.ctxs[:i], ka.ctxs[i+1:]...)
			ka.chs = append(ka.chs[:i], ka.chs[i+1:]...)
			break
		}
	}
	// remove if no one more listeners
	if len(ka.chs) == 0 {
		delete(l.keepAlives, id)
	}
}

func (l *lessor) keepAliveOnce(ctx context.Context, id LeaseID) (*LeaseKeepAliveResponse, error) {
	cctx, cancel := context.WithCancel(ctx)
	defer cancel()

	stream, err := l.remote.LeaseKeepAlive(cctx, grpc.FailFast(false))
	if err != nil {
		return nil, toErr(ctx, err)
	}

	err = stream.Send(&pb.LeaseKeepAliveRequest{ID: int64(id)})
	if err != nil {
		return nil, toErr(ctx, err)
	}

	resp, rerr := stream.Recv()
	if rerr != nil {
		return nil, toErr(ctx, rerr)
	}

	karesp := &LeaseKeepAliveResponse{
		ResponseHeader: resp.GetHeader(),
		ID:             LeaseID(resp.ID),
		TTL:            resp.TTL,
	}
	return karesp, nil
}

func (l *lessor) recvKeepAliveLoop() {
	defer func() {
		l.mu.Lock()
		close(l.donec)
		for _, ka := range l.keepAlives {
			ka.Close()
		}
		l.keepAlives = make(map[LeaseID]*keepAlive)
		l.mu.Unlock()
	}()

	stream, serr := l.resetRecv()
	for serr == nil {
		resp, err := stream.Recv()
		if err != nil {
			if isHaltErr(l.stopCtx, err) {
				return
			}
			stream, serr = l.resetRecv()
			continue
		}
		l.recvKeepAlive(resp)
	}
}

// resetRecv opens a new lease stream and starts sending LeaseKeepAliveRequests
func (l *lessor) resetRecv() (pb.Lease_LeaseKeepAliveClient, error) {
	if err := l.newStream(); err != nil {
		return nil, err
	}
	stream := l.getKeepAliveStream()
	go l.sendKeepAliveLoop(stream)
	return stream, nil
}

// recvKeepAlive updates a lease based on its LeaseKeepAliveResponse
func (l *lessor) recvKeepAlive(resp *pb.LeaseKeepAliveResponse) {
	karesp := &LeaseKeepAliveResponse{
		ResponseHeader: resp.GetHeader(),
		ID:             LeaseID(resp.ID),
		TTL:            resp.TTL,
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	ka, ok := l.keepAlives[karesp.ID]
	if !ok {
		return
	}

	if karesp.TTL <= 0 {
		// lease expired; close all keep alive channels
		delete(l.keepAlives, karesp.ID)
		ka.Close()
		return
	}

	// send update to all channels
	nextKeepAlive := time.Now().Add(1 + time.Duration(karesp.TTL/3)*time.Second)
	ka.deadline = time.Now().Add(time.Duration(karesp.TTL) * time.Second)
	for _, ch := range ka.chs {
		select {
		case ch <- karesp:
			ka.nextKeepAlive = nextKeepAlive
		default:
		}
	}
}

// deadlineLoop reaps any keep alive channels that have not received a response
// within the lease TTL
func (l *lessor) deadlineLoop() {
	for {
		select {
		case <-time.After(time.Second):
		case <-l.donec:
			return
		}
		now := time.Now()
		l.mu.Lock()
		for id, ka := range l.keepAlives {
			if ka.deadline.Before(now) {
				// waited too long for response; lease may be expired
				ka.Close()
				delete(l.keepAlives, id)
			}
		}
		l.mu.Unlock()
	}
}

// sendKeepAliveLoop sends LeaseKeepAliveRequests for the lifetime of a lease stream
func (l *lessor) sendKeepAliveLoop(stream pb.Lease_LeaseKeepAliveClient) {
	for {
		select {
		case <-time.After(500 * time.Millisecond):
		case <-stream.Context().Done():
			return
		case <-l.donec:
			return
		case <-l.stopCtx.Done():
			return
		}

		tosend := make([]LeaseID, 0)

		now := time.Now()
		l.mu.Lock()
		for id, ka := range l.keepAlives {
			if ka.nextKeepAlive.Before(now) {
				tosend = append(tosend, id)
			}
		}
		l.mu.Unlock()

		for _, id := range tosend {
			r := &pb.LeaseKeepAliveRequest{ID: int64(id)}
			if err := stream.Send(r); err != nil {
				// TODO do something with this error?
				return
			}
		}
	}
}

func (l *lessor) getKeepAliveStream() pb.Lease_LeaseKeepAliveClient {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.stream
}

func (l *lessor) newStream() error {
	sctx, cancel := context.WithCancel(l.stopCtx)
	stream, err := l.remote.LeaseKeepAlive(sctx, grpc.FailFast(false))
	if err != nil {
		cancel()
		return toErr(sctx, err)
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	if l.stream != nil && l.streamCancel != nil {
		l.stream.CloseSend()
		l.streamCancel()
	}

	l.streamCancel = cancel
	l.stream = stream
	return nil
}

func (ka *keepAlive) Close() {
	close(ka.donec)
	for _, ch := range ka.chs {
		close(ch)
	}
}

// cancelWhenStop calls cancel when the given stopc fires. It returns a done chan. done
// should be closed when the work is finished. When done fires, cancelWhenStop will release
// its internal resource.
func cancelWhenStop(cancel context.CancelFunc, stopc <-chan struct{}) chan<- struct{} {
	done := make(chan struct{}, 1)

	go func() {
		select {
		case <-stopc:
		case <-done:
		}
		cancel()
	}()

	return done
}
