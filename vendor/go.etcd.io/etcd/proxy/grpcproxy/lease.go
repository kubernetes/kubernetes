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

package grpcproxy

import (
	"context"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"go.etcd.io/etcd/clientv3"
	"go.etcd.io/etcd/etcdserver/api/v3rpc/rpctypes"
	pb "go.etcd.io/etcd/etcdserver/etcdserverpb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

type leaseProxy struct {
	// leaseClient handles req from LeaseGrant() that requires a lease ID.
	leaseClient pb.LeaseClient

	lessor clientv3.Lease

	ctx context.Context

	leader *leader

	// mu protects adding outstanding leaseProxyStream through wg.
	mu sync.RWMutex

	// wg waits until all outstanding leaseProxyStream quit.
	wg sync.WaitGroup
}

func NewLeaseProxy(c *clientv3.Client) (pb.LeaseServer, <-chan struct{}) {
	cctx, cancel := context.WithCancel(c.Ctx())
	lp := &leaseProxy{
		leaseClient: pb.NewLeaseClient(c.ActiveConnection()),
		lessor:      c.Lease,
		ctx:         cctx,
		leader:      newLeader(c.Ctx(), c.Watcher),
	}
	ch := make(chan struct{})
	go func() {
		defer close(ch)
		<-lp.leader.stopNotify()
		lp.mu.Lock()
		select {
		case <-lp.ctx.Done():
		case <-lp.leader.disconnectNotify():
			cancel()
		}
		<-lp.ctx.Done()
		lp.mu.Unlock()
		lp.wg.Wait()
	}()
	return lp, ch
}

func (lp *leaseProxy) LeaseGrant(ctx context.Context, cr *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error) {
	rp, err := lp.leaseClient.LeaseGrant(ctx, cr, grpc.FailFast(false))
	if err != nil {
		return nil, err
	}
	lp.leader.gotLeader()
	return rp, nil
}

func (lp *leaseProxy) LeaseRevoke(ctx context.Context, rr *pb.LeaseRevokeRequest) (*pb.LeaseRevokeResponse, error) {
	r, err := lp.lessor.Revoke(ctx, clientv3.LeaseID(rr.ID))
	if err != nil {
		return nil, err
	}
	lp.leader.gotLeader()
	return (*pb.LeaseRevokeResponse)(r), nil
}

func (lp *leaseProxy) LeaseTimeToLive(ctx context.Context, rr *pb.LeaseTimeToLiveRequest) (*pb.LeaseTimeToLiveResponse, error) {
	var (
		r   *clientv3.LeaseTimeToLiveResponse
		err error
	)
	if rr.Keys {
		r, err = lp.lessor.TimeToLive(ctx, clientv3.LeaseID(rr.ID), clientv3.WithAttachedKeys())
	} else {
		r, err = lp.lessor.TimeToLive(ctx, clientv3.LeaseID(rr.ID))
	}
	if err != nil {
		return nil, err
	}
	rp := &pb.LeaseTimeToLiveResponse{
		Header:     r.ResponseHeader,
		ID:         int64(r.ID),
		TTL:        r.TTL,
		GrantedTTL: r.GrantedTTL,
		Keys:       r.Keys,
	}
	return rp, err
}

func (lp *leaseProxy) LeaseLeases(ctx context.Context, rr *pb.LeaseLeasesRequest) (*pb.LeaseLeasesResponse, error) {
	r, err := lp.lessor.Leases(ctx)
	if err != nil {
		return nil, err
	}
	leases := make([]*pb.LeaseStatus, len(r.Leases))
	for i := range r.Leases {
		leases[i] = &pb.LeaseStatus{ID: int64(r.Leases[i].ID)}
	}
	rp := &pb.LeaseLeasesResponse{
		Header: r.ResponseHeader,
		Leases: leases,
	}
	return rp, err
}

func (lp *leaseProxy) LeaseKeepAlive(stream pb.Lease_LeaseKeepAliveServer) error {
	lp.mu.Lock()
	select {
	case <-lp.ctx.Done():
		lp.mu.Unlock()
		return lp.ctx.Err()
	default:
		lp.wg.Add(1)
	}
	lp.mu.Unlock()

	ctx, cancel := context.WithCancel(stream.Context())
	lps := leaseProxyStream{
		stream:          stream,
		lessor:          lp.lessor,
		keepAliveLeases: make(map[int64]*atomicCounter),
		respc:           make(chan *pb.LeaseKeepAliveResponse),
		ctx:             ctx,
		cancel:          cancel,
	}

	errc := make(chan error, 2)

	var lostLeaderC <-chan struct{}
	if md, ok := metadata.FromOutgoingContext(stream.Context()); ok {
		v := md[rpctypes.MetadataRequireLeaderKey]
		if len(v) > 0 && v[0] == rpctypes.MetadataHasLeader {
			lostLeaderC = lp.leader.lostNotify()
			// if leader is known to be lost at creation time, avoid
			// letting events through at all
			select {
			case <-lostLeaderC:
				lp.wg.Done()
				return rpctypes.ErrNoLeader
			default:
			}
		}
	}
	stopc := make(chan struct{}, 3)
	go func() {
		defer func() { stopc <- struct{}{} }()
		if err := lps.recvLoop(); err != nil {
			errc <- err
		}
	}()

	go func() {
		defer func() { stopc <- struct{}{} }()
		if err := lps.sendLoop(); err != nil {
			errc <- err
		}
	}()

	// tears down LeaseKeepAlive stream if leader goes down or entire leaseProxy is terminated.
	go func() {
		defer func() { stopc <- struct{}{} }()
		select {
		case <-lostLeaderC:
		case <-ctx.Done():
		case <-lp.ctx.Done():
		}
	}()

	var err error
	select {
	case <-stopc:
		stopc <- struct{}{}
	case err = <-errc:
	}
	cancel()

	// recv/send may only shutdown after function exits;
	// this goroutine notifies lease proxy that the stream is through
	go func() {
		<-stopc
		<-stopc
		<-stopc
		lps.close()
		close(errc)
		lp.wg.Done()
	}()

	select {
	case <-lostLeaderC:
		return rpctypes.ErrNoLeader
	case <-lp.leader.disconnectNotify():
		return status.Error(codes.Canceled, "the client connection is closing")
	default:
		if err != nil {
			return err
		}
		return ctx.Err()
	}
}

type leaseProxyStream struct {
	stream pb.Lease_LeaseKeepAliveServer

	lessor clientv3.Lease
	// wg tracks keepAliveLoop goroutines
	wg sync.WaitGroup
	// mu protects keepAliveLeases
	mu sync.RWMutex
	// keepAliveLeases tracks how many outstanding keepalive requests which need responses are on a lease.
	keepAliveLeases map[int64]*atomicCounter
	// respc receives lease keepalive responses from etcd backend
	respc chan *pb.LeaseKeepAliveResponse

	ctx    context.Context
	cancel context.CancelFunc
}

func (lps *leaseProxyStream) recvLoop() error {
	for {
		rr, err := lps.stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		lps.mu.Lock()
		neededResps, ok := lps.keepAliveLeases[rr.ID]
		if !ok {
			neededResps = &atomicCounter{}
			lps.keepAliveLeases[rr.ID] = neededResps
			lps.wg.Add(1)
			go func() {
				defer lps.wg.Done()
				if err := lps.keepAliveLoop(rr.ID, neededResps); err != nil {
					lps.cancel()
				}
			}()
		}
		neededResps.add(1)
		lps.mu.Unlock()
	}
}

func (lps *leaseProxyStream) keepAliveLoop(leaseID int64, neededResps *atomicCounter) error {
	cctx, ccancel := context.WithCancel(lps.ctx)
	defer ccancel()
	respc, err := lps.lessor.KeepAlive(cctx, clientv3.LeaseID(leaseID))
	if err != nil {
		return err
	}
	// ticker expires when loop hasn't received keepalive within TTL
	var ticker <-chan time.Time
	for {
		select {
		case <-ticker:
			lps.mu.Lock()
			// if there are outstanding keepAlive reqs at the moment of ticker firing,
			// don't close keepAliveLoop(), let it continuing to process the KeepAlive reqs.
			if neededResps.get() > 0 {
				lps.mu.Unlock()
				ticker = nil
				continue
			}
			delete(lps.keepAliveLeases, leaseID)
			lps.mu.Unlock()
			return nil
		case rp, ok := <-respc:
			if !ok {
				lps.mu.Lock()
				delete(lps.keepAliveLeases, leaseID)
				lps.mu.Unlock()
				if neededResps.get() == 0 {
					return nil
				}
				ttlResp, err := lps.lessor.TimeToLive(cctx, clientv3.LeaseID(leaseID))
				if err != nil {
					return err
				}
				r := &pb.LeaseKeepAliveResponse{
					Header: ttlResp.ResponseHeader,
					ID:     int64(ttlResp.ID),
					TTL:    ttlResp.TTL,
				}
				for neededResps.get() > 0 {
					select {
					case lps.respc <- r:
						neededResps.add(-1)
					case <-lps.ctx.Done():
						return nil
					}
				}
				return nil
			}
			if neededResps.get() == 0 {
				continue
			}
			ticker = time.After(time.Duration(rp.TTL) * time.Second)
			r := &pb.LeaseKeepAliveResponse{
				Header: rp.ResponseHeader,
				ID:     int64(rp.ID),
				TTL:    rp.TTL,
			}
			lps.replyToClient(r, neededResps)
		}
	}
}

func (lps *leaseProxyStream) replyToClient(r *pb.LeaseKeepAliveResponse, neededResps *atomicCounter) {
	timer := time.After(500 * time.Millisecond)
	for neededResps.get() > 0 {
		select {
		case lps.respc <- r:
			neededResps.add(-1)
		case <-timer:
			return
		case <-lps.ctx.Done():
			return
		}
	}
}

func (lps *leaseProxyStream) sendLoop() error {
	for {
		select {
		case lrp, ok := <-lps.respc:
			if !ok {
				return nil
			}
			if err := lps.stream.Send(lrp); err != nil {
				return err
			}
		case <-lps.ctx.Done():
			return lps.ctx.Err()
		}
	}
}

func (lps *leaseProxyStream) close() {
	lps.cancel()
	lps.wg.Wait()
	// only close respc channel if all the keepAliveLoop() goroutines have finished
	// this ensures those goroutines don't send resp to a closed resp channel
	close(lps.respc)
}

type atomicCounter struct {
	counter int64
}

func (ac *atomicCounter) add(delta int64) {
	atomic.AddInt64(&ac.counter, delta)
}

func (ac *atomicCounter) get() int64 {
	return atomic.LoadInt64(&ac.counter)
}
