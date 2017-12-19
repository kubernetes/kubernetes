/*
 *
 * Copyright 2016, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package grpc

import (
	"errors"
	"fmt"
	"math/rand"
	"net"
	"sync"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc/codes"
	lbpb "google.golang.org/grpc/grpclb/grpc_lb_v1"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/naming"
)

// Client API for LoadBalancer service.
// Mostly copied from generated pb.go file.
// To avoid circular dependency.
type loadBalancerClient struct {
	cc *ClientConn
}

func (c *loadBalancerClient) BalanceLoad(ctx context.Context, opts ...CallOption) (*balanceLoadClientStream, error) {
	desc := &StreamDesc{
		StreamName:    "BalanceLoad",
		ServerStreams: true,
		ClientStreams: true,
	}
	stream, err := NewClientStream(ctx, desc, c.cc, "/grpc.lb.v1.LoadBalancer/BalanceLoad", opts...)
	if err != nil {
		return nil, err
	}
	x := &balanceLoadClientStream{stream}
	return x, nil
}

type balanceLoadClientStream struct {
	ClientStream
}

func (x *balanceLoadClientStream) Send(m *lbpb.LoadBalanceRequest) error {
	return x.ClientStream.SendMsg(m)
}

func (x *balanceLoadClientStream) Recv() (*lbpb.LoadBalanceResponse, error) {
	m := new(lbpb.LoadBalanceResponse)
	if err := x.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

// AddressType indicates the address type returned by name resolution.
type AddressType uint8

const (
	// Backend indicates the server is a backend server.
	Backend AddressType = iota
	// GRPCLB indicates the server is a grpclb load balancer.
	GRPCLB
)

// AddrMetadataGRPCLB contains the information the name resolution for grpclb should provide. The
// name resolver used by grpclb balancer is required to provide this type of metadata in
// its address updates.
type AddrMetadataGRPCLB struct {
	// AddrType is the type of server (grpc load balancer or backend).
	AddrType AddressType
	// ServerName is the name of the grpc load balancer. Used for authentication.
	ServerName string
}

// NewGRPCLBBalancer creates a grpclb load balancer.
func NewGRPCLBBalancer(r naming.Resolver) Balancer {
	return &balancer{
		r: r,
	}
}

type remoteBalancerInfo struct {
	addr string
	// the server name used for authentication with the remote LB server.
	name string
}

// grpclbAddrInfo consists of the information of a backend server.
type grpclbAddrInfo struct {
	addr      Address
	connected bool
	// dropForRateLimiting indicates whether this particular request should be
	// dropped by the client for rate limiting.
	dropForRateLimiting bool
	// dropForLoadBalancing indicates whether this particular request should be
	// dropped by the client for load balancing.
	dropForLoadBalancing bool
}

type balancer struct {
	r        naming.Resolver
	target   string
	mu       sync.Mutex
	seq      int // a sequence number to make sure addrCh does not get stale addresses.
	w        naming.Watcher
	addrCh   chan []Address
	rbs      []remoteBalancerInfo
	addrs    []*grpclbAddrInfo
	next     int
	waitCh   chan struct{}
	done     bool
	expTimer *time.Timer
	rand     *rand.Rand

	clientStats lbpb.ClientStats
}

func (b *balancer) watchAddrUpdates(w naming.Watcher, ch chan []remoteBalancerInfo) error {
	updates, err := w.Next()
	if err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.done {
		return ErrClientConnClosing
	}
	for _, update := range updates {
		switch update.Op {
		case naming.Add:
			var exist bool
			for _, v := range b.rbs {
				// TODO: Is the same addr with different server name a different balancer?
				if update.Addr == v.addr {
					exist = true
					break
				}
			}
			if exist {
				continue
			}
			md, ok := update.Metadata.(*AddrMetadataGRPCLB)
			if !ok {
				// TODO: Revisit the handling here and may introduce some fallback mechanism.
				grpclog.Printf("The name resolution contains unexpected metadata %v", update.Metadata)
				continue
			}
			switch md.AddrType {
			case Backend:
				// TODO: Revisit the handling here and may introduce some fallback mechanism.
				grpclog.Printf("The name resolution does not give grpclb addresses")
				continue
			case GRPCLB:
				b.rbs = append(b.rbs, remoteBalancerInfo{
					addr: update.Addr,
					name: md.ServerName,
				})
			default:
				grpclog.Printf("Received unknow address type %d", md.AddrType)
				continue
			}
		case naming.Delete:
			for i, v := range b.rbs {
				if update.Addr == v.addr {
					copy(b.rbs[i:], b.rbs[i+1:])
					b.rbs = b.rbs[:len(b.rbs)-1]
					break
				}
			}
		default:
			grpclog.Println("Unknown update.Op ", update.Op)
		}
	}
	// TODO: Fall back to the basic round-robin load balancing if the resulting address is
	// not a load balancer.
	select {
	case <-ch:
	default:
	}
	ch <- b.rbs
	return nil
}

func (b *balancer) serverListExpire(seq int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	// TODO: gRPC interanls do not clear the connections when the server list is stale.
	// This means RPCs will keep using the existing server list until b receives new
	// server list even though the list is expired. Revisit this behavior later.
	if b.done || seq < b.seq {
		return
	}
	b.next = 0
	b.addrs = nil
	// Ask grpc internals to close all the corresponding connections.
	b.addrCh <- nil
}

func convertDuration(d *lbpb.Duration) time.Duration {
	if d == nil {
		return 0
	}
	return time.Duration(d.Seconds)*time.Second + time.Duration(d.Nanos)*time.Nanosecond
}

func (b *balancer) processServerList(l *lbpb.ServerList, seq int) {
	if l == nil {
		return
	}
	servers := l.GetServers()
	expiration := convertDuration(l.GetExpirationInterval())
	var (
		sl    []*grpclbAddrInfo
		addrs []Address
	)
	for _, s := range servers {
		md := metadata.Pairs("lb-token", s.LoadBalanceToken)
		addr := Address{
			Addr:     fmt.Sprintf("%s:%d", net.IP(s.IpAddress), s.Port),
			Metadata: &md,
		}
		sl = append(sl, &grpclbAddrInfo{
			addr:                 addr,
			dropForRateLimiting:  s.DropForRateLimiting,
			dropForLoadBalancing: s.DropForLoadBalancing,
		})
		addrs = append(addrs, addr)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.done || seq < b.seq {
		return
	}
	if len(sl) > 0 {
		// reset b.next to 0 when replacing the server list.
		b.next = 0
		b.addrs = sl
		b.addrCh <- addrs
		if b.expTimer != nil {
			b.expTimer.Stop()
			b.expTimer = nil
		}
		if expiration > 0 {
			b.expTimer = time.AfterFunc(expiration, func() {
				b.serverListExpire(seq)
			})
		}
	}
	return
}

func (b *balancer) sendLoadReport(s *balanceLoadClientStream, interval time.Duration, done <-chan struct{}) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
		case <-done:
			return
		}
		b.mu.Lock()
		stats := b.clientStats
		b.clientStats = lbpb.ClientStats{} // Clear the stats.
		b.mu.Unlock()
		t := time.Now()
		stats.Timestamp = &lbpb.Timestamp{
			Seconds: t.Unix(),
			Nanos:   int32(t.Nanosecond()),
		}
		if err := s.Send(&lbpb.LoadBalanceRequest{
			LoadBalanceRequestType: &lbpb.LoadBalanceRequest_ClientStats{
				ClientStats: &stats,
			},
		}); err != nil {
			return
		}
	}
}

func (b *balancer) callRemoteBalancer(lbc *loadBalancerClient, seq int) (retry bool) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := lbc.BalanceLoad(ctx)
	if err != nil {
		grpclog.Printf("Failed to perform RPC to the remote balancer %v", err)
		return
	}
	b.mu.Lock()
	if b.done {
		b.mu.Unlock()
		return
	}
	b.mu.Unlock()
	initReq := &lbpb.LoadBalanceRequest{
		LoadBalanceRequestType: &lbpb.LoadBalanceRequest_InitialRequest{
			InitialRequest: &lbpb.InitialLoadBalanceRequest{
				Name: b.target,
			},
		},
	}
	if err := stream.Send(initReq); err != nil {
		// TODO: backoff on retry?
		return true
	}
	reply, err := stream.Recv()
	if err != nil {
		// TODO: backoff on retry?
		return true
	}
	initResp := reply.GetInitialResponse()
	if initResp == nil {
		grpclog.Println("Failed to receive the initial response from the remote balancer.")
		return
	}
	// TODO: Support delegation.
	if initResp.LoadBalancerDelegate != "" {
		// delegation
		grpclog.Println("TODO: Delegation is not supported yet.")
		return
	}
	streamDone := make(chan struct{})
	defer close(streamDone)
	b.mu.Lock()
	b.clientStats = lbpb.ClientStats{} // Clear client stats.
	b.mu.Unlock()
	if d := convertDuration(initResp.ClientStatsReportInterval); d > 0 {
		go b.sendLoadReport(stream, d, streamDone)
	}
	// Retrieve the server list.
	for {
		reply, err := stream.Recv()
		if err != nil {
			break
		}
		b.mu.Lock()
		if b.done || seq < b.seq {
			b.mu.Unlock()
			return
		}
		b.seq++ // tick when receiving a new list of servers.
		seq = b.seq
		b.mu.Unlock()
		if serverList := reply.GetServerList(); serverList != nil {
			b.processServerList(serverList, seq)
		}
	}
	return true
}

func (b *balancer) Start(target string, config BalancerConfig) error {
	b.rand = rand.New(rand.NewSource(time.Now().Unix()))
	// TODO: Fall back to the basic direct connection if there is no name resolver.
	if b.r == nil {
		return errors.New("there is no name resolver installed")
	}
	b.target = target
	b.mu.Lock()
	if b.done {
		b.mu.Unlock()
		return ErrClientConnClosing
	}
	b.addrCh = make(chan []Address)
	w, err := b.r.Resolve(target)
	if err != nil {
		b.mu.Unlock()
		return err
	}
	b.w = w
	b.mu.Unlock()
	balancerAddrsCh := make(chan []remoteBalancerInfo, 1)
	// Spawn a goroutine to monitor the name resolution of remote load balancer.
	go func() {
		for {
			if err := b.watchAddrUpdates(w, balancerAddrsCh); err != nil {
				grpclog.Printf("grpc: the naming watcher stops working due to %v.\n", err)
				close(balancerAddrsCh)
				return
			}
		}
	}()
	// Spawn a goroutine to talk to the remote load balancer.
	go func() {
		var (
			cc *ClientConn
			// ccError is closed when there is an error in the current cc.
			// A new rb should be picked from rbs and connected.
			ccError chan struct{}
			rb      *remoteBalancerInfo
			rbs     []remoteBalancerInfo
			rbIdx   int
		)

		defer func() {
			if ccError != nil {
				select {
				case <-ccError:
				default:
					close(ccError)
				}
			}
			if cc != nil {
				cc.Close()
			}
		}()

		for {
			var ok bool
			select {
			case rbs, ok = <-balancerAddrsCh:
				if !ok {
					return
				}
				foundIdx := -1
				if rb != nil {
					for i, trb := range rbs {
						if trb == *rb {
							foundIdx = i
							break
						}
					}
				}
				if foundIdx >= 0 {
					if foundIdx >= 1 {
						// Move the address in use to the beginning of the list.
						b.rbs[0], b.rbs[foundIdx] = b.rbs[foundIdx], b.rbs[0]
						rbIdx = 0
					}
					continue // If found, don't dial new cc.
				} else if len(rbs) > 0 {
					// Pick a random one from the list, instead of always using the first one.
					if l := len(rbs); l > 1 && rb != nil {
						tmpIdx := b.rand.Intn(l - 1)
						b.rbs[0], b.rbs[tmpIdx] = b.rbs[tmpIdx], b.rbs[0]
					}
					rbIdx = 0
					rb = &rbs[0]
				} else {
					// foundIdx < 0 && len(rbs) <= 0.
					rb = nil
				}
			case <-ccError:
				ccError = nil
				if rbIdx < len(rbs)-1 {
					rbIdx++
					rb = &rbs[rbIdx]
				} else {
					rb = nil
				}
			}

			if rb == nil {
				continue
			}

			if cc != nil {
				cc.Close()
			}
			// Talk to the remote load balancer to get the server list.
			var err error
			creds := config.DialCreds
			ccError = make(chan struct{})
			if creds == nil {
				cc, err = Dial(rb.addr, WithInsecure())
			} else {
				if rb.name != "" {
					if err := creds.OverrideServerName(rb.name); err != nil {
						grpclog.Printf("Failed to override the server name in the credentials: %v", err)
						continue
					}
				}
				cc, err = Dial(rb.addr, WithTransportCredentials(creds))
			}
			if err != nil {
				grpclog.Printf("Failed to setup a connection to the remote balancer %v: %v", rb.addr, err)
				close(ccError)
				continue
			}
			b.mu.Lock()
			b.seq++ // tick when getting a new balancer address
			seq := b.seq
			b.next = 0
			b.mu.Unlock()
			go func(cc *ClientConn, ccError chan struct{}) {
				lbc := &loadBalancerClient{cc}
				b.callRemoteBalancer(lbc, seq)
				cc.Close()
				select {
				case <-ccError:
				default:
					close(ccError)
				}
			}(cc, ccError)
		}
	}()
	return nil
}

func (b *balancer) down(addr Address, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, a := range b.addrs {
		if addr == a.addr {
			a.connected = false
			break
		}
	}
}

func (b *balancer) Up(addr Address) func(error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.done {
		return nil
	}
	var cnt int
	for _, a := range b.addrs {
		if a.addr == addr {
			if a.connected {
				return nil
			}
			a.connected = true
		}
		if a.connected && !a.dropForRateLimiting && !a.dropForLoadBalancing {
			cnt++
		}
	}
	// addr is the only one which is connected. Notify the Get() callers who are blocking.
	if cnt == 1 && b.waitCh != nil {
		close(b.waitCh)
		b.waitCh = nil
	}
	return func(err error) {
		b.down(addr, err)
	}
}

func (b *balancer) Get(ctx context.Context, opts BalancerGetOptions) (addr Address, put func(), err error) {
	var ch chan struct{}
	b.mu.Lock()
	if b.done {
		b.mu.Unlock()
		err = ErrClientConnClosing
		return
	}
	seq := b.seq

	defer func() {
		if err != nil {
			return
		}
		put = func() {
			s, ok := rpcInfoFromContext(ctx)
			if !ok {
				return
			}
			b.mu.Lock()
			defer b.mu.Unlock()
			if b.done || seq < b.seq {
				return
			}
			b.clientStats.NumCallsFinished++
			if !s.bytesSent {
				b.clientStats.NumCallsFinishedWithClientFailedToSend++
			} else if s.bytesReceived {
				b.clientStats.NumCallsFinishedKnownReceived++
			}
		}
	}()

	b.clientStats.NumCallsStarted++
	if len(b.addrs) > 0 {
		if b.next >= len(b.addrs) {
			b.next = 0
		}
		next := b.next
		for {
			a := b.addrs[next]
			next = (next + 1) % len(b.addrs)
			if a.connected {
				if !a.dropForRateLimiting && !a.dropForLoadBalancing {
					addr = a.addr
					b.next = next
					b.mu.Unlock()
					return
				}
				if !opts.BlockingWait {
					b.next = next
					if a.dropForLoadBalancing {
						b.clientStats.NumCallsFinished++
						b.clientStats.NumCallsFinishedWithDropForLoadBalancing++
					} else if a.dropForRateLimiting {
						b.clientStats.NumCallsFinished++
						b.clientStats.NumCallsFinishedWithDropForRateLimiting++
					}
					b.mu.Unlock()
					err = Errorf(codes.Unavailable, "%s drops requests", a.addr.Addr)
					return
				}
			}
			if next == b.next {
				// Has iterated all the possible address but none is connected.
				break
			}
		}
	}
	if !opts.BlockingWait {
		if len(b.addrs) == 0 {
			b.clientStats.NumCallsFinished++
			b.clientStats.NumCallsFinishedWithClientFailedToSend++
			b.mu.Unlock()
			err = Errorf(codes.Unavailable, "there is no address available")
			return
		}
		// Returns the next addr on b.addrs for a failfast RPC.
		addr = b.addrs[b.next].addr
		b.next++
		b.mu.Unlock()
		return
	}
	// Wait on b.waitCh for non-failfast RPCs.
	if b.waitCh == nil {
		ch = make(chan struct{})
		b.waitCh = ch
	} else {
		ch = b.waitCh
	}
	b.mu.Unlock()
	for {
		select {
		case <-ctx.Done():
			b.mu.Lock()
			b.clientStats.NumCallsFinished++
			b.clientStats.NumCallsFinishedWithClientFailedToSend++
			b.mu.Unlock()
			err = ctx.Err()
			return
		case <-ch:
			b.mu.Lock()
			if b.done {
				b.clientStats.NumCallsFinished++
				b.clientStats.NumCallsFinishedWithClientFailedToSend++
				b.mu.Unlock()
				err = ErrClientConnClosing
				return
			}

			if len(b.addrs) > 0 {
				if b.next >= len(b.addrs) {
					b.next = 0
				}
				next := b.next
				for {
					a := b.addrs[next]
					next = (next + 1) % len(b.addrs)
					if a.connected {
						if !a.dropForRateLimiting && !a.dropForLoadBalancing {
							addr = a.addr
							b.next = next
							b.mu.Unlock()
							return
						}
						if !opts.BlockingWait {
							b.next = next
							if a.dropForLoadBalancing {
								b.clientStats.NumCallsFinished++
								b.clientStats.NumCallsFinishedWithDropForLoadBalancing++
							} else if a.dropForRateLimiting {
								b.clientStats.NumCallsFinished++
								b.clientStats.NumCallsFinishedWithDropForRateLimiting++
							}
							b.mu.Unlock()
							err = Errorf(codes.Unavailable, "drop requests for the addreess %s", a.addr.Addr)
							return
						}
					}
					if next == b.next {
						// Has iterated all the possible address but none is connected.
						break
					}
				}
			}
			// The newly added addr got removed by Down() again.
			if b.waitCh == nil {
				ch = make(chan struct{})
				b.waitCh = ch
			} else {
				ch = b.waitCh
			}
			b.mu.Unlock()
		}
	}
}

func (b *balancer) Notify() <-chan []Address {
	return b.addrCh
}

func (b *balancer) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.done = true
	if b.expTimer != nil {
		b.expTimer.Stop()
	}
	if b.waitCh != nil {
		close(b.waitCh)
	}
	if b.addrCh != nil {
		close(b.addrCh)
	}
	if b.w != nil {
		b.w.Close()
	}
	return nil
}
