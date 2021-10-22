/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package grpclb

import (
	"context"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	timestamppb "github.com/golang/protobuf/ptypes/timestamp"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/balancer"
	lbpb "google.golang.org/grpc/balancer/grpclb/grpc_lb_v1"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/backoff"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/resolver"
)

// processServerList updates balaner's internal state, create/remove SubConns
// and regenerates picker using the received serverList.
func (lb *lbBalancer) processServerList(l *lbpb.ServerList) {
	if grpclog.V(2) {
		grpclog.Infof("lbBalancer: processing server list: %+v", l)
	}
	lb.mu.Lock()
	defer lb.mu.Unlock()

	// Set serverListReceived to true so fallback will not take effect if it has
	// not hit timeout.
	lb.serverListReceived = true

	// If the new server list == old server list, do nothing.
	if cmp.Equal(lb.fullServerList, l.Servers, cmp.Comparer(proto.Equal)) {
		if grpclog.V(2) {
			grpclog.Infof("lbBalancer: new serverlist same as the previous one, ignoring")
		}
		return
	}
	lb.fullServerList = l.Servers

	var backendAddrs []resolver.Address
	for i, s := range l.Servers {
		if s.Drop {
			continue
		}

		md := metadata.Pairs(lbTokenKey, s.LoadBalanceToken)
		ip := net.IP(s.IpAddress)
		ipStr := ip.String()
		if ip.To4() == nil {
			// Add square brackets to ipv6 addresses, otherwise net.Dial() and
			// net.SplitHostPort() will return too many colons error.
			ipStr = fmt.Sprintf("[%s]", ipStr)
		}
		addr := resolver.Address{
			Addr:     fmt.Sprintf("%s:%d", ipStr, s.Port),
			Metadata: &md,
		}
		if grpclog.V(2) {
			grpclog.Infof("lbBalancer: server list entry[%d]: ipStr:|%s|, port:|%d|, load balancer token:|%v|",
				i, ipStr, s.Port, s.LoadBalanceToken)
		}
		backendAddrs = append(backendAddrs, addr)
	}

	// Call refreshSubConns to create/remove SubConns.  If we are in fallback,
	// this is also exiting fallback.
	lb.refreshSubConns(backendAddrs, false, lb.usePickFirst)
}

// refreshSubConns creates/removes SubConns with backendAddrs, and refreshes
// balancer state and picker.
//
// Caller must hold lb.mu.
func (lb *lbBalancer) refreshSubConns(backendAddrs []resolver.Address, fallback bool, pickFirst bool) {
	opts := balancer.NewSubConnOptions{}
	if !fallback {
		opts.CredsBundle = lb.grpclbBackendCreds
	}

	lb.backendAddrs = backendAddrs
	lb.backendAddrsWithoutMetadata = nil

	fallbackModeChanged := lb.inFallback != fallback
	lb.inFallback = fallback
	if fallbackModeChanged && lb.inFallback {
		// Clear previous received list when entering fallback, so if the server
		// comes back and sends the same list again, the new addresses will be
		// used.
		lb.fullServerList = nil
	}

	balancingPolicyChanged := lb.usePickFirst != pickFirst
	oldUsePickFirst := lb.usePickFirst
	lb.usePickFirst = pickFirst

	if fallbackModeChanged || balancingPolicyChanged {
		// Remove all SubConns when switching balancing policy or switching
		// fallback mode.
		//
		// For fallback mode switching with pickfirst, we want to recreate the
		// SubConn because the creds could be different.
		for a, sc := range lb.subConns {
			if oldUsePickFirst {
				// If old SubConn were created for pickfirst, bypass cache and
				// remove directly.
				lb.cc.cc.RemoveSubConn(sc)
			} else {
				lb.cc.RemoveSubConn(sc)
			}
			delete(lb.subConns, a)
		}
	}

	if lb.usePickFirst {
		var sc balancer.SubConn
		for _, sc = range lb.subConns {
			break
		}
		if sc != nil {
			sc.UpdateAddresses(backendAddrs)
			sc.Connect()
			return
		}
		// This bypasses the cc wrapper with SubConn cache.
		sc, err := lb.cc.cc.NewSubConn(backendAddrs, opts)
		if err != nil {
			grpclog.Warningf("grpclb: failed to create new SubConn: %v", err)
			return
		}
		sc.Connect()
		lb.subConns[backendAddrs[0]] = sc
		lb.scStates[sc] = connectivity.Idle
		return
	}

	// addrsSet is the set converted from backendAddrsWithoutMetadata, it's used to quick
	// lookup for an address.
	addrsSet := make(map[resolver.Address]struct{})
	// Create new SubConns.
	for _, addr := range backendAddrs {
		addrWithoutMD := addr
		addrWithoutMD.Metadata = nil
		addrsSet[addrWithoutMD] = struct{}{}
		lb.backendAddrsWithoutMetadata = append(lb.backendAddrsWithoutMetadata, addrWithoutMD)

		if _, ok := lb.subConns[addrWithoutMD]; !ok {
			// Use addrWithMD to create the SubConn.
			sc, err := lb.cc.NewSubConn([]resolver.Address{addr}, opts)
			if err != nil {
				grpclog.Warningf("grpclb: failed to create new SubConn: %v", err)
				continue
			}
			lb.subConns[addrWithoutMD] = sc // Use the addr without MD as key for the map.
			if _, ok := lb.scStates[sc]; !ok {
				// Only set state of new sc to IDLE. The state could already be
				// READY for cached SubConns.
				lb.scStates[sc] = connectivity.Idle
			}
			sc.Connect()
		}
	}

	for a, sc := range lb.subConns {
		// a was removed by resolver.
		if _, ok := addrsSet[a]; !ok {
			lb.cc.RemoveSubConn(sc)
			delete(lb.subConns, a)
			// Keep the state of this sc in b.scStates until sc's state becomes Shutdown.
			// The entry will be deleted in HandleSubConnStateChange.
		}
	}

	// Regenerate and update picker after refreshing subconns because with
	// cache, even if SubConn was newed/removed, there might be no state
	// changes (the subconn will be kept in cache, not actually
	// newed/removed).
	lb.updateStateAndPicker(true, true)
}

type remoteBalancerCCWrapper struct {
	cc      *grpc.ClientConn
	lb      *lbBalancer
	backoff backoff.Strategy
	done    chan struct{}

	// waitgroup to wait for all goroutines to exit.
	wg sync.WaitGroup
}

func (lb *lbBalancer) newRemoteBalancerCCWrapper() {
	var dopts []grpc.DialOption
	if creds := lb.opt.DialCreds; creds != nil {
		dopts = append(dopts, grpc.WithTransportCredentials(creds))
	} else if bundle := lb.grpclbClientConnCreds; bundle != nil {
		dopts = append(dopts, grpc.WithCredentialsBundle(bundle))
	} else {
		dopts = append(dopts, grpc.WithInsecure())
	}
	if lb.opt.Dialer != nil {
		dopts = append(dopts, grpc.WithContextDialer(lb.opt.Dialer))
	}
	// Explicitly set pickfirst as the balancer.
	dopts = append(dopts, grpc.WithDefaultServiceConfig(`{"loadBalancingPolicy":"pick_first"}`))
	dopts = append(dopts, grpc.WithResolvers(lb.manualResolver))
	if channelz.IsOn() {
		dopts = append(dopts, grpc.WithChannelzParentID(lb.opt.ChannelzParentID))
	}

	// Enable Keepalive for grpclb client.
	dopts = append(dopts, grpc.WithKeepaliveParams(keepalive.ClientParameters{
		Time:                20 * time.Second,
		Timeout:             10 * time.Second,
		PermitWithoutStream: true,
	}))

	// The dial target is not important.
	//
	// The grpclb server addresses will set field ServerName, and creds will
	// receive ServerName as authority.
	cc, err := grpc.DialContext(context.Background(), lb.manualResolver.Scheme()+":///grpclb.subClientConn", dopts...)
	if err != nil {
		grpclog.Fatalf("failed to dial: %v", err)
	}
	ccw := &remoteBalancerCCWrapper{
		cc:      cc,
		lb:      lb,
		backoff: lb.backoff,
		done:    make(chan struct{}),
	}
	lb.ccRemoteLB = ccw
	ccw.wg.Add(1)
	go ccw.watchRemoteBalancer()
}

// close closed the ClientConn to remote balancer, and waits until all
// goroutines to finish.
func (ccw *remoteBalancerCCWrapper) close() {
	close(ccw.done)
	ccw.cc.Close()
	ccw.wg.Wait()
}

func (ccw *remoteBalancerCCWrapper) readServerList(s *balanceLoadClientStream) error {
	for {
		reply, err := s.Recv()
		if err != nil {
			if err == io.EOF {
				return errServerTerminatedConnection
			}
			return fmt.Errorf("grpclb: failed to recv server list: %v", err)
		}
		if serverList := reply.GetServerList(); serverList != nil {
			ccw.lb.processServerList(serverList)
		}
	}
}

func (ccw *remoteBalancerCCWrapper) sendLoadReport(s *balanceLoadClientStream, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	lastZero := false
	for {
		select {
		case <-ticker.C:
		case <-s.Context().Done():
			return
		}
		stats := ccw.lb.clientStats.toClientStats()
		zero := isZeroStats(stats)
		if zero && lastZero {
			// Quash redundant empty load reports.
			continue
		}
		lastZero = zero
		t := time.Now()
		stats.Timestamp = &timestamppb.Timestamp{
			Seconds: t.Unix(),
			Nanos:   int32(t.Nanosecond()),
		}
		if err := s.Send(&lbpb.LoadBalanceRequest{
			LoadBalanceRequestType: &lbpb.LoadBalanceRequest_ClientStats{
				ClientStats: stats,
			},
		}); err != nil {
			return
		}
	}
}

func (ccw *remoteBalancerCCWrapper) callRemoteBalancer() (backoff bool, _ error) {
	lbClient := &loadBalancerClient{cc: ccw.cc}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := lbClient.BalanceLoad(ctx, grpc.WaitForReady(true))
	if err != nil {
		return true, fmt.Errorf("grpclb: failed to perform RPC to the remote balancer %v", err)
	}
	ccw.lb.mu.Lock()
	ccw.lb.remoteBalancerConnected = true
	ccw.lb.mu.Unlock()

	// grpclb handshake on the stream.
	initReq := &lbpb.LoadBalanceRequest{
		LoadBalanceRequestType: &lbpb.LoadBalanceRequest_InitialRequest{
			InitialRequest: &lbpb.InitialLoadBalanceRequest{
				Name: ccw.lb.target,
			},
		},
	}
	if err := stream.Send(initReq); err != nil {
		return true, fmt.Errorf("grpclb: failed to send init request: %v", err)
	}
	reply, err := stream.Recv()
	if err != nil {
		return true, fmt.Errorf("grpclb: failed to recv init response: %v", err)
	}
	initResp := reply.GetInitialResponse()
	if initResp == nil {
		return true, fmt.Errorf("grpclb: reply from remote balancer did not include initial response")
	}
	if initResp.LoadBalancerDelegate != "" {
		return true, fmt.Errorf("grpclb: Delegation is not supported")
	}

	ccw.wg.Add(1)
	go func() {
		defer ccw.wg.Done()
		if d := convertDuration(initResp.ClientStatsReportInterval); d > 0 {
			ccw.sendLoadReport(stream, d)
		}
	}()
	// No backoff if init req/resp handshake was successful.
	return false, ccw.readServerList(stream)
}

func (ccw *remoteBalancerCCWrapper) watchRemoteBalancer() {
	defer ccw.wg.Done()
	var retryCount int
	for {
		doBackoff, err := ccw.callRemoteBalancer()
		select {
		case <-ccw.done:
			return
		default:
			if err != nil {
				if err == errServerTerminatedConnection {
					grpclog.Info(err)
				} else {
					grpclog.Warning(err)
				}
			}
		}
		// Trigger a re-resolve when the stream errors.
		ccw.lb.cc.cc.ResolveNow(resolver.ResolveNowOptions{})

		ccw.lb.mu.Lock()
		ccw.lb.remoteBalancerConnected = false
		ccw.lb.fullServerList = nil
		// Enter fallback when connection to remote balancer is lost, and the
		// aggregated state is not Ready.
		if !ccw.lb.inFallback && ccw.lb.state != connectivity.Ready {
			// Entering fallback.
			ccw.lb.refreshSubConns(ccw.lb.resolvedBackendAddrs, true, ccw.lb.usePickFirst)
		}
		ccw.lb.mu.Unlock()

		if !doBackoff {
			retryCount = 0
			continue
		}

		timer := time.NewTimer(ccw.backoff.Backoff(retryCount)) // Copy backoff
		select {
		case <-timer.C:
		case <-ccw.done:
			timer.Stop()
			return
		}
		retryCount++
	}
}
