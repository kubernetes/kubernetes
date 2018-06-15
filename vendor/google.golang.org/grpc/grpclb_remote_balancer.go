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

package grpc

import (
	"fmt"
	"net"
	"reflect"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/channelz"

	"google.golang.org/grpc/connectivity"
	lbpb "google.golang.org/grpc/grpclb/grpc_lb_v1/messages"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/resolver"
)

// processServerList updates balaner's internal state, create/remove SubConns
// and regenerates picker using the received serverList.
func (lb *lbBalancer) processServerList(l *lbpb.ServerList) {
	grpclog.Infof("lbBalancer: processing server list: %+v", l)
	lb.mu.Lock()
	defer lb.mu.Unlock()

	// Set serverListReceived to true so fallback will not take effect if it has
	// not hit timeout.
	lb.serverListReceived = true

	// If the new server list == old server list, do nothing.
	if reflect.DeepEqual(lb.fullServerList, l.Servers) {
		grpclog.Infof("lbBalancer: new serverlist same as the previous one, ignoring")
		return
	}
	lb.fullServerList = l.Servers

	var backendAddrs []resolver.Address
	for _, s := range l.Servers {
		if s.DropForLoadBalancing || s.DropForRateLimiting {
			continue
		}

		md := metadata.Pairs(lbTokeyKey, s.LoadBalanceToken)
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

		backendAddrs = append(backendAddrs, addr)
	}

	// Call refreshSubConns to create/remove SubConns.
	lb.refreshSubConns(backendAddrs)
	// Regenerate and update picker no matter if there's update on backends (if
	// any SubConn will be newed/removed). Because since the full serverList was
	// different, there might be updates in drops or pick weights(different
	// number of duplicates). We need to update picker with the fulllist.
	//
	// Now with cache, even if SubConn was newed/removed, there might be no
	// state changes.
	lb.regeneratePicker()
	lb.cc.UpdateBalancerState(lb.state, lb.picker)
}

// refreshSubConns creates/removes SubConns with backendAddrs. It returns a bool
// indicating whether the backendAddrs are different from the cached
// backendAddrs (whether any SubConn was newed/removed).
// Caller must hold lb.mu.
func (lb *lbBalancer) refreshSubConns(backendAddrs []resolver.Address) bool {
	lb.backendAddrs = nil
	var backendsUpdated bool
	// addrsSet is the set converted from backendAddrs, it's used to quick
	// lookup for an address.
	addrsSet := make(map[resolver.Address]struct{})
	// Create new SubConns.
	for _, addr := range backendAddrs {
		addrWithoutMD := addr
		addrWithoutMD.Metadata = nil
		addrsSet[addrWithoutMD] = struct{}{}
		lb.backendAddrs = append(lb.backendAddrs, addrWithoutMD)

		if _, ok := lb.subConns[addrWithoutMD]; !ok {
			backendsUpdated = true

			// Use addrWithMD to create the SubConn.
			sc, err := lb.cc.NewSubConn([]resolver.Address{addr}, balancer.NewSubConnOptions{})
			if err != nil {
				grpclog.Warningf("roundrobinBalancer: failed to create new SubConn: %v", err)
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
			backendsUpdated = true

			lb.cc.RemoveSubConn(sc)
			delete(lb.subConns, a)
			// Keep the state of this sc in b.scStates until sc's state becomes Shutdown.
			// The entry will be deleted in HandleSubConnStateChange.
		}
	}

	return backendsUpdated
}

func (lb *lbBalancer) readServerList(s *balanceLoadClientStream) error {
	for {
		reply, err := s.Recv()
		if err != nil {
			return fmt.Errorf("grpclb: failed to recv server list: %v", err)
		}
		if serverList := reply.GetServerList(); serverList != nil {
			lb.processServerList(serverList)
		}
	}
}

func (lb *lbBalancer) sendLoadReport(s *balanceLoadClientStream, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
		case <-s.Context().Done():
			return
		}
		stats := lb.clientStats.toClientStats()
		t := time.Now()
		stats.Timestamp = &lbpb.Timestamp{
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

func (lb *lbBalancer) callRemoteBalancer() error {
	lbClient := &loadBalancerClient{cc: lb.ccRemoteLB}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := lbClient.BalanceLoad(ctx, FailFast(false))
	if err != nil {
		return fmt.Errorf("grpclb: failed to perform RPC to the remote balancer %v", err)
	}

	// grpclb handshake on the stream.
	initReq := &lbpb.LoadBalanceRequest{
		LoadBalanceRequestType: &lbpb.LoadBalanceRequest_InitialRequest{
			InitialRequest: &lbpb.InitialLoadBalanceRequest{
				Name: lb.target,
			},
		},
	}
	if err := stream.Send(initReq); err != nil {
		return fmt.Errorf("grpclb: failed to send init request: %v", err)
	}
	reply, err := stream.Recv()
	if err != nil {
		return fmt.Errorf("grpclb: failed to recv init response: %v", err)
	}
	initResp := reply.GetInitialResponse()
	if initResp == nil {
		return fmt.Errorf("grpclb: reply from remote balancer did not include initial response")
	}
	if initResp.LoadBalancerDelegate != "" {
		return fmt.Errorf("grpclb: Delegation is not supported")
	}

	go func() {
		if d := convertDuration(initResp.ClientStatsReportInterval); d > 0 {
			lb.sendLoadReport(stream, d)
		}
	}()
	return lb.readServerList(stream)
}

func (lb *lbBalancer) watchRemoteBalancer() {
	for {
		err := lb.callRemoteBalancer()
		select {
		case <-lb.doneCh:
			return
		default:
			if err != nil {
				grpclog.Error(err)
			}
		}

	}
}

func (lb *lbBalancer) dialRemoteLB(remoteLBName string) {
	var dopts []DialOption
	if creds := lb.opt.DialCreds; creds != nil {
		if err := creds.OverrideServerName(remoteLBName); err == nil {
			dopts = append(dopts, WithTransportCredentials(creds))
		} else {
			grpclog.Warningf("grpclb: failed to override the server name in the credentials: %v, using Insecure", err)
			dopts = append(dopts, WithInsecure())
		}
	} else {
		dopts = append(dopts, WithInsecure())
	}
	if lb.opt.Dialer != nil {
		// WithDialer takes a different type of function, so we instead use a
		// special DialOption here.
		dopts = append(dopts, withContextDialer(lb.opt.Dialer))
	}
	// Explicitly set pickfirst as the balancer.
	dopts = append(dopts, WithBalancerName(PickFirstBalancerName))
	dopts = append(dopts, withResolverBuilder(lb.manualResolver))
	if channelz.IsOn() {
		dopts = append(dopts, WithChannelzParentID(lb.opt.ChannelzParentID))
	}

	// DialContext using manualResolver.Scheme, which is a random scheme generated
	// when init grpclb. The target name is not important.
	cc, err := DialContext(context.Background(), "grpclb:///grpclb.server", dopts...)
	if err != nil {
		grpclog.Fatalf("failed to dial: %v", err)
	}
	lb.ccRemoteLB = cc
	go lb.watchRemoteBalancer()
}
