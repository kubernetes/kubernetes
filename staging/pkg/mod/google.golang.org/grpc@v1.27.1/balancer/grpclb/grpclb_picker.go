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
	"sync"
	"sync/atomic"

	"google.golang.org/grpc/balancer"
	lbpb "google.golang.org/grpc/balancer/grpclb/grpc_lb_v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/internal/grpcrand"
	"google.golang.org/grpc/status"
)

// rpcStats is same as lbpb.ClientStats, except that numCallsDropped is a map
// instead of a slice.
type rpcStats struct {
	// Only access the following fields atomically.
	numCallsStarted                        int64
	numCallsFinished                       int64
	numCallsFinishedWithClientFailedToSend int64
	numCallsFinishedKnownReceived          int64

	mu sync.Mutex
	// map load_balance_token -> num_calls_dropped
	numCallsDropped map[string]int64
}

func newRPCStats() *rpcStats {
	return &rpcStats{
		numCallsDropped: make(map[string]int64),
	}
}

func isZeroStats(stats *lbpb.ClientStats) bool {
	return len(stats.CallsFinishedWithDrop) == 0 &&
		stats.NumCallsStarted == 0 &&
		stats.NumCallsFinished == 0 &&
		stats.NumCallsFinishedWithClientFailedToSend == 0 &&
		stats.NumCallsFinishedKnownReceived == 0
}

// toClientStats converts rpcStats to lbpb.ClientStats, and clears rpcStats.
func (s *rpcStats) toClientStats() *lbpb.ClientStats {
	stats := &lbpb.ClientStats{
		NumCallsStarted:                        atomic.SwapInt64(&s.numCallsStarted, 0),
		NumCallsFinished:                       atomic.SwapInt64(&s.numCallsFinished, 0),
		NumCallsFinishedWithClientFailedToSend: atomic.SwapInt64(&s.numCallsFinishedWithClientFailedToSend, 0),
		NumCallsFinishedKnownReceived:          atomic.SwapInt64(&s.numCallsFinishedKnownReceived, 0),
	}
	s.mu.Lock()
	dropped := s.numCallsDropped
	s.numCallsDropped = make(map[string]int64)
	s.mu.Unlock()
	for token, count := range dropped {
		stats.CallsFinishedWithDrop = append(stats.CallsFinishedWithDrop, &lbpb.ClientStatsPerToken{
			LoadBalanceToken: token,
			NumCalls:         count,
		})
	}
	return stats
}

func (s *rpcStats) drop(token string) {
	atomic.AddInt64(&s.numCallsStarted, 1)
	s.mu.Lock()
	s.numCallsDropped[token]++
	s.mu.Unlock()
	atomic.AddInt64(&s.numCallsFinished, 1)
}

func (s *rpcStats) failedToSend() {
	atomic.AddInt64(&s.numCallsStarted, 1)
	atomic.AddInt64(&s.numCallsFinishedWithClientFailedToSend, 1)
	atomic.AddInt64(&s.numCallsFinished, 1)
}

func (s *rpcStats) knownReceived() {
	atomic.AddInt64(&s.numCallsStarted, 1)
	atomic.AddInt64(&s.numCallsFinishedKnownReceived, 1)
	atomic.AddInt64(&s.numCallsFinished, 1)
}

type errPicker struct {
	// Pick always returns this err.
	err error
}

func (p *errPicker) Pick(balancer.PickInfo) (balancer.PickResult, error) {
	return balancer.PickResult{}, p.err
}

// rrPicker does roundrobin on subConns. It's typically used when there's no
// response from remote balancer, and grpclb falls back to the resolved
// backends.
//
// It guaranteed that len(subConns) > 0.
type rrPicker struct {
	mu           sync.Mutex
	subConns     []balancer.SubConn // The subConns that were READY when taking the snapshot.
	subConnsNext int
}

func newRRPicker(readySCs []balancer.SubConn) *rrPicker {
	return &rrPicker{
		subConns:     readySCs,
		subConnsNext: grpcrand.Intn(len(readySCs)),
	}
}

func (p *rrPicker) Pick(balancer.PickInfo) (balancer.PickResult, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	sc := p.subConns[p.subConnsNext]
	p.subConnsNext = (p.subConnsNext + 1) % len(p.subConns)
	return balancer.PickResult{SubConn: sc}, nil
}

// lbPicker does two layers of picks:
//
// First layer: roundrobin on all servers in serverList, including drops and backends.
// - If it picks a drop, the RPC will fail as being dropped.
// - If it picks a backend, do a second layer pick to pick the real backend.
//
// Second layer: roundrobin on all READY backends.
//
// It's guaranteed that len(serverList) > 0.
type lbPicker struct {
	mu             sync.Mutex
	serverList     []*lbpb.Server
	serverListNext int
	subConns       []balancer.SubConn // The subConns that were READY when taking the snapshot.
	subConnsNext   int

	stats *rpcStats
}

func newLBPicker(serverList []*lbpb.Server, readySCs []balancer.SubConn, stats *rpcStats) *lbPicker {
	return &lbPicker{
		serverList:   serverList,
		subConns:     readySCs,
		subConnsNext: grpcrand.Intn(len(readySCs)),
		stats:        stats,
	}
}

func (p *lbPicker) Pick(balancer.PickInfo) (balancer.PickResult, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Layer one roundrobin on serverList.
	s := p.serverList[p.serverListNext]
	p.serverListNext = (p.serverListNext + 1) % len(p.serverList)

	// If it's a drop, return an error and fail the RPC.
	if s.Drop {
		p.stats.drop(s.LoadBalanceToken)
		return balancer.PickResult{}, status.Errorf(codes.Unavailable, "request dropped by grpclb")
	}

	// If not a drop but there's no ready subConns.
	if len(p.subConns) <= 0 {
		return balancer.PickResult{}, balancer.ErrNoSubConnAvailable
	}

	// Return the next ready subConn in the list, also collect rpc stats.
	sc := p.subConns[p.subConnsNext]
	p.subConnsNext = (p.subConnsNext + 1) % len(p.subConns)
	done := func(info balancer.DoneInfo) {
		if !info.BytesSent {
			p.stats.failedToSend()
		} else if info.BytesReceived {
			p.stats.knownReceived()
		}
	}
	return balancer.PickResult{SubConn: sc, Done: done}, nil
}

func (p *lbPicker) updateReadySCs(readySCs []balancer.SubConn) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.subConns = readySCs
	p.subConnsNext = p.subConnsNext % len(readySCs)
}
