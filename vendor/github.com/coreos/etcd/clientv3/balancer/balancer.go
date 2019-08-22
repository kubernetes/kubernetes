// Copyright 2018 The etcd Authors
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

// Package balancer implements client balancer.
package balancer

import (
	"strconv"
	"sync"
	"time"

	"github.com/coreos/etcd/clientv3/balancer/connectivity"
	"github.com/coreos/etcd/clientv3/balancer/picker"

	"go.uber.org/zap"
	"google.golang.org/grpc/balancer"
	grpcconnectivity "google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/resolver"
	_ "google.golang.org/grpc/resolver/dns"         // register DNS resolver
	_ "google.golang.org/grpc/resolver/passthrough" // register passthrough resolver
)

// Config defines balancer configurations.
type Config struct {
	// Policy configures balancer policy.
	Policy picker.Policy

	// Picker implements gRPC picker.
	// Leave empty if "Policy" field is not custom.
	// TODO: currently custom policy is not supported.
	// Picker picker.Picker

	// Name defines an additional name for balancer.
	// Useful for balancer testing to avoid register conflicts.
	// If empty, defaults to policy name.
	Name string

	// Logger configures balancer logging.
	// If nil, logs are discarded.
	Logger *zap.Logger
}

// RegisterBuilder creates and registers a builder. Since this function calls balancer.Register, it
// must be invoked at initialization time.
func RegisterBuilder(cfg Config) {
	bb := &builder{cfg}
	balancer.Register(bb)

	bb.cfg.Logger.Debug(
		"registered balancer",
		zap.String("policy", bb.cfg.Policy.String()),
		zap.String("name", bb.cfg.Name),
	)
}

type builder struct {
	cfg Config
}

// Build is called initially when creating "ccBalancerWrapper".
// "grpc.Dial" is called to this client connection.
// Then, resolved addresses will be handled via "HandleResolvedAddrs".
func (b *builder) Build(cc balancer.ClientConn, opt balancer.BuildOptions) balancer.Balancer {
	bb := &baseBalancer{
		id:     strconv.FormatInt(time.Now().UnixNano(), 36),
		policy: b.cfg.Policy,
		name:   b.cfg.Name,
		lg:     b.cfg.Logger,

		addrToSc: make(map[resolver.Address]balancer.SubConn),
		scToAddr: make(map[balancer.SubConn]resolver.Address),
		scToSt:   make(map[balancer.SubConn]grpcconnectivity.State),

		currentConn:          nil,
		connectivityRecorder: connectivity.New(b.cfg.Logger),

		// initialize picker always returns "ErrNoSubConnAvailable"
		picker: picker.NewErr(balancer.ErrNoSubConnAvailable),
	}

	// TODO: support multiple connections
	bb.mu.Lock()
	bb.currentConn = cc
	bb.mu.Unlock()

	bb.lg.Info(
		"built balancer",
		zap.String("balancer-id", bb.id),
		zap.String("policy", bb.policy.String()),
		zap.String("resolver-target", cc.Target()),
	)
	return bb
}

// Name implements "grpc/balancer.Builder" interface.
func (b *builder) Name() string { return b.cfg.Name }

// Balancer defines client balancer interface.
type Balancer interface {
	// Balancer is called on specified client connection. Client initiates gRPC
	// connection with "grpc.Dial(addr, grpc.WithBalancerName)", and then those resolved
	// addresses are passed to "grpc/balancer.Balancer.HandleResolvedAddrs".
	// For each resolved address, balancer calls "balancer.ClientConn.NewSubConn".
	// "grpc/balancer.Balancer.HandleSubConnStateChange" is called when connectivity state
	// changes, thus requires failover logic in this method.
	balancer.Balancer

	// Picker calls "Pick" for every client request.
	picker.Picker
}

type baseBalancer struct {
	id     string
	policy picker.Policy
	name   string
	lg     *zap.Logger

	mu sync.RWMutex

	addrToSc map[resolver.Address]balancer.SubConn
	scToAddr map[balancer.SubConn]resolver.Address
	scToSt   map[balancer.SubConn]grpcconnectivity.State

	currentConn          balancer.ClientConn
	connectivityRecorder connectivity.Recorder

	picker picker.Picker
}

// HandleResolvedAddrs implements "grpc/balancer.Balancer" interface.
// gRPC sends initial or updated resolved addresses from "Build".
func (bb *baseBalancer) HandleResolvedAddrs(addrs []resolver.Address, err error) {
	if err != nil {
		bb.lg.Warn("HandleResolvedAddrs called with error", zap.String("balancer-id", bb.id), zap.Error(err))
		return
	}
	bb.lg.Info("resolved",
		zap.String("picker", bb.picker.String()),
		zap.String("balancer-id", bb.id),
		zap.Strings("addresses", addrsToStrings(addrs)),
	)

	bb.mu.Lock()
	defer bb.mu.Unlock()

	resolved := make(map[resolver.Address]struct{})
	for _, addr := range addrs {
		resolved[addr] = struct{}{}
		if _, ok := bb.addrToSc[addr]; !ok {
			sc, err := bb.currentConn.NewSubConn([]resolver.Address{addr}, balancer.NewSubConnOptions{})
			if err != nil {
				bb.lg.Warn("NewSubConn failed", zap.String("picker", bb.picker.String()), zap.String("balancer-id", bb.id), zap.Error(err), zap.String("address", addr.Addr))
				continue
			}
			bb.lg.Info("created subconn", zap.String("address", addr.Addr))
			bb.addrToSc[addr] = sc
			bb.scToAddr[sc] = addr
			bb.scToSt[sc] = grpcconnectivity.Idle
			sc.Connect()
		}
	}

	for addr, sc := range bb.addrToSc {
		if _, ok := resolved[addr]; !ok {
			// was removed by resolver or failed to create subconn
			bb.currentConn.RemoveSubConn(sc)
			delete(bb.addrToSc, addr)

			bb.lg.Info(
				"removed subconn",
				zap.String("picker", bb.picker.String()),
				zap.String("balancer-id", bb.id),
				zap.String("address", addr.Addr),
				zap.String("subconn", scToString(sc)),
			)

			// Keep the state of this sc in bb.scToSt until sc's state becomes Shutdown.
			// The entry will be deleted in HandleSubConnStateChange.
			// (DO NOT) delete(bb.scToAddr, sc)
			// (DO NOT) delete(bb.scToSt, sc)
		}
	}
}

// HandleSubConnStateChange implements "grpc/balancer.Balancer" interface.
func (bb *baseBalancer) HandleSubConnStateChange(sc balancer.SubConn, s grpcconnectivity.State) {
	bb.mu.Lock()
	defer bb.mu.Unlock()

	old, ok := bb.scToSt[sc]
	if !ok {
		bb.lg.Warn(
			"state change for an unknown subconn",
			zap.String("picker", bb.picker.String()),
			zap.String("balancer-id", bb.id),
			zap.String("subconn", scToString(sc)),
			zap.Int("subconn-size", len(bb.scToAddr)),
			zap.String("state", s.String()),
		)
		return
	}

	bb.lg.Info(
		"state changed",
		zap.String("picker", bb.picker.String()),
		zap.String("balancer-id", bb.id),
		zap.Bool("connected", s == grpcconnectivity.Ready),
		zap.String("subconn", scToString(sc)),
		zap.Int("subconn-size", len(bb.scToAddr)),
		zap.String("address", bb.scToAddr[sc].Addr),
		zap.String("old-state", old.String()),
		zap.String("new-state", s.String()),
	)

	bb.scToSt[sc] = s
	switch s {
	case grpcconnectivity.Idle:
		sc.Connect()
	case grpcconnectivity.Shutdown:
		// When an address was removed by resolver, b called RemoveSubConn but
		// kept the sc's state in scToSt. Remove state for this sc here.
		delete(bb.scToAddr, sc)
		delete(bb.scToSt, sc)
	}

	oldAggrState := bb.connectivityRecorder.GetCurrentState()
	bb.connectivityRecorder.RecordTransition(old, s)

	// Update balancer picker when one of the following happens:
	//  - this sc became ready from not-ready
	//  - this sc became not-ready from ready
	//  - the aggregated state of balancer became TransientFailure from non-TransientFailure
	//  - the aggregated state of balancer became non-TransientFailure from TransientFailure
	if (s == grpcconnectivity.Ready) != (old == grpcconnectivity.Ready) ||
		(bb.connectivityRecorder.GetCurrentState() == grpcconnectivity.TransientFailure) != (oldAggrState == grpcconnectivity.TransientFailure) {
		bb.updatePicker()
	}

	bb.currentConn.UpdateBalancerState(bb.connectivityRecorder.GetCurrentState(), bb.picker)
}

func (bb *baseBalancer) updatePicker() {
	if bb.connectivityRecorder.GetCurrentState() == grpcconnectivity.TransientFailure {
		bb.picker = picker.NewErr(balancer.ErrTransientFailure)
		bb.lg.Info(
			"updated picker to transient error picker",
			zap.String("picker", bb.picker.String()),
			zap.String("balancer-id", bb.id),
			zap.String("policy", bb.policy.String()),
		)
		return
	}

	// only pass ready subconns to picker
	scToAddr := make(map[balancer.SubConn]resolver.Address)
	for addr, sc := range bb.addrToSc {
		if st, ok := bb.scToSt[sc]; ok && st == grpcconnectivity.Ready {
			scToAddr[sc] = addr
		}
	}

	bb.picker = picker.New(picker.Config{
		Policy:                   bb.policy,
		Logger:                   bb.lg,
		SubConnToResolverAddress: scToAddr,
	})
	bb.lg.Info(
		"updated picker",
		zap.String("picker", bb.picker.String()),
		zap.String("balancer-id", bb.id),
		zap.String("policy", bb.policy.String()),
		zap.Strings("subconn-ready", scsToStrings(scToAddr)),
		zap.Int("subconn-size", len(scToAddr)),
	)
}

// Close implements "grpc/balancer.Balancer" interface.
// Close is a nop because base balancer doesn't have internal state to clean up,
// and it doesn't need to call RemoveSubConn for the SubConns.
func (bb *baseBalancer) Close() {
	// TODO
}
