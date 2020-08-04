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
	"strings"
	"sync"

	"golang.org/x/net/context"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/status"
)

type balancerWrapperBuilder struct {
	b Balancer // The v1 balancer.
}

func (bwb *balancerWrapperBuilder) Build(cc balancer.ClientConn, opts balancer.BuildOptions) balancer.Balancer {
	targetAddr := cc.Target()
	targetSplitted := strings.Split(targetAddr, ":///")
	if len(targetSplitted) >= 2 {
		targetAddr = targetSplitted[1]
	}

	bwb.b.Start(targetAddr, BalancerConfig{
		DialCreds: opts.DialCreds,
		Dialer:    opts.Dialer,
	})
	_, pickfirst := bwb.b.(*pickFirst)
	bw := &balancerWrapper{
		balancer:   bwb.b,
		pickfirst:  pickfirst,
		cc:         cc,
		targetAddr: targetAddr,
		startCh:    make(chan struct{}),
		conns:      make(map[resolver.Address]balancer.SubConn),
		connSt:     make(map[balancer.SubConn]*scState),
		csEvltr:    &balancer.ConnectivityStateEvaluator{},
		state:      connectivity.Idle,
	}
	cc.UpdateBalancerState(connectivity.Idle, bw)
	go bw.lbWatcher()
	return bw
}

func (bwb *balancerWrapperBuilder) Name() string {
	return "wrapper"
}

type scState struct {
	addr Address // The v1 address type.
	s    connectivity.State
	down func(error)
}

type balancerWrapper struct {
	balancer  Balancer // The v1 balancer.
	pickfirst bool

	cc         balancer.ClientConn
	targetAddr string // Target without the scheme.

	mu     sync.Mutex
	conns  map[resolver.Address]balancer.SubConn
	connSt map[balancer.SubConn]*scState
	// This channel is closed when handling the first resolver result.
	// lbWatcher blocks until this is closed, to avoid race between
	// - NewSubConn is created, cc wants to notify balancer of state changes;
	// - Build hasn't return, cc doesn't have access to balancer.
	startCh chan struct{}

	// To aggregate the connectivity state.
	csEvltr *balancer.ConnectivityStateEvaluator
	state   connectivity.State
}

// lbWatcher watches the Notify channel of the balancer and manages
// connections accordingly.
func (bw *balancerWrapper) lbWatcher() {
	<-bw.startCh
	notifyCh := bw.balancer.Notify()
	if notifyCh == nil {
		// There's no resolver in the balancer. Connect directly.
		a := resolver.Address{
			Addr: bw.targetAddr,
			Type: resolver.Backend,
		}
		sc, err := bw.cc.NewSubConn([]resolver.Address{a}, balancer.NewSubConnOptions{})
		if err != nil {
			grpclog.Warningf("Error creating connection to %v. Err: %v", a, err)
		} else {
			bw.mu.Lock()
			bw.conns[a] = sc
			bw.connSt[sc] = &scState{
				addr: Address{Addr: bw.targetAddr},
				s:    connectivity.Idle,
			}
			bw.mu.Unlock()
			sc.Connect()
		}
		return
	}

	for addrs := range notifyCh {
		grpclog.Infof("balancerWrapper: got update addr from Notify: %v\n", addrs)
		if bw.pickfirst {
			var (
				oldA  resolver.Address
				oldSC balancer.SubConn
			)
			bw.mu.Lock()
			for oldA, oldSC = range bw.conns {
				break
			}
			bw.mu.Unlock()
			if len(addrs) <= 0 {
				if oldSC != nil {
					// Teardown old sc.
					bw.mu.Lock()
					delete(bw.conns, oldA)
					delete(bw.connSt, oldSC)
					bw.mu.Unlock()
					bw.cc.RemoveSubConn(oldSC)
				}
				continue
			}

			var newAddrs []resolver.Address
			for _, a := range addrs {
				newAddr := resolver.Address{
					Addr:       a.Addr,
					Type:       resolver.Backend, // All addresses from balancer are all backends.
					ServerName: "",
					Metadata:   a.Metadata,
				}
				newAddrs = append(newAddrs, newAddr)
			}
			if oldSC == nil {
				// Create new sc.
				sc, err := bw.cc.NewSubConn(newAddrs, balancer.NewSubConnOptions{})
				if err != nil {
					grpclog.Warningf("Error creating connection to %v. Err: %v", newAddrs, err)
				} else {
					bw.mu.Lock()
					// For pickfirst, there should be only one SubConn, so the
					// address doesn't matter. All states updating (up and down)
					// and picking should all happen on that only SubConn.
					bw.conns[resolver.Address{}] = sc
					bw.connSt[sc] = &scState{
						addr: addrs[0], // Use the first address.
						s:    connectivity.Idle,
					}
					bw.mu.Unlock()
					sc.Connect()
				}
			} else {
				bw.mu.Lock()
				bw.connSt[oldSC].addr = addrs[0]
				bw.mu.Unlock()
				oldSC.UpdateAddresses(newAddrs)
			}
		} else {
			var (
				add []resolver.Address // Addresses need to setup connections.
				del []balancer.SubConn // Connections need to tear down.
			)
			resAddrs := make(map[resolver.Address]Address)
			for _, a := range addrs {
				resAddrs[resolver.Address{
					Addr:       a.Addr,
					Type:       resolver.Backend, // All addresses from balancer are all backends.
					ServerName: "",
					Metadata:   a.Metadata,
				}] = a
			}
			bw.mu.Lock()
			for a := range resAddrs {
				if _, ok := bw.conns[a]; !ok {
					add = append(add, a)
				}
			}
			for a, c := range bw.conns {
				if _, ok := resAddrs[a]; !ok {
					del = append(del, c)
					delete(bw.conns, a)
					// Keep the state of this sc in bw.connSt until its state becomes Shutdown.
				}
			}
			bw.mu.Unlock()
			for _, a := range add {
				sc, err := bw.cc.NewSubConn([]resolver.Address{a}, balancer.NewSubConnOptions{})
				if err != nil {
					grpclog.Warningf("Error creating connection to %v. Err: %v", a, err)
				} else {
					bw.mu.Lock()
					bw.conns[a] = sc
					bw.connSt[sc] = &scState{
						addr: resAddrs[a],
						s:    connectivity.Idle,
					}
					bw.mu.Unlock()
					sc.Connect()
				}
			}
			for _, c := range del {
				bw.cc.RemoveSubConn(c)
			}
		}
	}
}

func (bw *balancerWrapper) HandleSubConnStateChange(sc balancer.SubConn, s connectivity.State) {
	bw.mu.Lock()
	defer bw.mu.Unlock()
	scSt, ok := bw.connSt[sc]
	if !ok {
		return
	}
	if s == connectivity.Idle {
		sc.Connect()
	}
	oldS := scSt.s
	scSt.s = s
	if oldS != connectivity.Ready && s == connectivity.Ready {
		scSt.down = bw.balancer.Up(scSt.addr)
	} else if oldS == connectivity.Ready && s != connectivity.Ready {
		if scSt.down != nil {
			scSt.down(errConnClosing)
		}
	}
	sa := bw.csEvltr.RecordTransition(oldS, s)
	if bw.state != sa {
		bw.state = sa
	}
	bw.cc.UpdateBalancerState(bw.state, bw)
	if s == connectivity.Shutdown {
		// Remove state for this sc.
		delete(bw.connSt, sc)
	}
}

func (bw *balancerWrapper) HandleResolvedAddrs([]resolver.Address, error) {
	bw.mu.Lock()
	defer bw.mu.Unlock()
	select {
	case <-bw.startCh:
	default:
		close(bw.startCh)
	}
	// There should be a resolver inside the balancer.
	// All updates here, if any, are ignored.
}

func (bw *balancerWrapper) Close() {
	bw.mu.Lock()
	defer bw.mu.Unlock()
	select {
	case <-bw.startCh:
	default:
		close(bw.startCh)
	}
	bw.balancer.Close()
}

// The picker is the balancerWrapper itself.
// Pick should never return ErrNoSubConnAvailable.
// It either blocks or returns error, consistent with v1 balancer Get().
func (bw *balancerWrapper) Pick(ctx context.Context, opts balancer.PickOptions) (balancer.SubConn, func(balancer.DoneInfo), error) {
	failfast := true // Default failfast is true.
	if ss, ok := rpcInfoFromContext(ctx); ok {
		failfast = ss.failfast
	}
	a, p, err := bw.balancer.Get(ctx, BalancerGetOptions{BlockingWait: !failfast})
	if err != nil {
		return nil, nil, err
	}
	var done func(balancer.DoneInfo)
	if p != nil {
		done = func(i balancer.DoneInfo) { p() }
	}
	var sc balancer.SubConn
	bw.mu.Lock()
	defer bw.mu.Unlock()
	if bw.pickfirst {
		// Get the first sc in conns.
		for _, sc = range bw.conns {
			break
		}
	} else {
		var ok bool
		sc, ok = bw.conns[resolver.Address{
			Addr:       a.Addr,
			Type:       resolver.Backend,
			ServerName: "",
			Metadata:   a.Metadata,
		}]
		if !ok && failfast {
			return nil, nil, status.Errorf(codes.Unavailable, "there is no connection available")
		}
		if s, ok := bw.connSt[sc]; failfast && (!ok || s.s != connectivity.Ready) {
			// If the returned sc is not ready and RPC is failfast,
			// return error, and this RPC will fail.
			return nil, nil, status.Errorf(codes.Unavailable, "there is no connection available")
		}
	}

	return sc, done, nil
}
