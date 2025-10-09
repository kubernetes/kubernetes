/*
 *
 * Copyright 2025 gRPC authors.
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

// Package lazy contains a load balancer that starts in IDLE instead of
// CONNECTING. Once it starts connecting, it instantiates its delegate.
//
// # Experimental
//
// Notice: This package is EXPERIMENTAL and may be changed or removed in a
// later release.
package lazy

import (
	"fmt"
	"sync"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/resolver"

	internalgrpclog "google.golang.org/grpc/internal/grpclog"
)

var (
	logger = grpclog.Component("lazy-lb")
)

const (
	logPrefix = "[lazy-lb %p] "
)

// ChildBuilderFunc creates a new balancer with the ClientConn. It has the same
// type as the balancer.Builder.Build method.
type ChildBuilderFunc func(cc balancer.ClientConn, opts balancer.BuildOptions) balancer.Balancer

// NewBalancer is the constructor for the lazy balancer.
func NewBalancer(cc balancer.ClientConn, bOpts balancer.BuildOptions, childBuilder ChildBuilderFunc) balancer.Balancer {
	b := &lazyBalancer{
		cc:           cc,
		buildOptions: bOpts,
		childBuilder: childBuilder,
	}
	b.logger = internalgrpclog.NewPrefixLogger(logger, fmt.Sprintf(logPrefix, b))
	cc.UpdateState(balancer.State{
		ConnectivityState: connectivity.Idle,
		Picker: &idlePicker{exitIdle: sync.OnceFunc(func() {
			// Call ExitIdle in a new goroutine to avoid deadlocks while calling
			// back into the channel synchronously.
			go b.ExitIdle()
		})},
	})
	return b
}

type lazyBalancer struct {
	// The following fields are initialized at build time and read-only after
	// that and therefore do not need to be guarded by a mutex.
	cc           balancer.ClientConn
	buildOptions balancer.BuildOptions
	logger       *internalgrpclog.PrefixLogger
	childBuilder ChildBuilderFunc

	// The following fields are accessed while handling calls to the idlePicker
	// and when handling ClientConn state updates. They are guarded by a mutex.

	mu                    sync.Mutex
	delegate              balancer.Balancer
	latestClientConnState *balancer.ClientConnState
	latestResolverError   error
}

func (lb *lazyBalancer) Close() {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	if lb.delegate != nil {
		lb.delegate.Close()
		lb.delegate = nil
	}
}

func (lb *lazyBalancer) ResolverError(err error) {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	if lb.delegate != nil {
		lb.delegate.ResolverError(err)
		return
	}
	lb.latestResolverError = err
}

func (lb *lazyBalancer) UpdateClientConnState(ccs balancer.ClientConnState) error {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	if lb.delegate != nil {
		return lb.delegate.UpdateClientConnState(ccs)
	}

	lb.latestClientConnState = &ccs
	lb.latestResolverError = nil
	return nil
}

// UpdateSubConnState implements balancer.Balancer.
func (lb *lazyBalancer) UpdateSubConnState(balancer.SubConn, balancer.SubConnState) {
	// UpdateSubConnState is deprecated.
}

func (lb *lazyBalancer) ExitIdle() {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	if lb.delegate != nil {
		lb.delegate.ExitIdle()
		return
	}
	lb.delegate = lb.childBuilder(lb.cc, lb.buildOptions)
	if lb.latestClientConnState != nil {
		if err := lb.delegate.UpdateClientConnState(*lb.latestClientConnState); err != nil {
			if err == balancer.ErrBadResolverState {
				lb.cc.ResolveNow(resolver.ResolveNowOptions{})
			} else {
				lb.logger.Warningf("Error from child policy on receiving initial state: %v", err)
			}
		}
		lb.latestClientConnState = nil
	}
	if lb.latestResolverError != nil {
		lb.delegate.ResolverError(lb.latestResolverError)
		lb.latestResolverError = nil
	}
}

// idlePicker is used when the SubConn is IDLE and kicks the SubConn into
// CONNECTING when Pick is called.
type idlePicker struct {
	exitIdle func()
}

func (i *idlePicker) Pick(balancer.PickInfo) (balancer.PickResult, error) {
	i.exitIdle()
	return balancer.PickResult{}, balancer.ErrNoSubConnAvailable
}
