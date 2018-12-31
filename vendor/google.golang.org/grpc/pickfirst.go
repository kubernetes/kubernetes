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
	"golang.org/x/net/context"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/resolver"
)

func newPickfirstBuilder() balancer.Builder {
	return &pickfirstBuilder{}
}

type pickfirstBuilder struct{}

func (*pickfirstBuilder) Build(cc balancer.ClientConn, opt balancer.BuildOptions) balancer.Balancer {
	return &pickfirstBalancer{cc: cc}
}

func (*pickfirstBuilder) Name() string {
	return "pickfirst"
}

type pickfirstBalancer struct {
	cc balancer.ClientConn
	sc balancer.SubConn
}

func (b *pickfirstBalancer) HandleResolvedAddrs(addrs []resolver.Address, err error) {
	if err != nil {
		grpclog.Infof("pickfirstBalancer: HandleResolvedAddrs called with error %v", err)
		return
	}
	if b.sc == nil {
		b.sc, err = b.cc.NewSubConn(addrs, balancer.NewSubConnOptions{})
		if err != nil {
			grpclog.Errorf("pickfirstBalancer: failed to NewSubConn: %v", err)
			return
		}
		b.cc.UpdateBalancerState(connectivity.Idle, &picker{sc: b.sc})
	} else {
		b.sc.UpdateAddresses(addrs)
	}
}

func (b *pickfirstBalancer) HandleSubConnStateChange(sc balancer.SubConn, s connectivity.State) {
	grpclog.Infof("pickfirstBalancer: HandleSubConnStateChange: %p, %v", sc, s)
	if b.sc != sc || s == connectivity.Shutdown {
		b.sc = nil
		return
	}

	switch s {
	case connectivity.Ready, connectivity.Idle:
		b.cc.UpdateBalancerState(s, &picker{sc: sc})
	case connectivity.Connecting:
		b.cc.UpdateBalancerState(s, &picker{err: balancer.ErrNoSubConnAvailable})
	case connectivity.TransientFailure:
		b.cc.UpdateBalancerState(s, &picker{err: balancer.ErrTransientFailure})
	}
}

func (b *pickfirstBalancer) Close() {
}

type picker struct {
	err error
	sc  balancer.SubConn
}

func (p *picker) Pick(ctx context.Context, opts balancer.PickOptions) (balancer.SubConn, func(balancer.DoneInfo), error) {
	if p.err != nil {
		return nil, nil, p.err
	}
	return p.sc, nil, nil
}
