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

	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/resolver"
)

// ccResolverWrapper is a wrapper on top of cc for resolvers.
// It implements resolver.ClientConnection interface.
type ccResolverWrapper struct {
	cc       *ClientConn
	resolver resolver.Resolver
	addrCh   chan []resolver.Address
	scCh     chan string
	done     chan struct{}
}

// split2 returns the values from strings.SplitN(s, sep, 2).
// If sep is not found, it returns "", s instead.
func split2(s, sep string) (string, string) {
	spl := strings.SplitN(s, sep, 2)
	if len(spl) < 2 {
		return "", s
	}
	return spl[0], spl[1]
}

// parseTarget splits target into a struct containing scheme, authority and
// endpoint.
func parseTarget(target string) (ret resolver.Target) {
	ret.Scheme, ret.Endpoint = split2(target, "://")
	ret.Authority, ret.Endpoint = split2(ret.Endpoint, "/")
	return ret
}

// newCCResolverWrapper parses cc.target for scheme and gets the resolver
// builder for this scheme. It then builds the resolver and starts the
// monitoring goroutine for it.
//
// This function could return nil, nil, in tests for old behaviors.
// TODO(bar) never return nil, nil when DNS becomes the default resolver.
func newCCResolverWrapper(cc *ClientConn) (*ccResolverWrapper, error) {
	target := parseTarget(cc.target)
	grpclog.Infof("dialing to target with scheme: %q", target.Scheme)

	rb := resolver.Get(target.Scheme)
	if rb == nil {
		// TODO(bar) return error when DNS becomes the default (implemented and
		// registered by DNS package).
		grpclog.Infof("could not get resolver for scheme: %q", target.Scheme)
		return nil, nil
	}

	ccr := &ccResolverWrapper{
		cc:     cc,
		addrCh: make(chan []resolver.Address, 1),
		scCh:   make(chan string, 1),
		done:   make(chan struct{}),
	}

	var err error
	ccr.resolver, err = rb.Build(target, ccr, resolver.BuildOption{})
	if err != nil {
		return nil, err
	}
	go ccr.watcher()
	return ccr, nil
}

// watcher processes address updates and service config updates sequencially.
// Otherwise, we need to resolve possible races between address and service
// config (e.g. they specify different balancer types).
func (ccr *ccResolverWrapper) watcher() {
	for {
		select {
		case <-ccr.done:
			return
		default:
		}

		select {
		case addrs := <-ccr.addrCh:
			grpclog.Infof("ccResolverWrapper: sending new addresses to balancer wrapper: %v", addrs)
			// TODO(bar switching) this should never be nil. Pickfirst should be default.
			if ccr.cc.balancerWrapper != nil {
				// TODO(bar switching) create balancer if it's nil?
				ccr.cc.balancerWrapper.handleResolvedAddrs(addrs, nil)
			}
		case sc := <-ccr.scCh:
			grpclog.Infof("ccResolverWrapper: got new service config: %v", sc)
		case <-ccr.done:
			return
		}
	}
}

func (ccr *ccResolverWrapper) close() {
	ccr.resolver.Close()
	close(ccr.done)
}

// NewAddress is called by the resolver implemenetion to send addresses to gRPC.
func (ccr *ccResolverWrapper) NewAddress(addrs []resolver.Address) {
	select {
	case <-ccr.addrCh:
	default:
	}
	ccr.addrCh <- addrs
}

// NewServiceConfig is called by the resolver implemenetion to send service
// configs to gPRC.
func (ccr *ccResolverWrapper) NewServiceConfig(sc string) {
	select {
	case <-ccr.scCh:
	default:
	}
	ccr.scCh <- sc
}
