/*
 *
 * Copyright 2021 gRPC authors.
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

package clusterresolver

import (
	"fmt"
	"net/url"
	"sync"

	"google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/pretty"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
)

var (
	newDNS = func(target resolver.Target, cc resolver.ClientConn, opts resolver.BuildOptions) (resolver.Resolver, error) {
		// The dns resolver is registered by the grpc package. So, this call to
		// resolver.Get() is never expected to return nil.
		return resolver.Get("dns").Build(target, cc, opts)
	}
)

// dnsDiscoveryMechanism watches updates for the given DNS hostname.
//
// It implements resolver.ClientConn interface to work with the DNS resolver.
type dnsDiscoveryMechanism struct {
	target           string
	topLevelResolver topLevelResolver
	dnsR             resolver.Resolver
	logger           *grpclog.PrefixLogger

	mu             sync.Mutex
	endpoints      []resolver.Endpoint
	updateReceived bool
}

// newDNSResolver creates an endpoints resolver which uses a DNS resolver under
// the hood.
//
// An error in parsing the provided target string or an error in creating a DNS
// resolver means that we will never be able to resolve the provided target
// strings to endpoints. The topLevelResolver propagates address updates to the
// clusterresolver LB policy **only** after it receives updates from all its
// child resolvers. Therefore, an error here means that the topLevelResolver
// will never send address updates to the clusterresolver LB policy.
//
// Calling the onError() callback will ensure that this error is
// propagated to the child policy which eventually move the channel to
// transient failure.
//
// The `dnsR` field is unset if we run into errors in this function. Therefore, a
// nil check is required wherever we access that field.
func newDNSResolver(target string, topLevelResolver topLevelResolver, logger *grpclog.PrefixLogger) *dnsDiscoveryMechanism {
	ret := &dnsDiscoveryMechanism{
		target:           target,
		topLevelResolver: topLevelResolver,
		logger:           logger,
	}
	u, err := url.Parse("dns:///" + target)
	if err != nil {
		if ret.logger.V(2) {
			ret.logger.Infof("Failed to parse dns hostname %q in clusterresolver LB policy", target)
		}
		ret.updateReceived = true
		ret.topLevelResolver.onUpdate(func() {})
		return ret
	}

	r, err := newDNS(resolver.Target{URL: *u}, ret, resolver.BuildOptions{})
	if err != nil {
		if ret.logger.V(2) {
			ret.logger.Infof("Failed to build DNS resolver for target %q: %v", target, err)
		}
		ret.updateReceived = true
		ret.topLevelResolver.onUpdate(func() {})
		return ret
	}
	ret.dnsR = r
	return ret
}

func (dr *dnsDiscoveryMechanism) lastUpdate() (any, bool) {
	dr.mu.Lock()
	defer dr.mu.Unlock()

	if !dr.updateReceived {
		return nil, false
	}
	return dr.endpoints, true
}

func (dr *dnsDiscoveryMechanism) resolveNow() {
	if dr.dnsR != nil {
		dr.dnsR.ResolveNow(resolver.ResolveNowOptions{})
	}
}

// The definition of stop() mentions that implementations must not invoke any
// methods on the topLevelResolver once the call to `stop()` returns. The
// underlying dns resolver does not send any updates to the resolver.ClientConn
// interface passed to it (implemented by dnsDiscoveryMechanism in this case)
// after its `Close()` returns. Therefore, we can guarantee that no methods of
// the topLevelResolver are invoked after we return from this method.
func (dr *dnsDiscoveryMechanism) stop() {
	if dr.dnsR != nil {
		dr.dnsR.Close()
	}
}

// dnsDiscoveryMechanism needs to implement resolver.ClientConn interface to receive
// updates from the real DNS resolver.

func (dr *dnsDiscoveryMechanism) UpdateState(state resolver.State) error {
	if dr.logger.V(2) {
		dr.logger.Infof("DNS discovery mechanism for resource %q reported an update: %s", dr.target, pretty.ToJSON(state))
	}

	dr.mu.Lock()
	var endpoints = state.Endpoints
	if len(endpoints) == 0 {
		endpoints = make([]resolver.Endpoint, len(state.Addresses))
		for i, a := range state.Addresses {
			endpoints[i] = resolver.Endpoint{Addresses: []resolver.Address{a}}
			endpoints[i].Attributes = a.BalancerAttributes
		}
	}
	dr.endpoints = endpoints
	dr.updateReceived = true
	dr.mu.Unlock()

	dr.topLevelResolver.onUpdate(func() {})
	return nil
}

func (dr *dnsDiscoveryMechanism) ReportError(err error) {
	if dr.logger.V(2) {
		dr.logger.Infof("DNS discovery mechanism for resource %q reported error: %v", dr.target, err)
	}

	dr.mu.Lock()
	// If a previous good update was received, suppress the error and continue
	// using the previous update. If RPCs were succeeding prior to this, they
	// will continue to do so. Also suppress errors if we previously received an
	// error, since there will be no downstream effects of propagating this
	// error.
	if dr.updateReceived {
		dr.mu.Unlock()
		return
	}
	dr.endpoints = nil
	dr.updateReceived = true
	dr.mu.Unlock()

	dr.topLevelResolver.onUpdate(func() {})
}

func (dr *dnsDiscoveryMechanism) NewAddress(addresses []resolver.Address) {
	dr.UpdateState(resolver.State{Addresses: addresses})
}

func (dr *dnsDiscoveryMechanism) ParseServiceConfig(string) *serviceconfig.ParseResult {
	return &serviceconfig.ParseResult{Err: fmt.Errorf("service config not supported")}
}
