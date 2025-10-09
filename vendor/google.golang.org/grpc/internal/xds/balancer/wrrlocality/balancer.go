/*
 *
 * Copyright 2023 gRPC authors.
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

// Package wrrlocality provides an implementation of the wrr locality LB policy,
// as defined in [A52 - xDS Custom LB Policies].
//
// [A52 - xDS Custom LB Policies]: https://github.com/grpc/proposal/blob/master/A52-xds-custom-lb-policies.md
package wrrlocality

import (
	"encoding/json"
	"errors"
	"fmt"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/weightedtarget"
	"google.golang.org/grpc/internal/grpclog"
	internalserviceconfig "google.golang.org/grpc/internal/serviceconfig"
	xdsinternal "google.golang.org/grpc/internal/xds"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
)

// Name is the name of wrr_locality balancer.
const Name = "xds_wrr_locality_experimental"

func init() {
	balancer.Register(bb{})
}

type bb struct{}

func (bb) Name() string {
	return Name
}

// LBConfig is the config for the wrr locality balancer.
type LBConfig struct {
	serviceconfig.LoadBalancingConfig `json:"-"`
	// ChildPolicy is the config for the child policy.
	ChildPolicy *internalserviceconfig.BalancerConfig `json:"childPolicy,omitempty"`
}

// To plumb in a different child in tests.
var weightedTargetName = weightedtarget.Name

func (bb) Build(cc balancer.ClientConn, bOpts balancer.BuildOptions) balancer.Balancer {
	builder := balancer.Get(weightedTargetName)
	if builder == nil {
		// Shouldn't happen, registered through imported weighted target,
		// defensive programming.
		return nil
	}

	// Doesn't need to intercept any balancer.ClientConn operations; pass
	// through by just giving cc to child balancer.
	wtb := builder.Build(cc, bOpts)
	if wtb == nil {
		// shouldn't happen, defensive programming.
		return nil
	}
	wtbCfgParser, ok := builder.(balancer.ConfigParser)
	if !ok {
		// Shouldn't happen, imported weighted target builder has this method.
		return nil
	}
	wrrL := &wrrLocalityBalancer{
		child:       wtb,
		childParser: wtbCfgParser,
	}

	wrrL.logger = prefixLogger(wrrL)
	wrrL.logger.Infof("Created")
	return wrrL
}

func (bb) ParseConfig(s json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
	var lbCfg *LBConfig
	if err := json.Unmarshal(s, &lbCfg); err != nil {
		return nil, fmt.Errorf("xds_wrr_locality: invalid LBConfig: %s, error: %v", string(s), err)
	}
	if lbCfg == nil || lbCfg.ChildPolicy == nil {
		return nil, errors.New("xds_wrr_locality: invalid LBConfig: child policy field must be set")
	}
	return lbCfg, nil
}

type attributeKey struct{}

// Equal allows the values to be compared by Attributes.Equal.
func (a AddrInfo) Equal(o any) bool {
	oa, ok := o.(AddrInfo)
	return ok && oa.LocalityWeight == a.LocalityWeight
}

// AddrInfo is the locality weight of the locality an address is a part of.
type AddrInfo struct {
	LocalityWeight uint32
}

// SetAddrInfo returns a copy of addr in which the BalancerAttributes field is
// updated with AddrInfo.
func SetAddrInfo(addr resolver.Address, addrInfo AddrInfo) resolver.Address {
	addr.BalancerAttributes = addr.BalancerAttributes.WithValue(attributeKey{}, addrInfo)
	return addr
}

// SetAddrInfoInEndpoint returns a copy of endpoint in which the Attributes
// field is updated with AddrInfo.
func SetAddrInfoInEndpoint(endpoint resolver.Endpoint, addrInfo AddrInfo) resolver.Endpoint {
	endpoint.Attributes = endpoint.Attributes.WithValue(attributeKey{}, addrInfo)
	return endpoint
}

func (a AddrInfo) String() string {
	return fmt.Sprintf("Locality Weight: %d", a.LocalityWeight)
}

// getAddrInfo returns the AddrInfo stored in the BalancerAttributes field of
// addr. Returns false if no AddrInfo found.
func getAddrInfo(addr resolver.Address) (AddrInfo, bool) {
	v := addr.BalancerAttributes.Value(attributeKey{})
	ai, ok := v.(AddrInfo)
	return ai, ok
}

// wrrLocalityBalancer wraps a weighted target balancer, and builds
// configuration for the weighted target once it receives configuration
// specifying the weighted target child balancer and locality weight
// information.
type wrrLocalityBalancer struct {
	// child will be a weighted target balancer, and will be built it at
	// wrrLocalityBalancer build time. Other than preparing configuration, other
	// balancer operations are simply pass through.
	child balancer.Balancer

	childParser balancer.ConfigParser

	logger *grpclog.PrefixLogger
}

func (b *wrrLocalityBalancer) ExitIdle() {
	b.child.ExitIdle()
}

func (b *wrrLocalityBalancer) UpdateClientConnState(s balancer.ClientConnState) error {
	lbCfg, ok := s.BalancerConfig.(*LBConfig)
	if !ok {
		b.logger.Errorf("Received config with unexpected type %T: %v", s.BalancerConfig, s.BalancerConfig)
		return balancer.ErrBadResolverState
	}

	weightedTargets := make(map[string]weightedtarget.Target)
	for _, addr := range s.ResolverState.Addresses {
		// This get of LocalityID could potentially return a zero value. This
		// shouldn't happen though (this attribute that is set actually gets
		// used to build localities in the first place), and thus don't error
		// out, and just build a weighted target with undefined behavior.
		locality := xdsinternal.LocalityString(xdsinternal.GetLocalityID(addr))
		ai, ok := getAddrInfo(addr)
		if !ok {
			return fmt.Errorf("xds_wrr_locality: missing locality weight information in address %q", addr)
		}
		weightedTargets[locality] = weightedtarget.Target{Weight: ai.LocalityWeight, ChildPolicy: lbCfg.ChildPolicy}
	}
	wtCfg := &weightedtarget.LBConfig{Targets: weightedTargets}
	wtCfgJSON, err := json.Marshal(wtCfg)
	if err != nil {
		// Shouldn't happen.
		return fmt.Errorf("xds_wrr_locality: error marshalling prepared config: %v", wtCfg)
	}
	var sc serviceconfig.LoadBalancingConfig
	if sc, err = b.childParser.ParseConfig(wtCfgJSON); err != nil {
		return fmt.Errorf("xds_wrr_locality: config generated %v is invalid: %v", wtCfgJSON, err)
	}

	return b.child.UpdateClientConnState(balancer.ClientConnState{
		ResolverState:  s.ResolverState,
		BalancerConfig: sc,
	})
}

func (b *wrrLocalityBalancer) ResolverError(err error) {
	b.child.ResolverError(err)
}

func (b *wrrLocalityBalancer) UpdateSubConnState(sc balancer.SubConn, state balancer.SubConnState) {
	b.logger.Errorf("UpdateSubConnState(%v, %+v) called unexpectedly", sc, state)
}

func (b *wrrLocalityBalancer) Close() {
	b.child.Close()
}
