/*
 *
 * Copyright 2020 gRPC authors.
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

// Package weightedtarget implements the weighted_target balancer.
//
// All APIs in this package are experimental.
package weightedtarget

import (
	"encoding/json"
	"fmt"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/weightedtarget/weightedaggregator"
	"google.golang.org/grpc/internal/balancergroup"
	"google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/hierarchy"
	"google.golang.org/grpc/internal/pretty"
	"google.golang.org/grpc/internal/wrr"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
)

// Name is the name of the weighted_target balancer.
const Name = "weighted_target_experimental"

// NewRandomWRR is the WRR constructor used to pick sub-pickers from
// sub-balancers. It's to be modified in tests.
var NewRandomWRR = wrr.NewRandom

func init() {
	balancer.Register(bb{})
}

type bb struct{}

func (bb) Build(cc balancer.ClientConn, bOpts balancer.BuildOptions) balancer.Balancer {
	b := &weightedTargetBalancer{}
	b.logger = prefixLogger(b)
	b.stateAggregator = weightedaggregator.New(cc, b.logger, NewRandomWRR)
	b.stateAggregator.Start()
	b.bg = balancergroup.New(balancergroup.Options{
		CC:                      cc,
		BuildOpts:               bOpts,
		StateAggregator:         b.stateAggregator,
		Logger:                  b.logger,
		SubBalancerCloseTimeout: time.Duration(0), // Disable caching of removed child policies
	})
	b.logger.Infof("Created")
	return b
}

func (bb) Name() string {
	return Name
}

func (bb) ParseConfig(c json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
	return parseConfig(c)
}

type weightedTargetBalancer struct {
	logger *grpclog.PrefixLogger

	bg              *balancergroup.BalancerGroup
	stateAggregator *weightedaggregator.Aggregator

	targets map[string]Target
}

type localityKeyType string

const localityKey = localityKeyType("locality")

// LocalityFromResolverState returns the locality from the resolver.State
// provided, or an empty string if not present.
func LocalityFromResolverState(state resolver.State) string {
	locality, _ := state.Attributes.Value(localityKey).(string)
	return locality
}

// UpdateClientConnState takes the new targets in balancer group,
// creates/deletes sub-balancers and sends them update. addresses are split into
// groups based on hierarchy path.
func (b *weightedTargetBalancer) UpdateClientConnState(s balancer.ClientConnState) error {
	if b.logger.V(2) {
		b.logger.Infof("Received update from resolver, balancer config: %+v", pretty.ToJSON(s.BalancerConfig))
	}

	newConfig, ok := s.BalancerConfig.(*LBConfig)
	if !ok {
		return fmt.Errorf("unexpected balancer config with type: %T", s.BalancerConfig)
	}
	addressesSplit := hierarchy.Group(s.ResolverState.Addresses)
	endpointsSplit := hierarchy.GroupEndpoints(s.ResolverState.Endpoints)

	b.stateAggregator.PauseStateUpdates()
	defer b.stateAggregator.ResumeStateUpdates()

	// Remove sub-pickers and sub-balancers that are not in the new config.
	for name := range b.targets {
		if _, ok := newConfig.Targets[name]; !ok {
			b.stateAggregator.Remove(name)
			b.bg.Remove(name)
		}
	}

	// For sub-balancers in the new config
	// - if it's new. add to balancer group,
	// - if it's old, but has a new weight, update weight in balancer group.
	//
	// For all sub-balancers, forward the address/balancer config update.
	for name, newT := range newConfig.Targets {
		oldT, ok := b.targets[name]
		if !ok {
			// If this is a new sub-balancer, add weights to the picker map.
			b.stateAggregator.Add(name, newT.Weight)
			// Then add to the balancer group.
			b.bg.Add(name, balancer.Get(newT.ChildPolicy.Name))
			// Not trigger a state/picker update. Wait for the new sub-balancer
			// to send its updates.
		} else if newT.ChildPolicy.Name != oldT.ChildPolicy.Name {
			// If the child policy name is different, remove from balancer group
			// and re-add.
			b.stateAggregator.Remove(name)
			b.bg.Remove(name)
			b.stateAggregator.Add(name, newT.Weight)
			b.bg.Add(name, balancer.Get(newT.ChildPolicy.Name))
		} else if newT.Weight != oldT.Weight {
			// If this is an existing sub-balancer, update weight if necessary.
			b.stateAggregator.UpdateWeight(name, newT.Weight)
		}

		// Forwards all the update:
		// - addresses are from the map after splitting with hierarchy path,
		// - Top level service config and attributes are the same,
		// - Balancer config comes from the targets map.
		//
		// TODO: handle error? How to aggregate errors and return?
		_ = b.bg.UpdateClientConnState(name, balancer.ClientConnState{
			ResolverState: resolver.State{
				Addresses:     addressesSplit[name],
				Endpoints:     endpointsSplit[name],
				ServiceConfig: s.ResolverState.ServiceConfig,
				Attributes:    s.ResolverState.Attributes.WithValue(localityKey, name),
			},
			BalancerConfig: newT.ChildPolicy.Config,
		})
	}

	b.targets = newConfig.Targets

	// If the targets length is zero, it means we have removed all child
	// policies from the balancer group and aggregator.
	// At the start of this UpdateClientConnState() operation, a call to
	// b.stateAggregator.ResumeStateUpdates() is deferred. Thus, setting the
	// needUpdateStateOnResume bool to true here will ensure a new picker is
	// built as part of that deferred function. Since there are now no child
	// policies, the aggregated connectivity state reported form the Aggregator
	// will be TRANSIENT_FAILURE.
	if len(b.targets) == 0 {
		b.stateAggregator.NeedUpdateStateOnResume()
	}

	return nil
}

func (b *weightedTargetBalancer) ResolverError(err error) {
	b.bg.ResolverError(err)
}

func (b *weightedTargetBalancer) UpdateSubConnState(sc balancer.SubConn, state balancer.SubConnState) {
	b.logger.Errorf("UpdateSubConnState(%v, %+v) called unexpectedly", sc, state)
}

func (b *weightedTargetBalancer) Close() {
	b.stateAggregator.Stop()
	b.bg.Close()
}

func (b *weightedTargetBalancer) ExitIdle() {
	b.bg.ExitIdle()
}
