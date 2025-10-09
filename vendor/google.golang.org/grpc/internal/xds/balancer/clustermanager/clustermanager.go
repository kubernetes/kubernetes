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

// Package clustermanager implements the cluster manager LB policy for xds.
package clustermanager

import (
	"encoding/json"
	"fmt"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/base"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/balancergroup"
	internalgrpclog "google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/hierarchy"
	"google.golang.org/grpc/internal/pretty"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
)

const balancerName = "xds_cluster_manager_experimental"

func init() {
	balancer.Register(bb{})
}

type bb struct{}

func (bb) Build(cc balancer.ClientConn, opts balancer.BuildOptions) balancer.Balancer {
	b := &bal{}
	b.logger = prefixLogger(b)
	b.stateAggregator = newBalancerStateAggregator(cc, b.logger)
	b.bg = balancergroup.New(balancergroup.Options{
		CC:                      cc,
		BuildOpts:               opts,
		StateAggregator:         b.stateAggregator,
		Logger:                  b.logger,
		SubBalancerCloseTimeout: time.Duration(0), // Disable caching of removed child policies
	})
	b.logger.Infof("Created")
	return b
}

func (bb) Name() string {
	return balancerName
}

func (bb) ParseConfig(c json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
	return parseConfig(c)
}

type bal struct {
	logger          *internalgrpclog.PrefixLogger
	bg              *balancergroup.BalancerGroup
	stateAggregator *balancerStateAggregator

	children map[string]childConfig
}

func (b *bal) setErrorPickerForChild(childName string, err error) {
	b.stateAggregator.UpdateState(childName, balancer.State{
		ConnectivityState: connectivity.TransientFailure,
		Picker:            base.NewErrPicker(err),
	})
}

func (b *bal) updateChildren(s balancer.ClientConnState, newConfig *lbConfig) error {
	// TODO: Get rid of handling hierarchy in addresses. This LB policy never
	// gets addresses from the resolver.
	addressesSplit := hierarchy.Group(s.ResolverState.Addresses)
	endpointsSplit := hierarchy.GroupEndpoints(s.ResolverState.Endpoints)

	// Remove sub-balancers that are not in the new list from the aggregator and
	// balancergroup.
	for name := range b.children {
		if _, ok := newConfig.Children[name]; !ok {
			b.stateAggregator.remove(name)
			b.bg.Remove(name)
		}
	}

	var retErr error
	for childName, childCfg := range newConfig.Children {
		lbCfg := childCfg.ChildPolicy.Config
		if _, ok := b.children[childName]; !ok {
			// Add new sub-balancers to the aggregator and balancergroup.
			b.stateAggregator.add(childName)
			b.bg.Add(childName, balancer.Get(childCfg.ChildPolicy.Name))
		} else {
			// If the child policy type has changed for existing sub-balancers,
			// parse the new config and send down the config update to the
			// balancergroup, which will take care of gracefully switching the
			// child over to the new policy.
			//
			// If we run into errors here, we need to ensure that RPCs to this
			// child fail, while RPCs to other children with good configs
			// continue to succeed.
			newPolicyName, oldPolicyName := childCfg.ChildPolicy.Name, b.children[childName].ChildPolicy.Name
			if newPolicyName != oldPolicyName {
				var err error
				var cfgJSON []byte
				cfgJSON, err = childCfg.ChildPolicy.MarshalJSON()
				if err != nil {
					retErr = fmt.Errorf("failed to JSON marshal load balancing policy for child %q: %v", childName, err)
					b.setErrorPickerForChild(childName, retErr)
					continue
				}
				// This overwrites lbCfg to be in the format expected by the
				// gracefulswitch balancer. So, when this config is pushed to
				// the child (below), it will result in a graceful switch to the
				// new child policy.
				lbCfg, err = balancergroup.ParseConfig(cfgJSON)
				if err != nil {
					retErr = fmt.Errorf("failed to parse load balancing policy for child %q: %v", childName, err)
					b.setErrorPickerForChild(childName, retErr)
					continue
				}
			}
		}

		if err := b.bg.UpdateClientConnState(childName, balancer.ClientConnState{
			ResolverState: resolver.State{
				Addresses:     addressesSplit[childName],
				Endpoints:     endpointsSplit[childName],
				ServiceConfig: s.ResolverState.ServiceConfig,
				Attributes:    s.ResolverState.Attributes,
			},
			BalancerConfig: lbCfg,
		}); err != nil {
			retErr = fmt.Errorf("failed to push new configuration %v to child %q", childCfg.ChildPolicy.Config, childName)
			b.setErrorPickerForChild(childName, retErr)
		}

		// Picker update is sent to the parent ClientConn only after the
		// new child policy returns a picker. So, there is no need to
		// set needUpdateStateOnResume to true here.
	}

	b.children = newConfig.Children

	// If multiple sub-balancers run into errors, we will return only the last
	// one, which is still good enough, since the grpc channel will anyways
	// return this error as balancer.ErrBadResolver to the name resolver,
	// resulting in re-resolution attempts.
	return retErr

	// Adding or removing a sub-balancer will result in the
	// needUpdateStateOnResume bit to true which results in a picker update once
	// resumeStateUpdates() is called.
}

func (b *bal) UpdateClientConnState(s balancer.ClientConnState) error {
	if b.logger.V(2) {
		b.logger.Infof("Received update from resolver, balancer config: %+v", pretty.ToJSON(s.BalancerConfig))
	}

	newConfig, ok := s.BalancerConfig.(*lbConfig)
	if !ok {
		return fmt.Errorf("unexpected balancer config with type: %T", s.BalancerConfig)
	}

	b.stateAggregator.pauseStateUpdates()
	defer b.stateAggregator.resumeStateUpdates()
	return b.updateChildren(s, newConfig)
}

func (b *bal) ResolverError(err error) {
	b.bg.ResolverError(err)
}

func (b *bal) UpdateSubConnState(sc balancer.SubConn, state balancer.SubConnState) {
	b.logger.Errorf("UpdateSubConnState(%v, %+v) called unexpectedly", sc, state)
}

func (b *bal) Close() {
	b.stateAggregator.close()
	b.bg.Close()
	b.logger.Infof("Shutdown")
}

func (b *bal) ExitIdle() {
	b.bg.ExitIdle()
}

const prefix = "[xds-cluster-manager-lb %p] "

var logger = grpclog.Component("xds")

func prefixLogger(p *bal) *internalgrpclog.PrefixLogger {
	return internalgrpclog.NewPrefixLogger(logger, fmt.Sprintf(prefix, p))
}
