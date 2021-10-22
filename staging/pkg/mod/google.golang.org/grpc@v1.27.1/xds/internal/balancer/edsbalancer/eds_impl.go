/*
 * Copyright 2019 gRPC authors.
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
 */

package edsbalancer

import (
	"encoding/json"
	"reflect"
	"sync"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/balancer/weightedroundrobin"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/xds/internal"
	"google.golang.org/grpc/xds/internal/balancer/lrs"
	xdsclient "google.golang.org/grpc/xds/internal/client"
)

// TODO: make this a environment variable?
var defaultPriorityInitTimeout = 10 * time.Second

type localityConfig struct {
	weight uint32
	addrs  []resolver.Address
}

// balancerGroupWithConfig contains the localities with the same priority. It
// manages all localities using a balancerGroup.
type balancerGroupWithConfig struct {
	bg      *balancerGroup
	configs map[internal.Locality]*localityConfig
}

// edsBalancerImpl does load balancing based on the EDS responses. Note that it
// doesn't implement the balancer interface. It's intended to be used by a high
// level balancer implementation.
//
// The localities are picked as weighted round robin. A configurable child
// policy is used to manage endpoints in each locality.
type edsBalancerImpl struct {
	cc balancer.ClientConn

	subBalancerBuilder   balancer.Builder
	loadStore            lrs.Store
	priorityToLocalities map[priorityType]*balancerGroupWithConfig

	// There's no need to hold any mutexes at the same time. The order to take
	// mutex should be: priorityMu > subConnMu, but this is implicit via
	// balancers (starting balancer with next priority while holding priorityMu,
	// and the balancer may create new SubConn).

	priorityMu sync.Mutex
	// priorities are pointers, and will be nil when EDS returns empty result.
	priorityInUse   priorityType
	priorityLowest  priorityType
	priorityToState map[priorityType]*balancer.State
	// The timer to give a priority 10 seconds to connect. And if the priority
	// doesn't go into Ready/Failure, start the next priority.
	//
	// One timer is enough because there can be at most one priority in init
	// state.
	priorityInitTimer *time.Timer

	subConnMu         sync.Mutex
	subConnToPriority map[balancer.SubConn]priorityType

	pickerMu   sync.Mutex
	drops      []*dropper
	innerState balancer.State // The state of the picker without drop support.
}

// newEDSBalancerImpl create a new edsBalancerImpl.
func newEDSBalancerImpl(cc balancer.ClientConn, loadStore lrs.Store) *edsBalancerImpl {
	edsImpl := &edsBalancerImpl{
		cc:                 cc,
		subBalancerBuilder: balancer.Get(roundrobin.Name),

		priorityToLocalities: make(map[priorityType]*balancerGroupWithConfig),
		priorityToState:      make(map[priorityType]*balancer.State),
		subConnToPriority:    make(map[balancer.SubConn]priorityType),
		loadStore:            loadStore,
	}
	// Don't start balancer group here. Start it when handling the first EDS
	// response. Otherwise the balancer group will be started with round-robin,
	// and if users specify a different sub-balancer, all balancers in balancer
	// group will be closed and recreated when sub-balancer update happens.
	return edsImpl
}

// HandleChildPolicy updates the child balancers handling endpoints. Child
// policy is roundrobin by default. If the specified balancer is not installed,
// the old child balancer will be used.
//
// HandleChildPolicy and HandleEDSResponse must be called by the same goroutine.
func (edsImpl *edsBalancerImpl) HandleChildPolicy(name string, config json.RawMessage) {
	if edsImpl.subBalancerBuilder.Name() == name {
		return
	}
	newSubBalancerBuilder := balancer.Get(name)
	if newSubBalancerBuilder == nil {
		grpclog.Infof("edsBalancerImpl: failed to find balancer with name %q, keep using %q", name, edsImpl.subBalancerBuilder.Name())
		return
	}
	edsImpl.subBalancerBuilder = newSubBalancerBuilder
	for _, bgwc := range edsImpl.priorityToLocalities {
		if bgwc == nil {
			continue
		}
		for id, config := range bgwc.configs {
			// TODO: (eds) add support to balancer group to support smoothly
			//  switching sub-balancers (keep old balancer around until new
			//  balancer becomes ready).
			bgwc.bg.remove(id)
			bgwc.bg.add(id, config.weight, edsImpl.subBalancerBuilder)
			bgwc.bg.handleResolvedAddrs(id, config.addrs)
		}
	}
}

// updateDrops compares new drop policies with the old. If they are different,
// it updates the drop policies and send ClientConn an updated picker.
func (edsImpl *edsBalancerImpl) updateDrops(dropPolicies []xdsclient.OverloadDropConfig) {
	var (
		newDrops     []*dropper
		dropsChanged bool
	)
	for i, dropPolicy := range dropPolicies {
		var (
			numerator   = dropPolicy.Numerator
			denominator = dropPolicy.Denominator
		)
		newDrops = append(newDrops, newDropper(numerator, denominator, dropPolicy.Category))

		// The following reading edsImpl.drops doesn't need mutex because it can only
		// be updated by the code following.
		if dropsChanged {
			continue
		}
		if i >= len(edsImpl.drops) {
			dropsChanged = true
			continue
		}
		if oldDrop := edsImpl.drops[i]; numerator != oldDrop.numerator || denominator != oldDrop.denominator {
			dropsChanged = true
		}
	}
	if dropsChanged {
		edsImpl.pickerMu.Lock()
		edsImpl.drops = newDrops
		if edsImpl.innerState.Picker != nil {
			// Update picker with old inner picker, new drops.
			edsImpl.cc.UpdateState(balancer.State{
				ConnectivityState: edsImpl.innerState.ConnectivityState,
				Picker:            newDropPicker(edsImpl.innerState.Picker, newDrops, edsImpl.loadStore)},
			)
		}
		edsImpl.pickerMu.Unlock()
	}
}

// HandleEDSResponse handles the EDS response and creates/deletes localities and
// SubConns. It also handles drops.
//
// HandleChildPolicy and HandleEDSResponse must be called by the same goroutine.
func (edsImpl *edsBalancerImpl) HandleEDSResponse(edsResp *xdsclient.EDSUpdate) {
	// TODO: Unhandled fields from EDS response:
	//  - edsResp.GetPolicy().GetOverprovisioningFactor()
	//  - locality.GetPriority()
	//  - lbEndpoint.GetMetadata(): contains BNS name, send to sub-balancers
	//    - as service config or as resolved address
	//  - if socketAddress is not ip:port
	//     - socketAddress.GetNamedPort(), socketAddress.GetResolverName()
	//     - resolve endpoint's name with another resolver

	edsImpl.updateDrops(edsResp.Drops)

	// Filter out all localities with weight 0.
	//
	// Locality weighted load balancer can be enabled by setting an option in
	// CDS, and the weight of each locality. Currently, without the guarantee
	// that CDS is always sent, we assume locality weighted load balance is
	// always enabled, and ignore all weight 0 localities.
	//
	// In the future, we should look at the config in CDS response and decide
	// whether locality weight matters.
	newLocalitiesWithPriority := make(map[priorityType][]xdsclient.Locality)
	for _, locality := range edsResp.Localities {
		if locality.Weight == 0 {
			continue
		}
		priority := newPriorityType(locality.Priority)
		newLocalitiesWithPriority[priority] = append(newLocalitiesWithPriority[priority], locality)
	}

	var (
		priorityLowest  priorityType
		priorityChanged bool
	)

	for priority, newLocalities := range newLocalitiesWithPriority {
		if !priorityLowest.isSet() || priorityLowest.higherThan(priority) {
			priorityLowest = priority
		}

		bgwc, ok := edsImpl.priorityToLocalities[priority]
		if !ok {
			// Create balancer group if it's never created (this is the first
			// time this priority is received). We don't start it here. It may
			// be started when necessary (e.g. when higher is down, or if it's a
			// new lowest priority).
			bgwc = &balancerGroupWithConfig{
				bg: newBalancerGroup(
					edsImpl.ccWrapperWithPriority(priority), edsImpl.loadStore,
				),
				configs: make(map[internal.Locality]*localityConfig),
			}
			edsImpl.priorityToLocalities[priority] = bgwc
			priorityChanged = true
		}
		edsImpl.handleEDSResponsePerPriority(bgwc, newLocalities)
	}
	edsImpl.priorityLowest = priorityLowest

	// Delete priorities that are removed in the latest response, and also close
	// the balancer group.
	for p, bgwc := range edsImpl.priorityToLocalities {
		if _, ok := newLocalitiesWithPriority[p]; !ok {
			delete(edsImpl.priorityToLocalities, p)
			bgwc.bg.close()
			delete(edsImpl.priorityToState, p)
			priorityChanged = true
		}
	}

	// If priority was added/removed, it may affect the balancer group to use.
	// E.g. priorityInUse was removed, or all priorities are down, and a new
	// lower priority was added.
	if priorityChanged {
		edsImpl.handlePriorityChange()
	}
}

func (edsImpl *edsBalancerImpl) handleEDSResponsePerPriority(bgwc *balancerGroupWithConfig, newLocalities []xdsclient.Locality) {
	// newLocalitiesSet contains all names of localities in the new EDS response
	// for the same priority. It's used to delete localities that are removed in
	// the new EDS response.
	newLocalitiesSet := make(map[internal.Locality]struct{})
	for _, locality := range newLocalities {
		// One balancer for each locality.

		lid := locality.ID
		newLocalitiesSet[lid] = struct{}{}

		newWeight := locality.Weight
		var newAddrs []resolver.Address
		for _, lbEndpoint := range locality.Endpoints {
			// Filter out all "unhealthy" endpoints (unknown and
			// healthy are both considered to be healthy:
			// https://www.envoyproxy.io/docs/envoy/latest/api-v2/api/v2/core/health_check.proto#envoy-api-enum-core-healthstatus).
			if lbEndpoint.HealthStatus != xdsclient.EndpointHealthStatusHealthy &&
				lbEndpoint.HealthStatus != xdsclient.EndpointHealthStatusUnknown {
				continue
			}

			address := resolver.Address{
				Addr: lbEndpoint.Address,
			}
			if edsImpl.subBalancerBuilder.Name() == weightedroundrobin.Name && lbEndpoint.Weight != 0 {
				address.Metadata = &weightedroundrobin.AddrInfo{
					Weight: lbEndpoint.Weight,
				}
			}
			newAddrs = append(newAddrs, address)
		}
		var weightChanged, addrsChanged bool
		config, ok := bgwc.configs[lid]
		if !ok {
			// A new balancer, add it to balancer group and balancer map.
			bgwc.bg.add(lid, newWeight, edsImpl.subBalancerBuilder)
			config = &localityConfig{
				weight: newWeight,
			}
			bgwc.configs[lid] = config

			// weightChanged is false for new locality, because there's no need
			// to update weight in bg.
			addrsChanged = true
		} else {
			// Compare weight and addrs.
			if config.weight != newWeight {
				weightChanged = true
			}
			if !reflect.DeepEqual(config.addrs, newAddrs) {
				addrsChanged = true
			}
		}

		if weightChanged {
			config.weight = newWeight
			bgwc.bg.changeWeight(lid, newWeight)
		}

		if addrsChanged {
			config.addrs = newAddrs
			bgwc.bg.handleResolvedAddrs(lid, newAddrs)
		}
	}

	// Delete localities that are removed in the latest response.
	for lid := range bgwc.configs {
		if _, ok := newLocalitiesSet[lid]; !ok {
			bgwc.bg.remove(lid)
			delete(bgwc.configs, lid)
		}
	}
}

// HandleSubConnStateChange handles the state change and update pickers accordingly.
func (edsImpl *edsBalancerImpl) HandleSubConnStateChange(sc balancer.SubConn, s connectivity.State) {
	edsImpl.subConnMu.Lock()
	var bgwc *balancerGroupWithConfig
	if p, ok := edsImpl.subConnToPriority[sc]; ok {
		if s == connectivity.Shutdown {
			// Only delete sc from the map when state changed to Shutdown.
			delete(edsImpl.subConnToPriority, sc)
		}
		bgwc = edsImpl.priorityToLocalities[p]
	}
	edsImpl.subConnMu.Unlock()
	if bgwc == nil {
		grpclog.Infof("edsBalancerImpl: priority not found for sc state change")
		return
	}
	if bg := bgwc.bg; bg != nil {
		bg.handleSubConnStateChange(sc, s)
	}
}

// updateState first handles priority, and then wraps picker in a drop picker
// before forwarding the update.
func (edsImpl *edsBalancerImpl) updateState(priority priorityType, s balancer.State) {
	_, ok := edsImpl.priorityToLocalities[priority]
	if !ok {
		grpclog.Infof("eds: received picker update from unknown priority")
		return
	}

	if edsImpl.handlePriorityWithNewState(priority, s) {
		edsImpl.pickerMu.Lock()
		defer edsImpl.pickerMu.Unlock()
		edsImpl.innerState = s
		// Don't reset drops when it's a state change.
		edsImpl.cc.UpdateState(balancer.State{ConnectivityState: s.ConnectivityState, Picker: newDropPicker(s.Picker, edsImpl.drops, edsImpl.loadStore)})
	}
}

func (edsImpl *edsBalancerImpl) ccWrapperWithPriority(priority priorityType) *edsBalancerWrapperCC {
	return &edsBalancerWrapperCC{
		ClientConn: edsImpl.cc,
		priority:   priority,
		parent:     edsImpl,
	}
}

// edsBalancerWrapperCC implements the balancer.ClientConn API and get passed to
// each balancer group. It contains the locality priority.
type edsBalancerWrapperCC struct {
	balancer.ClientConn
	priority priorityType
	parent   *edsBalancerImpl
}

func (ebwcc *edsBalancerWrapperCC) NewSubConn(addrs []resolver.Address, opts balancer.NewSubConnOptions) (balancer.SubConn, error) {
	return ebwcc.parent.newSubConn(ebwcc.priority, addrs, opts)
}
func (ebwcc *edsBalancerWrapperCC) UpdateBalancerState(state connectivity.State, picker balancer.Picker) {
	grpclog.Fatalln("not implemented")
}
func (ebwcc *edsBalancerWrapperCC) UpdateState(state balancer.State) {
	ebwcc.parent.updateState(ebwcc.priority, state)
}

func (edsImpl *edsBalancerImpl) newSubConn(priority priorityType, addrs []resolver.Address, opts balancer.NewSubConnOptions) (balancer.SubConn, error) {
	sc, err := edsImpl.cc.NewSubConn(addrs, opts)
	if err != nil {
		return nil, err
	}
	edsImpl.subConnMu.Lock()
	edsImpl.subConnToPriority[sc] = priority
	edsImpl.subConnMu.Unlock()
	return sc, nil
}

// Close closes the balancer.
func (edsImpl *edsBalancerImpl) Close() {
	for _, bgwc := range edsImpl.priorityToLocalities {
		if bg := bgwc.bg; bg != nil {
			bg.close()
		}
	}
}

type dropPicker struct {
	drops     []*dropper
	p         balancer.V2Picker
	loadStore lrs.Store
}

func newDropPicker(p balancer.V2Picker, drops []*dropper, loadStore lrs.Store) *dropPicker {
	return &dropPicker{
		drops:     drops,
		p:         p,
		loadStore: loadStore,
	}
}

func (d *dropPicker) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	var (
		drop     bool
		category string
	)
	for _, dp := range d.drops {
		if dp.drop() {
			drop = true
			category = dp.category
			break
		}
	}
	if drop {
		if d.loadStore != nil {
			d.loadStore.CallDropped(category)
		}
		return balancer.PickResult{}, status.Errorf(codes.Unavailable, "RPC is dropped")
	}
	// TODO: (eds) don't drop unless the inner picker is READY. Similar to
	// https://github.com/grpc/grpc-go/issues/2622.
	return d.p.Pick(info)
}
