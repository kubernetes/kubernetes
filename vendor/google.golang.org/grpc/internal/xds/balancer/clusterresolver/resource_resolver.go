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
	"context"
	"sync"

	"google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/grpcsync"
	"google.golang.org/grpc/internal/xds/xdsclient/xdsresource"
	"google.golang.org/grpc/resolver"
)

// resourceUpdate is a combined update from all the resources, in the order of
// priority. For example, it can be {EDS, EDS, DNS}.
type resourceUpdate struct {
	// A discovery mechanism would return an empty update when it runs into
	// errors, and this would result in the priority LB policy reporting
	// TRANSIENT_FAILURE (if there was a single discovery mechanism), or would
	// fallback to the next highest priority that is available.
	priorities []priorityConfig
	// To be invoked once the update is completely processed, or is dropped in
	// favor of a newer update.
	onDone func()
}

// topLevelResolver is used by concrete endpointsResolver implementations for
// reporting updates and errors. The `resourceResolver` type implements this
// interface and takes appropriate actions upon receipt of updates and errors
// from underlying concrete resolvers.
type topLevelResolver interface {
	// onUpdate is called when a new update is received from the underlying
	// endpointsResolver implementation. The onDone callback is to be invoked
	// once the update is completely processed, or is dropped in favor of a
	// newer update.
	onUpdate(onDone func())
}

// endpointsResolver wraps the functionality to resolve a given resource name to
// a set of endpoints. The mechanism used by concrete implementations depend on
// the supported discovery mechanism type.
type endpointsResolver interface {
	// lastUpdate returns endpoint results from the most recent resolution.
	//
	// The type of the first return result is dependent on the resolver
	// implementation.
	//
	// The second return result indicates whether the resolver was able to
	// successfully resolve the resource name to endpoints. If set to false, the
	// first return result is invalid and must not be used.
	lastUpdate() (any, bool)

	// resolverNow triggers re-resolution of the resource.
	resolveNow()

	// stop stops resolution of the resource. Implementations must not invoke
	// any methods on the topLevelResolver interface once `stop()` returns.
	stop()
}

// discoveryMechanismKey is {type+resource_name}, it's used as the map key, so
// that the same resource resolver can be reused (e.g. when there are two
// mechanisms, both for the same EDS resource, but has different circuit
// breaking config).
type discoveryMechanismKey struct {
	typ  DiscoveryMechanismType
	name string
}

// discoveryMechanismAndResolver is needed to keep the resolver and the
// discovery mechanism together, because resolvers can be shared. And we need
// the mechanism for fields like circuit breaking, LRS etc when generating the
// balancer config.
type discoveryMechanismAndResolver struct {
	dm DiscoveryMechanism
	r  endpointsResolver

	childNameGen *nameGenerator
}

type resourceResolver struct {
	parent           *clusterResolverBalancer
	logger           *grpclog.PrefixLogger
	updateChannel    chan *resourceUpdate
	serializer       *grpcsync.CallbackSerializer
	serializerCancel context.CancelFunc

	// mu protects the slice and map, and content of the resolvers in the slice.
	mu         sync.Mutex
	mechanisms []DiscoveryMechanism
	children   []discoveryMechanismAndResolver
	// childrenMap's value only needs the resolver implementation (type
	// discoveryMechanism) and the childNameGen. The other two fields are not
	// used.
	//
	// TODO(cleanup): maybe we can make a new type with just the necessary
	// fields, and use it here instead.
	childrenMap map[discoveryMechanismKey]discoveryMechanismAndResolver
	// Each new discovery mechanism needs a child name generator to reuse child
	// policy names. But to make sure the names across discover mechanism
	// doesn't conflict, we need a seq ID. This ID is incremented for each new
	// discover mechanism.
	childNameGeneratorSeqID uint64
}

func newResourceResolver(parent *clusterResolverBalancer, logger *grpclog.PrefixLogger) *resourceResolver {
	rr := &resourceResolver{
		parent:        parent,
		logger:        logger,
		updateChannel: make(chan *resourceUpdate, 1),
		childrenMap:   make(map[discoveryMechanismKey]discoveryMechanismAndResolver),
	}
	ctx, cancel := context.WithCancel(context.Background())
	rr.serializer = grpcsync.NewCallbackSerializer(ctx)
	rr.serializerCancel = cancel
	return rr
}

func equalDiscoveryMechanisms(a, b []DiscoveryMechanism) bool {
	if len(a) != len(b) {
		return false
	}
	for i, aa := range a {
		bb := b[i]
		if !aa.Equal(bb) {
			return false
		}
	}
	return true
}

func discoveryMechanismToKey(dm DiscoveryMechanism) discoveryMechanismKey {
	switch dm.Type {
	case DiscoveryMechanismTypeEDS:
		nameToWatch := dm.EDSServiceName
		if nameToWatch == "" {
			nameToWatch = dm.Cluster
		}
		return discoveryMechanismKey{typ: dm.Type, name: nameToWatch}
	case DiscoveryMechanismTypeLogicalDNS:
		return discoveryMechanismKey{typ: dm.Type, name: dm.DNSHostname}
	default:
		return discoveryMechanismKey{}
	}
}

func (rr *resourceResolver) updateMechanisms(mechanisms []DiscoveryMechanism) {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	if equalDiscoveryMechanisms(rr.mechanisms, mechanisms) {
		return
	}
	rr.mechanisms = mechanisms
	rr.children = make([]discoveryMechanismAndResolver, len(mechanisms))
	newDMs := make(map[discoveryMechanismKey]bool)

	// Start one watch for each new discover mechanism {type+resource_name}.
	for i, dm := range mechanisms {
		dmKey := discoveryMechanismToKey(dm)
		newDMs[dmKey] = true
		dmAndResolver, ok := rr.childrenMap[dmKey]
		if ok {
			// If this is not new, keep the fields (especially childNameGen),
			// and only update the DiscoveryMechanism.
			//
			// Note that the same dmKey doesn't mean the same
			// DiscoveryMechanism. There are fields (e.g.
			// MaxConcurrentRequests) in DiscoveryMechanism that are not copied
			// to dmKey, we need to keep those updated.
			dmAndResolver.dm = dm
			rr.children[i] = dmAndResolver
			continue
		}

		// Create resolver for a newly seen resource.
		var resolver endpointsResolver
		switch dm.Type {
		case DiscoveryMechanismTypeEDS:
			resolver = newEDSResolver(dmKey.name, rr.parent.xdsClient, rr, rr.logger)
		case DiscoveryMechanismTypeLogicalDNS:
			resolver = newDNSResolver(dmKey.name, rr, rr.logger)
		}
		dmAndResolver = discoveryMechanismAndResolver{
			dm:           dm,
			r:            resolver,
			childNameGen: newNameGenerator(rr.childNameGeneratorSeqID),
		}
		rr.childrenMap[dmKey] = dmAndResolver
		rr.children[i] = dmAndResolver
		rr.childNameGeneratorSeqID++
	}

	// Stop the resources that were removed.
	for dm, r := range rr.childrenMap {
		if !newDMs[dm] {
			delete(rr.childrenMap, dm)
			go r.r.stop()
		}
	}
	// Regenerate even if there's no change in discovery mechanism, in case
	// priority order changed.
	rr.generateLocked(func() {})
}

// resolveNow is typically called to trigger re-resolve of DNS. The EDS
// resolveNow() is a noop.
func (rr *resourceResolver) resolveNow() {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	for _, r := range rr.childrenMap {
		r.r.resolveNow()
	}
}

func (rr *resourceResolver) stop(closing bool) {
	rr.mu.Lock()

	// Save the previous childrenMap to stop the children outside the mutex,
	// and reinitialize the map.  We only need to reinitialize to allow for the
	// policy to be reused if the resource comes back.  In practice, this does
	// not happen as the parent LB policy will also be closed, causing this to
	// be removed entirely, but a future use case might want to reuse the
	// policy instead.
	cm := rr.childrenMap
	rr.childrenMap = make(map[discoveryMechanismKey]discoveryMechanismAndResolver)
	rr.mechanisms = nil
	rr.children = nil

	rr.mu.Unlock()

	for _, r := range cm {
		r.r.stop()
	}

	if closing {
		rr.serializerCancel()
		<-rr.serializer.Done()
	}

	// stop() is called when the LB policy is closed or when the underlying
	// cluster resource is removed by the management server. In the latter case,
	// an empty config update needs to be pushed to the child policy to ensure
	// that a picker that fails RPCs is sent up to the channel.
	//
	// Resource resolver implementations are expected to not send any updates
	// after they are stopped. Therefore, we don't have to worry about another
	// write to this channel happening at the same time as this one.
	select {
	case ru := <-rr.updateChannel:
		if ru.onDone != nil {
			ru.onDone()
		}
	default:
	}
	rr.updateChannel <- &resourceUpdate{}
}

// generateLocked collects updates from all resolvers. It pushes the combined
// result on the update channel if all child resolvers have received at least
// one update. Otherwise it returns early.
//
// The onDone callback is invoked inline if not all child resolvers have
// received at least one update. If all child resolvers have received at least
// one update, onDone is invoked when the combined update is processed by the
// clusterresolver LB policy.
//
// Caller must hold rr.mu.
func (rr *resourceResolver) generateLocked(onDone func()) {
	var ret []priorityConfig
	for _, rDM := range rr.children {
		u, ok := rDM.r.lastUpdate()
		if !ok {
			// Don't send updates to parent until all resolvers have update to
			// send.
			onDone()
			return
		}
		switch uu := u.(type) {
		case xdsresource.EndpointsUpdate:
			ret = append(ret, priorityConfig{mechanism: rDM.dm, edsResp: uu, childNameGen: rDM.childNameGen})
		case []resolver.Endpoint:
			ret = append(ret, priorityConfig{mechanism: rDM.dm, endpoints: uu, childNameGen: rDM.childNameGen})
		}
	}
	select {
	// A previously unprocessed update is dropped in favor of the new one, and
	// the former's onDone callback is invoked to unblock the xDS client's
	// receive path.
	case ru := <-rr.updateChannel:
		if ru.onDone != nil {
			ru.onDone()
		}
	default:
	}
	rr.updateChannel <- &resourceUpdate{priorities: ret, onDone: onDone}
}

func (rr *resourceResolver) onUpdate(onDone func()) {
	handleUpdate := func(context.Context) {
		rr.mu.Lock()
		rr.generateLocked(onDone)
		rr.mu.Unlock()
	}
	rr.serializer.ScheduleOr(handleUpdate, func() { onDone() })
}
