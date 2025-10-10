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

// Package clusterimpl implements the xds_cluster_impl balancing policy. It
// handles the cluster features (e.g. circuit_breaking, RPC dropping).
//
// Note that it doesn't handle name resolution, which is done by policy
// xds_cluster_resolver.
package clusterimpl

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/internal/balancer/gracefulswitch"
	"google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/pretty"
	xdsinternal "google.golang.org/grpc/internal/xds"
	"google.golang.org/grpc/internal/xds/balancer/loadstore"
	"google.golang.org/grpc/internal/xds/bootstrap"
	"google.golang.org/grpc/internal/xds/clients"
	"google.golang.org/grpc/internal/xds/clients/lrsclient"
	"google.golang.org/grpc/internal/xds/xdsclient"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
)

const (
	// Name is the name of the cluster_impl balancer.
	Name                   = "xds_cluster_impl_experimental"
	defaultRequestCountMax = 1024
	loadStoreStopTimeout   = 1 * time.Second
)

var (
	connectedAddress = internal.ConnectedAddress.(func(balancer.SubConnState) resolver.Address)
	// Below function is no-op in actual code, but can be overridden in
	// tests to give tests visibility into exactly when certain events happen.
	clientConnUpdateHook = func() {}
	pickerUpdateHook     = func() {}
)

func init() {
	balancer.Register(bb{})
}

type bb struct{}

func (bb) Build(cc balancer.ClientConn, bOpts balancer.BuildOptions) balancer.Balancer {
	b := &clusterImplBalancer{
		ClientConn:      cc,
		loadWrapper:     loadstore.NewWrapper(),
		requestCountMax: defaultRequestCountMax,
	}
	b.logger = prefixLogger(b)
	b.child = gracefulswitch.NewBalancer(b, bOpts)
	b.logger.Infof("Created")
	return b
}

func (bb) Name() string {
	return Name
}

func (bb) ParseConfig(c json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
	return parseConfig(c)
}

type clusterImplBalancer struct {
	balancer.ClientConn

	// The following fields are set at creation time, and are read-only after
	// that, and therefore need not be protected by a mutex.
	logger *grpclog.PrefixLogger
	// TODO: #8366 -  Refactor usage of loadWrapper to easily plugin a test
	// load reporter from tests.
	loadWrapper *loadstore.Wrapper

	// The following fields are only accessed from balancer API methods, which
	// are guaranteed to be called serially by gRPC.
	xdsClient        xdsclient.XDSClient     // Sent down in ResolverState attributes.
	cancelLoadReport func(context.Context)   // To stop reporting load through the above xDS client.
	edsServiceName   string                  // EDS service name to report load for.
	lrsServer        *bootstrap.ServerConfig // Load reporting server configuration.
	dropCategories   []DropConfig            // The categories for drops.
	child            *gracefulswitch.Balancer

	// The following fields are protected by mu, since they are accessed in
	// balancer API methods and in methods called from the child policy.
	mu                    sync.Mutex
	clusterName           string                            // The cluster name for credentials handshaking.
	inhibitPickerUpdates  bool                              // Inhibits state updates from child policy when processing an update from the parent.
	pendingPickerUpdates  bool                              // True if a picker update from the child policy was inhibited when processing an update from the parent.
	childState            balancer.State                    // Most recent state update from the child policy.
	drops                 []*dropper                        // Drops implementation.
	requestCounterCluster string                            // The cluster name for the request counter, from LB config.
	requestCounterService string                            // The service name for the request counter, from LB config.
	requestCountMax       uint32                            // Max concurrent requests, from LB config.
	requestCounter        *xdsclient.ClusterRequestsCounter // Tracks total inflight requests for a given service.
	telemetryLabels       map[string]string                 // Telemetry labels to set on picks, from LB config.
}

// handleDropAndRequestCountLocked compares drop and request counter in newConfig with
// the one currently used by picker, and is protected by b.mu. It returns a boolean
// indicating if a new picker needs to be generated.
func (b *clusterImplBalancer) handleDropAndRequestCountLocked(newConfig *LBConfig) bool {
	var updatePicker bool
	if !slices.Equal(b.dropCategories, newConfig.DropCategories) {
		b.dropCategories = newConfig.DropCategories
		b.drops = make([]*dropper, 0, len(newConfig.DropCategories))
		for _, c := range newConfig.DropCategories {
			b.drops = append(b.drops, newDropper(c))
		}
		updatePicker = true
	}

	if b.requestCounterCluster != newConfig.Cluster || b.requestCounterService != newConfig.EDSServiceName {
		b.requestCounterCluster = newConfig.Cluster
		b.requestCounterService = newConfig.EDSServiceName
		b.requestCounter = xdsclient.GetClusterRequestsCounter(newConfig.Cluster, newConfig.EDSServiceName)
		updatePicker = true
	}
	var newRequestCountMax uint32 = 1024
	if newConfig.MaxConcurrentRequests != nil {
		newRequestCountMax = *newConfig.MaxConcurrentRequests
	}
	if b.requestCountMax != newRequestCountMax {
		b.requestCountMax = newRequestCountMax
		updatePicker = true
	}

	return updatePicker
}

func (b *clusterImplBalancer) newPickerLocked() *picker {
	return &picker{
		drops:           b.drops,
		s:               b.childState,
		loadStore:       b.loadWrapper,
		counter:         b.requestCounter,
		countMax:        b.requestCountMax,
		telemetryLabels: b.telemetryLabels,
	}
}

// updateLoadStore checks the config for load store, and decides whether it
// needs to restart the load reporting stream.
func (b *clusterImplBalancer) updateLoadStore(newConfig *LBConfig) error {
	var updateLoadClusterAndService bool

	// ClusterName is different, restart. ClusterName is from ClusterName and
	// EDSServiceName.
	clusterName := b.getClusterName()
	if clusterName != newConfig.Cluster {
		updateLoadClusterAndService = true
		b.setClusterName(newConfig.Cluster)
		clusterName = newConfig.Cluster
	}
	if b.edsServiceName != newConfig.EDSServiceName {
		updateLoadClusterAndService = true
		b.edsServiceName = newConfig.EDSServiceName
	}
	if updateLoadClusterAndService {
		// This updates the clusterName and serviceName that will be reported
		// for the loads. The update here is too early, the perfect timing is
		// when the picker is updated with the new connection. But from this
		// balancer's point of view, it's impossible to tell.
		//
		// On the other hand, this will almost never happen. Each LRS policy
		// shouldn't get updated config. The parent should do a graceful switch
		// when the clusterName or serviceName is changed.
		b.loadWrapper.UpdateClusterAndService(clusterName, b.edsServiceName)
	}

	var (
		stopOldLoadReport  bool
		startNewLoadReport bool
	)

	// Check if it's necessary to restart load report.
	if b.lrsServer == nil {
		if newConfig.LoadReportingServer != nil {
			// Old is nil, new is not nil, start new LRS.
			b.lrsServer = newConfig.LoadReportingServer
			startNewLoadReport = true
		}
		// Old is nil, new is nil, do nothing.
	} else if newConfig.LoadReportingServer == nil {
		// Old is not nil, new is nil, stop old, don't start new.
		b.lrsServer = newConfig.LoadReportingServer
		stopOldLoadReport = true
	} else {
		// Old is not nil, new is not nil, compare string values, if
		// different, stop old and start new.
		if !b.lrsServer.Equal(newConfig.LoadReportingServer) {
			b.lrsServer = newConfig.LoadReportingServer
			stopOldLoadReport = true
			startNewLoadReport = true
		}
	}

	if stopOldLoadReport {
		if b.cancelLoadReport != nil {
			stopCtx, stopCancel := context.WithTimeout(context.Background(), loadStoreStopTimeout)
			defer stopCancel()
			b.cancelLoadReport(stopCtx)
			b.cancelLoadReport = nil
			if !startNewLoadReport {
				// If a new LRS stream will be started later, no need to update
				// it to nil here.
				b.loadWrapper.UpdateLoadStore(nil)
			}
		}
	}
	if startNewLoadReport {
		var loadStore *lrsclient.LoadStore
		if b.xdsClient != nil {
			loadStore, b.cancelLoadReport = b.xdsClient.ReportLoad(b.lrsServer)
		}
		b.loadWrapper.UpdateLoadStore(loadStore)
	}

	return nil
}

func (b *clusterImplBalancer) UpdateClientConnState(s balancer.ClientConnState) error {
	defer clientConnUpdateHook()

	b.mu.Lock()
	b.inhibitPickerUpdates = true
	b.mu.Unlock()
	if b.logger.V(2) {
		b.logger.Infof("Received configuration: %s", pretty.ToJSON(s.BalancerConfig))
	}
	newConfig, ok := s.BalancerConfig.(*LBConfig)
	if !ok {
		return fmt.Errorf("unexpected balancer config with type: %T", s.BalancerConfig)
	}

	// Need to check for potential errors at the beginning of this function, so
	// that on errors, we reject the whole config, instead of applying part of
	// it.
	bb := balancer.Get(newConfig.ChildPolicy.Name)
	if bb == nil {
		return fmt.Errorf("child policy %q not registered", newConfig.ChildPolicy.Name)
	}

	if b.xdsClient == nil {
		c := xdsclient.FromResolverState(s.ResolverState)
		if c == nil {
			return balancer.ErrBadResolverState
		}
		b.xdsClient = c
	}

	// Update load reporting config. This needs to be done before updating the
	// child policy because we need the loadStore from the updated client to be
	// passed to the ccWrapper, so that the next picker from the child policy
	// will pick up the new loadStore.
	if err := b.updateLoadStore(newConfig); err != nil {
		return err
	}

	// Build config for the gracefulswitch balancer. It is safe to ignore JSON
	// marshaling errors here, since the config was already validated as part of
	// ParseConfig().
	cfg := []map[string]any{{newConfig.ChildPolicy.Name: newConfig.ChildPolicy.Config}}
	cfgJSON, _ := json.Marshal(cfg)
	parsedCfg, err := gracefulswitch.ParseConfig(cfgJSON)
	if err != nil {
		return err
	}

	// Addresses and sub-balancer config are sent to sub-balancer.
	err = b.child.UpdateClientConnState(balancer.ClientConnState{
		ResolverState:  s.ResolverState,
		BalancerConfig: parsedCfg,
	})

	b.mu.Lock()
	b.telemetryLabels = newConfig.TelemetryLabels
	// We want to send a picker update to the parent if one of the two
	// conditions are met:
	// - drop/request config has changed *and* there is already a picker from
	//   the child, or
	// - there is a pending picker update from the child (and this covers the
	//   case where the drop/request config has not changed, but the child sent
	//   a picker update while we were still processing config from our parent).
	if (b.handleDropAndRequestCountLocked(newConfig) && b.childState.Picker != nil) || b.pendingPickerUpdates {
		b.pendingPickerUpdates = false
		b.ClientConn.UpdateState(balancer.State{
			ConnectivityState: b.childState.ConnectivityState,
			Picker:            b.newPickerLocked(),
		})
	}
	b.inhibitPickerUpdates = false
	b.mu.Unlock()
	pickerUpdateHook()
	return err
}

func (b *clusterImplBalancer) ResolverError(err error) {
	b.child.ResolverError(err)
}

func (b *clusterImplBalancer) updateSubConnState(_ balancer.SubConn, s balancer.SubConnState, cb func(balancer.SubConnState)) {
	// Trigger re-resolution when a SubConn turns transient failure. This is
	// necessary for the LogicalDNS in cluster_resolver policy to re-resolve.
	//
	// Note that this happens not only for the addresses from DNS, but also for
	// EDS (cluster_impl doesn't know if it's DNS or EDS, only the parent
	// knows). The parent priority policy is configured to ignore re-resolution
	// signal from the EDS children.
	if s.ConnectivityState == connectivity.TransientFailure {
		b.ClientConn.ResolveNow(resolver.ResolveNowOptions{})
	}

	if cb != nil {
		cb(s)
	}
}

func (b *clusterImplBalancer) UpdateSubConnState(sc balancer.SubConn, s balancer.SubConnState) {
	b.logger.Errorf("UpdateSubConnState(%v, %+v) called unexpectedly", sc, s)
}

func (b *clusterImplBalancer) Close() {
	b.child.Close()
	b.childState = balancer.State{}

	if b.cancelLoadReport != nil {
		stopCtx, stopCancel := context.WithTimeout(context.Background(), loadStoreStopTimeout)
		defer stopCancel()
		b.cancelLoadReport(stopCtx)
		b.cancelLoadReport = nil
	}
	b.logger.Infof("Shutdown")
}

func (b *clusterImplBalancer) ExitIdle() {
	b.child.ExitIdle()
}

// Override methods to accept updates from the child LB.

func (b *clusterImplBalancer) UpdateState(state balancer.State) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Inhibit sending a picker update to our parent as part of handling new
	// state from the child, if we are currently handling an update from our
	// parent. Update the childState field regardless.
	b.childState = state
	if b.inhibitPickerUpdates {
		b.pendingPickerUpdates = true
		if b.logger.V(2) {
			b.logger.Infof("Received a picker update from the child when processing an update from the parent")
		}
		return
	}

	b.ClientConn.UpdateState(balancer.State{
		ConnectivityState: state.ConnectivityState,
		Picker:            b.newPickerLocked(),
	})
	pickerUpdateHook()
}

func (b *clusterImplBalancer) setClusterName(n string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.clusterName = n
}

func (b *clusterImplBalancer) getClusterName() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.clusterName
}

// scWrapper is a wrapper of SubConn with locality ID. The locality ID can be
// retrieved from the addresses when creating SubConn.
//
// All SubConns passed to the child policies are wrapped in this, so that the
// picker can get the localityID from the picked SubConn, and do load reporting.
//
// After wrapping, all SubConns to and from the parent ClientConn (e.g. for
// SubConn state update, update/remove SubConn) must be the original SubConns.
// All SubConns to and from the child policy (NewSubConn, forwarding SubConn
// state update) must be the wrapper. The balancer keeps a map from the original
// SubConn to the wrapper for this purpose.
type scWrapper struct {
	balancer.SubConn
	// locality needs to be atomic because it can be updated while being read by
	// the picker.
	locality atomic.Pointer[clients.Locality]
}

func (scw *scWrapper) updateLocalityID(lID clients.Locality) {
	scw.locality.Store(&lID)
}

func (scw *scWrapper) localityID() clients.Locality {
	lID := scw.locality.Load()
	if lID == nil {
		return clients.Locality{}
	}
	return *lID
}

func (b *clusterImplBalancer) NewSubConn(addrs []resolver.Address, opts balancer.NewSubConnOptions) (balancer.SubConn, error) {
	clusterName := b.getClusterName()
	newAddrs := make([]resolver.Address, len(addrs))
	for i, addr := range addrs {
		newAddrs[i] = xdsinternal.SetXDSHandshakeClusterName(addr, clusterName)
	}
	var sc balancer.SubConn
	scw := &scWrapper{}
	oldListener := opts.StateListener
	opts.StateListener = func(state balancer.SubConnState) {
		b.updateSubConnState(sc, state, oldListener)
		if state.ConnectivityState != connectivity.Ready {
			return
		}
		// Read connected address and call updateLocalityID() based on the connected
		// address's locality. https://github.com/grpc/grpc-go/issues/7339
		addr := connectedAddress(state)
		lID := xdsinternal.GetLocalityID(addr)
		if (lID == clients.Locality{}) {
			if b.logger.V(2) {
				b.logger.Infof("Locality ID for %s unexpectedly empty", addr)
			}
			return
		}
		scw.updateLocalityID(lID)
	}
	sc, err := b.ClientConn.NewSubConn(newAddrs, opts)
	if err != nil {
		return nil, err
	}
	scw.SubConn = sc
	return scw, nil
}

func (b *clusterImplBalancer) RemoveSubConn(sc balancer.SubConn) {
	b.logger.Errorf("RemoveSubConn(%v) called unexpectedly", sc)
}

func (b *clusterImplBalancer) UpdateAddresses(sc balancer.SubConn, addrs []resolver.Address) {
	clusterName := b.getClusterName()
	newAddrs := make([]resolver.Address, len(addrs))
	var lID clients.Locality
	for i, addr := range addrs {
		newAddrs[i] = xdsinternal.SetXDSHandshakeClusterName(addr, clusterName)
		lID = xdsinternal.GetLocalityID(newAddrs[i])
	}
	if scw, ok := sc.(*scWrapper); ok {
		scw.updateLocalityID(lID)
		// Need to get the original SubConn from the wrapper before calling
		// parent ClientConn.
		sc = scw.SubConn
	}
	b.ClientConn.UpdateAddresses(sc, newAddrs)
}
