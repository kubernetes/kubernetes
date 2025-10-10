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

// Package cdsbalancer implements a balancer to handle CDS responses.
package cdsbalancer

import (
	"context"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"unsafe"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/balancer/base"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/tls/certprovider"
	"google.golang.org/grpc/internal/balancer/nop"
	xdsinternal "google.golang.org/grpc/internal/credentials/xds"
	"google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/grpcsync"
	"google.golang.org/grpc/internal/pretty"
	"google.golang.org/grpc/internal/xds/balancer/clusterresolver"
	"google.golang.org/grpc/internal/xds/xdsclient"
	"google.golang.org/grpc/internal/xds/xdsclient/xdsresource"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
)

const (
	cdsName                  = "cds_experimental"
	aggregateClusterMaxDepth = 16
)

var (
	errBalancerClosed  = fmt.Errorf("cds_experimental LB policy is closed")
	errExceedsMaxDepth = fmt.Errorf("aggregate cluster graph exceeds max depth (%d)", aggregateClusterMaxDepth)

	// newChildBalancer is a helper function to build a new cluster_resolver
	// balancer and will be overridden in unittests.
	newChildBalancer = func(cc balancer.ClientConn, opts balancer.BuildOptions) (balancer.Balancer, error) {
		builder := balancer.Get(clusterresolver.Name)
		if builder == nil {
			return nil, fmt.Errorf("xds: no balancer builder with name %v", clusterresolver.Name)
		}
		// We directly pass the parent clientConn to the underlying
		// cluster_resolver balancer because the cdsBalancer does not deal with
		// subConns.
		return builder.Build(cc, opts), nil
	}
	buildProvider = buildProviderFunc

	// x509SystemCertPoolFunc is used for mocking the system cert pool for
	// tests.
	x509SystemCertPoolFunc = x509.SystemCertPool
)

func init() {
	balancer.Register(bb{})
}

// bb implements the balancer.Builder interface to help build a cdsBalancer.
// It also implements the balancer.ConfigParser interface to help parse the
// JSON service config, to be passed to the cdsBalancer.
type bb struct{}

// Build creates a new CDS balancer with the ClientConn.
func (bb) Build(cc balancer.ClientConn, opts balancer.BuildOptions) balancer.Balancer {
	builder := balancer.Get(clusterresolver.Name)
	if builder == nil {
		// Shouldn't happen, registered through imported Cluster Resolver,
		// defensive programming.
		logger.Errorf("%q LB policy is needed but not registered", clusterresolver.Name)
		return nop.NewBalancer(cc, fmt.Errorf("%q LB policy is needed but not registered", clusterresolver.Name))
	}
	parser, ok := builder.(balancer.ConfigParser)
	if !ok {
		// Shouldn't happen, imported Cluster Resolver builder has this method.
		logger.Errorf("%q LB policy does not implement a config parser", clusterresolver.Name)
		return nop.NewBalancer(cc, fmt.Errorf("%q LB policy does not implement a config parser", clusterresolver.Name))
	}

	ctx, cancel := context.WithCancel(context.Background())
	hi := xdsinternal.NewHandshakeInfo(nil, nil, nil, false)
	xdsHIPtr := unsafe.Pointer(hi)
	b := &cdsBalancer{
		bOpts:             opts,
		childConfigParser: parser,
		serializer:        grpcsync.NewCallbackSerializer(ctx),
		serializerCancel:  cancel,
		xdsHIPtr:          &xdsHIPtr,
		watchers:          make(map[string]*watcherState),
	}
	b.ccw = &ccWrapper{
		ClientConn: cc,
		xdsHIPtr:   b.xdsHIPtr,
	}
	b.logger = prefixLogger(b)
	b.logger.Infof("Created")

	var creds credentials.TransportCredentials
	switch {
	case opts.DialCreds != nil:
		creds = opts.DialCreds
	case opts.CredsBundle != nil:
		creds = opts.CredsBundle.TransportCredentials()
	}
	if xc, ok := creds.(interface{ UsesXDS() bool }); ok && xc.UsesXDS() {
		b.xdsCredsInUse = true
	}
	b.logger.Infof("xDS credentials in use: %v", b.xdsCredsInUse)
	return b
}

// Name returns the name of balancers built by this builder.
func (bb) Name() string {
	return cdsName
}

// lbConfig represents the loadBalancingConfig section of the service config
// for the cdsBalancer.
type lbConfig struct {
	serviceconfig.LoadBalancingConfig
	ClusterName string `json:"Cluster"`
}

// ParseConfig parses the JSON load balancer config provided into an
// internal form or returns an error if the config is invalid.
func (bb) ParseConfig(c json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
	var cfg lbConfig
	if err := json.Unmarshal(c, &cfg); err != nil {
		return nil, fmt.Errorf("xds: unable to unmarshal lbconfig: %s, error: %v", string(c), err)
	}
	return &cfg, nil
}

// cdsBalancer implements a CDS based LB policy. It instantiates a
// cluster_resolver balancer to further resolve the serviceName received from
// CDS, into localities and endpoints. Implements the balancer.Balancer
// interface which is exposed to gRPC and implements the balancer.ClientConn
// interface which is exposed to the cluster_resolver balancer.
type cdsBalancer struct {
	// The following fields are initialized at build time and are either
	// read-only after that or provide their own synchronization, and therefore
	// do not need to be guarded by a mutex.
	ccw               *ccWrapper            // ClientConn interface passed to child LB.
	bOpts             balancer.BuildOptions // BuildOptions passed to child LB.
	childConfigParser balancer.ConfigParser // Config parser for cluster_resolver LB policy.
	logger            *grpclog.PrefixLogger // Prefix logger for all logging.
	xdsCredsInUse     bool

	xdsHIPtr *unsafe.Pointer // Accessed atomically.

	// The serializer and its cancel func are initialized at build time, and the
	// rest of the fields here are only accessed from serializer callbacks (or
	// from balancer.Balancer methods, which themselves are guaranteed to be
	// mutually exclusive) and hence do not need to be guarded by a mutex.
	serializer       *grpcsync.CallbackSerializer // Serializes updates from gRPC and xDS client.
	serializerCancel context.CancelFunc           // Stops the above serializer.
	childLB          balancer.Balancer            // Child policy, built upon resolution of the cluster graph.
	xdsClient        xdsclient.XDSClient          // xDS client to watch Cluster resources.
	watchers         map[string]*watcherState     // Set of watchers and associated state, keyed by cluster name.
	lbCfg            *lbConfig                    // Current load balancing configuration.

	// The certificate providers are cached here to that they can be closed when
	// a new provider is to be created.
	cachedRoot     certprovider.Provider
	cachedIdentity certprovider.Provider
}

// handleSecurityConfig processes the security configuration received from the
// management server, creates appropriate certificate provider plugins, and
// updates the HandshakeInfo which is added as an address attribute in
// NewSubConn() calls.
//
// Only executed in the context of a serializer callback.
func (b *cdsBalancer) handleSecurityConfig(config *xdsresource.SecurityConfig) error {
	// If xdsCredentials are not in use, i.e, the user did not want to get
	// security configuration from an xDS server, we should not be acting on the
	// received security config here. Doing so poses a security threat.
	if !b.xdsCredsInUse {
		return nil
	}
	var xdsHI *xdsinternal.HandshakeInfo

	// Security config being nil is a valid case where the management server has
	// not sent any security configuration. The xdsCredentials implementation
	// handles this by delegating to its fallback credentials.
	if config == nil {
		// We need to explicitly set the fields to nil here since this might be
		// a case of switching from a good security configuration to an empty
		// one where fallback credentials are to be used.
		xdsHI = xdsinternal.NewHandshakeInfo(nil, nil, nil, false)
		atomic.StorePointer(b.xdsHIPtr, unsafe.Pointer(xdsHI))
		return nil

	}

	// A root provider is required whether we are using TLS or mTLS.
	cpc := b.xdsClient.BootstrapConfig().CertProviderConfigs()
	var rootProvider certprovider.Provider
	if config.UseSystemRootCerts {
		rootProvider = systemRootCertsProvider{}
	} else {
		rp, err := buildProvider(cpc, config.RootInstanceName, config.RootCertName, false, true)
		if err != nil {
			return err
		}
		rootProvider = rp
	}

	// The identity provider is only present when using mTLS.
	var identityProvider certprovider.Provider
	if name, cert := config.IdentityInstanceName, config.IdentityCertName; name != "" {
		var err error
		identityProvider, err = buildProvider(cpc, name, cert, true, false)
		if err != nil {
			return err
		}
	}

	// Close the old providers and cache the new ones.
	if b.cachedRoot != nil {
		b.cachedRoot.Close()
	}
	if b.cachedIdentity != nil {
		b.cachedIdentity.Close()
	}
	b.cachedRoot = rootProvider
	b.cachedIdentity = identityProvider
	xdsHI = xdsinternal.NewHandshakeInfo(rootProvider, identityProvider, config.SubjectAltNameMatchers, false)
	atomic.StorePointer(b.xdsHIPtr, unsafe.Pointer(xdsHI))
	return nil
}

func buildProviderFunc(configs map[string]*certprovider.BuildableConfig, instanceName, certName string, wantIdentity, wantRoot bool) (certprovider.Provider, error) {
	cfg, ok := configs[instanceName]
	if !ok {
		// Defensive programming. If a resource received from the management
		// server contains a certificate provider instance name that is not
		// found in the bootstrap, the resource is NACKed by the xDS client.
		return nil, fmt.Errorf("certificate provider instance %q not found in bootstrap file", instanceName)
	}
	provider, err := cfg.Build(certprovider.BuildOptions{
		CertName:     certName,
		WantIdentity: wantIdentity,
		WantRoot:     wantRoot,
	})
	if err != nil {
		// This error is not expected since the bootstrap process parses the
		// config and makes sure that it is acceptable to the plugin. Still, it
		// is possible that the plugin parses the config successfully, but its
		// Build() method errors out.
		return nil, fmt.Errorf("xds: failed to get security plugin instance (%+v): %v", cfg, err)
	}
	return provider, nil
}

// A convenience method to create a watcher for cluster `name`. It also
// registers the watch with the xDS client, and adds the newly created watcher
// to the list of watchers maintained by the LB policy.
func (b *cdsBalancer) createAndAddWatcherForCluster(name string) {
	w := &clusterWatcher{
		name:   name,
		parent: b,
	}
	ws := &watcherState{
		watcher:     w,
		cancelWatch: xdsresource.WatchCluster(b.xdsClient, name, w),
	}
	b.watchers[name] = ws
}

// UpdateClientConnState receives the serviceConfig (which contains the
// clusterName to watch for in CDS) and the xdsClient object from the
// xdsResolver.
func (b *cdsBalancer) UpdateClientConnState(state balancer.ClientConnState) error {
	if b.xdsClient == nil {
		c := xdsclient.FromResolverState(state.ResolverState)
		if c == nil {
			b.logger.Warningf("Received balancer config with no xDS client")
			return balancer.ErrBadResolverState
		}
		b.xdsClient = c
	}
	b.logger.Infof("Received balancer config update: %s", pretty.ToJSON(state.BalancerConfig))

	// The errors checked here should ideally never happen because the
	// ServiceConfig in this case is prepared by the xdsResolver and is not
	// something that is received on the wire.
	lbCfg, ok := state.BalancerConfig.(*lbConfig)
	if !ok {
		b.logger.Warningf("Received unexpected balancer config type: %T", state.BalancerConfig)
		return balancer.ErrBadResolverState
	}
	if lbCfg.ClusterName == "" {
		b.logger.Warningf("Received balancer config with no cluster name")
		return balancer.ErrBadResolverState
	}

	// Do nothing and return early if configuration has not changed.
	if b.lbCfg != nil && b.lbCfg.ClusterName == lbCfg.ClusterName {
		return nil
	}
	b.lbCfg = lbCfg

	// Handle the update in a blocking fashion.
	errCh := make(chan error, 1)
	callback := func(context.Context) {
		// A config update with a changed top-level cluster name means that none
		// of our old watchers make any sense any more.
		b.closeAllWatchers()

		// Create a new watcher for the top-level cluster. Upon resolution, it
		// could end up creating more watchers if turns out to be an aggregate
		// cluster.
		b.createAndAddWatcherForCluster(lbCfg.ClusterName)
		errCh <- nil
	}
	onFailure := func() {
		// The call to Schedule returns false *only* if the serializer has been
		// closed, which happens only when we receive an update after close.
		errCh <- errBalancerClosed
	}
	b.serializer.ScheduleOr(callback, onFailure)
	return <-errCh
}

// ResolverError handles errors reported by the xdsResolver.
func (b *cdsBalancer) ResolverError(err error) {
	b.serializer.TrySchedule(func(context.Context) {
		// Missing Listener or RouteConfiguration on the management server
		// results in a 'resource not found' error from the xDS resolver. In
		// these cases, we should stap watching all of the current clusters
		// being watched.
		if xdsresource.ErrType(err) == xdsresource.ErrorTypeResourceNotFound {
			b.closeAllWatchers()
			b.closeChildPolicyAndReportTF(err)
			return
		}
		var root string
		if b.lbCfg != nil {
			root = b.lbCfg.ClusterName
		}
		b.onClusterError(root, err)
	})
}

// UpdateSubConnState handles subConn updates from gRPC.
func (b *cdsBalancer) UpdateSubConnState(sc balancer.SubConn, state balancer.SubConnState) {
	b.logger.Errorf("UpdateSubConnState(%v, %+v) called unexpectedly", sc, state)
}

// Closes all registered cluster watchers and removes them from the internal map.
//
// Only executed in the context of a serializer callback.
func (b *cdsBalancer) closeAllWatchers() {
	for name, state := range b.watchers {
		state.cancelWatch()
		delete(b.watchers, name)
	}
}

// closeChildPolicyAndReportTF closes the child policy, if it exists, and
// updates the connectivity state of the channel to TransientFailure with an
// error picker.
//
// Only executed in the context of a serializer callback.
func (b *cdsBalancer) closeChildPolicyAndReportTF(err error) {
	if b.childLB != nil {
		b.childLB.Close()
		b.childLB = nil
	}
	b.ccw.UpdateState(balancer.State{
		ConnectivityState: connectivity.TransientFailure,
		Picker:            base.NewErrPicker(err),
	})
}

// Close cancels the CDS watch, closes the child policy and closes the
// cdsBalancer.
func (b *cdsBalancer) Close() {
	b.serializer.TrySchedule(func(context.Context) {
		b.closeAllWatchers()

		if b.childLB != nil {
			b.childLB.Close()
			b.childLB = nil
		}
		if b.cachedRoot != nil {
			b.cachedRoot.Close()
		}
		if b.cachedIdentity != nil {
			b.cachedIdentity.Close()
		}
		b.logger.Infof("Shutdown")
	})
	b.serializerCancel()
	<-b.serializer.Done()
}

func (b *cdsBalancer) ExitIdle() {
	b.serializer.TrySchedule(func(context.Context) {
		if b.childLB == nil {
			b.logger.Warningf("Received ExitIdle with no child policy")
			return
		}
		// This implementation assumes the child balancer supports
		// ExitIdle (but still checks for the interface's existence to
		// avoid a panic if not).  If the child does not, no subconns
		// will be connected.
		b.childLB.ExitIdle()
	})
}

// Node ID needs to be manually added to errors generated in the following
// scenarios:
//   - resource-does-not-exist: since the xDS watch API uses a separate callback
//     instead of returning an error value. TODO(gRFC A88): Once A88 is
//     implemented, the xDS client will be able to add the node ID to
//     resource-does-not-exist errors as well, and we can get rid of this
//     special handling.
//   - received a good update from the xDS client, but the update either contains
//     an invalid security configuration or contains invalid aggragate cluster
//     config.
func (b *cdsBalancer) annotateErrorWithNodeID(err error) error {
	nodeID := b.xdsClient.BootstrapConfig().Node().GetId()
	return fmt.Errorf("[xDS node id: %v]: %w", nodeID, err)
}

// Handles a good Cluster update from the xDS client. Kicks off the discovery
// mechanism generation process from the top-level cluster and if the cluster
// graph is resolved, generates child policy config and pushes it down.
//
// Only executed in the context of a serializer callback.
func (b *cdsBalancer) onClusterUpdate(name string, update xdsresource.ClusterUpdate) {
	state := b.watchers[name]
	if state == nil {
		// We are currently not watching this cluster anymore. Return early.
		return
	}

	b.logger.Infof("Received Cluster resource: %s", pretty.ToJSON(update))

	// Update the watchers map with the update for the cluster.
	state.lastUpdate = &update

	// For an aggregate cluster, always use the security configuration on the
	// root cluster.
	if name == b.lbCfg.ClusterName {
		// Process the security config from the received update before building the
		// child policy or forwarding the update to it. We do this because the child
		// policy may try to create a new subConn inline. Processing the security
		// configuration here and setting up the handshakeInfo will make sure that
		// such attempts are handled properly.
		if err := b.handleSecurityConfig(update.SecurityCfg); err != nil {
			// If the security config is invalid, for example, if the provider
			// instance is not found in the bootstrap config, we need to put the
			// channel in transient failure.
			b.onClusterError(name, b.annotateErrorWithNodeID(fmt.Errorf("received Cluster resource contains invalid security config: %v", err)))
			return
		}
	}

	clustersSeen := make(map[string]bool)
	dms, ok, err := b.generateDMsForCluster(b.lbCfg.ClusterName, 0, nil, clustersSeen)
	if err != nil {
		b.onClusterError(b.lbCfg.ClusterName, b.annotateErrorWithNodeID(fmt.Errorf("failed to generate discovery mechanisms: %v", err)))
		return
	}
	if ok {
		if len(dms) == 0 {
			b.onClusterError(b.lbCfg.ClusterName, b.annotateErrorWithNodeID(fmt.Errorf("aggregate cluster graph has no leaf clusters")))
			return
		}
		// Child policy is built the first time we resolve the cluster graph.
		if b.childLB == nil {
			childLB, err := newChildBalancer(b.ccw, b.bOpts)
			if err != nil {
				b.logger.Errorf("Failed to create child policy of type %s: %v", clusterresolver.Name, err)
				return
			}
			b.childLB = childLB
			b.logger.Infof("Created child policy %p of type %s", b.childLB, clusterresolver.Name)
		}

		// Prepare the child policy configuration, convert it to JSON, have it
		// parsed by the child policy to convert it into service config and push
		// an update to it.
		childCfg := &clusterresolver.LBConfig{
			DiscoveryMechanisms: dms,
			// The LB policy is configured by the root cluster.
			XDSLBPolicy: b.watchers[b.lbCfg.ClusterName].lastUpdate.LBPolicy,
		}
		cfgJSON, err := json.Marshal(childCfg)
		if err != nil {
			// Shouldn't happen, since we just prepared struct.
			b.logger.Errorf("cds_balancer: error marshalling prepared config: %v", childCfg)
			return
		}

		var sc serviceconfig.LoadBalancingConfig
		if sc, err = b.childConfigParser.ParseConfig(cfgJSON); err != nil {
			b.logger.Errorf("cds_balancer: cluster_resolver config generated %v is invalid: %v", string(cfgJSON), err)
			return
		}

		ccState := balancer.ClientConnState{
			ResolverState:  xdsclient.SetClient(resolver.State{}, b.xdsClient),
			BalancerConfig: sc,
		}
		if err := b.childLB.UpdateClientConnState(ccState); err != nil {
			b.logger.Errorf("Encountered error when sending config {%+v} to child policy: %v", ccState, err)
		}
	}
	// We no longer need the clusters that we did not see in this iteration of
	// generateDMsForCluster().
	for cluster, state := range b.watchers {
		if !clustersSeen[cluster] {
			state.cancelWatch()
			delete(b.watchers, cluster)
		}
	}
}

// Handles an ambient error Cluster update from the xDS client to not stop
// using the previously seen resource.
//
// Only executed in the context of a serializer callback.
func (b *cdsBalancer) onClusterAmbientError(name string, err error) {
	b.logger.Warningf("Cluster resource %q received ambient error update: %v", name, err)

	if xdsresource.ErrType(err) != xdsresource.ErrorTypeConnection && b.childLB != nil {
		// Connection errors will be sent to the child balancers directly.
		// There's no need to forward them.
		b.childLB.ResolverError(err)
	}
}

// Handles an error Cluster update from the xDS client to stop using the
// previously seen resource. Propagates the error down to the child policy
// if one exists, and puts the channel in TRANSIENT_FAILURE.
//
// Only executed in the context of a serializer callback.
func (b *cdsBalancer) onClusterResourceError(name string, err error) {
	b.logger.Warningf("CDS watch for resource %q reported resource error", name)
	b.closeChildPolicyAndReportTF(err)
}

// Generates discovery mechanisms for the cluster graph rooted at `name`. This
// method is called recursively if `name` corresponds to an aggregate cluster,
// with the base case for recursion being a leaf cluster. If a new cluster is
// encountered when traversing the graph, a watcher is created for it.
//
// Inputs:
// - name: name of the cluster to start from
// - depth: recursion depth of the current cluster, starting from root
// - dms: prioritized list of current discovery mechanisms
// - clustersSeen: cluster names seen so far in the graph traversal
//
// Outputs:
//   - new prioritized list of discovery mechanisms
//   - boolean indicating if traversal of the aggregate cluster graph is
//     complete. If false, the above list of discovery mechanisms is ignored.
//   - error indicating if any error was encountered as part of the graph
//     traversal. If error is non-nil, the other return values are ignored.
//
// Only executed in the context of a serializer callback.
func (b *cdsBalancer) generateDMsForCluster(name string, depth int, dms []clusterresolver.DiscoveryMechanism, clustersSeen map[string]bool) ([]clusterresolver.DiscoveryMechanism, bool, error) {
	if depth >= aggregateClusterMaxDepth {
		return dms, false, errExceedsMaxDepth
	}

	if clustersSeen[name] {
		// Discovery mechanism already seen through a different branch.
		return dms, true, nil
	}
	clustersSeen[name] = true

	state, ok := b.watchers[name]
	if !ok {
		// If we have not seen this cluster so far, create a watcher for it, add
		// it to the map, start the watch and return.
		b.createAndAddWatcherForCluster(name)

		// And since we just created the watcher, we know that we haven't
		// resolved the cluster graph yet.
		return dms, false, nil
	}

	// A watcher exists, but no update has been received yet.
	if state.lastUpdate == nil {
		return dms, false, nil
	}

	var dm clusterresolver.DiscoveryMechanism
	cluster := state.lastUpdate
	switch cluster.ClusterType {
	case xdsresource.ClusterTypeAggregate:
		// This boolean is used to track if any of the clusters in the graph is
		// not yet completely resolved or returns errors, thereby allowing us to
		// traverse as much of the graph as possible (and start the associated
		// watches where required) to ensure that clustersSeen contains all
		// clusters in the graph that we can traverse to.
		missingCluster := false
		var err error
		for _, child := range cluster.PrioritizedClusterNames {
			var ok bool
			dms, ok, err = b.generateDMsForCluster(child, depth+1, dms, clustersSeen)
			if err != nil || !ok {
				missingCluster = true
			}
		}
		return dms, !missingCluster, err
	case xdsresource.ClusterTypeEDS:
		dm = clusterresolver.DiscoveryMechanism{
			Type:                  clusterresolver.DiscoveryMechanismTypeEDS,
			Cluster:               cluster.ClusterName,
			EDSServiceName:        cluster.EDSServiceName,
			MaxConcurrentRequests: cluster.MaxRequests,
			LoadReportingServer:   cluster.LRSServerConfig,
		}
	case xdsresource.ClusterTypeLogicalDNS:
		dm = clusterresolver.DiscoveryMechanism{
			Type:                  clusterresolver.DiscoveryMechanismTypeLogicalDNS,
			Cluster:               cluster.ClusterName,
			DNSHostname:           cluster.DNSHostName,
			MaxConcurrentRequests: cluster.MaxRequests,
			LoadReportingServer:   cluster.LRSServerConfig,
		}
	}
	odJSON := cluster.OutlierDetection
	// "In the cds LB policy, if the outlier_detection field is not set in
	// the Cluster resource, a "no-op" outlier_detection config will be
	// generated in the corresponding DiscoveryMechanism config, with all
	// fields unset." - A50
	if odJSON == nil {
		// This will pick up top level defaults in Cluster Resolver
		// ParseConfig, but sre and fpe will be nil still so still a
		// "no-op" config.
		odJSON = json.RawMessage(`{}`)
	}
	dm.OutlierDetection = odJSON

	dm.TelemetryLabels = cluster.TelemetryLabels

	return append(dms, dm), true, nil
}

func (b *cdsBalancer) onClusterError(name string, err error) {
	if b.childLB != nil {
		b.onClusterAmbientError(name, err)
	} else {
		b.onClusterResourceError(name, err)
	}
}

// ccWrapper wraps the balancer.ClientConn passed to the CDS balancer at
// creation and intercepts the NewSubConn() and UpdateAddresses() call from the
// child policy to add security configuration required by xDS credentials.
//
// Other methods of the balancer.ClientConn interface are not overridden and
// hence get the original implementation.
type ccWrapper struct {
	balancer.ClientConn

	xdsHIPtr *unsafe.Pointer
}

// NewSubConn intercepts NewSubConn() calls from the child policy and adds an
// address attribute which provides all information required by the xdsCreds
// handshaker to perform the TLS handshake.
func (ccw *ccWrapper) NewSubConn(addrs []resolver.Address, opts balancer.NewSubConnOptions) (balancer.SubConn, error) {
	newAddrs := make([]resolver.Address, len(addrs))
	for i, addr := range addrs {
		newAddrs[i] = xdsinternal.SetHandshakeInfo(addr, ccw.xdsHIPtr)
	}

	// No need to override opts.StateListener; just forward all calls to the
	// child that created the SubConn.
	return ccw.ClientConn.NewSubConn(newAddrs, opts)
}

func (ccw *ccWrapper) UpdateAddresses(sc balancer.SubConn, addrs []resolver.Address) {
	newAddrs := make([]resolver.Address, len(addrs))
	for i, addr := range addrs {
		newAddrs[i] = xdsinternal.SetHandshakeInfo(addr, ccw.xdsHIPtr)
	}
	ccw.ClientConn.UpdateAddresses(sc, newAddrs)
}

// systemRootCertsProvider implements a certprovider.Provider that returns the
// system default root certificates for validation.
type systemRootCertsProvider struct{}

func (systemRootCertsProvider) Close() {}

func (systemRootCertsProvider) KeyMaterial(context.Context) (*certprovider.KeyMaterial, error) {
	rootCAs, err := x509SystemCertPoolFunc()
	if err != nil {
		return nil, err
	}
	return &certprovider.KeyMaterial{
		Roots: rootCAs,
	}, nil
}
