/*
 *
 * Copyright 2025 gRPC authors.
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

package xdsclient

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"google.golang.org/grpc/grpclog"
	igrpclog "google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/xds/clients"
	"google.golang.org/grpc/internal/xds/clients/internal/syncutil"
	"google.golang.org/grpc/internal/xds/clients/xdsclient/internal/xdsresource"
	"google.golang.org/grpc/internal/xds/clients/xdsclient/metrics"
	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/timestamppb"

	v3adminpb "github.com/envoyproxy/go-control-plane/envoy/admin/v3"
	v3statuspb "github.com/envoyproxy/go-control-plane/envoy/service/status/v3"
)

type resourceState struct {
	watchers          map[ResourceWatcher]bool       // Set of watchers for this resource.
	cache             ResourceData                   // Most recent ACKed update for this resource.
	md                xdsresource.UpdateMetadata     // Metadata for the most recent update.
	deletionIgnored   bool                           // True, if resource deletion was ignored for a prior update.
	xdsChannelConfigs map[*xdsChannelWithConfig]bool // Set of xdsChannels where this resource is subscribed.
}

// xdsChannelForADS is used to acquire a reference to an xdsChannel. This
// functionality is provided by the xdsClient.
//
// The arguments to the function are as follows:
//   - the server config for the xdsChannel
//   - the calling authority on which a set of callbacks are invoked by the
//     xdsChannel on ADS stream events
//
// Returns a reference to the xdsChannel and a function to release the same. A
// non-nil error is returned if the channel creation fails and the first two
// return values are meaningless in this case.
type xdsChannelForADS func(*ServerConfig, *authority) (*xdsChannel, func(), error)

// xdsChannelWithConfig is a struct that holds an xdsChannel and its associated
// ServerConfig, along with a cleanup function to release the xdsChannel.
type xdsChannelWithConfig struct {
	channel      *xdsChannel
	serverConfig *ServerConfig
	cleanup      func()
}

// authority provides the functionality required to communicate with a
// management server corresponding to an authority name specified in the
// xDS client configuration.
//
// It holds references to one or more xdsChannels, one for each server
// configuration in the config, to allow fallback from a primary management
// server to a secondary management server. Authorities that contain similar
// server configuration entries will end up sharing the xdsChannel for that
// server configuration. The xdsChannels are owned and managed by the xdsClient.
//
// It also contains a cache of resource state for resources requested from
// management server(s). This cache contains the list of registered watchers and
// the most recent resource configuration received from the management server.
type authority struct {
	// The following fields are initialized at creation time and are read-only
	// afterwards, and therefore don't need to be protected with a mutex.
	name                      string                       // Name of the authority from xDS client configuration.
	watcherCallbackSerializer *syncutil.CallbackSerializer // Serializer to run watcher callbacks, owned by the xDS client implementation.
	getChannelForADS          xdsChannelForADS             // Function to get an xdsChannel for ADS, provided by the xDS client implementation.
	xdsClientSerializer       *syncutil.CallbackSerializer // Serializer to run call ins from the xDS client, owned by this authority.
	xdsClientSerializerClose  func()                       // Function to close the above serializer.
	logger                    *igrpclog.PrefixLogger       // Logger for this authority.
	target                    string                       // The gRPC Channel target.
	metricsReporter           clients.MetricsReporter

	// The below defined fields must only be accessed in the context of the
	// serializer callback, owned by this authority.

	// A two level map containing the state of all the resources being watched.
	//
	// The first level map key is the ResourceType (Listener, Route etc). This
	// allows us to have a single map for all resources instead of having per
	// resource-type maps.
	//
	// The second level map key is the resource name, with the value being the
	// actual state of the resource.
	resources map[ResourceType]map[string]*resourceState

	// An ordered list of xdsChannels corresponding to the list of server
	// configurations specified for this authority in the config. The
	// ordering specifies the order in which these channels are preferred for
	// fallback.
	xdsChannelConfigs []*xdsChannelWithConfig

	// The current active xdsChannel. Here, active does not mean that the
	// channel has a working connection to the server. It simply points to the
	// channel that we are trying to work with, based on fallback logic.
	activeXDSChannel *xdsChannelWithConfig
}

// authorityBuildOptions wraps arguments required to create a new authority.
type authorityBuildOptions struct {
	serverConfigs    []ServerConfig               // Server configs for the authority
	name             string                       // Name of the authority
	serializer       *syncutil.CallbackSerializer // Callback serializer for invoking watch callbacks
	getChannelForADS xdsChannelForADS             // Function to acquire a reference to an xdsChannel
	logPrefix        string                       // Prefix for logging
	target           string                       // Target for the gRPC Channel that owns xDS Client/Authority
	metricsReporter  clients.MetricsReporter      // Metrics reporter for the authority
}

// newAuthority creates a new authority instance with the provided
// configuration. The authority is responsible for managing the state of
// resources requested from the management server, as well as acquiring and
// releasing references to channels used to communicate with the management
// server.
//
// Note that no channels to management servers are created at this time. Instead
// a channel to the first server configuration is created when the first watch
// is registered, and more channels are created as needed by the fallback logic.
func newAuthority(args authorityBuildOptions) *authority {
	ctx, cancel := context.WithCancel(context.Background())
	l := grpclog.Component("xds")
	logPrefix := args.logPrefix + fmt.Sprintf("[authority %q] ", args.name)
	ret := &authority{
		name:                      args.name,
		watcherCallbackSerializer: args.serializer,
		getChannelForADS:          args.getChannelForADS,
		xdsClientSerializer:       syncutil.NewCallbackSerializer(ctx),
		xdsClientSerializerClose:  cancel,
		logger:                    igrpclog.NewPrefixLogger(l, logPrefix),
		resources:                 make(map[ResourceType]map[string]*resourceState),
		target:                    args.target,
		metricsReporter:           args.metricsReporter,
	}

	// Create an ordered list of xdsChannels with their server configs. The
	// actual channel to the first server configuration is created when the
	// first watch is registered, and channels to other server configurations
	// are created as needed to support fallback.
	for _, sc := range args.serverConfigs {
		ret.xdsChannelConfigs = append(ret.xdsChannelConfigs, &xdsChannelWithConfig{serverConfig: &sc})
	}
	return ret
}

// adsStreamFailure is called to notify the authority about an ADS stream
// failure on an xdsChannel to the management server identified by the provided
// server config. The error is forwarded to all the resource watchers.
//
// This method is called by the xDS client implementation (on all interested
// authorities) when a stream error is reported by an xdsChannel.
//
// Errors of type xdsresource.ErrTypeStreamFailedAfterRecv are ignored.
func (a *authority) adsStreamFailure(serverConfig *ServerConfig, err error) {
	a.xdsClientSerializer.TrySchedule(func(context.Context) {
		a.handleADSStreamFailure(serverConfig, err)
	})
}

// Handles ADS stream failure by invoking watch callbacks and triggering
// fallback if the associated conditions are met.
//
// Only executed in the context of a serializer callback.
func (a *authority) handleADSStreamFailure(serverConfig *ServerConfig, err error) {
	if a.logger.V(2) {
		a.logger.Infof("Connection to server %s failed with error: %v", serverConfig, err)
	}

	// We do not consider it an error if the ADS stream was closed after having
	// received a response on the stream. This is because there are legitimate
	// reasons why the server may need to close the stream during normal
	// operations, such as needing to rebalance load or the underlying
	// connection hitting its max connection age limit. See gRFC A57 for more
	// details.
	if xdsresource.ErrType(err) == xdsresource.ErrTypeStreamFailedAfterRecv {
		a.logger.Warningf("Watchers not notified since ADS stream failed after having received at least one response: %v", err)
		return
	}

	// Two conditions need to be met for fallback to be triggered:
	// 1. There is a connectivity failure on the ADS stream, as described in
	//    gRFC A57. For us, this means that the ADS stream was closed before the
	//    first server response was received. We already checked that condition
	//    earlier in this method.
	// 2. There is at least one watcher for a resource that is not cached.
	//    Cached resources include ones that
	//    - have been successfully received and can be used.
	//    - are considered non-existent according to xDS Protocol Specification.
	if !a.watcherExistsForUncachedResource() {
		if a.logger.V(2) {
			a.logger.Infof("No watchers for uncached resources. Not triggering fallback")
		}
		// Since we are not triggering fallback, propagate the connectivity
		// error to all watchers and return early.
		a.propagateConnectivityErrorToAllWatchers(err)
		return
	}

	// Attempt to fallback to servers with lower priority than the failing one.
	currentServerIdx := a.serverIndexForConfig(serverConfig)
	for i := currentServerIdx + 1; i < len(a.xdsChannelConfigs); i++ {
		if a.fallbackToServer(a.xdsChannelConfigs[i]) {
			// Since we have successfully triggered fallback, we don't have to
			// notify watchers about the connectivity error.
			return
		}
	}

	// Having exhausted all available servers, we must notify watchers of the
	// connectivity error - A71.
	a.propagateConnectivityErrorToAllWatchers(err)
}

// propagateConnectivityErrorToAllWatchers propagates the given connection error
// to all watchers of all resources.
//
// Only executed in the context of a serializer callback.
func (a *authority) propagateConnectivityErrorToAllWatchers(err error) {
	for _, rType := range a.resources {
		for _, state := range rType {
			for watcher := range state.watchers {
				if state.cache == nil {
					a.watcherCallbackSerializer.TrySchedule(func(context.Context) {
						watcher.ResourceError(xdsresource.NewErrorf(xdsresource.ErrorTypeConnection, "xds: error received from xDS stream: %v", err), func() {})
					})
				} else {
					a.watcherCallbackSerializer.TrySchedule(func(context.Context) {
						watcher.AmbientError(xdsresource.NewErrorf(xdsresource.ErrorTypeConnection, "xds: error received from xDS stream: %v", err), func() {})
					})
				}
			}
		}
	}
}

// serverIndexForConfig returns the index of the xdsChannelConfig matching the
// provided server config, panicking if no match is found (which indicates a
// programming error).
func (a *authority) serverIndexForConfig(sc *ServerConfig) int {
	for i, cfg := range a.xdsChannelConfigs {
		if isServerConfigEqual(sc, cfg.serverConfig) {
			return i
		}
	}
	panic(fmt.Sprintf("no server config matching %v found", sc))
}

// Determines the server to fallback to and triggers fallback to the same. If
// required, creates an xdsChannel to that server, and re-subscribes to all
// existing resources.
//
// Only executed in the context of a serializer callback.
func (a *authority) fallbackToServer(xc *xdsChannelWithConfig) bool {
	if a.logger.V(2) {
		a.logger.Infof("Attempting to initiate fallback to server %q", xc.serverConfig)
	}

	if xc.channel != nil {
		if a.logger.V(2) {
			a.logger.Infof("Channel to the next server in the list %q already exists", xc.serverConfig)
		}
		return false
	}

	channel, cleanup, err := a.getChannelForADS(xc.serverConfig, a)
	if err != nil {
		a.logger.Errorf("Failed to create xDS channel: %v", err)
		return false
	}
	xc.channel = channel
	xc.cleanup = cleanup
	a.activeXDSChannel = xc

	// Subscribe to all existing resources from the new management server.
	for typ, resources := range a.resources {
		for name, state := range resources {
			if a.logger.V(2) {
				a.logger.Infof("Resubscribing to resource of type %q and name %q", typ.TypeName, name)
			}
			xc.channel.subscribe(typ, name)

			// Add the new channel to the list of xdsChannels from which this
			// resource has been requested from. Retain the cached resource and
			// the set of existing watchers (and other metadata fields) in the
			// resource state.
			state.xdsChannelConfigs[xc] = true
		}
	}
	return true
}

// adsResourceUpdate is called to notify the authority about a resource update
// received on the ADS stream.
//
// This method is called by the xDS client implementation (on all interested
// authorities) when a stream error is reported by an xdsChannel.
func (a *authority) adsResourceUpdate(serverConfig *ServerConfig, rType ResourceType, updates map[string]dataAndErrTuple, md xdsresource.UpdateMetadata, onDone func()) {
	a.xdsClientSerializer.TrySchedule(func(context.Context) {
		a.handleADSResourceUpdate(serverConfig, rType, updates, md, onDone)
	})
}

// handleADSResourceUpdate processes an update from the xDS client, updating the
// resource cache and notifying any registered watchers of the update.
//
// If the update is received from a higher priority xdsChannel that was
// previously down, we revert to it and close all lower priority xdsChannels.
//
// Once the update has been processed by all watchers, the authority is expected
// to invoke the onDone callback.
//
// Only executed in the context of a serializer callback.
func (a *authority) handleADSResourceUpdate(serverConfig *ServerConfig, rType ResourceType, updates map[string]dataAndErrTuple, md xdsresource.UpdateMetadata, onDone func()) {
	if !a.handleRevertingToPrimaryOnUpdate(serverConfig) {
		return
	}

	// We build a list of callback funcs to invoke, and invoke them at the end
	// of this method instead of inline (when handling the update for a
	// particular resource), because we want to make sure that all calls to
	// increment watcherCnt happen before any callbacks are invoked. This will
	// ensure that the onDone callback is never invoked before all watcher
	// callbacks are invoked, and the watchers have processed the update.
	watcherCnt := new(atomic.Int64)
	done := func() {
		if watcherCnt.Add(-1) == 0 {
			onDone()
		}
	}
	funcsToSchedule := []func(context.Context){}
	defer func() {
		if len(funcsToSchedule) == 0 {
			// When there are no watchers for the resources received as part of
			// this update, invoke onDone explicitly to unblock the next read on
			// the ADS stream.
			onDone()
			return
		}
		for _, f := range funcsToSchedule {
			a.watcherCallbackSerializer.ScheduleOr(f, onDone)
		}
	}()

	resourceStates := a.resources[rType]
	for name, uErr := range updates {
		state, ok := resourceStates[name]
		if !ok {
			continue
		}

		// On error, keep previous version of the resource. But update status
		// and error.
		if uErr.Err != nil {
			if a.metricsReporter != nil {
				a.metricsReporter.ReportMetric(&metrics.ResourceUpdateInvalid{
					ServerURI: serverConfig.ServerIdentifier.ServerURI, ResourceType: rType.TypeName,
				})
			}
			state.md.ErrState = md.ErrState
			state.md.Status = md.Status
			for watcher := range state.watchers {
				watcher := watcher
				err := uErr.Err
				watcherCnt.Add(1)
				if state.cache == nil {
					funcsToSchedule = append(funcsToSchedule, func(context.Context) { watcher.ResourceError(err, done) })
				} else {
					funcsToSchedule = append(funcsToSchedule, func(context.Context) { watcher.AmbientError(err, done) })
				}
			}
			continue
		}

		if a.metricsReporter != nil {
			a.metricsReporter.ReportMetric(&metrics.ResourceUpdateValid{
				ServerURI: serverConfig.ServerIdentifier.ServerURI, ResourceType: rType.TypeName,
			})
		}

		if state.deletionIgnored {
			state.deletionIgnored = false
			a.logger.Infof("A valid update was received for resource %q of type %q after previously ignoring a deletion", name, rType.TypeName)
		}
		// Notify watchers if any of these conditions are met:
		//   - this is the first update for this resource
		//   - this update is different from the one currently cached
		//   - the previous update for this resource was NACKed, but the update
		//     before that was the same as this update.
		if state.cache == nil || !state.cache.Equal(uErr.Resource) || state.md.ErrState != nil {
			// Update the resource cache.
			if a.logger.V(2) {
				a.logger.Infof("Resource type %q with name %q added to cache", rType.TypeName, name)
			}
			state.cache = uErr.Resource

			for watcher := range state.watchers {
				watcher := watcher
				resource := uErr.Resource
				watcherCnt.Add(1)
				funcsToSchedule = append(funcsToSchedule, func(context.Context) { watcher.ResourceChanged(resource, done) })
			}
		}

		// Set status to ACK, and clear error state. The metadata might be a
		// NACK metadata because some other resources in the same response
		// are invalid.
		state.md = md
		state.md.ErrState = nil
		state.md.Status = xdsresource.ServiceStatusACKed
		if md.ErrState != nil {
			state.md.Version = md.ErrState.Version
		}
	}

	// If this resource type requires that all resources be present in every
	// SotW response from the server, a response that does not include a
	// previously seen resource will be interpreted as a deletion of that
	// resource unless ignore_resource_deletion option was set in the server
	// config.
	if !rType.AllResourcesRequiredInSotW {
		return
	}
	for name, state := range resourceStates {
		if state.cache == nil {
			// If the resource state does not contain a cached update, which can
			// happen when:
			// - resource was newly requested but has not yet been received, or,
			// - resource was removed as part of a previous update,
			// we don't want to generate an error for the watchers.
			//
			// For the first of the above two conditions, this ADS response may
			// be in reaction to an earlier request that did not yet request the
			// new resource, so its absence from the response does not
			// necessarily indicate that the resource does not exist. For that
			// case, we rely on the request timeout instead.
			//
			// For the second of the above two conditions, we already generated
			// an error when we received the first response which removed this
			// resource. So, there is no need to generate another one.
			continue
		}
		if _, ok := updates[name]; ok {
			// If the resource was present in the response, move on.
			continue
		}
		if state.md.Status == xdsresource.ServiceStatusNotExist {
			// The metadata status is set to "ServiceStatusNotExist" if a
			// previous update deleted this resource, in which case we do not
			// want to repeatedly call the watch callbacks with a
			// "resource-not-found" error.
			continue
		}
		if serverConfig.IgnoreResourceDeletion {
			// Per A53, resource deletions are ignored if the
			// `ignore_resource_deletion` server feature is enabled through the
			// xDS client configuration. If the resource deletion is to be
			// ignored, the resource is not removed from the cache and the
			// corresponding OnResourceDoesNotExist() callback is not invoked on
			// the watchers.
			if !state.deletionIgnored {
				state.deletionIgnored = true
				a.logger.Warningf("Ignoring resource deletion for resource %q of type %q", name, rType.TypeName)
			}
			continue
		}

		// If we get here, it means that the resource exists in cache, but not
		// in the new update. Delete the resource from cache, and send a
		// resource not found error to indicate that the resource has been
		// removed. Metadata for the resource is still maintained, as this is
		// required by CSDS.
		state.cache = nil
		state.md = xdsresource.UpdateMetadata{Status: xdsresource.ServiceStatusNotExist}
		for watcher := range state.watchers {
			watcher := watcher
			watcherCnt.Add(1)
			funcsToSchedule = append(funcsToSchedule, func(context.Context) {
				watcher.ResourceError(xdsresource.NewErrorf(xdsresource.ErrorTypeResourceNotFound, "xds: resource %q of type %q has been removed", name, rType.TypeName), done)
			})
		}
	}
}

// adsResourceDoesNotExist is called by the xDS client implementation (on all
// interested authorities) to notify the authority that a subscribed resource
// does not exist.
func (a *authority) adsResourceDoesNotExist(rType ResourceType, resourceName string) {
	a.xdsClientSerializer.TrySchedule(func(context.Context) {
		a.handleADSResourceDoesNotExist(rType, resourceName)
	})
}

// handleADSResourceDoesNotExist is called when a subscribed resource does not
// exist. It removes the resource from the cache, updates the metadata status
// to ServiceStatusNotExist, and notifies all watchers that the resource does
// not exist.
func (a *authority) handleADSResourceDoesNotExist(rType ResourceType, resourceName string) {
	if a.logger.V(2) {
		a.logger.Infof("Watch for resource %q of type %s timed out", resourceName, rType.TypeName)
	}

	resourceStates := a.resources[rType]
	if resourceStates == nil {
		if a.logger.V(2) {
			a.logger.Infof("Resource %q of type %s currently not being watched", resourceName, rType.TypeName)
		}
		return
	}
	state, ok := resourceStates[resourceName]
	if !ok {
		if a.logger.V(2) {
			a.logger.Infof("Resource %q of type %s currently not being watched", resourceName, rType.TypeName)
		}
		return
	}

	state.cache = nil
	state.md = xdsresource.UpdateMetadata{Status: xdsresource.ServiceStatusNotExist}
	for watcher := range state.watchers {
		watcher := watcher
		a.watcherCallbackSerializer.TrySchedule(func(context.Context) {
			watcher.ResourceError(xdsresource.NewErrorf(xdsresource.ErrorTypeResourceNotFound, "xds: resource %q of type %q has been removed", resourceName, rType.TypeName), func() {})
		})
	}
}

// handleRevertingToPrimaryOnUpdate is called when a resource update is received
// from the xDS client.
//
// If the update is from the currently active server, nothing is done. Else, all
// lower priority servers are closed and the active server is reverted to the
// highest priority server that sent the update.
//
// The return value indicates whether subsequent processing of the resource
// update should continue or not.
//
// This method is only executed in the context of a serializer callback.
func (a *authority) handleRevertingToPrimaryOnUpdate(serverConfig *ServerConfig) bool {
	if a.activeXDSChannel == nil {
		// This can happen only when all watches on this authority have been
		// removed, and the xdsChannels have been closed. This update should
		// have been received prior to closing of the channel, and therefore
		// must be ignored.
		return false
	}

	if isServerConfigEqual(serverConfig, a.activeXDSChannel.serverConfig) {
		// If the resource update is from the current active server, nothing
		// needs to be done from fallback point of view.
		return true
	}

	if a.logger.V(2) {
		a.logger.Infof("Received update from non-active server %q", serverConfig)
	}

	// If the resource update is not from the current active server, it means
	// that we have received an update either from:
	// - a server that has a higher priority than the current active server and
	//   therefore we need to revert back to it and close all lower priority
	//   servers, or,
	// - a server that has a lower priority than the current active server. This
	//   can happen when the server close and the response race against each
	//   other. We can safely ignore this update, since we have already reverted
	//   to the higher priority server, and closed all lower priority servers.
	serverIdx := a.serverIndexForConfig(serverConfig)
	activeServerIdx := a.serverIndexForConfig(a.activeXDSChannel.serverConfig)
	if activeServerIdx < serverIdx {
		return false
	}

	// At this point, we are guaranteed that we have received a response from a
	// higher priority server compared to the current active server. So, we
	// revert to the higher priorty server and close all lower priority ones.
	a.activeXDSChannel = a.xdsChannelConfigs[serverIdx]

	// Close all lower priority channels.
	//
	// But before closing any channel, we need to unsubscribe from any resources
	// that were subscribed to on this channel. Resources could be subscribed to
	// from multiple channels as we fallback to lower priority servers. But when
	// a higher priority one comes back up, we need to unsubscribe from all
	// lower priority ones before releasing the reference to them.
	for i := serverIdx + 1; i < len(a.xdsChannelConfigs); i++ {
		cfg := a.xdsChannelConfigs[i]

		for rType, rState := range a.resources {
			for resourceName, state := range rState {
				for xcc := range state.xdsChannelConfigs {
					if xcc != cfg {
						continue
					}
					// If the current resource is subscribed to on this channel,
					// unsubscribe, and remove the channel from the list of
					// channels that this resource is subscribed to.
					xcc.channel.unsubscribe(rType, resourceName)
					delete(state.xdsChannelConfigs, xcc)
				}
			}
		}

		// Release the reference to the channel.
		if cfg.cleanup != nil {
			if a.logger.V(2) {
				a.logger.Infof("Closing lower priority server %q", cfg.serverConfig)
			}
			cfg.cleanup()
			cfg.cleanup = nil
		}
		cfg.channel = nil
	}
	return true
}

// watchResource registers a new watcher for the specified resource type and
// name. It returns a function that can be called to cancel the watch.
//
// If this is the first watch for any resource on this authority, an xdsChannel
// to the first management server (from the list of server configurations) will
// be created.
//
// If this is the first watch for the given resource name, it will subscribe to
// the resource with the xdsChannel. If a cached copy of the resource exists, it
// will immediately notify the new watcher. When the last watcher for a resource
// is removed, it will unsubscribe the resource from the xdsChannel.
func (a *authority) watchResource(rType ResourceType, resourceName string, watcher ResourceWatcher) func() {
	cleanup := func() {}
	done := make(chan struct{})

	a.xdsClientSerializer.ScheduleOr(func(context.Context) {
		defer close(done)

		if a.logger.V(2) {
			a.logger.Infof("New watch for type %q, resource name %q", rType.TypeName, resourceName)
		}

		xdsChannel, err := a.xdsChannelToUse()
		if err != nil {
			a.watcherCallbackSerializer.TrySchedule(func(context.Context) { watcher.ResourceError(err, func() {}) })
			return
		}

		// Lookup the entry for the resource type in the top-level map. If there is
		// no entry for this resource type, create one.
		resources := a.resources[rType]
		if resources == nil {
			resources = make(map[string]*resourceState)
			a.resources[rType] = resources
		}

		// Lookup the resource state for the particular resource name that the watch
		// is being registered for. If this is the first watch for this resource
		// name, request it from the management server.
		state := resources[resourceName]
		if state == nil {
			if a.logger.V(2) {
				a.logger.Infof("First watch for type %q, resource name %q", rType.TypeName, resourceName)
			}
			state = &resourceState{
				watchers:          make(map[ResourceWatcher]bool),
				md:                xdsresource.UpdateMetadata{Status: xdsresource.ServiceStatusRequested},
				xdsChannelConfigs: map[*xdsChannelWithConfig]bool{xdsChannel: true},
			}
			resources[resourceName] = state
			xdsChannel.channel.subscribe(rType, resourceName)
		}
		// Always add the new watcher to the set of watchers.
		state.watchers[watcher] = true

		// If we have a cached copy of the resource, notify the new watcher
		// immediately.
		if state.cache != nil {
			if a.logger.V(2) {
				a.logger.Infof("Resource type %q with resource name %q found in cache: %v", rType.TypeName, resourceName, state.cache)
			}
			// state can only be accessed in the context of an
			// xdsClientSerializer callback. Hence making a copy of the cached
			// resource here for watchCallbackSerializer.
			resource := state.cache
			a.watcherCallbackSerializer.TrySchedule(func(context.Context) { watcher.ResourceChanged(resource, func() {}) })
		}
		// If last update was NACK'd, notify the new watcher of error
		// immediately as well.
		if state.md.Status == xdsresource.ServiceStatusNACKed {
			if a.logger.V(2) {
				a.logger.Infof("Resource type %q with resource name %q was NACKed", rType.TypeName, resourceName)
			}
			// state can only be accessed in the context of an
			// xdsClientSerializer callback. Hence making a copy of the error
			// here for watchCallbackSerializer.
			err := state.md.ErrState.Err
			if state.cache == nil {
				a.watcherCallbackSerializer.TrySchedule(func(context.Context) { watcher.ResourceError(err, func() {}) })
			} else {
				a.watcherCallbackSerializer.TrySchedule(func(context.Context) { watcher.AmbientError(err, func() {}) })
			}
		}
		// If the metadata field is updated to indicate that the management
		// server does not have this resource, notify the new watcher.
		if state.md.Status == xdsresource.ServiceStatusNotExist {
			a.watcherCallbackSerializer.TrySchedule(func(context.Context) {
				watcher.ResourceError(xdsresource.NewErrorf(xdsresource.ErrorTypeResourceNotFound, "xds: resource %q of type %q has been removed", resourceName, rType.TypeName), func() {})
			})
		}
		cleanup = a.unwatchResource(rType, resourceName, watcher)
	}, func() {
		if a.logger.V(2) {
			a.logger.Infof("Failed to schedule a watch for type %q, resource name %q, because the xDS client is closed", rType.TypeName, resourceName)
		}
		close(done)
	})
	<-done
	return cleanup
}

func (a *authority) unwatchResource(rType ResourceType, resourceName string, watcher ResourceWatcher) func() {
	return sync.OnceFunc(func() {
		done := make(chan struct{})
		a.xdsClientSerializer.ScheduleOr(func(context.Context) {
			defer close(done)

			if a.logger.V(2) {
				a.logger.Infof("Canceling a watch for type %q, resource name %q", rType.TypeName, resourceName)
			}

			// Lookup the resource type from the resource cache. The entry is
			// guaranteed to be present, since *we* were the ones who added it in
			// there when the watch was registered.
			resources := a.resources[rType]
			state := resources[resourceName]

			// Delete this particular watcher from the list of watchers, so that its
			// callback will not be invoked in the future.
			delete(state.watchers, watcher)
			if len(state.watchers) > 0 {
				if a.logger.V(2) {
					a.logger.Infof("Other watchers exist for type %q, resource name %q", rType.TypeName, resourceName)
				}
				return
			}

			// There are no more watchers for this resource. Unsubscribe this
			// resource from all channels where it was subscribed to and delete
			// the state associated with it.
			if a.logger.V(2) {
				a.logger.Infof("Removing last watch for resource name %q", resourceName)
			}
			for xcc := range state.xdsChannelConfigs {
				xcc.channel.unsubscribe(rType, resourceName)
			}
			delete(resources, resourceName)

			// If there are no more watchers for this resource type, delete the
			// resource type from the top-level map.
			if len(resources) == 0 {
				if a.logger.V(2) {
					a.logger.Infof("Removing last watch for resource type %q", rType.TypeName)
				}
				delete(a.resources, rType)
			}
			// If there are no more watchers for any resource type, release the
			// reference to the xdsChannels.
			if len(a.resources) == 0 {
				if a.logger.V(2) {
					a.logger.Infof("Removing last watch for for any resource type, releasing reference to the xdsChannel")
				}
				a.closeXDSChannels()
			}
		}, func() { close(done) })
		<-done
	})
}

// xdsChannelToUse returns the xdsChannel to use for communicating with the
// management server. If an active channel is available, it returns that.
// Otherwise, it creates a new channel using the first server configuration in
// the list of configurations, and returns that.
//
// A non-nil error is returned if the channel creation fails.
//
// Only executed in the context of a serializer callback.
func (a *authority) xdsChannelToUse() (*xdsChannelWithConfig, error) {
	if a.activeXDSChannel != nil {
		return a.activeXDSChannel, nil
	}

	sc := a.xdsChannelConfigs[0].serverConfig
	xc, cleanup, err := a.getChannelForADS(sc, a)
	if err != nil {
		return nil, err
	}
	a.xdsChannelConfigs[0].channel = xc
	a.xdsChannelConfigs[0].cleanup = cleanup
	a.activeXDSChannel = a.xdsChannelConfigs[0]
	return a.activeXDSChannel, nil
}

// closeXDSChannels closes all the xDS channels associated with this authority,
// when there are no more watchers for any resource type.
//
// Only executed in the context of a serializer callback.
func (a *authority) closeXDSChannels() {
	for _, xcc := range a.xdsChannelConfigs {
		if xcc.cleanup != nil {
			xcc.cleanup()
			xcc.cleanup = nil
		}
		xcc.channel = nil
	}
	a.activeXDSChannel = nil
}

// watcherExistsForUncachedResource returns true if there is at least one
// watcher for a resource that has not yet been cached.
//
// Only executed in the context of a serializer callback.
func (a *authority) watcherExistsForUncachedResource() bool {
	for _, resourceStates := range a.resources {
		for _, state := range resourceStates {
			if state.md.Status == xdsresource.ServiceStatusRequested {
				return true
			}
		}
	}
	return false
}

// dumpResources returns a dump of the resource configuration cached by this
// authority, for CSDS purposes.
func (a *authority) dumpResources() []*v3statuspb.ClientConfig_GenericXdsConfig {
	var ret []*v3statuspb.ClientConfig_GenericXdsConfig
	done := make(chan struct{})

	a.xdsClientSerializer.ScheduleOr(func(context.Context) {
		defer close(done)
		ret = a.resourceConfig()
	}, func() { close(done) })
	<-done
	return ret
}

// resourceConfig returns a slice of GenericXdsConfig objects representing the
// current state of all resources managed by this authority. This is used for
// reporting the current state of the xDS client.
//
// Only executed in the context of a serializer callback.
func (a *authority) resourceConfig() []*v3statuspb.ClientConfig_GenericXdsConfig {
	var ret []*v3statuspb.ClientConfig_GenericXdsConfig
	for rType, resourceStates := range a.resources {
		typeURL := rType.TypeURL
		for name, state := range resourceStates {
			var raw *anypb.Any
			if state.cache != nil {
				raw = &anypb.Any{TypeUrl: typeURL, Value: state.cache.Bytes()}
			}
			config := &v3statuspb.ClientConfig_GenericXdsConfig{
				TypeUrl:      typeURL,
				Name:         name,
				VersionInfo:  state.md.Version,
				XdsConfig:    raw,
				LastUpdated:  timestamppb.New(state.md.Timestamp),
				ClientStatus: serviceStatusToProto(state.md.Status),
			}
			if errState := state.md.ErrState; errState != nil {
				config.ErrorState = &v3adminpb.UpdateFailureState{
					LastUpdateAttempt: timestamppb.New(errState.Timestamp),
					Details:           errState.Err.Error(),
					VersionInfo:       errState.Version,
				}
			}
			ret = append(ret, config)
		}
	}
	return ret
}

func (a *authority) close() {
	a.xdsClientSerializerClose()
	<-a.xdsClientSerializer.Done()
	if a.logger.V(2) {
		a.logger.Infof("Closed")
	}
}

func serviceStatusToProto(serviceStatus xdsresource.ServiceStatus) v3adminpb.ClientResourceStatus {
	switch serviceStatus {
	case xdsresource.ServiceStatusUnknown:
		return v3adminpb.ClientResourceStatus_UNKNOWN
	case xdsresource.ServiceStatusRequested:
		return v3adminpb.ClientResourceStatus_REQUESTED
	case xdsresource.ServiceStatusNotExist:
		return v3adminpb.ClientResourceStatus_DOES_NOT_EXIST
	case xdsresource.ServiceStatusACKed:
		return v3adminpb.ClientResourceStatus_ACKED
	case xdsresource.ServiceStatusNACKed:
		return v3adminpb.ClientResourceStatus_NACKED
	default:
		return v3adminpb.ClientResourceStatus_UNKNOWN
	}
}

func (a *authority) resourceWatchStateForTesting(rType ResourceType, resourceName string) (state xdsresource.ResourceWatchState, err error) {
	done := make(chan struct{})
	a.xdsClientSerializer.ScheduleOr(func(context.Context) {
		state, err = a.activeXDSChannel.channel.ads.adsResourceWatchStateForTesting(rType, resourceName)
		close(done)
	}, func() {
		err = errors.New("failed to retrieve resource watch state because the xDS client is closed")
		close(done)
	})
	<-done

	return state, err
}
