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

package server

import (
	"sync"

	igrpclog "google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/xds/xdsclient/xdsresource"
)

// rdsHandler handles any RDS queries that need to be started for a given server
// side listeners Filter Chains (i.e. not inline). It persists rdsWatcher
// updates for later use and also determines whether all the rdsWatcher updates
// needed have been received or not.
type rdsHandler struct {
	xdsC      XDSClient
	xdsNodeID string
	logger    *igrpclog.PrefixLogger

	callback func(string, rdsWatcherUpdate)

	// updates is a map from routeName to rdsWatcher update, including
	// RouteConfiguration resources and any errors received. If not written in
	// this map, no RouteConfiguration or error for that route name yet. If
	// update set in value, use that as valid route configuration, otherwise
	// treat as an error case and fail at L7 level.
	updates map[string]rdsWatcherUpdate

	mu      sync.Mutex
	cancels map[string]func()
}

// newRDSHandler creates a new rdsHandler to watch for RouteConfiguration
// resources. listenerWrapper updates the list of route names to watch by
// calling updateRouteNamesToWatch() upon receipt of new Listener configuration.
func newRDSHandler(cb func(string, rdsWatcherUpdate), xdsC XDSClient, logger *igrpclog.PrefixLogger) *rdsHandler {
	r := &rdsHandler{
		xdsC:     xdsC,
		logger:   logger,
		callback: cb,
		updates:  make(map[string]rdsWatcherUpdate),
		cancels:  make(map[string]func()),
	}
	r.xdsNodeID = xdsC.BootstrapConfig().Node().GetId()
	return r
}

// updateRouteNamesToWatch handles a list of route names to watch for a given
// server side listener (if a filter chain specifies dynamic
// RouteConfiguration). This function handles all the logic with respect to any
// routes that may have been added or deleted as compared to what was previously
// present. Must be called within an xDS Client callback.
func (rh *rdsHandler) updateRouteNamesToWatch(routeNamesToWatch map[string]bool) {
	rh.mu.Lock()
	defer rh.mu.Unlock()
	// Add and start watches for any new routes in routeNamesToWatch.
	for routeName := range routeNamesToWatch {
		if _, ok := rh.cancels[routeName]; !ok {
			// The xDS client keeps a reference to the watcher until the cancel
			// func is invoked. So, we don't need to keep a reference for fear
			// of it being garbage collected.
			w := &rdsWatcher{parent: rh, routeName: routeName}
			cancel := xdsresource.WatchRouteConfig(rh.xdsC, routeName, w)
			// Set bit on cancel function to eat any RouteConfiguration calls
			// for this watcher after it has been canceled.
			rh.cancels[routeName] = func() {
				w.mu.Lock()
				w.canceled = true
				w.mu.Unlock()
				cancel()
			}
		}
	}

	// Delete and cancel watches for any routes from persisted routeNamesToWatch
	// that are no longer present.
	for routeName := range rh.cancels {
		if _, ok := routeNamesToWatch[routeName]; !ok {
			rh.cancels[routeName]()
			delete(rh.cancels, routeName)
			delete(rh.updates, routeName)
		}
	}
}

// determines if all dynamic RouteConfiguration needed has received
// configuration or update. Must be called from an xDS Client Callback.
func (rh *rdsHandler) determineRouteConfigurationReady() bool {
	// Safe to read cancels because only written to in other parts of xDS Client
	// Callbacks, which are sync.
	return len(rh.updates) == len(rh.cancels)
}

// close() is meant to be called by wrapped listener when the wrapped listener
// is closed, and it cleans up resources by canceling all the active RDS
// watches.
func (rh *rdsHandler) close() {
	rh.mu.Lock()
	defer rh.mu.Unlock()
	for _, cancel := range rh.cancels {
		cancel()
	}
}

type rdsWatcherUpdate struct {
	data *xdsresource.RouteConfigUpdate
	err  error
}

// rdsWatcher implements the xdsresource.RouteConfigWatcher interface and is
// passed to the WatchRouteConfig API.
type rdsWatcher struct {
	parent    *rdsHandler
	logger    *igrpclog.PrefixLogger
	routeName string

	mu       sync.Mutex
	canceled bool // eats callbacks if true
}

func (rw *rdsWatcher) ResourceChanged(update *xdsresource.RouteConfigResourceData, onDone func()) {
	defer onDone()
	rw.mu.Lock()
	if rw.canceled {
		rw.mu.Unlock()
		return
	}
	rw.mu.Unlock()
	if rw.logger.V(2) {
		rw.logger.Infof("RDS watch for resource %q received update: %#v", rw.routeName, update.Resource)
	}

	routeName := rw.routeName
	rwu := rdsWatcherUpdate{data: &update.Resource}
	rw.parent.updates[routeName] = rwu
	rw.parent.callback(routeName, rwu)
}

func (rw *rdsWatcher) ResourceError(err error, onDone func()) {
	defer onDone()
	rw.mu.Lock()
	if rw.canceled {
		rw.mu.Unlock()
		return
	}
	rw.mu.Unlock()
	if rw.logger.V(2) {
		rw.logger.Infof("RDS watch for resource %q reported resource error", rw.routeName)
	}

	routeName := rw.routeName
	rwu := rdsWatcherUpdate{err: err}
	rw.parent.updates[routeName] = rwu
	rw.parent.callback(routeName, rwu)
}

func (rw *rdsWatcher) AmbientError(err error, onDone func()) {
	defer onDone()
	rw.mu.Lock()
	if rw.canceled {
		rw.mu.Unlock()
		return
	}
	rw.mu.Unlock()
	if rw.logger.V(2) {
		rw.logger.Infof("RDS watch for resource %q reported ambient error: %v", rw.routeName, err)
	}
	routeName := rw.routeName
	rwu := rw.parent.updates[routeName]
	rw.parent.callback(routeName, rwu)
}
