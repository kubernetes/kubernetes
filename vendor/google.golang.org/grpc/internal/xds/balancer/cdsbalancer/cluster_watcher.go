/*
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
 */

package cdsbalancer

import (
	"context"

	"google.golang.org/grpc/internal/xds/xdsclient/xdsresource"
)

// clusterWatcher implements the xdsresource.ClusterWatcher interface, and is
// passed to the xDS client as part of the WatchResource() API.
//
// It watches a single cluster and handles callbacks from the xDS client by
// scheduling them on the parent LB policy's serializer.
type clusterWatcher struct {
	name   string
	parent *cdsBalancer
}

func (cw *clusterWatcher) ResourceChanged(u *xdsresource.ClusterResourceData, onDone func()) {
	handleUpdate := func(context.Context) { cw.parent.onClusterUpdate(cw.name, u.Resource); onDone() }
	cw.parent.serializer.ScheduleOr(handleUpdate, onDone)
}

func (cw *clusterWatcher) ResourceError(err error, onDone func()) {
	handleResourceError := func(context.Context) { cw.parent.onClusterResourceError(cw.name, err); onDone() }
	cw.parent.serializer.ScheduleOr(handleResourceError, onDone)
}

func (cw *clusterWatcher) AmbientError(err error, onDone func()) {
	handleError := func(context.Context) { cw.parent.onClusterAmbientError(cw.name, err); onDone() }
	cw.parent.serializer.ScheduleOr(handleError, onDone)
}

// watcherState groups the state associated with a clusterWatcher.
type watcherState struct {
	watcher     *clusterWatcher            // The underlying watcher.
	cancelWatch func()                     // Cancel func to cancel the watch.
	lastUpdate  *xdsresource.ClusterUpdate // Most recent update received for this cluster.
}
