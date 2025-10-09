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

package resolver

import (
	"context"

	"google.golang.org/grpc/internal/xds/xdsclient/xdsresource"
)

type listenerWatcher struct {
	resourceName string
	cancel       func()
	parent       *xdsResolver
}

func newListenerWatcher(resourceName string, parent *xdsResolver) *listenerWatcher {
	lw := &listenerWatcher{resourceName: resourceName, parent: parent}
	lw.cancel = xdsresource.WatchListener(parent.xdsClient, resourceName, lw)
	return lw
}

func (l *listenerWatcher) ResourceChanged(update *xdsresource.ListenerResourceData, onDone func()) {
	handleUpdate := func(context.Context) { l.parent.onListenerResourceUpdate(update.Resource); onDone() }
	l.parent.serializer.ScheduleOr(handleUpdate, onDone)
}

func (l *listenerWatcher) ResourceError(err error, onDone func()) {
	handleError := func(context.Context) { l.parent.onListenerResourceError(err); onDone() }
	l.parent.serializer.ScheduleOr(handleError, onDone)
}

func (l *listenerWatcher) AmbientError(err error, onDone func()) {
	handleError := func(context.Context) { l.parent.onListenerResourceAmbientError(err); onDone() }
	l.parent.serializer.ScheduleOr(handleError, onDone)
}

func (l *listenerWatcher) stop() {
	l.cancel()
	l.parent.logger.Infof("Canceling watch on Listener resource %q", l.resourceName)
}

type routeConfigWatcher struct {
	resourceName string
	cancel       func()
	parent       *xdsResolver
}

func newRouteConfigWatcher(resourceName string, parent *xdsResolver) *routeConfigWatcher {
	rw := &routeConfigWatcher{resourceName: resourceName, parent: parent}
	rw.cancel = xdsresource.WatchRouteConfig(parent.xdsClient, resourceName, rw)
	return rw
}

func (r *routeConfigWatcher) ResourceChanged(u *xdsresource.RouteConfigResourceData, onDone func()) {
	handleUpdate := func(context.Context) {
		r.parent.onRouteConfigResourceUpdate(r.resourceName, u.Resource)
		onDone()
	}
	r.parent.serializer.ScheduleOr(handleUpdate, onDone)
}

func (r *routeConfigWatcher) ResourceError(err error, onDone func()) {
	handleError := func(context.Context) { r.parent.onRouteConfigResourceError(r.resourceName, err); onDone() }
	r.parent.serializer.ScheduleOr(handleError, onDone)
}

func (r *routeConfigWatcher) AmbientError(err error, onDone func()) {
	handleError := func(context.Context) { r.parent.onRouteConfigResourceAmbientError(r.resourceName, err); onDone() }
	r.parent.serializer.ScheduleOr(handleError, onDone)
}

func (r *routeConfigWatcher) stop() {
	r.cancel()
	r.parent.logger.Infof("Canceling watch on RouteConfiguration resource %q", r.resourceName)
}
