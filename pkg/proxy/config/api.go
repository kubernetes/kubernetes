/*
Copyright 2014 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package config

import (
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// TODO: to use Reflector, need to change the ServicesWatcher to a generic ListerWatcher.
// ServicesWatcher is capable of listing and watching for changes to services across ALL namespaces
type ServicesWatcher interface {
	List(label labels.Selector) (*api.ServiceList, error)
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// EndpointsWatcher is capable of listing and watching for changes to endpoints across ALL namespaces
type EndpointsWatcher interface {
	List(label labels.Selector) (*api.EndpointsList, error)
	Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error)
}

// SourceAPI implements a configuration source for services and endpoints that
// uses the client watch API to efficiently detect changes.
type SourceAPI struct {
	s servicesReflector
	e endpointsReflector
}

type servicesReflector struct {
	watcher           ServicesWatcher
	services          chan<- ServiceUpdate
	resourceVersion   string
	waitDuration      time.Duration
	reconnectDuration time.Duration
}

type endpointsReflector struct {
	watcher           EndpointsWatcher
	endpoints         chan<- EndpointsUpdate
	resourceVersion   string
	waitDuration      time.Duration
	reconnectDuration time.Duration
}

// NewSourceAPI creates a config source that watches for changes to the services and endpoints.
func NewSourceAPI(servicesWatcher ServicesWatcher, endpointsWatcher EndpointsWatcher, period time.Duration, services chan<- ServiceUpdate, endpoints chan<- EndpointsUpdate) *SourceAPI {
	config := &SourceAPI{
		s: servicesReflector{
			watcher:         servicesWatcher,
			services:        services,
			resourceVersion: "",
			waitDuration:    period,
			// prevent hot loops if the server starts to misbehave
			reconnectDuration: time.Second * 1,
		},
		e: endpointsReflector{
			watcher:         endpointsWatcher,
			endpoints:       endpoints,
			resourceVersion: "",
			waitDuration:    period,
			// prevent hot loops if the server starts to misbehave
			reconnectDuration: time.Second * 1,
		},
	}
	go util.Forever(func() { config.s.listAndWatch() }, period)
	go util.Forever(func() { config.e.listAndWatch() }, period)
	return config
}

func (r *servicesReflector) listAndWatch() {
	r.run(&r.resourceVersion)
	time.Sleep(wait.Jitter(r.reconnectDuration, 0.0))
}

func (r *endpointsReflector) listAndWatch() {
	r.run(&r.resourceVersion)
	time.Sleep(wait.Jitter(r.reconnectDuration, 0.0))
}

// run loops forever looking for changes to services.
func (s *servicesReflector) run(resourceVersion *string) {
	if len(*resourceVersion) == 0 {
		services, err := s.watcher.List(labels.Everything())
		if err != nil {
			glog.Errorf("Unable to load services: %v", err)
			// TODO: reconcile with pkg/client/cache which doesn't use reflector.
			time.Sleep(wait.Jitter(s.waitDuration, 0.0))
			return
		}
		*resourceVersion = services.ResourceVersion
		// TODO: replace with code to update the
		s.services <- ServiceUpdate{Op: SET, Services: services.Items}
	}

	watcher, err := s.watcher.Watch(labels.Everything(), fields.Everything(), *resourceVersion)
	if err != nil {
		glog.Errorf("Unable to watch for services changes: %v", err)
		if !client.IsTimeout(err) {
			// Reset so that we do a fresh get request
			*resourceVersion = ""
		}
		time.Sleep(wait.Jitter(s.waitDuration, 0.0))
		return
	}
	defer watcher.Stop()

	ch := watcher.ResultChan()
	s.watchHandler(resourceVersion, ch, s.services)
}

// watchHandler loops over an event channel and delivers config changes to an update channel.
func (s *servicesReflector) watchHandler(resourceVersion *string, ch <-chan watch.Event, updates chan<- ServiceUpdate) {
	for {
		select {
		case event, ok := <-ch:
			if !ok {
				glog.V(4).Infof("WatchServices channel closed")
				return
			}

			if event.Object == nil {
				glog.Errorf("Got nil over WatchServices channel")
				return
			}
			var service *api.Service
			switch obj := event.Object.(type) {
			case *api.Service:
				service = obj
			case *api.Status:
				glog.Warningf("Got error status on WatchServices channel: %+v", obj)
				*resourceVersion = ""
				return
			default:
				glog.Errorf("Got unexpected object over WatchServices channel: %+v", obj)
				*resourceVersion = ""
				return
			}

			*resourceVersion = service.ResourceVersion

			switch event.Type {
			case watch.Added, watch.Modified:
				updates <- ServiceUpdate{Op: ADD, Services: []api.Service{*service}}

			case watch.Deleted:
				updates <- ServiceUpdate{Op: REMOVE, Services: []api.Service{*service}}
			}
		}
	}
}

// run loops forever looking for changes to endpoints.
func (s *endpointsReflector) run(resourceVersion *string) {
	if len(*resourceVersion) == 0 {
		endpoints, err := s.watcher.List(labels.Everything())
		if err != nil {
			glog.Errorf("Unable to load endpoints: %v", err)
			time.Sleep(wait.Jitter(s.waitDuration, 0.0))
			return
		}
		*resourceVersion = endpoints.ResourceVersion
		s.endpoints <- EndpointsUpdate{Op: SET, Endpoints: endpoints.Items}
	}

	watcher, err := s.watcher.Watch(labels.Everything(), fields.Everything(), *resourceVersion)
	if err != nil {
		glog.Errorf("Unable to watch for endpoints changes: %v", err)
		if !client.IsTimeout(err) {
			// Reset so that we do a fresh get request
			*resourceVersion = ""
		}

		time.Sleep(wait.Jitter(s.waitDuration, 0.0))
		return
	}
	defer watcher.Stop()

	ch := watcher.ResultChan()
	s.watchHandler(resourceVersion, ch, s.endpoints)
}

// watchHandler loops over an event channel and delivers config changes to an update channel.
func (s *endpointsReflector) watchHandler(resourceVersion *string, ch <-chan watch.Event, updates chan<- EndpointsUpdate) {
	for {
		select {
		case event, ok := <-ch:
			if !ok {
				glog.V(4).Infof("WatchEndpoints channel closed")
				return
			}

			if event.Object == nil {
				glog.Errorf("Got nil over WatchEndpoints channel")
				return
			}
			var endpoints *api.Endpoints
			switch obj := event.Object.(type) {
			case *api.Endpoints:
				endpoints = obj
			case *api.Status:
				glog.Warningf("Got error status on WatchEndpoints channel: %+v", obj)
				*resourceVersion = ""
				return
			default:
				glog.Errorf("Got unexpected object over WatchEndpoints channel: %+v", obj)
				*resourceVersion = ""
				return
			}
			*resourceVersion = endpoints.ResourceVersion

			switch event.Type {
			case watch.Added, watch.Modified:
				updates <- EndpointsUpdate{Op: ADD, Endpoints: []api.Endpoints{*endpoints}}

			case watch.Deleted:
				updates <- EndpointsUpdate{Op: REMOVE, Endpoints: []api.Endpoints{*endpoints}}
			}
		}
	}
}
