/*
Copyright 2014 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// ServicesWatcher is capable of listing and watching for changes to services across ALL namespaces
type ServicesWatcher interface {
	List(label labels.Selector) (*api.ServiceList, error)
	Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error)
}

// EndpointsWatcher is capable of listing and watching for changes to endpoints across ALL namespaces
type EndpointsWatcher interface {
	List(label labels.Selector) (*api.EndpointsList, error)
	Watch(label, field labels.Selector, resourceVersion string) (watch.Interface, error)
}

// SourceAPI implements a configuration source for services and endpoints that
// uses the client watch API to efficiently detect changes.
type SourceAPI struct {
	servicesWatcher  ServicesWatcher
	endpointsWatcher EndpointsWatcher

	services  chan<- ServiceUpdate
	endpoints chan<- EndpointsUpdate

	waitDuration      time.Duration
	reconnectDuration time.Duration
}

// NewSourceAPI creates a config source that watches for changes to the services and endpoints.
func NewSourceAPI(servicesWatcher ServicesWatcher, endpointsWatcher EndpointsWatcher, period time.Duration, services chan<- ServiceUpdate, endpoints chan<- EndpointsUpdate) *SourceAPI {
	config := &SourceAPI{
		servicesWatcher:  servicesWatcher,
		endpointsWatcher: endpointsWatcher,
		services:         services,
		endpoints:        endpoints,

		waitDuration: period,
		// prevent hot loops if the server starts to misbehave
		reconnectDuration: time.Second * 1,
	}
	serviceVersion := ""
	go util.Forever(func() {
		config.runServices(&serviceVersion)
		time.Sleep(wait.Jitter(config.reconnectDuration, 0.0))
	}, period)
	endpointVersion := ""
	go util.Forever(func() {
		config.runEndpoints(&endpointVersion)
		time.Sleep(wait.Jitter(config.reconnectDuration, 0.0))
	}, period)
	return config
}

// runServices loops forever looking for changes to services.
func (s *SourceAPI) runServices(resourceVersion *string) {
	if len(*resourceVersion) == 0 {
		services, err := s.servicesWatcher.List(labels.Everything())
		if err != nil {
			glog.Errorf("Unable to load services: %v", err)
			time.Sleep(wait.Jitter(s.waitDuration, 0.0))
			return
		}
		*resourceVersion = services.ResourceVersion
		s.services <- ServiceUpdate{Op: SET, Services: services.Items}
	}

	watcher, err := s.servicesWatcher.Watch(labels.Everything(), labels.Everything(), *resourceVersion)
	if err != nil {
		glog.Errorf("Unable to watch for services changes: %v", err)
		time.Sleep(wait.Jitter(s.waitDuration, 0.0))
		return
	}
	defer watcher.Stop()

	ch := watcher.ResultChan()
	handleServicesWatch(resourceVersion, ch, s.services)
}

// handleServicesWatch loops over an event channel and delivers config changes to an update channel.
func handleServicesWatch(resourceVersion *string, ch <-chan watch.Event, updates chan<- ServiceUpdate) {
	for {
		select {
		case event, ok := <-ch:
			if !ok {
				glog.V(4).Infof("WatchServices channel closed")
				return
			}

			service := event.Object.(*api.Service)
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

// runEndpoints loops forever looking for changes to endpoints.
func (s *SourceAPI) runEndpoints(resourceVersion *string) {
	if len(*resourceVersion) == 0 {
		endpoints, err := s.endpointsWatcher.List(labels.Everything())
		if err != nil {
			glog.Errorf("Unable to load endpoints: %v", err)
			time.Sleep(wait.Jitter(s.waitDuration, 0.0))
			return
		}
		*resourceVersion = endpoints.ResourceVersion
		s.endpoints <- EndpointsUpdate{Op: SET, Endpoints: endpoints.Items}
	}

	watcher, err := s.endpointsWatcher.Watch(labels.Everything(), labels.Everything(), *resourceVersion)
	if err != nil {
		glog.Errorf("Unable to watch for endpoints changes: %v", err)
		time.Sleep(wait.Jitter(s.waitDuration, 0.0))
		return
	}
	defer watcher.Stop()

	ch := watcher.ResultChan()
	handleEndpointsWatch(resourceVersion, ch, s.endpoints)
}

// handleEndpointsWatch loops over an event channel and delivers config changes to an update channel.
func handleEndpointsWatch(resourceVersion *string, ch <-chan watch.Event, updates chan<- EndpointsUpdate) {
	for {
		select {
		case event, ok := <-ch:
			if !ok {
				glog.V(4).Infof("WatchEndpoints channel closed")
				return
			}

			endpoints := event.Object.(*api.Endpoints)
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
