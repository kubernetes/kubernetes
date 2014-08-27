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

// Watcher is the interface needed to receive changes to services and endpoints
type Watcher interface {
	WatchServices(label, field labels.Selector, resourceVersion uint64) (watch.Interface, error)
	WatchEndpoints(label, field labels.Selector, resourceVersion uint64) (watch.Interface, error)
}

// SourceAPI implements a configuration source for services and endpoints that
// uses the client watch API to efficiently detect changes.
type SourceAPI struct {
	client    Watcher
	services  chan<- ServiceUpdate
	endpoints chan<- EndpointsUpdate

	waitDuration      time.Duration
	reconnectDuration time.Duration
}

// NewSourceAPI creates a config source that watches for changes to the services and endpoints
func NewSourceAPI(client Watcher, period time.Duration, services chan<- ServiceUpdate, endpoints chan<- EndpointsUpdate) *SourceAPI {
	config := &SourceAPI{
		client:    client,
		services:  services,
		endpoints: endpoints,

		waitDuration: period,
		// prevent hot loops if the server starts to misbehave
		reconnectDuration: time.Second * 1,
	}
	serviceVersion := uint64(0)
	go util.Forever(func() {
		config.runServices(&serviceVersion)
		time.Sleep(wait.Jitter(config.reconnectDuration, 0.0))
	}, period)
	endpointVersion := uint64(0)
	go util.Forever(func() {
		config.runEndpoints(&endpointVersion)
		time.Sleep(wait.Jitter(config.reconnectDuration, 0.0))
	}, period)
	return config
}

// runServices loops forever looking for changes to services
func (s *SourceAPI) runServices(resourceVersion *uint64) {
	watcher, err := s.client.WatchServices(labels.Everything(), labels.Everything(), *resourceVersion)
	if err != nil {
		glog.Errorf("Unable to watch for services changes: %v", err)
		time.Sleep(wait.Jitter(s.waitDuration, 0.0))
		return
	}
	defer watcher.Stop()

	ch := watcher.ResultChan()
	handleServicesWatch(resourceVersion, ch, s.services)
}

// handleServicesWatch loops over an event channel and delivers config changes to an update channel
func handleServicesWatch(resourceVersion *uint64, ch <-chan watch.Event, updates chan<- ServiceUpdate) {
	for {
		select {
		case event, ok := <-ch:
			if !ok {
				glog.V(2).Infof("WatchServices channel closed")
				return
			}

			service := event.Object.(*api.Service)
			*resourceVersion = service.ResourceVersion + 1

			switch event.Type {
			case watch.Added, watch.Modified:
				updates <- ServiceUpdate{Op: SET, Services: []api.Service{*service}}

			case watch.Deleted:
				updates <- ServiceUpdate{Op: SET}
			}
		}
	}
}

// runEndpoints loops forever looking for changes to endpoints
func (s *SourceAPI) runEndpoints(resourceVersion *uint64) {
	watcher, err := s.client.WatchEndpoints(labels.Everything(), labels.Everything(), *resourceVersion)
	if err != nil {
		glog.Errorf("Unable to watch for endpoints changes: %v", err)
		time.Sleep(wait.Jitter(s.waitDuration, 0.0))
		return
	}
	defer watcher.Stop()

	ch := watcher.ResultChan()
	handleEndpointsWatch(resourceVersion, ch, s.endpoints)
}

// handleEndpointsWatch loops over an event channel and delivers config changes to an update channel
func handleEndpointsWatch(resourceVersion *uint64, ch <-chan watch.Event, updates chan<- EndpointsUpdate) {
	for {
		select {
		case event, ok := <-ch:
			if !ok {
				glog.V(2).Infof("WatchEndpoints channel closed")
				return
			}

			endpoints := event.Object.(*api.Endpoints)
			*resourceVersion = endpoints.ResourceVersion + 1

			switch event.Type {
			case watch.Added, watch.Modified:
				updates <- EndpointsUpdate{Op: SET, Endpoints: []api.Endpoints{*endpoints}}

			case watch.Deleted:
				updates <- EndpointsUpdate{Op: SET}
			}
		}
	}
}
