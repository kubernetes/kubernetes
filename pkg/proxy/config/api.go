/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/fields"
)

// NewSourceAPI creates config source that watches for changes to the services and endpoints.
func NewSourceAPI(c cache.Getter, period time.Duration, servicesChan chan<- ServiceUpdate, endpointsChan chan<- EndpointsUpdate) {
	servicesLW := cache.NewListWatchFromClient(c, "services", api.NamespaceAll, fields.Everything())
	cache.NewReflector(servicesLW, &api.Service{}, NewServiceStore(nil, servicesChan), period).Run()

	endpointsLW := cache.NewListWatchFromClient(c, "endpoints", api.NamespaceAll, fields.Everything())
	cache.NewReflector(endpointsLW, &api.Endpoints{}, NewEndpointsStore(nil, endpointsChan), period).Run()
}

// NewServiceStore creates an undelta store that expands updates to the store into
// ServiceUpdate events on the channel. If no store is passed, a default store will
// be initialized. Allows reuse of a cache store across multiple components.
func NewServiceStore(store cache.Store, ch chan<- ServiceUpdate) cache.Store {
	fn := func(objs []interface{}) {
		var services []api.Service
		for _, o := range objs {
			services = append(services, *(o.(*api.Service)))
		}
		ch <- ServiceUpdate{Op: SET, Services: services}
	}
	if store == nil {
		store = cache.NewStore(cache.MetaNamespaceKeyFunc)
	}
	return &cache.UndeltaStore{
		Store:    store,
		PushFunc: fn,
	}
}

// NewEndpointsStore creates an undelta store that expands updates to the store into
// EndpointsUpdate events on the channel. If no store is passed, a default store will
// be initialized. Allows reuse of a cache store across multiple components.
func NewEndpointsStore(store cache.Store, ch chan<- EndpointsUpdate) cache.Store {
	fn := func(objs []interface{}) {
		var endpoints []api.Endpoints
		for _, o := range objs {
			endpoints = append(endpoints, *(o.(*api.Endpoints)))
		}
		ch <- EndpointsUpdate{Op: SET, Endpoints: endpoints}
	}
	if store == nil {
		store = cache.NewStore(cache.MetaNamespaceKeyFunc)
	}
	return &cache.UndeltaStore{
		Store:    store,
		PushFunc: fn,
	}
}
