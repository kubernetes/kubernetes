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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/fields"
)

// NewSourceAPIserver creates config source that watches for changes to the services and endpoints.
func NewSourceAPI(c *client.Client, period time.Duration, servicesChan chan<- ServiceUpdate, endpointsChan chan<- EndpointsUpdate) {
	servicesLW := cache.NewListWatchFromClient(c, "services", api.NamespaceAll, fields.Everything())
	endpointsLW := cache.NewListWatchFromClient(c, "endpoints", api.NamespaceAll, fields.Everything())

	newServicesSourceApiFromLW(servicesLW, period, servicesChan)
	newEndpointsSourceApiFromLW(endpointsLW, period, endpointsChan)
}

func newServicesSourceApiFromLW(servicesLW cache.ListerWatcher, period time.Duration, servicesChan chan<- ServiceUpdate) {
	servicesPush := func(objs []interface{}) {
		var services []api.Service
		for _, o := range objs {
			services = append(services, *(o.(*api.Service)))
		}
		servicesChan <- ServiceUpdate{Op: SET, Services: services}
	}

	serviceQueue := cache.NewUndeltaStore(servicesPush, cache.MetaNamespaceKeyFunc)
	cache.NewReflector(servicesLW, &api.Service{}, serviceQueue, period).Run()
}

func newEndpointsSourceApiFromLW(endpointsLW cache.ListerWatcher, period time.Duration, endpointsChan chan<- EndpointsUpdate) {
	endpointsPush := func(objs []interface{}) {
		var endpoints []api.Endpoints
		for _, o := range objs {
			endpoints = append(endpoints, *(o.(*api.Endpoints)))
		}
		endpointsChan <- EndpointsUpdate{Op: SET, Endpoints: endpoints}
	}

	endpointQueue := cache.NewUndeltaStore(endpointsPush, cache.MetaNamespaceKeyFunc)
	cache.NewReflector(endpointsLW, &api.Endpoints{}, endpointQueue, period).Run()
}
