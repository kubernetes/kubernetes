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
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api"
)

// NewSourceAPI creates config source that watches for changes to the services and endpoints.
func NewSourceAPI(c cache.Getter, period time.Duration, servicesChan chan<- ServiceUpdate, endpointsChan chan<- EndpointsUpdate) {
	servicesLW := cache.NewListWatchFromClient(c, "services", metav1.NamespaceAll, fields.Everything())
	endpointsLW := cache.NewListWatchFromClient(c, "endpoints", metav1.NamespaceAll, fields.Everything())
	newSourceAPI(servicesLW, endpointsLW, period, servicesChan, endpointsChan, wait.NeverStop)
}

func newSourceAPI(
	servicesLW cache.ListerWatcher,
	endpointsLW cache.ListerWatcher,
	period time.Duration,
	servicesChan chan<- ServiceUpdate,
	endpointsChan chan<- EndpointsUpdate,
	stopCh <-chan struct{}) {
	serviceController := NewServiceController(servicesLW, period, servicesChan)
	go serviceController.Run(stopCh)

	endpointsController := NewEndpointsController(endpointsLW, period, endpointsChan)
	go endpointsController.Run(stopCh)

	if !cache.WaitForCacheSync(stopCh, serviceController.HasSynced, endpointsController.HasSynced) {
		utilruntime.HandleError(fmt.Errorf("source controllers not synced"))
		return
	}
	servicesChan <- ServiceUpdate{Op: SYNCED}
	endpointsChan <- EndpointsUpdate{Op: SYNCED}
}

func sendAddService(servicesChan chan<- ServiceUpdate) func(obj interface{}) {
	return func(obj interface{}) {
		service, ok := obj.(*api.Service)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("cannot convert to *api.Service: %v", obj))
			return
		}
		servicesChan <- ServiceUpdate{Op: ADD, Service: service}
	}
}

func sendUpdateService(servicesChan chan<- ServiceUpdate) func(oldObj, newObj interface{}) {
	return func(_, newObj interface{}) {
		service, ok := newObj.(*api.Service)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("cannot convert to *api.Service: %v", newObj))
			return
		}
		servicesChan <- ServiceUpdate{Op: UPDATE, Service: service}
	}
}

func sendDeleteService(servicesChan chan<- ServiceUpdate) func(obj interface{}) {
	return func(obj interface{}) {
		var service *api.Service
		switch t := obj.(type) {
		case *api.Service:
			service = t
		case cache.DeletedFinalStateUnknown:
			var ok bool
			service, ok = t.Obj.(*api.Service)
			if !ok {
				utilruntime.HandleError(fmt.Errorf("cannot convert to *api.Service: %v", t.Obj))
				return
			}
		default:
			utilruntime.HandleError(fmt.Errorf("cannot convert to *api.Service: %v", t))
			return
		}
		servicesChan <- ServiceUpdate{Op: REMOVE, Service: service}
	}
}

func sendAddEndpoints(endpointsChan chan<- EndpointsUpdate) func(obj interface{}) {
	return func(obj interface{}) {
		endpoints, ok := obj.(*api.Endpoints)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("cannot convert to *api.Endpoints: %v", obj))
			return
		}
		endpointsChan <- EndpointsUpdate{Op: ADD, Endpoints: endpoints}
	}
}

func sendUpdateEndpoints(endpointsChan chan<- EndpointsUpdate) func(oldObj, newObj interface{}) {
	return func(_, newObj interface{}) {
		endpoints, ok := newObj.(*api.Endpoints)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("cannot convert to *api.Endpoints: %v", newObj))
			return
		}
		endpointsChan <- EndpointsUpdate{Op: UPDATE, Endpoints: endpoints}
	}
}

func sendDeleteEndpoints(endpointsChan chan<- EndpointsUpdate) func(obj interface{}) {
	return func(obj interface{}) {
		var endpoints *api.Endpoints
		switch t := obj.(type) {
		case *api.Endpoints:
			endpoints = t
		case cache.DeletedFinalStateUnknown:
			var ok bool
			endpoints, ok = t.Obj.(*api.Endpoints)
			if !ok {
				utilruntime.HandleError(fmt.Errorf("cannot convert to *api.Endpoints: %v", t.Obj))
				return
			}
		default:
			utilruntime.HandleError(fmt.Errorf("cannot convert to *api.Endpoints: %v", obj))
			return
		}
		endpointsChan <- EndpointsUpdate{Op: REMOVE, Endpoints: endpoints}
	}
}

// NewServiceController creates a controller that is watching services and sending
// updates into ServiceUpdate channel.
func NewServiceController(lw cache.ListerWatcher, period time.Duration, ch chan<- ServiceUpdate) cache.Controller {
	_, serviceController := cache.NewInformer(
		lw,
		&api.Service{},
		period,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    sendAddService(ch),
			UpdateFunc: sendUpdateService(ch),
			DeleteFunc: sendDeleteService(ch),
		},
	)
	return serviceController
}

// NewEndpointsController creates a controller that is watching endpoints and sending
// updates into EndpointsUpdate channel.
func NewEndpointsController(lw cache.ListerWatcher, period time.Duration, ch chan<- EndpointsUpdate) cache.Controller {
	_, endpointsController := cache.NewInformer(
		lw,
		&api.Endpoints{},
		period,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    sendAddEndpoints(ch),
			UpdateFunc: sendUpdateEndpoints(ch),
			DeleteFunc: sendDeleteEndpoints(ch),
		},
	)
	return endpointsController
}
