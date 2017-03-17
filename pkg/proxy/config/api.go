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
func NewSourceAPI(c cache.Getter, period time.Duration, servicesChan chan<- ServiceUpdate) {
	servicesLW := cache.NewListWatchFromClient(c, "services", metav1.NamespaceAll, fields.Everything())
	newSourceAPI(servicesLW, period, servicesChan, wait.NeverStop)
}

func newSourceAPI(
	servicesLW cache.ListerWatcher,
	period time.Duration,
	servicesChan chan<- ServiceUpdate,
	stopCh <-chan struct{}) {
	serviceController := NewServiceController(servicesLW, period, servicesChan)
	go serviceController.Run(stopCh)

	if !cache.WaitForCacheSync(stopCh, serviceController.HasSynced) {
		utilruntime.HandleError(fmt.Errorf("source controllers not synced"))
		return
	}
	servicesChan <- ServiceUpdate{Op: SYNCED}
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
