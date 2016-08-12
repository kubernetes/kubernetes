/*
Copyright 2016 The Kubernetes Authors.

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

// Package healthcheck LoadBalancer Healthcheck responder library for kubernetes network proxies
package healthcheck // import "k8s.io/kubernetes/pkg/proxy/healthcheck"

import (
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
)

var once sync.Once
var healthchecker *proxyHC

func run() {
	allocatedCh := make(chan bool)
	go func() {
		healthchecker = proxyHealthCheckFactory()
		allocatedCh <- true
		wait.Forever(func() {
			healthchecker.handlerLoop()
		}, 0)
	}()
	// Return only after factory function allocation
	<-allocatedCh
}

// handlerLoop Serializes all requests to prevent concurrent access to the maps
func (h *proxyHC) handlerLoop() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	for {
		select {
		case req := <-h.mutationRequestChannel:
			h.handleMutationRequest(req)
		case req := <-h.listenerRequestChannel:
			req.responseChannel <- h.handleServiceListenerRequest(req)
		case <-ticker.C:
			h.sync()
		}
	}
}

func (h *proxyHC) sync() {
	glog.V(2).Infof("%d Health Check Listeners", len(h.serviceResponderMap))
	glog.V(2).Infof("%d Services registered for health checking", len(h.serviceEndpointsMap.List()))
	for _, svc := range h.serviceEndpointsMap.ListKeys() {
		if e, ok := h.serviceEndpointsMap.Get(svc); ok {
			endpointList := e.(*serviceEndpointsList)
			glog.V(2).Infof("Service %s has %d local endpoints", svc, endpointList.endpoints.Len())
		}
	}
}
