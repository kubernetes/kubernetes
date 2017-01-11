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

package healthcheck

import (
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
)

// All public API Methods for this package

// UpdateEndpoints Update the set of local endpoints for a service
func UpdateEndpoints(serviceName types.NamespacedName, endpointUids sets.String) {
	req := &proxyMutationRequest{
		serviceName:  serviceName,
		endpointUids: &endpointUids,
	}
	healthchecker.mutationRequestChannel <- req
}

func updateServiceListener(serviceName types.NamespacedName, listenPort int, add bool) bool {
	responseChannel := make(chan bool)
	req := &proxyListenerRequest{
		serviceName:     serviceName,
		listenPort:      uint16(listenPort),
		add:             add,
		responseChannel: responseChannel,
	}
	healthchecker.listenerRequestChannel <- req
	return <-responseChannel
}

// AddServiceListener Request addition of a listener for a service's health check
func AddServiceListener(serviceName types.NamespacedName, listenPort int) bool {
	return updateServiceListener(serviceName, listenPort, true)
}

// DeleteServiceListener Request deletion of a listener for a service's health check
func DeleteServiceListener(serviceName types.NamespacedName, listenPort int) bool {
	return updateServiceListener(serviceName, listenPort, false)
}

// Run Start the healthchecker main loop
func Run() {
	healthchecker = proxyHealthCheckFactory()
	// Wrap with a wait.Forever to handle panics.
	go wait.Forever(func() {
		healthchecker.handlerLoop()
	}, 0)
}
