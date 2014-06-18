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

package registry

import (
	"fmt"
	"log"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

func MakeEndpointController(serviceRegistry ServiceRegistry, podRegistry PodRegistry) *EndpointController {
	return &EndpointController{
		serviceRegistry: serviceRegistry,
		podRegistry:     podRegistry,
	}
}

type EndpointController struct {
	serviceRegistry ServiceRegistry
	podRegistry     PodRegistry
}

func (e *EndpointController) SyncServiceEndpoints() error {
	services, err := e.serviceRegistry.ListServices()
	if err != nil {
		return err
	}
	var resultErr error
	for _, service := range services.Items {
		pods, err := e.podRegistry.ListPods(labels.Set(service.Selector).AsSelector())
		if err != nil {
			log.Printf("Error syncing service: %#v, skipping.", service)
			resultErr = err
			continue
		}
		endpoints := make([]string, len(pods))
		for ix, pod := range pods {
			// TODO: Use port names in the service object, don't just use port #0
			endpoints[ix] = fmt.Sprintf("%s:%d", pod.CurrentState.Host, pod.DesiredState.Manifest.Containers[0].Ports[0].HostPort)
		}
		err = e.serviceRegistry.UpdateEndpoints(api.Endpoints{
			Name:      service.ID,
			Endpoints: endpoints,
		})
		if err != nil {
			log.Printf("Error updating endpoints: %#v", err)
			continue
		}
	}
	return resultErr
}
