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

package service

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
)

// EndpointController manages selector-based service endpoints.
type EndpointController struct {
	client *client.Client
}

// NewEndpointController returns a new *EndpointController.
func NewEndpointController(client *client.Client) *EndpointController {
	return &EndpointController{
		client: client,
	}
}

// SyncServiceEndpoints syncs endpoints for services with selectors.
func (e *EndpointController) SyncServiceEndpoints() error {
	services, err := e.client.Services(api.NamespaceAll).List(labels.Everything())
	if err != nil {
		glog.Errorf("Failed to list services: %v", err)
		return err
	}
	var resultErr error
	for _, service := range services.Items {
		if service.Spec.Selector == nil {
			// services without a selector receive no endpoints from this controller;
			// these services will receive the endpoints that are created out-of-band via the REST API.
			continue
		}

		glog.V(5).Infof("About to update endpoints for service %s/%s", service.Namespace, service.Name)
		pods, err := e.client.Pods(service.Namespace).List(labels.Set(service.Spec.Selector).AsSelector())
		if err != nil {
			glog.Errorf("Error syncing service: %s/%s, skipping", service.Namespace, service.Name)
			resultErr = err
			continue
		}
		endpoints := []api.Endpoint{}

		for _, pod := range pods.Items {
			port, err := findPort(&pod, service.Spec.ContainerPort)
			if err != nil {
				glog.Errorf("Failed to find port for service %s/%s: %v", service.Namespace, service.Name, err)
				continue
			}
			if len(pod.Status.PodIP) == 0 {
				glog.Errorf("Failed to find an IP for pod %s/%s", pod.Namespace, pod.Name)
				continue
			}

			inService := false
			for _, c := range pod.Status.Conditions {
				if c.Type == api.PodReady && c.Status == api.ConditionFull {
					inService = true
					break
				}
			}
			if !inService {
				glog.V(5).Infof("Pod is out of service: %v/%v", pod.Namespace, pod.Name)
				continue
			}

			endpoints = append(endpoints, api.Endpoint{IP: pod.Status.PodIP, Port: port})
		}
		currentEndpoints, err := e.client.Endpoints(service.Namespace).Get(service.Name)
		if err != nil {
			if errors.IsNotFound(err) {
				currentEndpoints = &api.Endpoints{
					ObjectMeta: api.ObjectMeta{
						Name: service.Name,
					},
					Protocol: service.Spec.Protocol,
				}
			} else {
				glog.Errorf("Error getting endpoints: %v", err)
				continue
			}
		}
		newEndpoints := &api.Endpoints{}
		*newEndpoints = *currentEndpoints
		newEndpoints.Endpoints = endpoints

		if len(currentEndpoints.ResourceVersion) == 0 {
			// No previous endpoints, create them
			_, err = e.client.Endpoints(service.Namespace).Create(newEndpoints)
		} else {
			// Pre-existing
			if currentEndpoints.Protocol == service.Spec.Protocol && endpointsEqual(currentEndpoints, endpoints) {
				glog.V(5).Infof("protocol and endpoints are equal for %s/%s, skipping update", service.Namespace, service.Name)
				continue
			}
			_, err = e.client.Endpoints(service.Namespace).Update(newEndpoints)
		}
		if err != nil {
			glog.Errorf("Error updating endpoints: %v", err)
			continue
		}
	}
	return resultErr
}

func containsEndpoint(haystack *api.Endpoints, needle *api.Endpoint) bool {
	if haystack == nil || needle == nil {
		return false
	}
	for ix := range haystack.Endpoints {
		if haystack.Endpoints[ix] == *needle {
			return true
		}
	}
	return false
}

func endpointsEqual(eps *api.Endpoints, endpoints []api.Endpoint) bool {
	if len(eps.Endpoints) != len(endpoints) {
		return false
	}
	for i := range endpoints {
		if !containsEndpoint(eps, &endpoints[i]) {
			return false
		}
	}
	return true
}

// findPort locates the container port for the given manifest and portName.
func findPort(pod *api.Pod, portName util.IntOrString) (int, error) {
	firstContainerPort := 0
	if len(pod.Spec.Containers) > 0 && len(pod.Spec.Containers[0].Ports) > 0 {
		firstContainerPort = pod.Spec.Containers[0].Ports[0].ContainerPort
	}

	switch portName.Kind {
	case util.IntstrString:
		if len(portName.StrVal) == 0 {
			if firstContainerPort != 0 {
				return firstContainerPort, nil
			}
			break
		}
		name := portName.StrVal
		for _, container := range pod.Spec.Containers {
			for _, port := range container.Ports {
				if port.Name == name {
					return port.ContainerPort, nil
				}
			}
		}
	case util.IntstrInt:
		if portName.IntVal == 0 {
			if firstContainerPort != 0 {
				return firstContainerPort, nil
			}
			break
		}
		return portName.IntVal, nil
	}

	return 0, fmt.Errorf("no suitable port for manifest: %s", pod.UID)
}
