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
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"reflect"
	"sort"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
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
	for i := range services.Items {
		service := &services.Items[i]

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

		subsets := []api.EndpointSubset{}
		for i := range pods.Items {
			pod := &pods.Items[i]

			// TODO: Once v1beta1 and v1beta2 are EOL'ed, this can
			// assume that service.Spec.TargetPort is populated.
			_ = v1beta1.Dependency
			_ = v1beta2.Dependency
			// TODO: Add multiple-ports to Service and expose them here.
			portName := ""
			portProto := service.Spec.Protocol
			portNum, err := findPort(pod, service)
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

			epp := api.EndpointPort{Name: portName, Port: portNum, Protocol: portProto}
			epa := api.EndpointAddress{IP: pod.Status.PodIP, TargetRef: &api.ObjectReference{
				Kind:            "Pod",
				Namespace:       pod.ObjectMeta.Namespace,
				Name:            pod.ObjectMeta.Name,
				UID:             pod.ObjectMeta.UID,
				ResourceVersion: pod.ObjectMeta.ResourceVersion,
			}}
			subsets = append(subsets, api.EndpointSubset{Addresses: []api.EndpointAddress{epa}, Ports: []api.EndpointPort{epp}})
		}
		subsets = packSubsets(subsets)

		// See if there's actually an update here.
		currentEndpoints, err := e.client.Endpoints(service.Namespace).Get(service.Name)
		if err != nil {
			if errors.IsNotFound(err) {
				currentEndpoints = &api.Endpoints{
					ObjectMeta: api.ObjectMeta{
						Name: service.Name,
					},
				}
			} else {
				glog.Errorf("Error getting endpoints: %v", err)
				continue
			}
		}
		if reflect.DeepEqual(currentEndpoints.Subsets, subsets) {
			glog.V(5).Infof("endpoints are equal for %s/%s, skipping update", service.Namespace, service.Name)
			continue
		}
		newEndpoints := currentEndpoints
		newEndpoints.Subsets = subsets

		if len(currentEndpoints.ResourceVersion) == 0 {
			// No previous endpoints, create them
			_, err = e.client.Endpoints(service.Namespace).Create(newEndpoints)
		} else {
			// Pre-existing
			_, err = e.client.Endpoints(service.Namespace).Update(newEndpoints)
		}
		if err != nil {
			glog.Errorf("Error updating endpoints: %v", err)
			continue
		}
	}
	return resultErr
}

func packSubsets(subsets []api.EndpointSubset) []api.EndpointSubset {
	// First map each unique port definition to the sets of hosts
	// that offer it.
	portsToAddrs := map[api.EndpointPort][]api.EndpointAddress{}
	for i := range subsets {
		for j := range subsets[i].Ports {
			epp := &subsets[i].Ports[j]
			for k := range subsets[i].Addresses {
				epa := &subsets[i].Addresses[k]
				portsToAddrs[*epp] = append(portsToAddrs[*epp], *epa)
			}
		}
	}

	// Next, map the sets of hosts to the sets of ports they offer.
	// Go does not allow maps or slices as keys to maps, so we have
	// to synthesize and artificial key and do a sort of 2-part
	// associative entity.
	addrSets := map[string][]api.EndpointAddress{}
	addrSetsToPorts := map[string][]api.EndpointPort{}
	for epp, pods := range portsToAddrs {
		key := makePodsKey(pods)
		addrSets[key] = pods
		addrSetsToPorts[key] = append(addrSetsToPorts[key], epp)
	}

	// Next, build the N-to-M association the API wants.
	final := []api.EndpointSubset{}
	for key, ports := range addrSetsToPorts {
		final = append(final, api.EndpointSubset{Addresses: addrSets[key], Ports: ports})
	}

	// Finally, sort it.
	return sortSubsets(final)
}

func makePodsKey(addrs []api.EndpointAddress) string {
	// Flatten the list of addresses into a string so it can be used as a
	// map key.
	hasher := md5.New()
	util.DeepHashObject(hasher, addrs)
	return hex.EncodeToString(hasher.Sum(nil)[0:])
}

func sortSubsets(subsets []api.EndpointSubset) []api.EndpointSubset {
	for i := range subsets {
		ss := &subsets[i]
		sort.Sort(addrsByHash(ss.Addresses))
		sort.Sort(portsByHash(ss.Ports))
	}
	sort.Sort(subsetsByHash(subsets))
	return subsets
}

type subsetsByHash []api.EndpointSubset

func (sl subsetsByHash) Len() int      { return len(sl) }
func (sl subsetsByHash) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl subsetsByHash) Less(i, j int) bool {
	hasher := md5.New()
	util.DeepHashObject(hasher, sl[i])
	h1 := hasher.Sum(nil)
	util.DeepHashObject(hasher, sl[j])
	h2 := hasher.Sum(nil)
	return bytes.Compare(h1, h2) < 0
}

type addrsByHash []api.EndpointAddress

func (sl addrsByHash) Len() int      { return len(sl) }
func (sl addrsByHash) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl addrsByHash) Less(i, j int) bool {
	hasher := md5.New()
	util.DeepHashObject(hasher, sl[i])
	h1 := hasher.Sum(nil)
	util.DeepHashObject(hasher, sl[j])
	h2 := hasher.Sum(nil)
	return bytes.Compare(h1, h2) < 0
}

type portsByHash []api.EndpointPort

func (sl portsByHash) Len() int      { return len(sl) }
func (sl portsByHash) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl portsByHash) Less(i, j int) bool {
	hasher := md5.New()
	util.DeepHashObject(hasher, sl[i])
	h1 := hasher.Sum(nil)
	util.DeepHashObject(hasher, sl[j])
	h2 := hasher.Sum(nil)
	return bytes.Compare(h1, h2) < 0
}

func findDefaultPort(pod *api.Pod, servicePort int) int {
	firstPort := 0
	for _, container := range pod.Spec.Containers {
		for _, port := range container.Ports {
			if port.Name == "" {
				return port.ContainerPort
			}
			if firstPort == 0 {
				firstPort = port.ContainerPort
			}
		}
	}
	if firstPort == 0 {
		return servicePort
	}
	return firstPort
}

// findPort locates the container port for the given manifest and portName.
func findPort(pod *api.Pod, service *api.Service) (int, error) {
	portName := service.Spec.TargetPort
	switch portName.Kind {
	case util.IntstrString:
		if len(portName.StrVal) == 0 {
			return findDefaultPort(pod, service.Spec.Port), nil
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
			return findDefaultPort(pod, service.Spec.Port), nil
		}
		return portName.IntVal, nil
	}

	return 0, fmt.Errorf("no suitable port for manifest: %s", pod.UID)
}
