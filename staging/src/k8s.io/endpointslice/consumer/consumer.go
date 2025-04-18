/*
Copyright 2024 The Kubernetes Authors.

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

package consumer

import (
	"fmt"
	"reflect"
	"sort"
	"sync"

	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
)

// EndpointSliceConsumer provides a unified view of endpoints for services
// across multiple EndpointSlice objects.
type EndpointSliceConsumer struct {
	// lock protects slicesByService.
	lock sync.RWMutex

	// slicesByService maps service namespaced names to a map of EndpointSlices
	// keyed by their names.
	slicesByService map[types.NamespacedName]map[string]*discovery.EndpointSlice

	// handlers for endpoint changes
	handlers []EndpointChangeHandler

	// nodeName is the name of the node this consumer is running on.
	// Used to determine if an endpoint is local.
	nodeName string
}

// EndpointChangeHandler is called when endpoints for a service change.
type EndpointChangeHandler interface {
	// OnEndpointsChange is called when endpoints for a service change.
	OnEndpointsChange(serviceNN types.NamespacedName, endpoints []*discovery.EndpointSlice)
}

// EndpointChangeHandlerFunc is a function that implements EndpointChangeHandler.
type EndpointChangeHandlerFunc func(serviceNN types.NamespacedName, endpoints []*discovery.EndpointSlice)

// OnEndpointsChange calls the function.
func (f EndpointChangeHandlerFunc) OnEndpointsChange(serviceNN types.NamespacedName, endpoints []*discovery.EndpointSlice) {
	f(serviceNN, endpoints)
}

// NewEndpointSliceConsumer creates a new EndpointSliceConsumer.
func NewEndpointSliceConsumer(nodeName string) *EndpointSliceConsumer {
	return &EndpointSliceConsumer{
		slicesByService: make(map[types.NamespacedName]map[string]*discovery.EndpointSlice),
		nodeName:        nodeName,
	}
}

// AddEventHandler adds a handler for endpoint changes.
func (c *EndpointSliceConsumer) AddEventHandler(handler EndpointChangeHandler) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.handlers = append(c.handlers, handler)
}

// OnEndpointSliceAdd is called when an EndpointSlice is added.
func (c *EndpointSliceConsumer) OnEndpointSliceAdd(endpointSlice *discovery.EndpointSlice) {
	serviceNN, sliceName, err := endpointSliceCacheKeys(endpointSlice)
	if err != nil {
		klog.ErrorS(err, "Error getting endpoint slice cache keys")
		return
	}

	c.lock.Lock()
	defer c.lock.Unlock()

	slices, ok := c.slicesByService[serviceNN]
	if !ok {
		slices = make(map[string]*discovery.EndpointSlice)
		c.slicesByService[serviceNN] = slices
	}

	// Check if this is a new slice or an update to an existing one
	existingSlice, exists := slices[sliceName]
	slices[sliceName] = endpointSlice.DeepCopy()

	// Only notify handlers if this is a new slice or the endpoints have changed
	if !exists || !reflect.DeepEqual(existingSlice.Endpoints, endpointSlice.Endpoints) {
		c.notifyHandlersLocked(serviceNN)
	}
}

// OnEndpointSliceUpdate is called when an EndpointSlice is updated.
func (c *EndpointSliceConsumer) OnEndpointSliceUpdate(_, newEndpointSlice *discovery.EndpointSlice) {
	c.OnEndpointSliceAdd(newEndpointSlice)
}

// OnEndpointSliceDelete is called when an EndpointSlice is deleted.
func (c *EndpointSliceConsumer) OnEndpointSliceDelete(endpointSlice *discovery.EndpointSlice) {
	serviceNN, sliceName, err := endpointSliceCacheKeys(endpointSlice)
	if err != nil {
		klog.ErrorS(err, "Error getting endpoint slice cache keys")
		return
	}

	c.lock.Lock()
	defer c.lock.Unlock()

	slices, ok := c.slicesByService[serviceNN]
	if !ok {
		return
	}

	if _, exists := slices[sliceName]; exists {
		delete(slices, sliceName)
		if len(slices) == 0 {
			delete(c.slicesByService, serviceNN)
		}
		c.notifyHandlersLocked(serviceNN)
	}
}

// GetEndpointSlices returns all EndpointSlices for a service.
func (c *EndpointSliceConsumer) GetEndpointSlices(serviceNN types.NamespacedName) []*discovery.EndpointSlice {
	c.lock.RLock()
	defer c.lock.RUnlock()

	slices, ok := c.slicesByService[serviceNN]
	if !ok {
		return nil
	}

	result := make([]*discovery.EndpointSlice, 0, len(slices))
	for _, slice := range slices {
		result = append(result, slice.DeepCopy())
	}

	// Sort slices by name for consistent results
	sort.Slice(result, func(i, j int) bool {
		return result[i].Name < result[j].Name
	})

	return result
}

// GetEndpoints returns all endpoints for a service, merging and deduplicating
// endpoints from all EndpointSlices for the service.
func (c *EndpointSliceConsumer) GetEndpoints(serviceNN types.NamespacedName) []discovery.Endpoint {
	slices := c.GetEndpointSlices(serviceNN)
	if len(slices) == 0 {
		return nil
	}

	// Use a map to deduplicate endpoints by address
	endpointMap := make(map[string]discovery.Endpoint)

	for _, slice := range slices {
		for _, endpoint := range slice.Endpoints {
			if len(endpoint.Addresses) == 0 {
				continue
			}

			// Use the first address as the key for deduplication
			key := endpoint.Addresses[0]
			
			// If we already have this endpoint, only replace it if the existing one
			// is not local but the new one is
			existingEp, exists := endpointMap[key]
			isLocal := endpoint.NodeName != nil && *endpoint.NodeName == c.nodeName
			existingIsLocal := exists && existingEp.NodeName != nil && *existingEp.NodeName == c.nodeName
			
			if !exists || (isLocal && !existingIsLocal) {
				endpointMap[key] = *endpoint.DeepCopy()
			}
		}
	}

	// Convert map to slice
	result := make([]discovery.Endpoint, 0, len(endpointMap))
	for _, endpoint := range endpointMap {
		result = append(result, endpoint)
	}

	// Sort endpoints by address for consistent results
	sort.Slice(result, func(i, j int) bool {
		if len(result[i].Addresses) == 0 || len(result[j].Addresses) == 0 {
			return len(result[i].Addresses) > len(result[j].Addresses)
		}
		return result[i].Addresses[0] < result[j].Addresses[0]
	})

	return result
}

// notifyHandlersLocked notifies all handlers of an endpoint change.
// The caller must hold the lock.
func (c *EndpointSliceConsumer) notifyHandlersLocked(serviceNN types.NamespacedName) {
	if len(c.handlers) == 0 {
		return
	}

	slices, ok := c.slicesByService[serviceNN]
	if !ok {
		// Service has been deleted, notify with empty slice
		for _, handler := range c.handlers {
			handler.OnEndpointsChange(serviceNN, nil)
		}
		return
	}

	// Convert map to slice
	sliceList := make([]*discovery.EndpointSlice, 0, len(slices))
	for _, slice := range slices {
		sliceList = append(sliceList, slice.DeepCopy())
	}

	// Sort slices by name for consistent results
	sort.Slice(sliceList, func(i, j int) bool {
		return sliceList[i].Name < sliceList[j].Name
	})

	// Notify handlers
	for _, handler := range c.handlers {
		handler.OnEndpointsChange(serviceNN, sliceList)
	}
}

// endpointSliceCacheKeys returns cache keys used for a given EndpointSlice.
func endpointSliceCacheKeys(endpointSlice *discovery.EndpointSlice) (types.NamespacedName, string, error) {
	var err error
	serviceName, ok := endpointSlice.Labels[discovery.LabelServiceName]
	if !ok || serviceName == "" {
		err = fmt.Errorf("no %s label set on endpoint slice: %s", discovery.LabelServiceName, endpointSlice.Name)
	} else if endpointSlice.Namespace == "" || endpointSlice.Name == "" {
		err = fmt.Errorf("expected EndpointSlice name and namespace to be set: %v", endpointSlice)
	}
	return types.NamespacedName{Namespace: endpointSlice.Namespace, Name: serviceName}, endpointSlice.Name, err
}
