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

	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	discoverylisters "k8s.io/client-go/listers/discovery/v1"
)

// EndpointSliceLister provides a lister-like interface for EndpointSlices
// that handles merging multiple slices for the same service.
type EndpointSliceLister struct {
	// consumer is the underlying EndpointSliceConsumer.
	consumer *EndpointSliceConsumer

	// lister is the underlying EndpointSlice lister.
	lister discoverylisters.EndpointSliceLister
}

// NewEndpointSliceLister creates a new EndpointSliceLister.
func NewEndpointSliceLister(
	lister discoverylisters.EndpointSliceLister,
	nodeName string,
) *EndpointSliceLister {
	consumer := NewEndpointSliceConsumer(nodeName)

	return &EndpointSliceLister{
		consumer: consumer,
		lister:   lister,
	}
}

// List lists all EndpointSlices in the indexer.
func (l *EndpointSliceLister) List(selector labels.Selector) ([]*discovery.EndpointSlice, error) {
	return l.lister.List(selector)
}

// EndpointSlices returns an object that can list and get EndpointSlices for a given namespace.
func (l *EndpointSliceLister) EndpointSlices(namespace string) EndpointSliceNamespaceLister {
	return &endpointSliceNamespaceLister{
		consumer: l.consumer,
		lister:   l.lister.EndpointSlices(namespace),
	}
}

// EndpointSliceNamespaceLister helps list and get EndpointSlices for a given namespace.
type EndpointSliceNamespaceLister interface {
	// List lists all EndpointSlices in the indexer for a given namespace.
	List(selector labels.Selector) ([]*discovery.EndpointSlice, error)
	// Get retrieves all EndpointSlices for a given service.
	Get(serviceName string) ([]*discovery.EndpointSlice, error)
	// GetEndpoints retrieves all endpoints for a given service, merging and
	// deduplicating endpoints from all EndpointSlices for the service.
	GetEndpoints(serviceName string) ([]discovery.Endpoint, error)
}

// endpointSliceNamespaceLister implements EndpointSliceNamespaceLister.
type endpointSliceNamespaceLister struct {
	consumer *EndpointSliceConsumer
	lister   discoverylisters.EndpointSliceNamespaceLister
}

// List lists all EndpointSlices in the indexer for a given namespace.
func (l *endpointSliceNamespaceLister) List(selector labels.Selector) ([]*discovery.EndpointSlice, error) {
	return l.lister.List(selector)
}

// Get retrieves all EndpointSlices for a given service.
func (l *endpointSliceNamespaceLister) Get(serviceName string) ([]*discovery.EndpointSlice, error) {
	// Get all EndpointSlices for the namespace
	allSlices, err := l.List(labels.Everything())
	if err != nil {
		return nil, err
	}

	// Filter by service name
	var serviceSlices []*discovery.EndpointSlice
	for _, slice := range allSlices {
		if slice.Labels[discovery.LabelServiceName] == serviceName {
			// Add to the consumer cache
			l.consumer.OnEndpointSliceAdd(slice)
			serviceSlices = append(serviceSlices, slice)
		}
	}

	if len(serviceSlices) == 0 {
		return nil, fmt.Errorf("no EndpointSlices found for service %s", serviceName)
	}

	return serviceSlices, nil
}

// GetEndpoints retrieves all endpoints for a given service, merging and
// deduplicating endpoints from all EndpointSlices for the service.
func (l *endpointSliceNamespaceLister) GetEndpoints(serviceName string) ([]discovery.Endpoint, error) {
	// Get all EndpointSlices for the service
	_, err := l.Get(serviceName)
	if err != nil {
		return nil, err
	}

	// Get the merged endpoints from the consumer
	serviceNN := types.NamespacedName{
		Namespace: l.lister.Namespace(),
		Name:      serviceName,
	}
	return l.consumer.GetEndpoints(serviceNN), nil
}
