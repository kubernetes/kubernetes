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
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/labels"
	discoverylisters "k8s.io/client-go/listers/discovery/v1"
)

// EndpointSliceLister provides a lister-like interface for EndpointSlices
// that handles listing multiple slices for the same service.
type EndpointSliceLister struct {
	// lister is the underlying EndpointSlice lister.
	lister discoverylisters.EndpointSliceLister
}

// NewEndpointSliceLister creates a new EndpointSliceLister.
func NewEndpointSliceLister(
	lister discoverylisters.EndpointSliceLister,
) *EndpointSliceLister {
	return &EndpointSliceLister{
		lister: lister,
	}
}

// List lists all EndpointSlices in the indexer.
func (l *EndpointSliceLister) List(selector labels.Selector) ([]*discovery.EndpointSlice, error) {
	return l.lister.List(selector)
}

// EndpointSlices returns an object that can list and get EndpointSlices for a given namespace.
func (l *EndpointSliceLister) EndpointSlices(namespace string) EndpointSliceNamespaceLister {
	return &endpointSliceNamespaceLister{
		lister: l.lister.EndpointSlices(namespace),
	}
}

// EndpointSliceNamespaceLister helps list and get EndpointSlices for a given namespace.
type EndpointSliceNamespaceLister interface {
	// List lists all EndpointSlices in the indexer for a given namespace.
	List(selector labels.Selector) ([]*discovery.EndpointSlice, error)
	// Get retrieves all EndpointSlices for a given service.
	Get(serviceName string) ([]*discovery.EndpointSlice, error)
}

// endpointSliceNamespaceLister implements EndpointSliceNamespaceLister.
type endpointSliceNamespaceLister struct {
	lister discoverylisters.EndpointSliceNamespaceLister
}

// List lists all EndpointSlices in the indexer for a given namespace.
func (l *endpointSliceNamespaceLister) List(selector labels.Selector) ([]*discovery.EndpointSlice, error) {
	return l.lister.List(selector)
}

// Get retrieves all EndpointSlices for a given service.
func (l *endpointSliceNamespaceLister) Get(serviceName string) ([]*discovery.EndpointSlice, error) {
	selector := labels.SelectorFromSet(labels.Set{discovery.LabelServiceName: serviceName})
	return l.lister.List(selector)
}
