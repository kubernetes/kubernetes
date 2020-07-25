/*
Copyright 2020 The Kubernetes Authors.

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

package endpointslice

import (
	"sync"

	discovery "k8s.io/api/discovery/v1beta1"
	"k8s.io/apimachinery/pkg/types"
)

// ResourceVersionsByName tracks expected EndpointSlice resource versions
// by EndpointSlice name.
type ResourceVersionsByName map[string]string

// Tracker tracks EndpointSlices and their associated resource versions to help
// determine if a change to an EndpointSlice has been processed by the
// EndpointSlice controller.
type Tracker struct {
	// lock protects resourceVersionsByService.
	lock sync.Mutex
	// resourceVersionsByService tracks the list of EndpointSlices and
	// associated resource versions expected for a given Service.
	resourceVersionsByService map[types.NamespacedName]ResourceVersionsByName
}

// NewTracker creates and initializes a new Tracker.
func NewTracker() *Tracker {
	return &Tracker{
		resourceVersionsByService: map[types.NamespacedName]ResourceVersionsByName{},
	}
}

// Has returns true if the Tracker has a resource version for the provided
// EndpointSlice.
func (t *Tracker) Has(endpointSlice *discovery.EndpointSlice) bool {
	t.lock.Lock()
	defer t.lock.Unlock()

	rrv, ok := t.RelatedResourceVersions(endpointSlice)
	if !ok {
		return false
	}
	_, ok = rrv[endpointSlice.Name]
	return ok
}

// Stale returns true if this EndpointSliceTracker does not have a resource
// version for the provided EndpointSlice or it does not match the resource
// version of the provided EndpointSlice.
func (t *Tracker) Stale(endpointSlice *discovery.EndpointSlice) bool {
	t.lock.Lock()
	defer t.lock.Unlock()

	rrv, ok := t.RelatedResourceVersions(endpointSlice)
	if !ok {
		return true
	}
	return rrv[endpointSlice.Name] != endpointSlice.ResourceVersion
}

// Update adds or updates the resource version in this EndpointSliceTracker for
// the provided EndpointSlice.
func (t *Tracker) Update(endpointSlice *discovery.EndpointSlice) {
	t.lock.Lock()
	defer t.lock.Unlock()

	rrv, ok := t.RelatedResourceVersions(endpointSlice)
	if !ok {
		rrv = ResourceVersionsByName{}
		t.resourceVersionsByService[getServiceNN(endpointSlice)] = rrv
	}
	rrv[endpointSlice.Name] = endpointSlice.ResourceVersion
}

// DeleteService removes the set of resource versions tracked for the Service.
func (t *Tracker) DeleteService(namespace, name string) {
	t.lock.Lock()
	defer t.lock.Unlock()

	serviceNN := types.NamespacedName{Name: name, Namespace: namespace}
	delete(t.resourceVersionsByService, serviceNN)
}

// Delete removes the resource version in this EndpointSliceTracker for the
// provided EndpointSlice.
func (t *Tracker) Delete(endpointSlice *discovery.EndpointSlice) {
	t.lock.Lock()
	defer t.lock.Unlock()

	rrv, ok := t.RelatedResourceVersions(endpointSlice)
	if ok {
		delete(rrv, endpointSlice.Name)
	}
}

// RelatedResourceVersions returns the set of resource versions tracked for the
// Service corresponding to the provided EndpointSlice, and a bool to indicate
// if it exists.
func (t *Tracker) RelatedResourceVersions(endpointSlice *discovery.EndpointSlice) (ResourceVersionsByName, bool) {
	serviceNN := getServiceNN(endpointSlice)
	vers, ok := t.resourceVersionsByService[serviceNN]
	return vers, ok
}

// getServiceNN returns a namespaced name for the Service corresponding to the
// provided EndpointSlice.
func getServiceNN(endpointSlice *discovery.EndpointSlice) types.NamespacedName {
	serviceName, _ := endpointSlice.Labels[discovery.LabelServiceName]
	return types.NamespacedName{Name: serviceName, Namespace: endpointSlice.Namespace}
}
