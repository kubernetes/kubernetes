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

package endpointslicemirroring

import (
	"sync"

	discovery "k8s.io/api/discovery/v1beta1"
	"k8s.io/apimachinery/pkg/types"
)

// endpointSliceResourceVersions tracks expected EndpointSlice resource versions
// by EndpointSlice name.
type endpointSliceResourceVersions map[string]string

// endpointSliceTracker tracks EndpointSlices and their associated resource
// versions to help determine if a change to an EndpointSlice has been processed
// by the EndpointSlice controller.
type endpointSliceTracker struct {
	// lock protects resourceVersionsByService.
	lock sync.Mutex
	// resourceVersionsByService tracks the list of EndpointSlices and
	// associated resource versions expected for a given Service.
	resourceVersionsByService map[types.NamespacedName]endpointSliceResourceVersions
}

// newEndpointSliceTracker creates and initializes a new endpointSliceTracker.
func newEndpointSliceTracker() *endpointSliceTracker {
	return &endpointSliceTracker{
		resourceVersionsByService: map[types.NamespacedName]endpointSliceResourceVersions{},
	}
}

// Has returns true if the endpointSliceTracker has a resource version for the
// provided EndpointSlice.
func (est *endpointSliceTracker) Has(endpointSlice *discovery.EndpointSlice) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	rrv, ok := est.relatedResourceVersions(endpointSlice)
	if !ok {
		return false
	}
	_, ok = rrv[endpointSlice.Name]
	return ok
}

// Stale returns true if this endpointSliceTracker does not have a resource
// version for the provided EndpointSlice or it does not match the resource
// version of the provided EndpointSlice.
func (est *endpointSliceTracker) Stale(endpointSlice *discovery.EndpointSlice) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	rrv, ok := est.relatedResourceVersions(endpointSlice)
	if !ok {
		return true
	}
	return rrv[endpointSlice.Name] != endpointSlice.ResourceVersion
}

// Update adds or updates the resource version in this endpointSliceTracker for
// the provided EndpointSlice.
func (est *endpointSliceTracker) Update(endpointSlice *discovery.EndpointSlice) {
	est.lock.Lock()
	defer est.lock.Unlock()

	rrv, ok := est.relatedResourceVersions(endpointSlice)
	if !ok {
		rrv = endpointSliceResourceVersions{}
		est.resourceVersionsByService[getServiceNN(endpointSlice)] = rrv
	}
	rrv[endpointSlice.Name] = endpointSlice.ResourceVersion
}

// DeleteService removes the set of resource versions tracked for the Service.
func (est *endpointSliceTracker) DeleteService(namespace, name string) {
	est.lock.Lock()
	defer est.lock.Unlock()

	serviceNN := types.NamespacedName{Name: name, Namespace: namespace}
	delete(est.resourceVersionsByService, serviceNN)
}

// Delete removes the resource version in this endpointSliceTracker for the
// provided EndpointSlice.
func (est *endpointSliceTracker) Delete(endpointSlice *discovery.EndpointSlice) {
	est.lock.Lock()
	defer est.lock.Unlock()

	rrv, ok := est.relatedResourceVersions(endpointSlice)
	if ok {
		delete(rrv, endpointSlice.Name)
	}
}

// relatedResourceVersions returns the set of resource versions tracked for the
// Service corresponding to the provided EndpointSlice, and a bool to indicate
// if it exists.
func (est *endpointSliceTracker) relatedResourceVersions(endpointSlice *discovery.EndpointSlice) (endpointSliceResourceVersions, bool) {
	serviceNN := getServiceNN(endpointSlice)
	vers, ok := est.resourceVersionsByService[serviceNN]
	return vers, ok
}

// getServiceNN returns a namespaced name for the Service corresponding to the
// provided EndpointSlice.
func getServiceNN(endpointSlice *discovery.EndpointSlice) types.NamespacedName {
	serviceName, _ := endpointSlice.Labels[discovery.LabelServiceName]
	return types.NamespacedName{Name: serviceName, Namespace: endpointSlice.Namespace}
}

// managedByChanged returns true if one of the provided EndpointSlices is
// managed by the EndpointSlice controller while the other is not.
func managedByChanged(endpointSlice1, endpointSlice2 *discovery.EndpointSlice) bool {
	return managedByController(endpointSlice1) != managedByController(endpointSlice2)
}

// managedByController returns true if the controller of the provided
// EndpointSlices is the EndpointSlice controller.
func managedByController(endpointSlice *discovery.EndpointSlice) bool {
	managedBy, _ := endpointSlice.Labels[discovery.LabelManagedBy]
	return managedBy == controllerName
}
