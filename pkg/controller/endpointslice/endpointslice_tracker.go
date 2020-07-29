/*
Copyright 2019 The Kubernetes Authors.

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
	"time"

	discovery "k8s.io/api/discovery/v1beta1"
	"k8s.io/apimachinery/pkg/types"
)

// serviceStatus tracks the last time an EndpointSlice for the Service has been
// updated along with status for each EndpointSlice.
type serviceStatus struct {
	// lastUpdated tracks the last time an EndpointSlice for this Service was
	// updated.
	lastUpdated time.Time
	// statusBySlice tracks expected EndpointSlice resource versions by
	// EndpointSlice name.
	statusBySlice map[string]*sliceStatus
}

// sliceStatus tracks the last known resource version for an EndpointSlice
// and if that has been updated in the informer cache.
type sliceStatus struct {
	resourceVersion string
	cacheUpdated    bool
}

// endpointSliceTracker tracks EndpointSlices and their associated resource
// versions to help determine if a change to an EndpointSlice has been processed
// by the EndpointSlice controller.
type endpointSliceTracker struct {
	// lock protects resourceVersionsByService.
	lock sync.Mutex
	// statusByService tracks the status of each Service and the associated
	// EndpointSlices.
	statusByService map[types.NamespacedName]*serviceStatus
}

// newEndpointSliceTracker creates and initializes a new endpointSliceTracker.
func newEndpointSliceTracker() *endpointSliceTracker {
	return &endpointSliceTracker{
		statusByService: map[types.NamespacedName]*serviceStatus{},
	}
}

// newServiceStatus returns a new serviceStatus.
func newServiceStatus() *serviceStatus {
	return &serviceStatus{
		statusBySlice: map[string]*sliceStatus{},
	}
}

// Has returns true if the endpointSliceTracker has a resource version for the
// provided EndpointSlice.
func (est *endpointSliceTracker) Has(endpointSlice *discovery.EndpointSlice) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	ss, ok := est.serviceStatusForSlice(endpointSlice)
	if !ok {
		return false
	}
	_, ok = ss.statusBySlice[endpointSlice.Name]
	return ok
}

// Stale returns true if this endpointSliceTracker does not have a resource
// version for the provided EndpointSlice or it does not match the resource
// version of the provided EndpointSlice.
func (est *endpointSliceTracker) Stale(endpointSlice *discovery.EndpointSlice) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	ss, ok := est.serviceStatusForSlice(endpointSlice)
	if !ok {
		return true
	}
	rvs, ok := ss.statusBySlice[endpointSlice.Name]
	if !ok {
		return true
	}
	return rvs.resourceVersion != endpointSlice.ResourceVersion
}

// Update adds or updates the resource version in this endpointSliceTracker for
// the provided EndpointSlice.
func (est *endpointSliceTracker) Update(endpointSlice *discovery.EndpointSlice) {
	est.lock.Lock()
	defer est.lock.Unlock()

	ss, ok := est.serviceStatusForSlice(endpointSlice)
	if !ok {
		ss = newServiceStatus()
		est.statusByService[getServiceNN(endpointSlice)] = ss
	}
	ss.lastUpdated = time.Now()
	ss.statusBySlice[endpointSlice.Name] = &sliceStatus{
		resourceVersion: endpointSlice.ResourceVersion,
		cacheUpdated:    false,
	}
}

// DeleteService removes the set of resource versions tracked for the Service.
func (est *endpointSliceTracker) DeleteService(namespace, name string) {
	est.lock.Lock()
	defer est.lock.Unlock()

	serviceNN := types.NamespacedName{Name: name, Namespace: namespace}
	delete(est.statusByService, serviceNN)
}

// Delete removes the resource version in this endpointSliceTracker for the
// provided EndpointSlice.
func (est *endpointSliceTracker) Delete(endpointSlice *discovery.EndpointSlice) {
	est.lock.Lock()
	defer est.lock.Unlock()

	ss, ok := est.serviceStatusForSlice(endpointSlice)
	if ok {
		delete(ss.statusBySlice, endpointSlice.Name)
	}
}

// MarkCacheUpdated sets cacheUpdated to true for an EndpointSlice if the
// EndpointSlice resource version matches what is tracked.
func (est *endpointSliceTracker) MarkCacheUpdated(endpointSlice *discovery.EndpointSlice) {
	est.lock.Lock()
	defer est.lock.Unlock()

	ss, ok := est.serviceStatusForSlice(endpointSlice)
	if !ok {
		return
	}
	epStatus, ok := ss.statusBySlice[endpointSlice.Name]
	if !ok {
		return
	}
	if epStatus.resourceVersion == endpointSlice.ResourceVersion {
		epStatus.cacheUpdated = true
	}
}

// MarkServiceCacheUpdated sets cacheUpdated to true for all EndpointSlices
// owned by a Service.
func (est *endpointSliceTracker) MarkServiceCacheUpdated(namespace, name string) {
	est.lock.Lock()
	defer est.lock.Unlock()

	ss, ok := est.statusByService[types.NamespacedName{Namespace: namespace, Name: name}]
	if !ok {
		return
	}
	for _, sliceStatus := range ss.statusBySlice {
		sliceStatus.cacheUpdated = true
	}
}

// ServiceCacheOutdated returns true if the cache is out of date for any
// EndpointSlices tracked for the provided Service namespace and name.
func (est *endpointSliceTracker) ServiceCacheOutdated(namespace, name string) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	ss, ok := est.statusByService[types.NamespacedName{Namespace: namespace, Name: name}]
	if !ok {
		// If we don't have any record of this Service it is technically up to
		// date. This likely means this is a new Service and we should not be
		// delaying any syncs in this case.
		return false
	}
	for _, epStatus := range ss.statusBySlice {
		if !epStatus.cacheUpdated {
			return true
		}
	}
	return false
}

// ServiceCacheUpdatedSince returns true if the cache for a Service has been
// updated since the provided time.
func (est *endpointSliceTracker) ServiceCacheUpdatedSince(namespace, name string, updatedSince time.Time) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	ss, ok := est.statusByService[types.NamespacedName{Namespace: namespace, Name: name}]
	if !ok {
		// If we don't have any record of this Service it is technically up to
		// date. This likely means this is a new Service and we should not be
		// delaying any syncs in this case.
		return false
	}

	return ss.lastUpdated.After(updatedSince)
}

// serviceStatusForSlice returns the serviceStatus for the Service corresponding
// to the provided EndpointSlice, and a bool to indicate if it exists.
func (est *endpointSliceTracker) serviceStatusForSlice(endpointSlice *discovery.EndpointSlice) (*serviceStatus, bool) {
	serviceNN := getServiceNN(endpointSlice)
	serviceStatus, ok := est.statusByService[serviceNN]
	return serviceStatus, ok
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
