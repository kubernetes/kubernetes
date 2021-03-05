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

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
)

const (
	deletionExpected = -1
)

// generationsBySlice tracks expected EndpointSlice generations by EndpointSlice
// uid. A value of deletionExpected (-1) may be used here to indicate that we
// expect this EndpointSlice to be deleted.
type generationsBySlice map[types.UID]int64

// endpointSliceTracker tracks EndpointSlices and their associated generation to
// help determine if a change to an EndpointSlice has been processed by the
// EndpointSlice controller.
type endpointSliceTracker struct {
	// lock protects generationsByService.
	lock sync.Mutex
	// generationsByService tracks the generations of EndpointSlices for each
	// Service.
	generationsByService map[types.NamespacedName]generationsBySlice
}

// newEndpointSliceTracker creates and initializes a new endpointSliceTracker.
func newEndpointSliceTracker() *endpointSliceTracker {
	return &endpointSliceTracker{
		generationsByService: map[types.NamespacedName]generationsBySlice{},
	}
}

// Has returns true if the endpointSliceTracker has a generation for the
// provided EndpointSlice.
func (est *endpointSliceTracker) Has(endpointSlice *discovery.EndpointSlice) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	gfs, ok := est.generationsForSliceUnsafe(endpointSlice)
	if !ok {
		return false
	}
	_, ok = gfs[endpointSlice.UID]
	return ok
}

// ShouldSync returns true if this endpointSliceTracker does not have a
// generation for the provided EndpointSlice or it is greater than the
// generation of the tracked EndpointSlice.
func (est *endpointSliceTracker) ShouldSync(endpointSlice *discovery.EndpointSlice) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	gfs, ok := est.generationsForSliceUnsafe(endpointSlice)
	if !ok {
		return true
	}
	g, ok := gfs[endpointSlice.UID]
	return !ok || endpointSlice.Generation > g
}

// StaleSlices returns true if one or more of the provided EndpointSlices
// have older generations than the corresponding tracked ones or if the tracker
// is expecting one or more of the provided EndpointSlices to be deleted.
func (est *endpointSliceTracker) StaleSlices(service *v1.Service, endpointSlices []*discovery.EndpointSlice) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	nn := types.NamespacedName{Name: service.Name, Namespace: service.Namespace}
	gfs, ok := est.generationsByService[nn]
	if !ok {
		return false
	}
	for _, endpointSlice := range endpointSlices {
		g, ok := gfs[endpointSlice.UID]
		if ok && (g == deletionExpected || g > endpointSlice.Generation) {
			return true
		}
	}
	return false
}

// Update adds or updates the generation in this endpointSliceTracker for the
// provided EndpointSlice.
func (est *endpointSliceTracker) Update(endpointSlice *discovery.EndpointSlice) {
	est.lock.Lock()
	defer est.lock.Unlock()

	gfs, ok := est.generationsForSliceUnsafe(endpointSlice)

	if !ok {
		gfs = generationsBySlice{}
		est.generationsByService[getServiceNN(endpointSlice)] = gfs
	}
	gfs[endpointSlice.UID] = endpointSlice.Generation
}

// DeleteService removes the set of generations tracked for the Service.
func (est *endpointSliceTracker) DeleteService(namespace, name string) {
	est.lock.Lock()
	defer est.lock.Unlock()

	serviceNN := types.NamespacedName{Name: name, Namespace: namespace}
	delete(est.generationsByService, serviceNN)
}

// ExpectDeletion sets the generation to deletionExpected in this
// endpointSliceTracker for the provided EndpointSlice.
func (est *endpointSliceTracker) ExpectDeletion(endpointSlice *discovery.EndpointSlice) {
	est.lock.Lock()
	defer est.lock.Unlock()

	gfs, ok := est.generationsForSliceUnsafe(endpointSlice)

	if !ok {
		gfs = generationsBySlice{}
		est.generationsByService[getServiceNN(endpointSlice)] = gfs
	}
	gfs[endpointSlice.UID] = deletionExpected
}

// HandleDeletion removes the generation in this endpointSliceTracker for the
// provided EndpointSlice. This returns true if the tracker expected this
// EndpointSlice to be deleted and false if not.
func (est *endpointSliceTracker) HandleDeletion(endpointSlice *discovery.EndpointSlice) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	gfs, ok := est.generationsForSliceUnsafe(endpointSlice)

	if ok {
		g, ok := gfs[endpointSlice.UID]
		delete(gfs, endpointSlice.UID)
		if ok && g != deletionExpected {
			return false
		}
	}

	return true
}

// generationsForSliceUnsafe returns the generations for the Service
// corresponding to the provided EndpointSlice, and a bool to indicate if it
// exists. A lock must be applied before calling this function.
func (est *endpointSliceTracker) generationsForSliceUnsafe(endpointSlice *discovery.EndpointSlice) (generationsBySlice, bool) {
	serviceNN := getServiceNN(endpointSlice)
	generations, ok := est.generationsByService[serviceNN]
	return generations, ok
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
