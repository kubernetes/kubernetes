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

// GenerationsBySlice tracks expected EndpointSlice generations by EndpointSlice
// uid. A value of deletionExpected (-1) may be used here to indicate that we
// expect this EndpointSlice to be deleted.
type GenerationsBySlice map[types.UID]int64

// EndpointSliceTracker tracks EndpointSlices and their associated generation to
// help determine if a change to an EndpointSlice has been processed by the
// EndpointSlice controller.
type EndpointSliceTracker struct {
	// lock protects generationsByService.
	lock sync.Mutex
	// generationsByService tracks the generations of EndpointSlices for each
	// Service.
	generationsByService map[types.NamespacedName]GenerationsBySlice
}

// NewEndpointSliceTracker creates and initializes a new endpointSliceTracker.
func NewEndpointSliceTracker() *EndpointSliceTracker {
	return &EndpointSliceTracker{
		generationsByService: map[types.NamespacedName]GenerationsBySlice{},
	}
}

// Has returns true if the endpointSliceTracker has a generation for the
// provided EndpointSlice.
func (est *EndpointSliceTracker) Has(endpointSlice *discovery.EndpointSlice) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	gfs, ok := est.GenerationsForSliceUnsafe(endpointSlice)
	if !ok {
		return false
	}
	_, ok = gfs[endpointSlice.UID]
	return ok
}

// ShouldSync returns true if this endpointSliceTracker does not have a
// generation for the provided EndpointSlice or it is greater than the
// generation of the tracked EndpointSlice.
func (est *EndpointSliceTracker) ShouldSync(endpointSlice *discovery.EndpointSlice) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	gfs, ok := est.GenerationsForSliceUnsafe(endpointSlice)
	if !ok {
		return true
	}
	g, ok := gfs[endpointSlice.UID]
	return !ok || endpointSlice.Generation > g
}

// StaleSlices returns true if any of the following are true:
// 1. One or more of the provided EndpointSlices have older generations than the
//    corresponding tracked ones.
// 2. The tracker is expecting one or more of the provided EndpointSlices to be
//    deleted. (EndpointSlices that have already been marked for deletion are ignored here.)
// 3. The tracker is tracking EndpointSlices that have not been provided.
func (est *EndpointSliceTracker) StaleSlices(service *v1.Service, endpointSlices []*discovery.EndpointSlice) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	nn := types.NamespacedName{Name: service.Name, Namespace: service.Namespace}
	gfs, ok := est.generationsByService[nn]
	if !ok {
		return false
	}
	providedSlices := map[types.UID]int64{}
	for _, endpointSlice := range endpointSlices {
		providedSlices[endpointSlice.UID] = endpointSlice.Generation
		g, ok := gfs[endpointSlice.UID]
		if ok && (g == deletionExpected || g > endpointSlice.Generation) {
			return true
		}
	}
	for uid, generation := range gfs {
		if generation == deletionExpected {
			continue
		}
		_, ok := providedSlices[uid]
		if !ok {
			return true
		}
	}
	return false
}

// Update adds or updates the generation in this endpointSliceTracker for the
// provided EndpointSlice.
func (est *EndpointSliceTracker) Update(endpointSlice *discovery.EndpointSlice) {
	est.lock.Lock()
	defer est.lock.Unlock()

	gfs, ok := est.GenerationsForSliceUnsafe(endpointSlice)

	if !ok {
		gfs = GenerationsBySlice{}
		est.generationsByService[getServiceNN(endpointSlice)] = gfs
	}
	gfs[endpointSlice.UID] = endpointSlice.Generation
}

// DeleteService removes the set of generations tracked for the Service.
func (est *EndpointSliceTracker) DeleteService(namespace, name string) {
	est.lock.Lock()
	defer est.lock.Unlock()

	serviceNN := types.NamespacedName{Name: name, Namespace: namespace}
	delete(est.generationsByService, serviceNN)
}

// ExpectDeletion sets the generation to deletionExpected in this
// endpointSliceTracker for the provided EndpointSlice.
func (est *EndpointSliceTracker) ExpectDeletion(endpointSlice *discovery.EndpointSlice) {
	est.lock.Lock()
	defer est.lock.Unlock()

	gfs, ok := est.GenerationsForSliceUnsafe(endpointSlice)

	if !ok {
		gfs = GenerationsBySlice{}
		est.generationsByService[getServiceNN(endpointSlice)] = gfs
	}
	gfs[endpointSlice.UID] = deletionExpected
}

// HandleDeletion removes the generation in this endpointSliceTracker for the
// provided EndpointSlice. This returns true if the tracker expected this
// EndpointSlice to be deleted and false if not.
func (est *EndpointSliceTracker) HandleDeletion(endpointSlice *discovery.EndpointSlice) bool {
	est.lock.Lock()
	defer est.lock.Unlock()

	gfs, ok := est.GenerationsForSliceUnsafe(endpointSlice)

	if ok {
		g, ok := gfs[endpointSlice.UID]
		delete(gfs, endpointSlice.UID)
		if ok && g != deletionExpected {
			return false
		}
	}

	return true
}

// GenerationsForSliceUnsafe returns the generations for the Service
// corresponding to the provided EndpointSlice, and a bool to indicate if it
// exists. A lock must be applied before calling this function.
func (est *EndpointSliceTracker) GenerationsForSliceUnsafe(endpointSlice *discovery.EndpointSlice) (GenerationsBySlice, bool) {
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
