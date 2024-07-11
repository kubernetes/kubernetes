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

package metrics

import (
	"math"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	endpointsliceutil "k8s.io/endpointslice/util"
)

// NewCache returns a new Cache with the specified endpointsPerSlice.
func NewCache(endpointsPerSlice int32, endpointSliceMetrics *EndpointSliceMetrics, placeholderEnabled bool) *Cache {
	return &Cache{
		endpointSliceMetrics:          endpointSliceMetrics,
		maxEndpointsPerSlice:          endpointsPerSlice,
		cache:                         map[types.NamespacedName]*ObjectPortCache{},
		servicesByTrafficDistribution: make(map[string]map[types.NamespacedName]bool),
		placeholderEnabled:            placeholderEnabled,
	}
}

// Cache tracks values for total numbers of desired endpoints as well as the
// efficiency of EndpointSlice endpoints distribution.
type Cache struct {
	endpointSliceMetrics *EndpointSliceMetrics

	// maxEndpointsPerSlice references the maximum number of endpoints that
	// should be added to an EndpointSlice.
	maxEndpointsPerSlice int32

	// lock protects changes to numEndpoints, numSlicesActual, numSlicesDesired,
	// cache and servicesByTrafficDistribution
	lock sync.Mutex
	// numEndpoints represents the total number of endpoints stored in
	// EndpointSlices.
	numEndpoints int
	// numSlicesActual represents the total number of EndpointSlices.
	numSlicesActual int
	// numSlicesDesired represents the desired number of EndpointSlices.
	numSlicesDesired int
	// cache stores a ObjectPortCache grouped by NamespacedNames representing
	// objects (services, entpoints...).
	cache map[types.NamespacedName]*ObjectPortCache
	// Tracks all services partitioned by their trafficDistribution field.
	//
	// The type should be read as map[trafficDistribution]setOfServices
	servicesByTrafficDistribution map[string]map[types.NamespacedName]bool
	// placeholderEnabled indicates if the placeholder endpointslices must
	// be created or not.
	placeholderEnabled bool
}

const (
	// Label value for cases when service.spec.trafficDistribution is set to an
	// unknown value.
	trafficDistributionImplementationSpecific = "ImplementationSpecific"
)

// ObjectPortCache tracks values for total numbers of desired endpoints as well
// as the efficiency of EndpointSlice endpoints distribution for each unique
// object (services, endpoints...) Port combination.
type ObjectPortCache struct {
	items map[endpointsliceutil.PortMapKey]EfficiencyInfo
}

// EfficiencyInfo stores the number of Endpoints and Slices for calculating
// total numbers of desired endpoints and the efficiency of EndpointSlice
// endpoints distribution.
type EfficiencyInfo struct {
	Endpoints int
	Slices    int
}

// NewObjectPortCache initializes and returns a new ObjectPortCache.
func NewObjectPortCache() *ObjectPortCache {
	return &ObjectPortCache{
		items: map[endpointsliceutil.PortMapKey]EfficiencyInfo{},
	}
}

// Set updates the ObjectPortCache to contain the provided EfficiencyInfo
// for the provided PortMapKey.
func (opc *ObjectPortCache) Set(pmKey endpointsliceutil.PortMapKey, eInfo EfficiencyInfo) {
	opc.items[pmKey] = eInfo
}

// totals returns the total number of endpoints and slices represented by a
// ObjectPortCache.
func (opc *ObjectPortCache) totals(maxEndpointsPerSlice int, placeholderEnabled bool) (int, int, int) {
	var actualSlices, desiredSlices, endpoints int
	for _, eInfo := range opc.items {
		endpoints += eInfo.Endpoints
		actualSlices += eInfo.Slices
		desiredSlices += numDesiredSlices(eInfo.Endpoints, maxEndpointsPerSlice)
	}
	// there is always a placeholder slice
	if placeholderEnabled && desiredSlices == 0 {
		desiredSlices = 1
	}
	return actualSlices, desiredSlices, endpoints
}

// UpdateObjectPortCache updates a ObjectPortCache in the global cache for a
// given Object (Service, Endpoints...) and updates the corresponding metrics.
// Parameters:
// * objectNN refers to a NamespacedName representing the Object (Service, Endpoints...).
// * spCache refers to a ObjectPortCache for the specified Object (Service, Endpoints...).
func (c *Cache) UpdateObjectPortCache(objectNN types.NamespacedName, spCache *ObjectPortCache) {
	c.lock.Lock()
	defer c.lock.Unlock()

	var prevActualSlices, prevDesiredSlices, prevEndpoints int
	if existingSPCache, ok := c.cache[objectNN]; ok {
		prevActualSlices, prevDesiredSlices, prevEndpoints = existingSPCache.totals(int(c.maxEndpointsPerSlice), c.placeholderEnabled)
	}

	currActualSlices, currDesiredSlices, currEndpoints := spCache.totals(int(c.maxEndpointsPerSlice), c.placeholderEnabled)
	// To keep numEndpoints up to date, add the difference between the number of
	// endpoints in the provided spCache and any spCache it might be replacing.
	c.numEndpoints = c.numEndpoints + currEndpoints - prevEndpoints

	c.numSlicesDesired += currDesiredSlices - prevDesiredSlices
	c.numSlicesActual += currActualSlices - prevActualSlices

	c.cache[objectNN] = spCache
	c.updateMetrics()
}

func (c *Cache) UpdateTrafficDistributionForService(objectNN types.NamespacedName, trafficDistributionPtr *string) {
	c.lock.Lock()
	defer c.lock.Unlock()

	defer c.updateMetrics()

	for _, serviceSet := range c.servicesByTrafficDistribution {
		delete(serviceSet, objectNN)
	}

	if trafficDistributionPtr == nil {
		return
	}

	trafficDistribution := *trafficDistributionPtr
	// If we don't explicitly recognize a value for trafficDistribution, it should
	// be treated as an implementation specific value. All such implementation
	// specific values should use the label value "ImplementationSpecific" to not
	// explode the metric labels cardinality.
	if trafficDistribution != corev1.ServiceTrafficDistributionPreferClose {
		trafficDistribution = trafficDistributionImplementationSpecific
	}
	serviceSet, ok := c.servicesByTrafficDistribution[trafficDistribution]
	if !ok {
		serviceSet = make(map[types.NamespacedName]bool)
		c.servicesByTrafficDistribution[trafficDistribution] = serviceSet
	}
	serviceSet[objectNN] = true
}

// DeleteObject removes references of a Object (Service, Endpoints...) from the global cache and
// updates the corresponding metrics.
func (c *Cache) DeleteObject(objectNN types.NamespacedName) {
	c.lock.Lock()
	defer c.lock.Unlock()

	for _, serviceSet := range c.servicesByTrafficDistribution {
		delete(serviceSet, objectNN)
	}

	if spCache, ok := c.cache[objectNN]; ok {
		actualSlices, desiredSlices, endpoints := spCache.totals(int(c.maxEndpointsPerSlice), c.placeholderEnabled)
		c.numEndpoints = c.numEndpoints - endpoints
		c.numSlicesDesired -= desiredSlices
		c.numSlicesActual -= actualSlices
		c.updateMetrics()
		delete(c.cache, objectNN)
	}
}

// updateMetrics updates metrics with the values from this Cache.
// Must be called holding lock.
func (c *Cache) updateMetrics() {
	c.endpointSliceMetrics.NumEndpointSlices.WithLabelValues().Set(float64(c.numSlicesActual))
	c.endpointSliceMetrics.DesiredEndpointSlices.WithLabelValues().Set(float64(c.numSlicesDesired))
	c.endpointSliceMetrics.EndpointsDesired.WithLabelValues().Set(float64(c.numEndpoints))

	c.endpointSliceMetrics.ServicesCountByTrafficDistribution.Reset()
	for trafficDistribution, services := range c.servicesByTrafficDistribution {
		c.endpointSliceMetrics.ServicesCountByTrafficDistribution.WithLabelValues(trafficDistribution).Set(float64(len(services)))
	}
}

// numDesiredSlices calculates the number of EndpointSlices that would exist
// with ideal endpoint distribution.
func numDesiredSlices(numEndpoints, maxEndpointsPerSlice int) int {
	if numEndpoints == 0 {
		return 0
	}
	if numEndpoints <= maxEndpointsPerSlice {
		return 1
	}
	return int(math.Ceil(float64(numEndpoints) / float64(maxEndpointsPerSlice)))
}
