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

package metrics

import (
	"math"
	"sync"

	"k8s.io/apimachinery/pkg/types"
	endpointutil "k8s.io/kubernetes/pkg/controller/util/endpoint"
)

// NewCache returns a new Cache with the specified endpointsPerSlice.
func NewCache(endpointsPerSlice int32) *Cache {
	return &Cache{
		maxEndpointsPerSlice: endpointsPerSlice,
		cache:                map[types.NamespacedName]*ServicePortCache{},
	}
}

// Cache tracks values for total numbers of desired endpoints as well as the
// efficiency of EndpointSlice endpoints distribution.
type Cache struct {
	// maxEndpointsPerSlice references the maximum number of endpoints that
	// should be added to an EndpointSlice.
	maxEndpointsPerSlice int32

	// lock protects changes to numEndpoints and cache.
	lock sync.Mutex
	// numEndpoints represents the total number of endpoints stored in
	// EndpointSlices.
	numEndpoints int
	// cache stores a ServicePortCache grouped by NamespacedNames representing
	// Services.
	cache map[types.NamespacedName]*ServicePortCache
}

// ServicePortCache tracks values for total numbers of desired endpoints as well
// as the efficiency of EndpointSlice endpoints distribution for each unique
// Service Port combination.
type ServicePortCache struct {
	items map[endpointutil.PortMapKey]EfficiencyInfo
}

// EfficiencyInfo stores the number of Endpoints and Slices for calculating
// total numbers of desired endpoints and the efficiency of EndpointSlice
// endpoints distribution.
type EfficiencyInfo struct {
	Endpoints int
	Slices    int
}

// NewServicePortCache initializes and returns a new ServicePortCache.
func NewServicePortCache() *ServicePortCache {
	return &ServicePortCache{
		items: map[endpointutil.PortMapKey]EfficiencyInfo{},
	}
}

// Set updates the ServicePortCache to contain the provided EfficiencyInfo
// for the provided PortMapKey.
func (spc *ServicePortCache) Set(pmKey endpointutil.PortMapKey, eInfo EfficiencyInfo) {
	spc.items[pmKey] = eInfo
}

// numEndpoints returns the total number of endpoints represented by a
// ServicePortCache.
func (spc *ServicePortCache) numEndpoints() int {
	num := 0
	for _, eInfo := range spc.items {
		num += eInfo.Endpoints
	}
	return num
}

// UpdateServicePortCache updates a ServicePortCache in the global cache for a
// given Service and updates the corresponding metrics.
// Parameters:
// * serviceNN refers to a NamespacedName representing the Service.
// * spCache refers to a ServicePortCache for the specified Service.
func (c *Cache) UpdateServicePortCache(serviceNN types.NamespacedName, spCache *ServicePortCache) {
	c.lock.Lock()
	defer c.lock.Unlock()

	prevNumEndpoints := 0
	if existingSPCache, ok := c.cache[serviceNN]; ok {
		prevNumEndpoints = existingSPCache.numEndpoints()
	}

	currNumEndpoints := spCache.numEndpoints()
	// To keep numEndpoints up to date, add the difference between the number of
	// endpoints in the provided spCache and any spCache it might be replacing.
	c.numEndpoints = c.numEndpoints + currNumEndpoints - prevNumEndpoints

	c.cache[serviceNN] = spCache
	c.updateMetrics()
}

// DeleteService removes references of a Service from the global cache and
// updates the corresponding metrics.
func (c *Cache) DeleteService(serviceNN types.NamespacedName) {
	c.lock.Lock()
	defer c.lock.Unlock()

	if spCache, ok := c.cache[serviceNN]; ok {
		c.numEndpoints = c.numEndpoints - spCache.numEndpoints()
		delete(c.cache, serviceNN)
		c.updateMetrics()
	}
}

// metricsUpdate stores a desired and actual number of EndpointSlices.
type metricsUpdate struct {
	desired, actual int
}

// desiredAndActualSlices returns a metricsUpdate with the desired and actual
// number of EndpointSlices given the current values in the cache.
// Must be called holding lock.
func (c *Cache) desiredAndActualSlices() metricsUpdate {
	mUpdate := metricsUpdate{}
	for _, spCache := range c.cache {
		for _, eInfo := range spCache.items {
			mUpdate.actual += eInfo.Slices
			mUpdate.desired += numDesiredSlices(eInfo.Endpoints, int(c.maxEndpointsPerSlice))
		}
	}
	return mUpdate
}

// updateMetrics updates metrics with the values from this Cache.
// Must be called holding lock.
func (c *Cache) updateMetrics() {
	mUpdate := c.desiredAndActualSlices()
	NumEndpointSlices.WithLabelValues().Set(float64(mUpdate.actual))
	DesiredEndpointSlices.WithLabelValues().Set(float64(mUpdate.desired))
	EndpointsDesired.WithLabelValues().Set(float64(c.numEndpoints))
}

// numDesiredSlices calculates the number of EndpointSlices that would exist
// with ideal endpoint distribution.
func numDesiredSlices(numEndpoints, maxPerSlice int) int {
	if numEndpoints <= maxPerSlice {
		return 1
	}
	return int(math.Ceil(float64(numEndpoints) / float64(maxPerSlice)))
}
