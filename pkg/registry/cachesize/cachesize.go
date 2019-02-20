/*
Copyright 2014 The Kubernetes Authors.

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

package cachesize

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// NewHeuristicWatchCacheSizes returns a map of suggested watch cache sizes based on total
// memory.
func NewHeuristicWatchCacheSizes(expectedRAMCapacityMB int) map[schema.GroupResource]int {
	// From our documentation, we officially recommend 120GB machines for
	// 2000 nodes, and we scale from that point. Thus we assume ~60MB of
	// capacity per node.
	// TODO: Revisit this heuristics
	clusterSize := expectedRAMCapacityMB / 60

	// We should specify cache size for a given resource only if it
	// is supposed to have non-default value.
	//
	// TODO: Figure out which resource we should have non-default value.
	watchCacheSizes := make(map[schema.GroupResource]int)
	watchCacheSizes[schema.GroupResource{Resource: "replicationcontrollers"}] = maxInt(5*clusterSize, 100)
	watchCacheSizes[schema.GroupResource{Resource: "endpoints"}] = maxInt(10*clusterSize, 1000)
	watchCacheSizes[schema.GroupResource{Resource: "nodes"}] = maxInt(5*clusterSize, 1000)
	watchCacheSizes[schema.GroupResource{Resource: "pods"}] = maxInt(50*clusterSize, 1000)
	watchCacheSizes[schema.GroupResource{Resource: "services"}] = maxInt(5*clusterSize, 1000)
	watchCacheSizes[schema.GroupResource{Resource: "events"}] = 0
	watchCacheSizes[schema.GroupResource{Resource: "apiservices", Group: "apiregistration.k8s.io"}] = maxInt(5*clusterSize, 1000)
	return watchCacheSizes
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
