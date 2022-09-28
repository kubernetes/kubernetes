/*
Copyright 2022 The Kubernetes Authors.

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

package ipam

import (
	"math"

	cidrset "k8s.io/kubernetes/pkg/controller/nodeipam/ipam/multicidrset"
)

// A PriorityQueue implementation based on https://pkg.go.dev/container/heap#example-package-PriorityQueue

// An PriorityQueueItem is something we manage in a priority queue.
type PriorityQueueItem struct {
	clusterCIDR *cidrset.ClusterCIDR
	// labelMatchCount is the first determinant of priority.
	labelMatchCount int
	// selectorString is a string representation of the labelSelector associated with the cidrSet.
	selectorString string
	// index is needed by update and is maintained by the heap.Interface methods.
	index int // The index of the item in the heap.
}

// A PriorityQueue implements heap.Interface and holds PriorityQueueItems.
type PriorityQueue []*PriorityQueueItem

func (pq PriorityQueue) Len() int { return len(pq) }

// Less compares the priority queue items, to store in a min heap.
// Less(i,j) == true denotes i has higher priority than j.
func (pq PriorityQueue) Less(i, j int) bool {
	if pq[i].labelMatchCount != pq[j].labelMatchCount {
		// P0: CidrSet with higher number of matching labels has the highest priority.
		return pq[i].labelMatchCount > pq[j].labelMatchCount
	}

	// If the count of matching labels is equal, compare the max allocatable pod CIDRs.
	if pq[i].maxAllocatable() != pq[j].maxAllocatable() {
		// P1: CidrSet with fewer allocatable pod CIDRs has higher priority.
		return pq[i].maxAllocatable() < pq[j].maxAllocatable()
	}

	// If the value of allocatable pod CIDRs is equal, compare the node mask size.
	if pq[i].nodeMaskSize() != pq[j].nodeMaskSize() {
		// P2: CidrSet with a PerNodeMaskSize having fewer IPs has higher priority.
		// For example, `27` (32 IPs) picked before `25` (128 IPs).
		return pq[i].nodeMaskSize() > pq[j].nodeMaskSize()
	}

	// If the per node mask size are equal compare the CIDR labels.
	if pq[i].selectorString != pq[j].selectorString {
		// P3: CidrSet having label with lower alphanumeric value has higher priority.
		return pq[i].selectorString < pq[j].selectorString
	}

	// P4: CidrSet having an alpha-numerically smaller IP address value has a higher priority.
	return pq[i].cidrLabel() < pq[j].cidrLabel()
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	if item, ok := x.(*PriorityQueueItem); ok {
		item.index = n
		*pq = append(*pq, item)
	}
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil  // avoid memory leak.
	item.index = -1 // for safety.
	*pq = old[0 : n-1]
	return item
}

// maxAllocatable computes the minimum value of the MaxCIDRs for a ClusterCIDR.
// It compares the MaxCIDRs for each CIDR family and returns the minimum.
// e.g. IPv4 - 10.0.0.0/16  PerNodeMaskSize: 24   MaxCIDRs = 256
// IPv6 - ff:ff::/120  PerNodeMaskSize: 120  MaxCIDRs = 1
// MaxAllocatable for this ClusterCIDR = 1
func (pqi *PriorityQueueItem) maxAllocatable() int {
	ipv4Allocatable := math.MaxInt
	ipv6Allocatable := math.MaxInt

	if pqi.clusterCIDR.IPv4CIDRSet != nil {
		ipv4Allocatable = pqi.clusterCIDR.IPv4CIDRSet.MaxCIDRs
	}

	if pqi.clusterCIDR.IPv6CIDRSet != nil {
		ipv6Allocatable = pqi.clusterCIDR.IPv6CIDRSet.MaxCIDRs
	}

	if ipv4Allocatable < ipv6Allocatable {
		return ipv4Allocatable
	}

	return ipv6Allocatable
}

// nodeMaskSize returns IPv4 NodeMaskSize if present, else returns IPv6 NodeMaskSize.
// Note the requirement: 32 - IPv4 NodeMaskSize == 128 - IPv6 NodeMaskSize
// Due to the above requirement it does not matter which NodeMaskSize we compare.
func (pqi *PriorityQueueItem) nodeMaskSize() int {
	if pqi.clusterCIDR.IPv4CIDRSet != nil {
		return pqi.clusterCIDR.IPv4CIDRSet.NodeMaskSize
	}

	return pqi.clusterCIDR.IPv6CIDRSet.NodeMaskSize
}

// cidrLabel returns IPv4 CIDR if present, else returns IPv6 CIDR.
func (pqi *PriorityQueueItem) cidrLabel() string {
	if pqi.clusterCIDR.IPv4CIDRSet != nil {
		return pqi.clusterCIDR.IPv4CIDRSet.Label
	}

	return pqi.clusterCIDR.IPv6CIDRSet.Label
}
