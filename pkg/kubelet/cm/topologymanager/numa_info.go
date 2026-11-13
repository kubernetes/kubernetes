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

package topologymanager

import (
	"fmt"
	"runtime"

	cadvisorapi "github.com/google/cadvisor/lib/model"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

type NUMADistances map[int][]uint64

type NUMAInfo struct {
	Nodes         []int
	NUMADistances NUMADistances
}

func NewNUMAInfo(topology []cadvisorapi.Node, opts PolicyOptions) (*NUMAInfo, error) {
	var numaNodes []int
	distances := map[int][]uint64{}
	for _, node := range topology {
		numaNodes = append(numaNodes, node.Id)

		var nodeDistance []uint64
		if opts.PreferClosestNUMA {
			nodeDistance = node.Distances
			if nodeDistance == nil {
				if runtime.GOOS == "windows" {
					return nil, fmt.Errorf("the %q policy option is not supported on Windows because NUMA node distances are not available", PreferClosestNUMANodes)
				}
				return nil, fmt.Errorf("error getting NUMA distances from cadvisor")
			}
		}
		distances[node.Id] = nodeDistance
	}

	numaInfo := &NUMAInfo{
		Nodes:         numaNodes,
		NUMADistances: distances,
	}

	return numaInfo, nil
}

// Narrowest returns the narrower of two bitmasks. The second return value is
// true when both masks have the same width (equally fit under the narrowest
// criterion), and the result is chosen by an arbitrary tiebreaker.
func (n *NUMAInfo) Narrowest(m1 bitmask.BitMask, m2 bitmask.BitMask) (bitmask.BitMask, bool) {
	equal := m1.Count() == m2.Count()
	if m1.IsNarrowerThan(m2) {
		return m1, equal
	}
	return m2, equal
}

// Closest returns the mask with the shortest average distance between its NUMA
// nodes. The second return value is true when both masks are equally fit under
// the closest criterion (same width and same average distance), and the result
// is chosen by an arbitrary tiebreaker.
func (n *NUMAInfo) Closest(m1 bitmask.BitMask, m2 bitmask.BitMask) (bitmask.BitMask, bool) {
	// If the length of both bitmasks aren't the same, choose the one that is narrowest.
	if m1.Count() != m2.Count() {
		return n.Narrowest(m1, m2)
	}

	m1Distance := n.NUMADistances.CalculateAverageFor(m1)
	m2Distance := n.NUMADistances.CalculateAverageFor(m2)
	// If average distance is the same, take bitmask with more lower-number bits set.
	// The masks are equally fit: same width, same distance.
	if m1Distance == m2Distance {
		if m1.IsLessThan(m2) {
			return m1, true
		}
		return m2, true
	}

	// Otherwise, return the bitmask with the shortest average distance between NUMA nodes.
	// Not equally fit: one has a better (shorter) distance.
	if m1Distance < m2Distance {
		return m1, false
	}

	return m2, false
}

func (n NUMAInfo) DefaultAffinityMask() bitmask.BitMask {
	defaultAffinity, _ := bitmask.NewBitMask(n.Nodes...)
	return defaultAffinity
}

func (d NUMADistances) CalculateAverageFor(bm bitmask.BitMask) float64 {
	// This should never happen, but just in case make sure we do not divide by zero.
	if bm.Count() == 0 {
		return 0
	}

	var count float64 = 0
	var sum float64 = 0
	for _, node1 := range bm.GetBits() {
		for _, node2 := range bm.GetBits() {
			sum += float64(d[node1][node2])
			count++
		}
	}

	return sum / count
}
