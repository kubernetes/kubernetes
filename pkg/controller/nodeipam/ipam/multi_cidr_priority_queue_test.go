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
	"container/heap"
	"testing"

	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam/multicidrset"
	utilnet "k8s.io/utils/net"
)

func createTestPriorityQueueItem(name, cidr, selectorString string, labelMatchCount, perNodeHostBits int) *PriorityQueueItem {
	_, clusterCIDR, _ := utilnet.ParseCIDRSloppy(cidr)
	cidrSet, _ := multicidrset.NewMultiCIDRSet(clusterCIDR, perNodeHostBits)

	return &PriorityQueueItem{
		clusterCIDR: &multicidrset.ClusterCIDR{
			Name:        name,
			IPv4CIDRSet: cidrSet,
		},
		labelMatchCount: labelMatchCount,
		selectorString:  selectorString,
	}
}

func TestPriorityQueue(t *testing.T) {

	pqi1 := createTestPriorityQueueItem("cidr1", "192.168.0.0/16", "foo=bar,name=test1", 1, 8)
	pqi2 := createTestPriorityQueueItem("cidr2", "10.1.0.0/24", "foo=bar,name=test2", 2, 8)
	pqi3 := createTestPriorityQueueItem("cidr3", "172.16.0.0/16", "foo=bar,name=test3", 2, 8)
	pqi4 := createTestPriorityQueueItem("cidr4", "10.1.1.0/26", "abc=bar,name=test4", 2, 6)
	pqi5 := createTestPriorityQueueItem("cidr5", "10.1.2.0/26", "foo=bar,name=test5", 2, 6)
	pqi6 := createTestPriorityQueueItem("cidr6", "10.1.3.0/26", "abc=bar,name=test4", 2, 6)

	for _, testQueue := range []struct {
		name  string
		items []*PriorityQueueItem
		want  *PriorityQueueItem
	}{
		{"Test queue with single item", []*PriorityQueueItem{pqi1}, pqi1},
		{"Test queue with items having different labelMatchCount", []*PriorityQueueItem{pqi1, pqi2}, pqi2},
		{"Test queue with items having same labelMatchCount, different max Allocatable Pod CIDRs", []*PriorityQueueItem{pqi1, pqi2, pqi3}, pqi2},
		{"Test queue with items having same labelMatchCount, max Allocatable Pod CIDRs, different PerNodeMaskSize", []*PriorityQueueItem{pqi1, pqi2, pqi4}, pqi4},
		{"Test queue with items having same labelMatchCount, max Allocatable Pod CIDRs, PerNodeMaskSize, different labels", []*PriorityQueueItem{pqi1, pqi2, pqi4, pqi5}, pqi4},
		{"Test queue with items having same labelMatchCount, max Allocatable Pod CIDRs, PerNodeMaskSize, labels, different IP addresses", []*PriorityQueueItem{pqi1, pqi2, pqi4, pqi5, pqi6}, pqi4},
	} {
		pq := make(PriorityQueue, 0)
		for _, pqi := range testQueue.items {
			heap.Push(&pq, pqi)
		}

		got := heap.Pop(&pq)

		if got != testQueue.want {
			t.Errorf("Error, wanted: %+v, got: %+v", testQueue.want, got)
		}
	}
}

func TestLess(t *testing.T) {

	for _, testQueue := range []struct {
		name  string
		items []*PriorityQueueItem
		want  bool
	}{
		{
			name: "different labelMatchCount, i higher priority than j",
			items: []*PriorityQueueItem{
				createTestPriorityQueueItem("cidr1", "192.168.0.0/16", "foo=bar,name=test1", 2, 8),
				createTestPriorityQueueItem("cidr2", "10.1.0.0/24", "foo=bar,name=test2", 1, 8),
			},
			want: true,
		},
		{
			name: "different labelMatchCount, i lower priority than j",
			items: []*PriorityQueueItem{
				createTestPriorityQueueItem("cidr1", "192.168.0.0/16", "foo=bar,name=test1", 1, 8),
				createTestPriorityQueueItem("cidr2", "10.1.0.0/24", "foo=bar,name=test2", 2, 8),
			},
			want: false,
		},
		{
			name: "same labelMatchCount, different max allocatable cidrs, i higher priority than j",
			items: []*PriorityQueueItem{
				createTestPriorityQueueItem("cidr2", "10.1.0.0/24", "foo=bar,name=test2", 2, 8),
				createTestPriorityQueueItem("cidr3", "172.16.0.0/16", "foo=bar,name=test3", 2, 8),
			},
			want: true,
		},
		{
			name: "same labelMatchCount, different max allocatable cidrs, i lower priority than j",
			items: []*PriorityQueueItem{
				createTestPriorityQueueItem("cidr2", "10.1.0.0/16", "foo=bar,name=test2", 2, 8),
				createTestPriorityQueueItem("cidr3", "172.16.0.0/24", "foo=bar,name=test3", 2, 8),
			},
			want: false,
		},
		{
			name: "same labelMatchCount, max allocatable cidrs, different PerNodeMaskSize i higher priority than j",
			items: []*PriorityQueueItem{
				createTestPriorityQueueItem("cidr2", "10.1.0.0/26", "foo=bar,name=test2", 2, 6),
				createTestPriorityQueueItem("cidr4", "10.1.1.0/24", "abc=bar,name=test4", 2, 8),
			},
			want: true,
		},
		{
			name: "same labelMatchCount, max allocatable cidrs, different PerNodeMaskSize i lower priority than j",
			items: []*PriorityQueueItem{
				createTestPriorityQueueItem("cidr2", "10.1.0.0/24", "foo=bar,name=test2", 2, 8),
				createTestPriorityQueueItem("cidr4", "10.1.1.0/26", "abc=bar,name=test4", 2, 6),
			},
			want: false,
		},
		{
			name: "same labelMatchCount, max Allocatable Pod CIDRs, PerNodeMaskSize, different labels i higher priority than j",
			items: []*PriorityQueueItem{
				createTestPriorityQueueItem("cidr4", "10.1.1.0/26", "abc=bar,name=test4", 2, 6),
				createTestPriorityQueueItem("cidr5", "10.1.2.0/26", "foo=bar,name=test5", 2, 6),
			},
			want: true,
		},
		{
			name: "same labelMatchCount, max Allocatable Pod CIDRs, PerNodeMaskSize, different labels i lower priority than j",
			items: []*PriorityQueueItem{
				createTestPriorityQueueItem("cidr4", "10.1.1.0/26", "xyz=bar,name=test4", 2, 6),
				createTestPriorityQueueItem("cidr5", "10.1.2.0/26", "foo=bar,name=test5", 2, 6),
			},
			want: false,
		},
		{
			name: "same labelMatchCount, max Allocatable Pod CIDRs, PerNodeMaskSize, labels, different IP addresses i higher priority than j",
			items: []*PriorityQueueItem{
				createTestPriorityQueueItem("cidr4", "10.1.1.0/26", "abc=bar,name=test4", 2, 6),
				createTestPriorityQueueItem("cidr6", "10.1.3.0/26", "abc=bar,name=test4", 2, 6),
			},
			want: true,
		},
		{
			name: "same labelMatchCount, max Allocatable Pod CIDRs, PerNodeMaskSize, labels, different IP addresses i lower priority than j",
			items: []*PriorityQueueItem{
				createTestPriorityQueueItem("cidr4", "10.1.1.0/26", "xyz=bar,name=test4", 2, 6),
				createTestPriorityQueueItem("cidr6", "10.0.3.0/26", "abc=bar,name=test4", 2, 6),
			},
			want: false,
		},
	} {
		var pq PriorityQueue
		pq = testQueue.items
		got := pq.Less(0, 1)
		if got != testQueue.want {
			t.Errorf("Error, wanted: %v, got: %v\nTest %q \npq[0]: %+v \npq[1]: %+v ", testQueue.want, got, testQueue.name, pq[0], pq[1])
		}
	}
}
