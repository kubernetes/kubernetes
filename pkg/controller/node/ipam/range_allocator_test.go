/*
Copyright 2016 The Kubernetes Authors.

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
	"net"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/controller/testutil"
)

const (
	nodePollInterval = 100 * time.Millisecond
)

func waitForUpdatedNodeWithTimeout(nodeHandler *testutil.FakeNodeHandler, number int, timeout time.Duration) error {
	return wait.Poll(nodePollInterval, timeout, func() (bool, error) {
		if len(nodeHandler.GetUpdatedNodesCopy()) >= number {
			return true, nil
		}
		return false, nil
	})
}

func TestAllocateOrOccupyCIDRSuccess(t *testing.T) {
	testCases := []struct {
		description           string
		fakeNodeHandler       *testutil.FakeNodeHandler
		clusterCIDR           []*net.IPNet
		serviceCIDR           *net.IPNet
		subNetMaskSize        []int
		expectedAllocatedCIDR string
		allocatedCIDRs        []string
	}{
		{
			description: "When there's no ServiceCIDR return first CIDR in range",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			clusterCIDR: func() []*net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/24")
				c := []*net.IPNet{clusterCIDR}
				return c
			}(),
			serviceCIDR:           nil,
			subNetMaskSize:        []int{30},
			expectedAllocatedCIDR: "127.123.234.0/30",
		},
		{
			description: "When there's no ServiceCIDR return first CIDR in range for v6",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			clusterCIDR: func() []*net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("2001:abcd:1234:5600::/56")
				c := []*net.IPNet{clusterCIDR}
				return c
			}(),
			serviceCIDR:           nil,
			subNetMaskSize:        []int{62},
			expectedAllocatedCIDR: "2001:abcd:1234:5600::/62",
		},
		{
			description: "Correctly filter out ServiceCIDR",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			clusterCIDR: func() []*net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/24")
				_, v6clusterCIDR, _ := net.ParseCIDR("2001:abcd:1234:5600::/56")
				c := []*net.IPNet{clusterCIDR, v6clusterCIDR}
				return c
			}(),
			serviceCIDR: func() *net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/26")
				return clusterCIDR
			}(),
			subNetMaskSize: []int{30, 62},
			// it should return first /30 CIDR after service range
			expectedAllocatedCIDR: "127.123.234.64/30",
		},
		{
			description: "Correctly filter out ServiceCIDR for v6",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			clusterCIDR: func() []*net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("2001:abcd:1234:5600::/56")
				c := []*net.IPNet{clusterCIDR}
				return c
			}(),
			serviceCIDR: func() *net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("2001:abcd:1234:5600::/58")
				return clusterCIDR
			}(),
			subNetMaskSize: []int{62},
			// it should return first /62 CIDR after service range
			expectedAllocatedCIDR: "2001:abcd:1234:5640::/62",
		},
		{
			description: "Correctly ignore already allocated CIDRs",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			clusterCIDR: func() []*net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/24")
				c := []*net.IPNet{clusterCIDR}
				return c
			}(),
			serviceCIDR: func() *net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/26")
				return clusterCIDR
			}(),
			subNetMaskSize:        []int{30},
			allocatedCIDRs:        []string{"127.123.234.64/30", "127.123.234.68/30", "127.123.234.72/30", "127.123.234.80/30"},
			expectedAllocatedCIDR: "127.123.234.76/30",
		},
		{
			description: "Correctly ignore already allocated CIDRs for v6",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			clusterCIDR: func() []*net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("2001:abcd:1234:5600::/56")
				c := []*net.IPNet{clusterCIDR}
				return c
			}(),
			serviceCIDR: func() *net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("2001:abcd:1234:5600::/58")
				return clusterCIDR
			}(),
			subNetMaskSize:        []int{62},
			allocatedCIDRs:        []string{"2001:abcd:1234:5640::/62", "2001:abcd:1234:5644::/62", "2001:abcd:1234:5648::/62", "2001:abcd:1234:5650::/62"},
			expectedAllocatedCIDR: "2001:abcd:1234:564c::/62",
		},
		{
			description: "Correctly ignore already allocated CIDRs for v4 and v6",
			// Note, only the first is allocated until nodes accept two
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			clusterCIDR: func() []*net.IPNet {
				_, v6clusterCIDR, _ := net.ParseCIDR("2001:abcd:1234:5600::/56")
				_, v4clusterCIDR, _ := net.ParseCIDR("127.123.234.0/24")
				c := []*net.IPNet{v6clusterCIDR, v4clusterCIDR}
				return c
			}(),
			serviceCIDR: func() *net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("2001:abcd:1234:5600::/58")
				return clusterCIDR
			}(),
			subNetMaskSize:        []int{62, 30},
			allocatedCIDRs:        []string{"2001:abcd:1234:5640::/62", "2001:abcd:1234:5644::/62", "2001:abcd:1234:5648::/62", "2001:abcd:1234:5650::/62"},
			expectedAllocatedCIDR: "2001:abcd:1234:564c::/62",
		},
	}

	testFunc := func(tc struct {
		description           string
		fakeNodeHandler       *testutil.FakeNodeHandler
		clusterCIDR           []*net.IPNet
		serviceCIDR           *net.IPNet
		subNetMaskSize        []int
		expectedAllocatedCIDR string
		allocatedCIDRs        []string
	}) {
		allocator, _ := NewCIDRRangeAllocator(tc.fakeNodeHandler, tc.clusterCIDR, tc.serviceCIDR, tc.subNetMaskSize, nil)
		// this is a bit of white box testing
		for _, allocated := range tc.allocatedCIDRs {
			_, cidr, err := net.ParseCIDR(allocated)
			if err != nil {
				t.Fatalf("%v: unexpected error when parsing CIDR %v: %v", tc.description, allocated, err)
			}
			rangeAllocator, ok := allocator.(*rangeAllocator)
			if !ok {
				t.Logf("%v: found non-default implementation of CIDRAllocator, skipping white-box test...", tc.description)
				return
			}
			rangeAllocator.recorder = testutil.NewFakeRecorder()
			for _, c := range rangeAllocator.cidrs {
				if c.Contains(cidr.IP) {
					if err = c.Occupy(cidr); err != nil {
						t.Fatalf("%v: unexpected error when occupying CIDR %v: %v", tc.description, allocated, err)
					}
				}
			}
		}
		if err := allocator.AllocateOrOccupyCIDR(tc.fakeNodeHandler.Existing[0]); err != nil {
			t.Errorf("%v: unexpected error in AllocateOrOccupyCIDR: %v", tc.description, err)
		}
		if err := waitForUpdatedNodeWithTimeout(tc.fakeNodeHandler, 1, wait.ForeverTestTimeout); err != nil {
			t.Fatalf("%v: timeout while waiting for Node update: %v", tc.description, err)
		}
		found := false
		seenCIDRs := []string{}
		for _, updatedNode := range tc.fakeNodeHandler.GetUpdatedNodesCopy() {
			seenCIDRs = append(seenCIDRs, updatedNode.Spec.PodCIDR)
			if updatedNode.Spec.PodCIDR == tc.expectedAllocatedCIDR {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("%v: Unable to find allocated CIDR %v, found updated Nodes with CIDRs: %v",
				tc.description, tc.expectedAllocatedCIDR, seenCIDRs)
		}
	}

	for _, tc := range testCases {
		testFunc(tc)
	}
}

func TestAllocateOrOccupyCIDRFailure(t *testing.T) {
	testCases := []struct {
		description     string
		fakeNodeHandler *testutil.FakeNodeHandler
		clusterCIDR     []*net.IPNet
		serviceCIDR     *net.IPNet
		subNetMaskSize  []int
		allocatedCIDRs  []string
	}{
		{
			description: "When there's no ServiceCIDR return first CIDR in range",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			clusterCIDR: func() []*net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/28")
				c := []*net.IPNet{clusterCIDR}
				return c
			}(),
			serviceCIDR:    nil,
			subNetMaskSize: []int{30},
			allocatedCIDRs: []string{"127.123.234.0/30", "127.123.234.4/30", "127.123.234.8/30", "127.123.234.12/30"},
		},
	}

	testFunc := func(tc struct {
		description     string
		fakeNodeHandler *testutil.FakeNodeHandler
		clusterCIDR     []*net.IPNet
		serviceCIDR     *net.IPNet
		subNetMaskSize  []int
		allocatedCIDRs  []string
	}) {
		allocator, _ := NewCIDRRangeAllocator(tc.fakeNodeHandler, tc.clusterCIDR, tc.serviceCIDR, tc.subNetMaskSize, nil)
		// this is a bit of white box testing
		for _, allocated := range tc.allocatedCIDRs {
			_, cidr, err := net.ParseCIDR(allocated)
			if err != nil {
				t.Fatalf("%v: unexpected error when parsing CIDR %v: %v", tc.description, allocated, err)
			}
			rangeAllocator, ok := allocator.(*rangeAllocator)
			if !ok {
				t.Logf("%v: found non-default implementation of CIDRAllocator, skipping white-box test...", tc.description)
				return
			}
			rangeAllocator.recorder = testutil.NewFakeRecorder()
			for _, c := range rangeAllocator.cidrs {
				if c.Contains(cidr.IP) {
					err = c.Occupy(cidr)
					if err != nil {
						t.Fatalf("%v: unexpected error when occupying CIDR %v: %v", tc.description, allocated, err)
					}
				}
			}
		}
		if err := allocator.AllocateOrOccupyCIDR(tc.fakeNodeHandler.Existing[0]); err == nil {
			t.Errorf("%v: unexpected success in AllocateOrOccupyCIDR: %v", tc.description, err)
		}
		// We don't expect any updates, so just sleep for some time
		time.Sleep(time.Second)
		if len(tc.fakeNodeHandler.GetUpdatedNodesCopy()) != 0 {
			t.Fatalf("%v: unexpected update of nodes: %v", tc.description, tc.fakeNodeHandler.GetUpdatedNodesCopy())
		}
		seenCIDRs := []string{}
		for _, updatedNode := range tc.fakeNodeHandler.GetUpdatedNodesCopy() {
			if updatedNode.Spec.PodCIDR != "" {
				seenCIDRs = append(seenCIDRs, updatedNode.Spec.PodCIDR)
			}
		}
		if len(seenCIDRs) != 0 {
			t.Errorf("%v: Seen assigned CIDRs when not expected: %v",
				tc.description, seenCIDRs)
		}
	}
	for _, tc := range testCases {
		testFunc(tc)
	}
}

func TestReleaseCIDRSuccess(t *testing.T) {
	testCases := []struct {
		description                      string
		fakeNodeHandler                  *testutil.FakeNodeHandler
		clusterCIDR                      []*net.IPNet
		serviceCIDR                      *net.IPNet
		subNetMaskSize                   []int
		expectedAllocatedCIDRFirstRound  string
		expectedAllocatedCIDRSecondRound string
		allocatedCIDRs                   []string
		cidrsToRelease                   []string
	}{
		{
			description: "Correctly release preallocated CIDR",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			clusterCIDR: func() []*net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/28")
				c := []*net.IPNet{clusterCIDR}
				return c
			}(),
			serviceCIDR:                      nil,
			subNetMaskSize:                   []int{30},
			allocatedCIDRs:                   []string{"127.123.234.0/30", "127.123.234.4/30", "127.123.234.8/30", "127.123.234.12/30"},
			expectedAllocatedCIDRFirstRound:  "",
			cidrsToRelease:                   []string{"127.123.234.4/30"},
			expectedAllocatedCIDRSecondRound: "127.123.234.4/30",
		},
		{
			description: "Correctly recycle CIDR",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			clusterCIDR: func() []*net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/28")
				c := []*net.IPNet{clusterCIDR}
				return c
				//return clusterCIDR
			}(),
			serviceCIDR:                      nil,
			subNetMaskSize:                   []int{30},
			expectedAllocatedCIDRFirstRound:  "127.123.234.0/30",
			cidrsToRelease:                   []string{"127.123.234.0/30"},
			expectedAllocatedCIDRSecondRound: "127.123.234.0/30",
		},
	}

	testFunc := func(tc struct {
		description                      string
		fakeNodeHandler                  *testutil.FakeNodeHandler
		clusterCIDR                      []*net.IPNet
		serviceCIDR                      *net.IPNet
		subNetMaskSize                   []int
		expectedAllocatedCIDRFirstRound  string
		expectedAllocatedCIDRSecondRound string
		allocatedCIDRs                   []string
		cidrsToRelease                   []string
	}) {
		allocator, _ := NewCIDRRangeAllocator(tc.fakeNodeHandler, tc.clusterCIDR, tc.serviceCIDR, tc.subNetMaskSize, nil)
		// this is a bit of white box testing
		for _, allocated := range tc.allocatedCIDRs {
			_, cidr, err := net.ParseCIDR(allocated)
			if err != nil {
				t.Fatalf("%v: unexpected error when parsing CIDR %v: %v", tc.description, allocated, err)
			}
			rangeAllocator, ok := allocator.(*rangeAllocator)
			if !ok {
				t.Logf("%v: found non-default implementation of CIDRAllocator, skipping white-box test...", tc.description)
				return
			}
			rangeAllocator.recorder = testutil.NewFakeRecorder()
			err = rangeAllocator.cidrs[0].Occupy(cidr)
			if err != nil {
				t.Fatalf("%v: unexpected error when occupying CIDR %v: %v", tc.description, allocated, err)
			}
		}
		err := allocator.AllocateOrOccupyCIDR(tc.fakeNodeHandler.Existing[0])
		if tc.expectedAllocatedCIDRFirstRound != "" {
			if err != nil {
				t.Fatalf("%v: unexpected error in AllocateOrOccupyCIDR: %v", tc.description, err)
			}
			if err := waitForUpdatedNodeWithTimeout(tc.fakeNodeHandler, 1, wait.ForeverTestTimeout); err != nil {
				t.Fatalf("%v: timeout while waiting for Node update: %v", tc.description, err)
			}
		} else {
			if err == nil {
				t.Fatalf("%v: unexpected success in AllocateOrOccupyCIDR: %v", tc.description, err)
			}
			// We don't expect any updates here
			time.Sleep(time.Second)
			if len(tc.fakeNodeHandler.GetUpdatedNodesCopy()) != 0 {
				t.Fatalf("%v: unexpected update of nodes: %v", tc.description, tc.fakeNodeHandler.GetUpdatedNodesCopy())
			}
		}

		for _, cidrToRelease := range tc.cidrsToRelease {
			nodeToRelease := v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node0",
				},
			}
			nodeToRelease.Spec.PodCIDR = cidrToRelease
			err = allocator.ReleaseCIDR(&nodeToRelease)
			if err != nil {
				t.Fatalf("%v: unexpected error in ReleaseCIDR: %v", tc.description, err)
			}
		}

		if err = allocator.AllocateOrOccupyCIDR(tc.fakeNodeHandler.Existing[0]); err != nil {
			t.Fatalf("%v: unexpected error in AllocateOrOccupyCIDR: %v", tc.description, err)
		}
		if err := waitForUpdatedNodeWithTimeout(tc.fakeNodeHandler, 1, wait.ForeverTestTimeout); err != nil {
			t.Fatalf("%v: timeout while waiting for Node update: %v", tc.description, err)
		}

		found := false
		seenCIDRs := []string{}
		for _, updatedNode := range tc.fakeNodeHandler.GetUpdatedNodesCopy() {
			seenCIDRs = append(seenCIDRs, updatedNode.Spec.PodCIDR)
			if updatedNode.Spec.PodCIDR == tc.expectedAllocatedCIDRSecondRound {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("%v: Unable to find allocated CIDR %v, found updated Nodes with CIDRs: %v",
				tc.description, tc.expectedAllocatedCIDRSecondRound, seenCIDRs)
		}
	}
	for _, tc := range testCases {
		testFunc(tc)
	}
}
