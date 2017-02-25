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

package node

import (
	"net"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/controller/node/testutil"
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
		clusterCIDR           *net.IPNet
		serviceCIDR           *net.IPNet
		subNetMaskSize        int
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
			clusterCIDR: func() *net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/24")
				return clusterCIDR
			}(),
			serviceCIDR:           nil,
			subNetMaskSize:        30,
			expectedAllocatedCIDR: "127.123.234.0/30",
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
			clusterCIDR: func() *net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/24")
				return clusterCIDR
			}(),
			serviceCIDR: func() *net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/26")
				return clusterCIDR
			}(),
			subNetMaskSize: 30,
			// it should return first /30 CIDR after service range
			expectedAllocatedCIDR: "127.123.234.64/30",
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
			clusterCIDR: func() *net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/24")
				return clusterCIDR
			}(),
			serviceCIDR: func() *net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/26")
				return clusterCIDR
			}(),
			subNetMaskSize:        30,
			allocatedCIDRs:        []string{"127.123.234.64/30", "127.123.234.68/30", "127.123.234.72/30", "127.123.234.80/30"},
			expectedAllocatedCIDR: "127.123.234.76/30",
		},
	}

	testFunc := func(tc struct {
		description           string
		fakeNodeHandler       *testutil.FakeNodeHandler
		clusterCIDR           *net.IPNet
		serviceCIDR           *net.IPNet
		subNetMaskSize        int
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
			if err = rangeAllocator.cidrs.occupy(cidr); err != nil {
				t.Fatalf("%v: unexpected error when occupying CIDR %v: %v", tc.description, allocated, err)
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
		clusterCIDR     *net.IPNet
		serviceCIDR     *net.IPNet
		subNetMaskSize  int
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
			clusterCIDR: func() *net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/28")
				return clusterCIDR
			}(),
			serviceCIDR:    nil,
			subNetMaskSize: 30,
			allocatedCIDRs: []string{"127.123.234.0/30", "127.123.234.4/30", "127.123.234.8/30", "127.123.234.12/30"},
		},
	}

	testFunc := func(tc struct {
		description     string
		fakeNodeHandler *testutil.FakeNodeHandler
		clusterCIDR     *net.IPNet
		serviceCIDR     *net.IPNet
		subNetMaskSize  int
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
			err = rangeAllocator.cidrs.occupy(cidr)
			if err != nil {
				t.Fatalf("%v: unexpected error when occupying CIDR %v: %v", tc.description, allocated, err)
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
		clusterCIDR                      *net.IPNet
		serviceCIDR                      *net.IPNet
		subNetMaskSize                   int
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
			clusterCIDR: func() *net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/28")
				return clusterCIDR
			}(),
			serviceCIDR:                      nil,
			subNetMaskSize:                   30,
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
			clusterCIDR: func() *net.IPNet {
				_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/28")
				return clusterCIDR
			}(),
			serviceCIDR:                      nil,
			subNetMaskSize:                   30,
			expectedAllocatedCIDRFirstRound:  "127.123.234.0/30",
			cidrsToRelease:                   []string{"127.123.234.0/30"},
			expectedAllocatedCIDRSecondRound: "127.123.234.0/30",
		},
	}

	testFunc := func(tc struct {
		description                      string
		fakeNodeHandler                  *testutil.FakeNodeHandler
		clusterCIDR                      *net.IPNet
		serviceCIDR                      *net.IPNet
		subNetMaskSize                   int
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
			err = rangeAllocator.cidrs.occupy(cidr)
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
