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
	"context"
	"fmt"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	networkingv1alpha1 "k8s.io/api/networking/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	cidrset "k8s.io/kubernetes/pkg/controller/nodeipam/ipam/multicidrset"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam/test"
	"k8s.io/kubernetes/pkg/controller/testutil"
	utilnet "k8s.io/utils/net"
)

type testCaseMultiCIDR struct {
	description     string
	fakeNodeHandler *testutil.FakeNodeHandler
	allocatorParams CIDRAllocatorParams
	testCIDRMap     map[string][]*cidrset.ClusterCIDR
	// key is index of the cidr allocated.
	expectedAllocatedCIDR map[int]string
	allocatedCIDRs        map[int][]string
	// should controller creation fail?
	ctrlCreateFail bool
}

type testClusterCIDR struct {
	perNodeHostBits int32
	ipv4CIDR        string
	ipv6CIDR        string
	name            string
}

type testNodeSelectorRequirement struct {
	key      string
	operator v1.NodeSelectorOperator
	values   []string
}

func getTestNodeSelector(requirements []testNodeSelectorRequirement) string {
	testNodeSelector := &v1.NodeSelector{}

	for _, nsr := range requirements {
		nst := v1.NodeSelectorTerm{
			MatchExpressions: []v1.NodeSelectorRequirement{
				{
					Key:      nsr.key,
					Operator: nsr.operator,
					Values:   nsr.values,
				},
			},
		}
		testNodeSelector.NodeSelectorTerms = append(testNodeSelector.NodeSelectorTerms, nst)
	}

	selector, _ := nodeSelectorAsSelector(testNodeSelector)
	return selector.String()
}

func getTestCidrMap(testClusterCIDRMap map[string][]*testClusterCIDR) map[string][]*cidrset.ClusterCIDR {
	cidrMap := make(map[string][]*cidrset.ClusterCIDR, 0)
	for labels, testClusterCIDRList := range testClusterCIDRMap {
		clusterCIDRList := make([]*cidrset.ClusterCIDR, 0)
		for _, testClusterCIDR := range testClusterCIDRList {
			clusterCIDR := &cidrset.ClusterCIDR{
				Name:            testClusterCIDR.name,
				AssociatedNodes: make(map[string]bool, 0),
			}

			if testClusterCIDR.ipv4CIDR != "" {
				_, testCIDR, _ := utilnet.ParseCIDRSloppy(testClusterCIDR.ipv4CIDR)
				testCIDRSet, _ := cidrset.NewMultiCIDRSet(testCIDR, int(testClusterCIDR.perNodeHostBits))
				clusterCIDR.IPv4CIDRSet = testCIDRSet
			}
			if testClusterCIDR.ipv6CIDR != "" {
				_, testCIDR, _ := utilnet.ParseCIDRSloppy(testClusterCIDR.ipv6CIDR)
				testCIDRSet, _ := cidrset.NewMultiCIDRSet(testCIDR, int(testClusterCIDR.perNodeHostBits))
				clusterCIDR.IPv6CIDRSet = testCIDRSet
			}
			clusterCIDRList = append(clusterCIDRList, clusterCIDR)
		}
		cidrMap[labels] = clusterCIDRList
	}
	return cidrMap
}

func getClusterCIDRList(nodeName string, cidrMap map[string][]*cidrset.ClusterCIDR) ([]*cidrset.ClusterCIDR, error) {
	labelSelector := getTestNodeSelector([]testNodeSelectorRequirement{
		{
			key:      "testLabel-0",
			operator: v1.NodeSelectorOpIn,
			values:   []string{nodeName},
		},
	})
	if clusterCIDRList, ok := cidrMap[labelSelector]; ok {
		return clusterCIDRList, nil
	}
	return nil, fmt.Errorf("unable to get clusterCIDR for node: %s", nodeName)
}

func TestMultiCIDROccupyPreExistingCIDR(t *testing.T) {
	// all tests operate on a single node.
	testCaseMultiCIDRs := []testCaseMultiCIDR{
		{
			description: "success, single stack no node allocation",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR:          nil,
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "single-stack-cidr",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.10.0.0/16",
						},
					},
				}),
			allocatedCIDRs:        nil,
			expectedAllocatedCIDR: nil,
			ctrlCreateFail:        false,
		},
		{
			description: "success, dual stack no node allocation",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR:          nil,
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "dual-stack-cidr",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.10.0.0/16",
							ipv6CIDR:        "ace:cab:deca::/112",
						},
					},
				}),
			allocatedCIDRs:        nil,
			expectedAllocatedCIDR: nil,
			ctrlCreateFail:        false,
		},
		{
			description: "success, single stack correct node allocation",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
						Spec: v1.NodeSpec{
							PodCIDRs: []string{"10.10.0.1/24"},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR:          nil,
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "single-stack-cidr-allocated",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.10.0.0/16",
						},
					},
				}),
			allocatedCIDRs:        nil,
			expectedAllocatedCIDR: nil,
			ctrlCreateFail:        false,
		},
		{
			description: "success, dual stack both allocated correctly",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
						Spec: v1.NodeSpec{
							PodCIDRs: []string{"10.10.0.1/24", "ace:cab:deca::1/120"},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR:          nil,
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "dual-stack-cidr-allocated",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.10.0.0/16",
							ipv6CIDR:        "ace:cab:deca::/112",
						},
					},
				}),
			allocatedCIDRs:        nil,
			expectedAllocatedCIDR: nil,
			ctrlCreateFail:        false,
		},
		// failure cases.
		{
			description: "fail, single stack incorrect node allocation",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
						Spec: v1.NodeSpec{
							PodCIDRs: []string{"172.10.0.1/24"},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR:          nil,
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "single-stack-cidr-allocate-fail",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.10.0.0/16",
						},
					},
				}),
			allocatedCIDRs:        nil,
			expectedAllocatedCIDR: nil,
			ctrlCreateFail:        true,
		},
		{
			description: "fail, dualstack node allocating from non existing cidr",

			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
						Spec: v1.NodeSpec{
							PodCIDRs: []string{"10.10.0.1/24", "a00::/86"},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR:          nil,
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "dual-stack-cidr-allocate-fail",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.10.0.0/16",
							ipv6CIDR:        "ace:cab:deca::/112",
						},
					},
				}),
			allocatedCIDRs:        nil,
			expectedAllocatedCIDR: nil,
			ctrlCreateFail:        true,
		},
		{
			description: "fail, dualstack node allocating bad v4",

			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
						Spec: v1.NodeSpec{
							PodCIDRs: []string{"172.10.0.1/24", "ace:cab:deca::1/120"},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR:          nil,
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "dual-stack-cidr-bad-v4",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.10.0.0/16",
							ipv6CIDR:        "ace:cab:deca::/112",
						},
					},
				}),
			allocatedCIDRs:        nil,
			expectedAllocatedCIDR: nil,
			ctrlCreateFail:        true,
		},
		{
			description: "fail, dualstack node allocating bad v6",

			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
						Spec: v1.NodeSpec{
							PodCIDRs: []string{"10.10.0.1/24", "cdd::/86"},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR:          nil,
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "dual-stack-cidr-bad-v6",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.10.0.0/16",
							ipv6CIDR:        "ace:cab:deca::/112",
						},
					},
				}),
			allocatedCIDRs:        nil,
			expectedAllocatedCIDR: nil,
			ctrlCreateFail:        true,
		},
	}

	// test function
	_, ctx := ktesting.NewTestContext(t)
	for _, tc := range testCaseMultiCIDRs {
		t.Run(tc.description, func(t *testing.T) {
			// Initialize the range allocator.
			fakeNodeInformer := test.FakeNodeInformer(tc.fakeNodeHandler)
			fakeClient := &fake.Clientset{}
			fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, controller.NoResyncPeriodFunc())
			fakeClusterCIDRInformer := fakeInformerFactory.Networking().V1alpha1().ClusterCIDRs()
			nodeList, _ := tc.fakeNodeHandler.List(context.TODO(), metav1.ListOptions{})

			_, err := NewMultiCIDRRangeAllocator(ctx, tc.fakeNodeHandler, fakeNodeInformer, fakeClusterCIDRInformer, tc.allocatorParams, nodeList, tc.testCIDRMap)
			if err == nil && tc.ctrlCreateFail {
				t.Fatalf("creating range allocator was expected to fail, but it did not")
			}
			if err != nil && !tc.ctrlCreateFail {
				t.Fatalf("creating range allocator was expected to succeed, but it did not")
			}
		})
	}
}

func TestMultiCIDRAllocateOrOccupyCIDRSuccess(t *testing.T) {
	// Non-parallel test (overrides global var).
	oldNodePollInterval := nodePollInterval
	nodePollInterval = test.NodePollInterval
	defer func() {
		nodePollInterval = oldNodePollInterval
	}()

	// all tests operate on a single node.
	testCaseMultiCIDRs := []testCaseMultiCIDR{
		{
			description: "When there's no ServiceCIDR return first CIDR in range",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR:          nil,
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "single-stack-cidr",
							perNodeHostBits: 2,
							ipv4CIDR:        "127.123.234.0/24",
						},
					},
				}),
			expectedAllocatedCIDR: map[int]string{
				0: "127.123.234.0/30",
			},
		},
		{
			description: "Correctly filter out ServiceCIDR",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR: func() *net.IPNet {
					_, serviceCIDR, _ := utilnet.ParseCIDRSloppy("127.123.234.0/26")
					return serviceCIDR
				}(),
				SecondaryServiceCIDR: nil,
				NodeCIDRMaskSizes:    []int{30},
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "single-stack-cidr",
							perNodeHostBits: 2,
							ipv4CIDR:        "127.123.234.0/24",
						},
					},
				}),
			// it should return first /30 CIDR after service range.
			expectedAllocatedCIDR: map[int]string{
				0: "127.123.234.64/30",
			},
		},
		{
			description: "Correctly ignore already allocated CIDRs",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR: func() *net.IPNet {
					_, serviceCIDR, _ := utilnet.ParseCIDRSloppy("127.123.234.0/26")
					return serviceCIDR
				}(),
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "single-stack-cidr",
							perNodeHostBits: 2,
							ipv4CIDR:        "127.123.234.0/24",
						},
					},
				}),
			allocatedCIDRs: map[int][]string{
				0: {"127.123.234.64/30", "127.123.234.68/30", "127.123.234.72/30", "127.123.234.80/30"},
			},
			expectedAllocatedCIDR: map[int]string{
				0: "127.123.234.76/30",
			},
		},
		{
			description: "Dualstack CIDRs, prioritize clusterCIDR with higher label match count",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
								"testLabel-1": "label1",
								"testLabel-2": "label2",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR: func() *net.IPNet {
					_, serviceCIDR, _ := utilnet.ParseCIDRSloppy("127.123.234.0/26")
					return serviceCIDR
				}(),
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "dual-stack-cidr-1",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.0.0.0/8",
							ipv6CIDR:        "ace:cab:deca::/112",
						},
					},
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
						{
							key:      "testLabel-1",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"label1"},
						},
					}): {
						{
							name:            "dual-stack-cidr-2",
							perNodeHostBits: 8,
							ipv4CIDR:        "127.123.234.0/8",
							ipv6CIDR:        "abc:def:deca::/112",
						},
					},
				}),
			expectedAllocatedCIDR: map[int]string{
				0: "127.0.0.0/24",
				1: "abc:def:deca::/120",
			},
		},
		{
			description: "Dualstack CIDRs, prioritize clusterCIDR with higher label match count, overlapping CIDRs",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
								"testLabel-1": "label1",
								"testLabel-2": "label2",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR: func() *net.IPNet {
					_, serviceCIDR, _ := utilnet.ParseCIDRSloppy("127.123.234.0/26")
					return serviceCIDR
				}(),
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "dual-stack-cidr-1",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.0.0.0/8",
							ipv6CIDR:        "ace:cab:deca::/112",
						},
					},
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
						{
							key:      "testLabel-1",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"label1"},
						},
					}): {
						{
							name:            "dual-stack-cidr-2",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.0.0.0/16",
							ipv6CIDR:        "ace:cab:deca::/112",
						},
					},
				}),
			allocatedCIDRs: map[int][]string{
				0: {"10.0.0.0/24", "10.0.1.0/24", "10.0.2.0/24", "10.0.4.0/24"},
				1: {"ace:cab:deca::/120"},
			},
			expectedAllocatedCIDR: map[int]string{
				0: "10.0.3.0/24",
				1: "ace:cab:deca::100/120",
			},
		},
		{
			description: "Dualstack CIDRs, clusterCIDR with equal label match count, prioritize clusterCIDR with fewer allocatable pod CIDRs",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
								"testLabel-1": "label1",
								"testLabel-2": "label2",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR: func() *net.IPNet {
					_, serviceCIDR, _ := utilnet.ParseCIDRSloppy("127.123.234.0/26")
					return serviceCIDR
				}(),
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
						{
							key:      "testLabel-1",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"label1"},
						},
					}): {
						{
							name:            "dual-stack-cidr-1",
							perNodeHostBits: 8,
							ipv4CIDR:        "127.123.234.0/8",
							ipv6CIDR:        "abc:def:deca::/112",
						},
					},
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
						{
							key:      "testLabel-2",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"label2"},
						},
					}): {
						{
							name:            "dual-stack-cidr-2",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.0.0.0/24",
							ipv6CIDR:        "ace:cab:deca::/120",
						},
					},
				}),
			expectedAllocatedCIDR: map[int]string{
				0: "10.0.0.0/24",
				1: "ace:cab:deca::/120",
			},
		},
		{
			description: "Dualstack CIDRs, clusterCIDR with equal label count, non comparable allocatable pod CIDRs, prioritize clusterCIDR with lower perNodeMaskSize",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
								"testLabel-1": "label1",
								"testLabel-2": "label2",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR: func() *net.IPNet {
					_, serviceCIDR, _ := utilnet.ParseCIDRSloppy("127.123.234.0/26")
					return serviceCIDR
				}(),
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
						{
							key:      "testLabel-1",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"label1"},
						},
					}): {
						{
							name:            "dual-stack-cidr-1",
							perNodeHostBits: 8,
							ipv4CIDR:        "127.123.234.0/23",
						},
					},
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
						{
							key:      "testLabel-2",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"label2"},
						},
					}): {
						{
							name:            "dual-stack-cidr-2",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.0.0.0/16",
							ipv6CIDR:        "ace:cab:deca::/120",
						},
					},
				}),
			expectedAllocatedCIDR: map[int]string{
				0: "10.0.0.0/24",
				1: "ace:cab:deca::/120",
			},
		},
		{
			description: "Dualstack CIDRs, clusterCIDR with equal label count and allocatable pod CIDRs, prioritize clusterCIDR with lower perNodeMaskSize",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
								"testLabel-1": "label1",
								"testLabel-2": "label2",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR: func() *net.IPNet {
					_, serviceCIDR, _ := utilnet.ParseCIDRSloppy("127.123.234.0/26")
					return serviceCIDR
				}(),
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
						{
							key:      "testLabel-1",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"label1"},
						},
					}): {
						{
							name:            "dual-stack-cidr-1",
							perNodeHostBits: 8,
							ipv4CIDR:        "127.123.234.0/24",
							ipv6CIDR:        "abc:def:deca::/120",
						},
					},
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
						{
							key:      "testLabel-2",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"label2"},
						},
					}): {
						{
							name:            "dual-stack-cidr-2",
							perNodeHostBits: 0,
							ipv4CIDR:        "10.0.0.0/32",
							ipv6CIDR:        "ace:cab:deca::/128",
						},
					},
				}),
			expectedAllocatedCIDR: map[int]string{
				0: "10.0.0.0/32",
				1: "ace:cab:deca::/128",
			},
		},
		{
			description: "Dualstack CIDRs, clusterCIDR with equal label count, allocatable pod CIDRs and allocatable IPs, prioritize clusterCIDR with lower alphanumeric label",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
								"testLabel-1": "label1",
								"testLabel-2": "label2",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR: func() *net.IPNet {
					_, serviceCIDR, _ := utilnet.ParseCIDRSloppy("127.123.234.0/26")
					return serviceCIDR
				}(),
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
						{
							key:      "testLabel-1",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"label1"},
						},
					}): {
						{
							name:            "dual-stack-cidr-1",
							perNodeHostBits: 8,
							ipv4CIDR:        "127.123.234.0/16",
							ipv6CIDR:        "abc:def:deca::/112",
						},
					},
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
						{
							key:      "testLabel-2",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"label2"},
						},
					}): {
						{
							name:            "dual-stack-cidr-2",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.0.0.0/16",
							ipv6CIDR:        "ace:cab:deca::/112",
						},
					},
				}),
			expectedAllocatedCIDR: map[int]string{
				0: "127.123.0.0/24",
				1: "abc:def:deca::/120",
			},
		},
		{
			description: "Dualstack CIDRs, clusterCIDR with equal label count, allocatable pod CIDRs, allocatable IPs and labels, prioritize clusterCIDR with smaller IP",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
								"testLabel-1": "label1",
								"testLabel-2": "label2",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR: func() *net.IPNet {
					_, serviceCIDR, _ := utilnet.ParseCIDRSloppy("127.123.234.0/26")
					return serviceCIDR
				}(),
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
						{
							key:      "testLabel-1",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"label1"},
						},
					}): {
						{
							name:            "dual-stack-cidr-1",
							perNodeHostBits: 8,
							ipv4CIDR:        "127.123.234.0/16",
							ipv6CIDR:        "abc:def:deca::/112",
						},
					},
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
						{
							key:      "testLabel-1",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"label1"},
						},
					}): {
						{
							name:            "dual-stack-cidr-2",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.0.0.0/16",
							ipv6CIDR:        "ace:cab:deca::/112",
						},
					},
				}),
			expectedAllocatedCIDR: map[int]string{
				0: "10.0.0.0/24",
				1: "ace:cab:deca::/120",
			},
		},
		{
			description: "no double counting",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "nodepool1",
							},
						},
						Spec: v1.NodeSpec{
							PodCIDRs: []string{"10.10.0.0/24"},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
							Labels: map[string]string{
								"testLabel-0": "nodepool1",
							},
						},
						Spec: v1.NodeSpec{
							PodCIDRs: []string{"10.10.2.0/24"},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node2",
							Labels: map[string]string{
								"testLabel-0": "nodepool1",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR:          nil,
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"nodepool1"},
						},
					}): {
						{
							name:            "no-double-counting",
							perNodeHostBits: 8,
							ipv4CIDR:        "10.10.0.0/22",
						},
					},
				}),
			expectedAllocatedCIDR: map[int]string{
				0: "10.10.1.0/24",
			},
		},
	}

	logger, ctx := ktesting.NewTestContext(t)

	// test function
	testFunc := func(tc testCaseMultiCIDR) {
		nodeList, _ := tc.fakeNodeHandler.List(context.TODO(), metav1.ListOptions{})
		// Initialize the range allocator.

		fakeClient := &fake.Clientset{}
		fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, controller.NoResyncPeriodFunc())
		fakeClusterCIDRInformer := fakeInformerFactory.Networking().V1alpha1().ClusterCIDRs()
		allocator, err := NewMultiCIDRRangeAllocator(ctx, tc.fakeNodeHandler, test.FakeNodeInformer(tc.fakeNodeHandler), fakeClusterCIDRInformer, tc.allocatorParams, nodeList, tc.testCIDRMap)
		if err != nil {
			t.Errorf("%v: failed to create CIDRRangeAllocator with error %v", tc.description, err)
			return
		}
		rangeAllocator, ok := allocator.(*multiCIDRRangeAllocator)
		if !ok {
			t.Logf("%v: found non-default implementation of CIDRAllocator, skipping white-box test...", tc.description)
			return
		}
		rangeAllocator.nodesSynced = test.AlwaysReady
		rangeAllocator.recorder = testutil.NewFakeRecorder()

		// this is a bit of white box testing
		// pre allocate the CIDRs as per the test
		for _, allocatedList := range tc.allocatedCIDRs {
			for _, allocated := range allocatedList {
				_, cidr, err := utilnet.ParseCIDRSloppy(allocated)
				if err != nil {
					t.Fatalf("%v: unexpected error when parsing CIDR %v: %v", tc.description, allocated, err)
				}

				clusterCIDRList, err := getClusterCIDRList("node0", rangeAllocator.cidrMap)
				if err != nil {
					t.Fatalf("%v: unexpected error when getting associated clusterCIDR for node %v %v", tc.description, "node0", err)
				}

				occupied := false
				for _, clusterCIDR := range clusterCIDRList {
					if err := rangeAllocator.Occupy(clusterCIDR, cidr); err == nil {
						occupied = true
						break
					}
				}
				if !occupied {
					t.Fatalf("%v: unable to occupy CIDR %v", tc.description, allocated)
				}
			}
		}

		updateCount := 0
		for _, node := range tc.fakeNodeHandler.Existing {
			if node.Spec.PodCIDRs == nil {
				updateCount++
			}
			if err := allocator.AllocateOrOccupyCIDR(logger, node); err != nil {
				t.Errorf("%v: unexpected error in AllocateOrOccupyCIDR: %v", tc.description, err)
			}
		}
		if updateCount != 1 {
			t.Fatalf("test error: all tests must update exactly one node")
		}
		if err := test.WaitForUpdatedNodeWithTimeout(tc.fakeNodeHandler, updateCount, wait.ForeverTestTimeout); err != nil {
			t.Fatalf("%v: timeout while waiting for Node update: %v", tc.description, err)
		}

		if len(tc.expectedAllocatedCIDR) == 0 {
			// nothing further expected
			return
		}
		for _, updatedNode := range tc.fakeNodeHandler.GetUpdatedNodesCopy() {
			if len(updatedNode.Spec.PodCIDRs) == 0 {
				continue // not assigned yet
			}
			//match
			for podCIDRIdx, expectedPodCIDR := range tc.expectedAllocatedCIDR {
				if updatedNode.Spec.PodCIDRs[podCIDRIdx] != expectedPodCIDR {
					t.Errorf("%v: Unable to find allocated CIDR %v, found updated Nodes with CIDRs: %v", tc.description, expectedPodCIDR, updatedNode.Spec.PodCIDRs)
					break
				}
			}
		}
	}

	// run the test cases
	for _, tc := range testCaseMultiCIDRs {
		testFunc(tc)
	}
}

func TestMultiCIDRAllocateOrOccupyCIDRFailure(t *testing.T) {
	testCaseMultiCIDRs := []testCaseMultiCIDR{
		{
			description: "When there's no ServiceCIDR return first CIDR in range",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR:          nil,
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "allocate-fail",
							perNodeHostBits: 2,
							ipv4CIDR:        "127.123.234.0/28",
						},
					},
				}),
			allocatedCIDRs: map[int][]string{
				0: {"127.123.234.0/30", "127.123.234.4/30", "127.123.234.8/30", "127.123.234.12/30"},
			},
		},
	}

	logger, ctx := ktesting.NewTestContext(t)

	testFunc := func(tc testCaseMultiCIDR) {
		fakeClient := &fake.Clientset{}
		fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, controller.NoResyncPeriodFunc())
		fakeClusterCIDRInformer := fakeInformerFactory.Networking().V1alpha1().ClusterCIDRs()

		// Initialize the range allocator.
		allocator, err := NewMultiCIDRRangeAllocator(ctx, tc.fakeNodeHandler, test.FakeNodeInformer(tc.fakeNodeHandler), fakeClusterCIDRInformer, tc.allocatorParams, nil, tc.testCIDRMap)
		if err != nil {
			t.Logf("%v: failed to create CIDRRangeAllocator with error %v", tc.description, err)
		}
		rangeAllocator, ok := allocator.(*multiCIDRRangeAllocator)
		if !ok {
			t.Logf("%v: found non-default implementation of CIDRAllocator, skipping white-box test...", tc.description)
			return
		}
		rangeAllocator.nodesSynced = test.AlwaysReady
		rangeAllocator.recorder = testutil.NewFakeRecorder()

		// this is a bit of white box testing
		// pre allocate the CIDRs as per the test
		for _, allocatedList := range tc.allocatedCIDRs {
			for _, allocated := range allocatedList {
				_, cidr, err := utilnet.ParseCIDRSloppy(allocated)
				if err != nil {
					t.Fatalf("%v: unexpected error when parsing CIDR %v: %v", tc.description, allocated, err)
				}

				clusterCIDRList, err := getClusterCIDRList("node0", rangeAllocator.cidrMap)
				if err != nil {
					t.Fatalf("%v: unexpected error when getting associated clusterCIDR for node %v %v", tc.description, "node0", err)
				}

				occupied := false
				for _, clusterCIDR := range clusterCIDRList {
					if err := rangeAllocator.Occupy(clusterCIDR, cidr); err == nil {
						occupied = true
						break
					}
				}
				if !occupied {
					t.Fatalf("%v: unable to occupy CIDR %v", tc.description, allocated)
				}
			}
		}

		if err := allocator.AllocateOrOccupyCIDR(logger, tc.fakeNodeHandler.Existing[0]); err == nil {
			t.Errorf("%v: unexpected success in AllocateOrOccupyCIDR: %v", tc.description, err)
		}
		// We don't expect any updates, so just sleep for some time
		time.Sleep(time.Second)
		if len(tc.fakeNodeHandler.GetUpdatedNodesCopy()) != 0 {
			t.Fatalf("%v: unexpected update of nodes: %v", tc.description, tc.fakeNodeHandler.GetUpdatedNodesCopy())
		}
		if len(tc.expectedAllocatedCIDR) == 0 {
			// nothing further expected
			return
		}
		for _, updatedNode := range tc.fakeNodeHandler.GetUpdatedNodesCopy() {
			if len(updatedNode.Spec.PodCIDRs) == 0 {
				continue // not assigned yet
			}
			//match
			for podCIDRIdx, expectedPodCIDR := range tc.expectedAllocatedCIDR {
				if updatedNode.Spec.PodCIDRs[podCIDRIdx] == expectedPodCIDR {
					t.Errorf("%v: found cidr %v that should not be allocated on node with CIDRs:%v", tc.description, expectedPodCIDR, updatedNode.Spec.PodCIDRs)
					break
				}
			}
		}
	}
	for _, tc := range testCaseMultiCIDRs {
		testFunc(tc)
	}
}

type releasetestCaseMultiCIDR struct {
	description                      string
	fakeNodeHandler                  *testutil.FakeNodeHandler
	testCIDRMap                      map[string][]*cidrset.ClusterCIDR
	allocatorParams                  CIDRAllocatorParams
	expectedAllocatedCIDRFirstRound  map[int]string
	expectedAllocatedCIDRSecondRound map[int]string
	allocatedCIDRs                   map[int][]string
	cidrsToRelease                   [][]string
}

func TestMultiCIDRReleaseCIDRSuccess(t *testing.T) {
	// Non-parallel test (overrides global var)
	oldNodePollInterval := nodePollInterval
	nodePollInterval = test.NodePollInterval
	defer func() {
		nodePollInterval = oldNodePollInterval
	}()

	testCaseMultiCIDRs := []releasetestCaseMultiCIDR{
		{
			description: "Correctly release preallocated CIDR",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR:          nil,
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "cidr-release",
							perNodeHostBits: 2,
							ipv4CIDR:        "127.123.234.0/28",
						},
					},
				}),
			allocatedCIDRs: map[int][]string{
				0: {"127.123.234.0/30", "127.123.234.4/30", "127.123.234.8/30", "127.123.234.12/30"},
			},
			expectedAllocatedCIDRFirstRound: nil,
			cidrsToRelease: [][]string{
				{"127.123.234.4/30"},
			},
			expectedAllocatedCIDRSecondRound: map[int]string{
				0: "127.123.234.4/30",
			},
		},
		{
			description: "Correctly recycle CIDR",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node0",
							Labels: map[string]string{
								"testLabel-0": "node0",
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(),
			},
			allocatorParams: CIDRAllocatorParams{
				ServiceCIDR:          nil,
				SecondaryServiceCIDR: nil,
			},
			testCIDRMap: getTestCidrMap(
				map[string][]*testClusterCIDR{
					getTestNodeSelector([]testNodeSelectorRequirement{
						{
							key:      "testLabel-0",
							operator: v1.NodeSelectorOpIn,
							values:   []string{"node0"},
						},
					}): {
						{
							name:            "cidr-release",
							perNodeHostBits: 2,
							ipv4CIDR:        "127.123.234.0/28",
						},
					},
				}),
			allocatedCIDRs: map[int][]string{
				0: {"127.123.234.4/30", "127.123.234.8/30", "127.123.234.12/30"},
			},
			expectedAllocatedCIDRFirstRound: map[int]string{
				0: "127.123.234.0/30",
			},
			cidrsToRelease: [][]string{
				{"127.123.234.0/30"},
			},
			expectedAllocatedCIDRSecondRound: map[int]string{
				0: "127.123.234.0/30",
			},
		},
	}
	logger, ctx := ktesting.NewTestContext(t)
	testFunc := func(tc releasetestCaseMultiCIDR) {
		fakeClient := &fake.Clientset{}
		fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, controller.NoResyncPeriodFunc())
		fakeClusterCIDRInformer := fakeInformerFactory.Networking().V1alpha1().ClusterCIDRs()
		// Initialize the range allocator.
		allocator, _ := NewMultiCIDRRangeAllocator(ctx, tc.fakeNodeHandler, test.FakeNodeInformer(tc.fakeNodeHandler), fakeClusterCIDRInformer, tc.allocatorParams, nil, tc.testCIDRMap)
		rangeAllocator, ok := allocator.(*multiCIDRRangeAllocator)
		if !ok {
			t.Logf("%v: found non-default implementation of CIDRAllocator, skipping white-box test...", tc.description)
			return
		}
		rangeAllocator.nodesSynced = test.AlwaysReady
		rangeAllocator.recorder = testutil.NewFakeRecorder()

		// this is a bit of white box testing
		for _, allocatedList := range tc.allocatedCIDRs {
			for _, allocated := range allocatedList {
				_, cidr, err := utilnet.ParseCIDRSloppy(allocated)
				if err != nil {
					t.Fatalf("%v: unexpected error when parsing CIDR %v: %v", tc.description, allocated, err)
				}

				clusterCIDRList, err := getClusterCIDRList("node0", rangeAllocator.cidrMap)
				if err != nil {
					t.Fatalf("%v: unexpected error when getting associated clusterCIDR for node %v %v", tc.description, "node0", err)
				}

				occupied := false
				for _, clusterCIDR := range clusterCIDRList {
					if err := rangeAllocator.Occupy(clusterCIDR, cidr); err == nil {
						occupied = true
						clusterCIDR.AssociatedNodes["fakeNode"] = true
						break
					}
				}
				if !occupied {
					t.Fatalf("%v: unable to occupy CIDR %v", tc.description, allocated)
				}
			}
		}

		err := allocator.AllocateOrOccupyCIDR(logger, tc.fakeNodeHandler.Existing[0])
		if len(tc.expectedAllocatedCIDRFirstRound) != 0 {
			if err != nil {
				t.Fatalf("%v: unexpected error in AllocateOrOccupyCIDR: %v", tc.description, err)
			}
			if err := test.WaitForUpdatedNodeWithTimeout(tc.fakeNodeHandler, 1, wait.ForeverTestTimeout); err != nil {
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
					Name: "fakeNode",
					Labels: map[string]string{
						"testLabel-0": "node0",
					},
				},
			}
			nodeToRelease.Spec.PodCIDRs = cidrToRelease
			err = allocator.ReleaseCIDR(logger, &nodeToRelease)
			if err != nil {
				t.Fatalf("%v: unexpected error in ReleaseCIDR: %v", tc.description, err)
			}
		}
		if err = allocator.AllocateOrOccupyCIDR(logger, tc.fakeNodeHandler.Existing[0]); err != nil {
			t.Fatalf("%v: unexpected error in AllocateOrOccupyCIDR: %v", tc.description, err)
		}
		if err := test.WaitForUpdatedNodeWithTimeout(tc.fakeNodeHandler, 1, wait.ForeverTestTimeout); err != nil {
			t.Fatalf("%v: timeout while waiting for Node update: %v", tc.description, err)
		}

		if len(tc.expectedAllocatedCIDRSecondRound) == 0 {
			// nothing further expected
			return
		}
		for _, updatedNode := range tc.fakeNodeHandler.GetUpdatedNodesCopy() {
			if len(updatedNode.Spec.PodCIDRs) == 0 {
				continue // not assigned yet
			}
			//match
			for podCIDRIdx, expectedPodCIDR := range tc.expectedAllocatedCIDRSecondRound {
				if updatedNode.Spec.PodCIDRs[podCIDRIdx] != expectedPodCIDR {
					t.Errorf("%v: found cidr %v that should not be allocated on node with CIDRs:%v", tc.description, expectedPodCIDR, updatedNode.Spec.PodCIDRs)
					break
				}
			}
		}
	}

	for _, tc := range testCaseMultiCIDRs {
		testFunc(tc)
	}
}

// ClusterCIDR tests.

var alwaysReady = func() bool { return true }

type clusterCIDRController struct {
	*multiCIDRRangeAllocator
	clusterCIDRStore cache.Store
}

func newController(ctx context.Context) (*fake.Clientset, *clusterCIDRController) {
	client := fake.NewSimpleClientset()

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	cccInformer := informerFactory.Networking().V1alpha1().ClusterCIDRs()
	cccIndexer := cccInformer.Informer().GetIndexer()

	nodeInformer := informerFactory.Core().V1().Nodes()

	// These reactors are required to mock functionality that would be covered
	// automatically if we weren't using the fake client.
	client.PrependReactor("create", "clustercidrs", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		clusterCIDR := action.(k8stesting.CreateAction).GetObject().(*networkingv1alpha1.ClusterCIDR)

		if clusterCIDR.ObjectMeta.GenerateName != "" {
			clusterCIDR.ObjectMeta.Name = fmt.Sprintf("%s-%s", clusterCIDR.ObjectMeta.GenerateName, rand.String(8))
			clusterCIDR.ObjectMeta.GenerateName = ""
		}
		clusterCIDR.Generation = 1
		cccIndexer.Add(clusterCIDR)

		return false, clusterCIDR, nil
	}))
	client.PrependReactor("update", "clustercidrs", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		clusterCIDR := action.(k8stesting.CreateAction).GetObject().(*networkingv1alpha1.ClusterCIDR)
		clusterCIDR.Generation++
		cccIndexer.Update(clusterCIDR)

		return false, clusterCIDR, nil
	}))

	_, clusterCIDR, _ := utilnet.ParseCIDRSloppy("192.168.0.0/16")
	_, serviceCIDR, _ := utilnet.ParseCIDRSloppy("10.1.0.0/16")

	allocatorParams := CIDRAllocatorParams{
		ClusterCIDRs:         []*net.IPNet{clusterCIDR},
		ServiceCIDR:          serviceCIDR,
		SecondaryServiceCIDR: nil,
		NodeCIDRMaskSizes:    []int{24},
	}
	testCIDRMap := make(map[string][]*cidrset.ClusterCIDR, 0)

	// Initialize the range allocator.
	ra, _ := NewMultiCIDRRangeAllocator(ctx, client, nodeInformer, cccInformer, allocatorParams, nil, testCIDRMap)
	cccController := ra.(*multiCIDRRangeAllocator)

	cccController.clusterCIDRSynced = alwaysReady

	return client, &clusterCIDRController{
		cccController,
		informerFactory.Networking().V1alpha1().ClusterCIDRs().Informer().GetStore(),
	}
}

// Ensure default ClusterCIDR is created during bootstrap.
func TestClusterCIDRDefault(t *testing.T) {
	defaultCCC := makeClusterCIDR(defaultClusterCIDRName, "192.168.0.0/16", "", 8, nil)
	_, ctx := ktesting.NewTestContext(t)
	client, _ := newController(ctx)
	createdCCC, err := client.NetworkingV1alpha1().ClusterCIDRs().Get(context.TODO(), defaultClusterCIDRName, metav1.GetOptions{})
	assert.Nil(t, err, "Expected no error getting clustercidr objects")
	assert.Equal(t, defaultCCC.Spec, createdCCC.Spec)
}

// Ensure SyncClusterCIDR creates a new valid ClusterCIDR.
func TestSyncClusterCIDRCreate(t *testing.T) {
	tests := []struct {
		name    string
		ccc     *networkingv1alpha1.ClusterCIDR
		wantErr bool
	}{
		{
			name:    "valid IPv4 ClusterCIDR with no NodeSelector",
			ccc:     makeClusterCIDR("ipv4-ccc", "10.2.0.0/16", "", 8, nil),
			wantErr: false,
		},
		{
			name:    "valid IPv4 ClusterCIDR with NodeSelector",
			ccc:     makeClusterCIDR("ipv4-ccc-label", "10.3.0.0/16", "", 8, makeNodeSelector("foo", v1.NodeSelectorOpIn, []string{"bar"})),
			wantErr: false,
		},
		{
			name:    "valid IPv4 ClusterCIDR with overlapping CIDRs",
			ccc:     makeClusterCIDR("ipv4-ccc-overlap", "10.2.0.0/24", "", 8, makeNodeSelector("foo", v1.NodeSelectorOpIn, []string{"bar"})),
			wantErr: false,
		},
		{
			name:    "valid IPv6 ClusterCIDR with no NodeSelector",
			ccc:     makeClusterCIDR("ipv6-ccc", "", "fd00:1::/112", 8, nil),
			wantErr: false,
		},
		{
			name:    "valid IPv6 ClusterCIDR with NodeSelector",
			ccc:     makeClusterCIDR("ipv6-ccc-label", "", "fd00:2::/112", 8, makeNodeSelector("foo", v1.NodeSelectorOpIn, []string{"bar"})),
			wantErr: false,
		},
		{
			name:    "valid IPv6 ClusterCIDR with overlapping CIDRs",
			ccc:     makeClusterCIDR("ipv6-ccc-overlap", "", "fd00:1:1::/112", 8, makeNodeSelector("foo", v1.NodeSelectorOpIn, []string{"bar"})),
			wantErr: false,
		},
		{
			name:    "valid Dualstack ClusterCIDR with no NodeSelector",
			ccc:     makeClusterCIDR("dual-ccc", "10.2.0.0/16", "fd00:1::/112", 8, nil),
			wantErr: false,
		},
		{
			name:    "valid DualStack ClusterCIDR with NodeSelector",
			ccc:     makeClusterCIDR("dual-ccc-label", "10.3.0.0/16", "fd00:2::/112", 8, makeNodeSelector("foo", v1.NodeSelectorOpIn, []string{"bar"})),
			wantErr: false,
		},
		{
			name:    "valid Dualstack ClusterCIDR with overlapping CIDRs",
			ccc:     makeClusterCIDR("dual-ccc-overlap", "10.2.0.0/16", "fd00:1:1::/112", 8, makeNodeSelector("foo", v1.NodeSelectorOpIn, []string{"bar"})),
			wantErr: false,
		},
		// invalid ClusterCIDRs.
		{
			name:    "invalid ClusterCIDR with both IPv4 and IPv6 CIDRs nil",
			ccc:     makeClusterCIDR("invalid-ccc", "", "", 0, nil),
			wantErr: true,
		},
		{
			name:    "invalid IPv4 ClusterCIDR",
			ccc:     makeClusterCIDR("invalid-ipv4-ccc", "1000.2.0.0/16", "", 8, nil),
			wantErr: true,
		},
		{
			name:    "invalid IPv6 ClusterCIDR",
			ccc:     makeClusterCIDR("invalid-ipv6-ccc", "", "aaaaa:1:1::/112", 8, nil),
			wantErr: true,
		},
		{
			name:    "invalid dualstack ClusterCIDR",
			ccc:     makeClusterCIDR("invalid-dual-ccc", "10.2.0.0/16", "aaaaa:1:1::/112", 8, makeNodeSelector("foo", v1.NodeSelectorOpIn, []string{"bar"})),
			wantErr: true,
		},
	}
	_, ctx := ktesting.NewTestContext(t)
	client, cccController := newController(ctx)
	for _, tc := range tests {
		cccController.clusterCIDRStore.Add(tc.ccc)
		err := cccController.syncClusterCIDR(ctx, tc.ccc.Name)
		if tc.wantErr {
			assert.Error(t, err)
			continue
		}
		assert.NoError(t, err)
		expectActions(t, client.Actions(), 1, "create", "clustercidrs")

		createdCCC, err := client.NetworkingV1alpha1().ClusterCIDRs().Get(context.TODO(), tc.ccc.Name, metav1.GetOptions{})
		assert.Nil(t, err, "Expected no error getting clustercidr object")
		assert.Equal(t, tc.ccc.Spec, createdCCC.Spec)
		assert.Equal(t, []string{clusterCIDRFinalizer}, createdCCC.Finalizers)
	}
}

// Ensure syncClusterCIDR for ClusterCIDR delete removes the ClusterCIDR.
func TestSyncClusterCIDRDelete(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	_, cccController := newController(ctx)

	testCCC := makeClusterCIDR("testing-1", "10.1.0.0/16", "", 8, makeNodeSelector("foo", v1.NodeSelectorOpIn, []string{"bar"}))

	cccController.clusterCIDRStore.Add(testCCC)
	err := cccController.syncClusterCIDR(ctx, testCCC.Name)
	assert.NoError(t, err)

	deletionTimestamp := metav1.Now()
	testCCC.DeletionTimestamp = &deletionTimestamp
	cccController.clusterCIDRStore.Update(testCCC)
	err = cccController.syncClusterCIDR(ctx, testCCC.Name)
	assert.NoError(t, err)
}

// Ensure syncClusterCIDR for ClusterCIDR delete does not remove ClusterCIDR
// if a node is associated with the ClusterCIDR.
func TestSyncClusterCIDRDeleteWithNodesAssociated(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	client, cccController := newController(ctx)

	testCCC := makeClusterCIDR("testing-1", "10.1.0.0/16", "", 8, makeNodeSelector("foo", v1.NodeSelectorOpIn, []string{"bar"}))

	cccController.clusterCIDRStore.Add(testCCC)
	err := cccController.syncClusterCIDR(ctx, testCCC.Name)
	assert.NoError(t, err)

	// Mock the IPAM controller behavior associating node with ClusterCIDR.
	nodeSelectorKey, _ := cccController.nodeSelectorKey(testCCC)
	clusterCIDRs, _ := cccController.cidrMap[nodeSelectorKey]
	clusterCIDRs[0].AssociatedNodes["test-node"] = true

	createdCCC, err := client.NetworkingV1alpha1().ClusterCIDRs().Get(context.TODO(), testCCC.Name, metav1.GetOptions{})
	assert.Nil(t, err, "Expected no error getting clustercidr object")

	deletionTimestamp := metav1.Now()
	createdCCC.DeletionTimestamp = &deletionTimestamp
	cccController.clusterCIDRStore.Update(createdCCC)
	err = cccController.syncClusterCIDR(ctx, createdCCC.Name)
	assert.Error(t, err, fmt.Sprintf("ClusterCIDR %s marked as terminating, won't be deleted until all associated nodes are deleted", createdCCC.Name))
}

func expectActions(t *testing.T, actions []k8stesting.Action, num int, verb, resource string) {
	t.Helper()
	// if actions are less, the below logic will panic.
	if num > len(actions) {
		t.Fatalf("len of actions %v is unexpected. Expected to be at least %v", len(actions), num+1)
	}

	for i := 0; i < num; i++ {
		relativePos := len(actions) - i - 1
		assert.Equal(t, verb, actions[relativePos].GetVerb(), "Expected action -%d verb to be %s", i, verb)
		assert.Equal(t, resource, actions[relativePos].GetResource().Resource, "Expected action -%d resource to be %s", i, resource)
	}
}

func makeNodeSelector(key string, op v1.NodeSelectorOperator, values []string) *v1.NodeSelector {
	return &v1.NodeSelector{
		NodeSelectorTerms: []v1.NodeSelectorTerm{
			{
				MatchExpressions: []v1.NodeSelectorRequirement{
					{
						Key:      key,
						Operator: op,
						Values:   values,
					},
				},
			},
		},
	}
}

// makeClusterCIDR returns a mock ClusterCIDR object.
func makeClusterCIDR(cccName, ipv4CIDR, ipv6CIDR string, perNodeHostBits int32, nodeSelector *v1.NodeSelector) *networkingv1alpha1.ClusterCIDR {
	testCCC := &networkingv1alpha1.ClusterCIDR{
		ObjectMeta: metav1.ObjectMeta{Name: cccName},
		Spec:       networkingv1alpha1.ClusterCIDRSpec{},
	}

	testCCC.Spec.PerNodeHostBits = perNodeHostBits

	if ipv4CIDR != "" {
		testCCC.Spec.IPv4 = ipv4CIDR
	}

	if ipv6CIDR != "" {
		testCCC.Spec.IPv6 = ipv6CIDR
	}

	if nodeSelector != nil {
		testCCC.Spec.NodeSelector = nodeSelector
	}

	return testCCC
}
