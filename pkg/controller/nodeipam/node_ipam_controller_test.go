//go:build !providerless
// +build !providerless

/*
Copyright 2018 The Kubernetes Authors.

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

package nodeipam

import (
	"context"
	"errors"
	"net"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam"
	"k8s.io/kubernetes/pkg/controller/testutil"
	"k8s.io/legacy-cloud-providers/gce"
	netutils "k8s.io/utils/net"
)

func newTestNodeIpamController(ctx context.Context, clusterCIDR []*net.IPNet, serviceCIDR *net.IPNet, secondaryServiceCIDR *net.IPNet, nodeCIDRMaskSizes []int, allocatorType ipam.CIDRAllocatorType) (*Controller, error) {
	clientSet := fake.NewSimpleClientset()
	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			{ObjectMeta: metav1.ObjectMeta{Name: "node0"}},
		},
		Clientset: fake.NewSimpleClientset(),
	}
	fakeClient := &fake.Clientset{}
	fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, 0)
	fakeNodeInformer := fakeInformerFactory.Core().V1().Nodes()

	for _, node := range fakeNodeHandler.Existing {
		fakeNodeInformer.Informer().GetStore().Add(node)
	}

	fakeGCE := gce.NewFakeGCECloud(gce.DefaultTestClusterValues())
	return NewNodeIpamController(
		ctx,
		fakeNodeInformer, fakeGCE, clientSet,
		clusterCIDR, serviceCIDR, secondaryServiceCIDR, nodeCIDRMaskSizes, allocatorType,
	)
}

// TestNewNodeIpamControllerWithCIDRMasks tests if the controller can be
// created with combinations of network CIDRs and masks.
func TestNewNodeIpamControllerWithCIDRMasks(t *testing.T) {
	emptyServiceCIDR := ""
	for _, tc := range []struct {
		desc                 string
		clusterCIDR          string
		serviceCIDR          string
		secondaryServiceCIDR string
		maskSize             []int
		allocatorType        ipam.CIDRAllocatorType
		expectedError        error
	}{
		{"valid_range_allocator", "10.0.0.0/21", "10.1.0.0/21", emptyServiceCIDR, []int{24}, ipam.RangeAllocatorType, nil},
		{"valid_range_allocator_dualstack", "10.0.0.0/21,2000::/48", "10.1.0.0/21", emptyServiceCIDR, []int{24, 64}, ipam.RangeAllocatorType, nil},
		{"valid_range_allocator_dualstack_dualstackservice", "10.0.0.0/21,2000::/48", "10.1.0.0/21", "3000::/112", []int{24, 64}, ipam.RangeAllocatorType, nil},
		{"valid_cloud_allocator", "10.0.0.0/21", "10.1.0.0/21", emptyServiceCIDR, []int{24}, ipam.CloudAllocatorType, nil},
		{"valid_ipam_from_cluster", "10.0.0.0/21", "10.1.0.0/21", emptyServiceCIDR, []int{24}, ipam.IPAMFromClusterAllocatorType, nil},
		{"valid_ipam_from_cloud", "10.0.0.0/21", "10.1.0.0/21", emptyServiceCIDR, []int{24}, ipam.IPAMFromCloudAllocatorType, nil},
		{"valid_skip_cluster_CIDR_validation_for_cloud_allocator", "invalid", "10.1.0.0/21", emptyServiceCIDR, []int{24}, ipam.CloudAllocatorType, nil},
		{"valid_CIDR_larger_than_mask_cloud_allocator", "10.0.0.0/16", "10.1.0.0/21", emptyServiceCIDR, []int{24}, ipam.CloudAllocatorType, nil},
		{"invalid_cluster_CIDR", "", "10.1.0.0/21", emptyServiceCIDR, []int{24}, ipam.IPAMFromClusterAllocatorType, errors.New("Controller: Must specify --cluster-cidr if --allocate-node-cidrs is set")},
		{"invalid_CIDR_smaller_than_mask_other_allocators", "10.0.0.0/26", "10.1.0.0/21", emptyServiceCIDR, []int{24}, ipam.IPAMFromCloudAllocatorType, errors.New("Controller: Invalid --cluster-cidr, mask size of cluster CIDR must be less than or equal to --node-cidr-mask-size configured for CIDR family")},
		{"invalid_serviceCIDR_contains_clusterCIDR", "10.0.0.0/16", "10.0.0.0/8", emptyServiceCIDR, []int{24}, ipam.IPAMFromClusterAllocatorType, errors.New("error creating ipam controller: failed after occupy serviceCIDR: CIDR allocation failed; there are no remaining CIDRs left to allocate in the accepted range")},
		{"invalid_CIDR_mask_size", "10.0.0.0/24,2000::/64", "10.1.0.0/21", emptyServiceCIDR, []int{24, 48}, ipam.IPAMFromClusterAllocatorType, errors.New("Controller: Invalid --cluster-cidr, mask size of cluster CIDR must be less than or equal to --node-cidr-mask-size configured for CIDR family")},
	} {
		test := tc
		_, ctx := ktesting.NewTestContext(t)
		t.Run(test.desc, func(t *testing.T) {
			t.Parallel()
			clusterCidrs, err := netutils.ParseCIDRs(strings.Split(test.clusterCIDR, ","))
			if err != nil {
				clusterCidrs = nil
			}
			_, serviceCIDRIpNet, err := netutils.ParseCIDRSloppy(test.serviceCIDR)
			if err != nil {
				serviceCIDRIpNet = nil
			}
			_, secondaryServiceCIDRIpNet, err := netutils.ParseCIDRSloppy(test.secondaryServiceCIDR)
			if err != nil {
				secondaryServiceCIDRIpNet = nil
			}
			_, err = newTestNodeIpamController(ctx, clusterCidrs, serviceCIDRIpNet, secondaryServiceCIDRIpNet, test.maskSize, test.allocatorType)
			if test.expectedError == nil {
				if err != nil {
					t.Errorf("Test %s, unexpected error: %v", test.desc, err)
				}
			} else {
				if err.Error() != test.expectedError.Error() {
					t.Errorf("Test %s, got error: %v, expected error: %v", test.desc, err, test.expectedError)
				}
			}
		})
	}
}
