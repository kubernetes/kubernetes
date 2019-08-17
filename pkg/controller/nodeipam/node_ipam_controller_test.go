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
	"net"
	"os"
	"os/exec"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam"
	"k8s.io/kubernetes/pkg/controller/testutil"
	"k8s.io/legacy-cloud-providers/gce"
	netutils "k8s.io/utils/net"
)

func newTestNodeIpamController(clusterCIDR []*net.IPNet, serviceCIDR *net.IPNet, nodeCIDRMaskSize int, allocatorType ipam.CIDRAllocatorType) (*Controller, error) {
	clientSet := fake.NewSimpleClientset()
	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			{ObjectMeta: metav1.ObjectMeta{Name: "node0"}},
		},
		Clientset: fake.NewSimpleClientset(),
	}
	fakeClient := &fake.Clientset{}
	fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, controller.NoResyncPeriodFunc())
	fakeNodeInformer := fakeInformerFactory.Core().V1().Nodes()

	for _, node := range fakeNodeHandler.Existing {
		fakeNodeInformer.Informer().GetStore().Add(node)
	}

	fakeGCE := gce.NewFakeGCECloud(gce.DefaultTestClusterValues())
	return NewNodeIpamController(
		fakeNodeInformer, fakeGCE, clientSet,
		clusterCIDR, serviceCIDR, nodeCIDRMaskSize, allocatorType,
	)
}

// TestNewNodeIpamControllerWithCIDRMasks tests if the controller can be
// created with combinations of network CIDRs and masks.
func TestNewNodeIpamControllerWithCIDRMasks(t *testing.T) {
	for _, tc := range []struct {
		desc          string
		clusterCIDR   string
		serviceCIDR   string
		maskSize      int
		allocatorType ipam.CIDRAllocatorType
		wantFatal     bool
	}{
		{"valid_range_allocator", "10.0.0.0/21", "10.1.0.0/21", 24, ipam.RangeAllocatorType, false},
		{"valid_range_allocator_dualstack", "10.0.0.0/21,2000::/10", "10.1.0.0/21", 24, ipam.RangeAllocatorType, false},
		{"valid_cloud_allocator", "10.0.0.0/21", "10.1.0.0/21", 24, ipam.CloudAllocatorType, false},
		{"valid_ipam_from_cluster", "10.0.0.0/21", "10.1.0.0/21", 24, ipam.IPAMFromClusterAllocatorType, false},
		{"valid_ipam_from_cloud", "10.0.0.0/21", "10.1.0.0/21", 24, ipam.IPAMFromCloudAllocatorType, false},
		{"valid_skip_cluster_CIDR_validation_for_cloud_allocator", "invalid", "10.1.0.0/21", 24, ipam.CloudAllocatorType, false},
		{"invalid_cluster_CIDR", "invalid", "10.1.0.0/21", 24, ipam.IPAMFromClusterAllocatorType, true},
		{"valid_CIDR_smaller_than_mask_cloud_allocator", "10.0.0.0/26", "10.1.0.0/21", 24, ipam.CloudAllocatorType, false},
		{"invalid_CIDR_smaller_than_mask_other_allocators", "10.0.0.0/26", "10.1.0.0/21", 24, ipam.IPAMFromCloudAllocatorType, true},
		{"invalid_serviceCIDR_contains_clusterCIDR", "10.0.0.0/23", "10.0.0.0/21", 24, ipam.IPAMFromClusterAllocatorType, true},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			clusterCidrs, _ := netutils.ParseCIDRs(strings.Split(tc.clusterCIDR, ","))
			_, serviceCIDRIpNet, _ := net.ParseCIDR(tc.serviceCIDR)

			if os.Getenv("EXIT_ON_FATAL") == "1" {
				// This is the subprocess which runs the actual code.
				newTestNodeIpamController(clusterCidrs, serviceCIDRIpNet, tc.maskSize, tc.allocatorType)
				return
			}
			// This is the host process that monitors the exit code of the subprocess.
			cmd := exec.Command(os.Args[0], "-test.run=TestNewNodeIpamControllerWithCIDRMasks/"+tc.desc)
			cmd.Env = append(os.Environ(), "EXIT_ON_FATAL=1")
			err := cmd.Run()
			var gotFatal bool
			if err != nil {
				exitErr, ok := err.(*exec.ExitError)
				if !ok {
					t.Fatalf("Failed to run subprocess: %v", err)
				}
				gotFatal = !exitErr.Success()
			}
			if gotFatal != tc.wantFatal {
				t.Errorf("newTestNodeIpamController(%v, %v, %v, %v) : gotFatal = %t ; wantFatal = %t", clusterCidrs, serviceCIDRIpNet, tc.maskSize, tc.allocatorType, gotFatal, tc.wantFatal)
			}
		})
	}
}
