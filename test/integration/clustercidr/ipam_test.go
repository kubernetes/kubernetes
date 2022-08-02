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

package clustercidr

import (
	"context"
	"fmt"
	"net"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingv1alpha1 "k8s.io/api/networking/v1alpha1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/controller/nodeipam"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam"
	"k8s.io/kubernetes/test/integration/framework"
	netutils "k8s.io/utils/net"
)

func TestIPAMMultiCIDRRangeAllocatorType(t *testing.T) {

	// parameters
	// TODO convert to table tests and use different variations
	// 1. single ipv4
	// 2. single ipv6
	// 3. dual stack ipv4-ipv6
	// 4. dual stack ipv6-ipv4
	_, clusterCIDRv4, _ := netutils.ParseCIDRSloppy("10.96.0.0/12")     // allows up to 8K nodes
	_, clusterCIDRv6, _ := netutils.ParseCIDRSloppy("fd00:10:96::/112") // allows up to 8K nodes
	_, serviceCIDR, _ := netutils.ParseCIDRSloppy("10.94.0.0/24")       // does not matter for test - pick upto  250 services
	_, secServiceCIDR, _ := netutils.ParseCIDRSloppy("2001:db2::/120")  // does not matter for test - pick upto  250 services

	// order is ipv4 - ipv6 by convention for dual stack
	clusterCIDRs := make([]*net.IPNet, 2)
	clusterCIDRs[0] = clusterCIDRv4
	clusterCIDRs[1] = clusterCIDRv6
	nodeMaskCIDRs := make([]int, len(clusterCIDRs))
	nodeMaskCIDRs[0] = 24
	nodeMaskCIDRs[1] = 120

	// set the feature gate accordingly
	// defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)()

	_, kubeConfig, tearDownFn := framework.StartTestServer(t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount", "TaintNodesByCondition"}
			opts.APIEnablement.RuntimeConfig.Set("networking.k8s.io/v1alpha1=true")
		},
	})
	defer tearDownFn()

	clientSet := clientset.NewForConfigOrDie(kubeConfig)
	sharedInformer := informers.NewSharedInformerFactory(clientSet, 1*time.Hour)

	// set the current state of the informer, we can preseed nodes and ClusterCIDRs so we
	// can simulate the bootstrap
	ccc := networking.ClusterCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: "initial-ccc",
		},
		Spec: networking.ClusterCIDRSpec{
			PerNodeHostBits: 8,
			IPv4:            clusterCIDRv4.String(),
			IPv6:            clusterCIDRv6.String(),
		},
	}
	sharedInformer.Networking().V1alpha1().ClusterCIDRs().Informer().GetStore().Add(ccc)
	node := v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "initial-node",
		},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods:   *resource.NewQuantity(110, resource.DecimalSI),
				v1.ResourceCPU:    resource.MustParse("4"),
				v1.ResourceMemory: resource.MustParse("32Gi"),
			},
			Phase: v1.NodeRunning,
			Conditions: []v1.NodeCondition{
				{Type: v1.NodeReady, Status: v1.ConditionTrue},
			},
		},
	}
	sharedInformer.Core().V1().Nodes().Informer().GetStore().Add(node)

	ipamController, err := nodeipam.NewNodeIpamController(
		sharedInformer.Core().V1().Nodes(),
		sharedInformer.Networking().V1alpha1().ClusterCIDRs(),
		nil,
		clientSet,
		clusterCIDRs,
		serviceCIDR,
		secServiceCIDR,
		nodeMaskCIDRs,
		ipam.MultiCIDRRangeAllocatorType,
	)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go ipamController.Run(ctx.Done())
	sharedInformer.Start(ctx.Done())

	// Create Nodes so they can be processed by the controller
	baseNodeTemplate := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "sample-node-",
		},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods:   *resource.NewQuantity(110, resource.DecimalSI),
				v1.ResourceCPU:    resource.MustParse("4"),
				v1.ResourceMemory: resource.MustParse("32Gi"),
			},
			Phase: v1.NodeRunning,
			Conditions: []v1.NodeCondition{
				{Type: v1.NodeReady, Status: v1.ConditionTrue},
			},
		},
	}

	cidrs := getNodesPodCIDRs(t, clientSet)
	fmt.Println("CIDRs: ", cidrs)

	numNodes := 3
	for j := 0; j < numNodes; j++ {
		if _, err = clientSet.CoreV1().Nodes().Create(context.TODO(), baseNodeTemplate, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}

	}

	// assert nodes get the correct PodCIDRs

	cidrs = getNodesPodCIDRs(t, clientSet)
	fmt.Println("CIDRs: ", cidrs)

	// Create ClusterCIDRs so they can be processed by the controller
	clusterCidr := &networkingv1alpha1.ClusterCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: "cluster-cidr-1",
		},
		Spec: networkingv1alpha1.ClusterCIDRSpec{
			PerNodeHostBits: 8,
			IPv4:            "10.0.0.0/24",
			IPv6:            "fd00::/112",
		},
	}
	if _, err = clientSet.NetworkingV1alpha1().ClusterCIDRs().Create(context.TODO(), clusterCidr, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	if list, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().List(context.TODO(), metav1.ListOptions{}); err != nil {
		t.Fatal(err)
	} else {
		fmt.Println("ClusterCIDRs: ", list)
	}

	// assert nodes get the correct PodCIDRs
	cidrs = getNodesPodCIDRs(t, clientSet)
	fmt.Println("CIDRs: ", cidrs)

}

func getNodesPodCIDRs(t *testing.T, c clientset.Interface) []string {
	nodeList, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	cidrs := []string{}
	for _, node := range nodeList.Items {
		if len(node.Spec.PodCIDRs) > 0 {
			cidrs = append(cidrs, node.Spec.PodCIDRs...)
		}
	}
	return cidrs
}
