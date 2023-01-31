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
	"net"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingv1alpha1 "k8s.io/api/networking/v1alpha1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controller/nodeipam"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	netutils "k8s.io/utils/net"
)

func TestIPAMMultiCIDRRangeAllocatorType(t *testing.T) {

	// set the feature gate to enable MultiCIDRRangeAllocator
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MultiCIDRRangeAllocator, true)()

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

	ipamController := booststrapMultiCIDRRangeAllocator(t, clientSet, sharedInformer)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go ipamController.Run(ctx.Done())
	sharedInformer.Start(ctx.Done())

	tests := []struct {
		name             string
		clusterCIDR      *networkingv1alpha1.ClusterCIDR
		node             *v1.Node
		expectedPodCIDRs []string
	}{
		{
			name:             "Default dualstack Pod CIDRs assigned to a node, node labels matching no ClusterCIDR nodeSelectors",
			clusterCIDR:      nil,
			node:             makeNode("default-node", map[string]string{"label": "unmatched"}),
			expectedPodCIDRs: []string{"10.96.0.0/24", "fd00:10:96::/120"},
		},
		{
			name:             "Dualstack Pod CIDRs assigned to a node from a CC created during bootstrap",
			clusterCIDR:      nil,
			node:             makeNode("bootstrap-node", map[string]string{"bootstrap": "true"}),
			expectedPodCIDRs: []string{"10.2.1.0/24", "fd00:20:96::100/120"},
		},
		{
			name:             "Single stack IPv4 Pod CIDR assigned to a node",
			clusterCIDR:      makeClusterCIDR("ipv4-cc", "10.0.0.0/16", "", nodeSelector(map[string][]string{"ipv4": {"true"}, "singlestack": {"true"}})),
			node:             makeNode("ipv4-node", map[string]string{"ipv4": "true", "singlestack": "true"}),
			expectedPodCIDRs: []string{"10.0.0.0/24"},
		},
		{
			name:             "Single stack IPv6 Pod CIDR assigned to a node",
			clusterCIDR:      makeClusterCIDR("ipv6-cc", "", "fd00:20:100::/112", nodeSelector(map[string][]string{"ipv6": {"true"}})),
			node:             makeNode("ipv6-node", map[string]string{"ipv6": "true"}),
			expectedPodCIDRs: []string{"fd00:20:100::/120"},
		},
		{
			name:             "DualStack Pod CIDRs assigned to a node",
			clusterCIDR:      makeClusterCIDR("dualstack-cc", "192.168.0.0/16", "fd00:30:100::/112", nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
			node:             makeNode("dualstack-node", map[string]string{"ipv4": "true", "ipv6": "true"}),
			expectedPodCIDRs: []string{"192.168.0.0/24", "fd00:30:100::/120"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.clusterCIDR != nil {
				// Create the test ClusterCIDR
				if _, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Create(context.TODO(), test.clusterCIDR, metav1.CreateOptions{}); err != nil {
					t.Fatal(err)
				}

				// Wait for the ClusterCIDR to be created
				if err := wait.PollImmediate(time.Second, 5*time.Second, func() (bool, error) {
					cc, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Get(context.TODO(), test.clusterCIDR.Name, metav1.GetOptions{})
					if err != nil {
						return false, err
					}
					return cc != nil, nil
				}); err != nil {
					t.Fatalf("failed while waiting for ClusterCIDR %q to be created: %v", test.clusterCIDR.Name, err)
				}
			}

			// Sleep for one second to make sure the controller process the new created ClusterCIDR.
			time.Sleep(1 * time.Second)

			if _, err := clientSet.CoreV1().Nodes().Create(context.TODO(), test.node, metav1.CreateOptions{}); err != nil {
				t.Fatal(err)
			}
			if gotPodCIDRs, err := nodePodCIDRs(clientSet, test.node.Name); err != nil {
				t.Fatal(err)
			} else if !reflect.DeepEqual(test.expectedPodCIDRs, gotPodCIDRs) {
				t.Errorf("unexpected result, expected Pod CIDRs %v but got %v", test.expectedPodCIDRs, gotPodCIDRs)
			}
		})
	}
}

func booststrapMultiCIDRRangeAllocator(t *testing.T,
	clientSet clientset.Interface,
	sharedInformer informers.SharedInformerFactory,
) *nodeipam.Controller {
	_, clusterCIDRv4, _ := netutils.ParseCIDRSloppy("10.96.0.0/12")     // allows up to 8K nodes
	_, clusterCIDRv6, _ := netutils.ParseCIDRSloppy("fd00:10:96::/112") // allows up to 8K nodes
	_, serviceCIDR, _ := netutils.ParseCIDRSloppy("10.94.0.0/24")       // does not matter for test - pick upto  250 services
	_, secServiceCIDR, _ := netutils.ParseCIDRSloppy("2001:db2::/120")  // does not matter for test - pick upto  250 services

	// order is ipv4 - ipv6 by convention for dual stack
	clusterCIDRs := []*net.IPNet{clusterCIDRv4, clusterCIDRv6}
	nodeMaskCIDRs := []int{24, 120}

	// set the current state of the informer, we can preseed nodes and ClusterCIDRs so we
	// can simulate the bootstrap
	initialCC := makeClusterCIDR("initial-cc", "10.2.0.0/16", "fd00:20:96::/112", nodeSelector(map[string][]string{"bootstrap": {"true"}}))
	if _, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Create(context.TODO(), initialCC, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	initialNode := makeNode("initial-node", map[string]string{"bootstrap": "true"})
	if _, err := clientSet.CoreV1().Nodes().Create(context.TODO(), initialNode, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

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

	return ipamController
}

func makeNode(name string, labels map[string]string) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
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
}

func makeClusterCIDR(name, ipv4CIDR, ipv6CIDR string, nodeSelector *v1.NodeSelector) *networkingv1alpha1.ClusterCIDR {
	return &networkingv1alpha1.ClusterCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: networkingv1alpha1.ClusterCIDRSpec{
			PerNodeHostBits: 8,
			IPv4:            ipv4CIDR,
			IPv6:            ipv6CIDR,
			NodeSelector:    nodeSelector,
		},
	}
}

func nodeSelector(labels map[string][]string) *v1.NodeSelector {
	testNodeSelector := &v1.NodeSelector{}

	for key, values := range labels {
		nst := v1.NodeSelectorTerm{
			MatchExpressions: []v1.NodeSelectorRequirement{
				{
					Key:      key,
					Operator: v1.NodeSelectorOpIn,
					Values:   values,
				},
			},
		}
		testNodeSelector.NodeSelectorTerms = append(testNodeSelector.NodeSelectorTerms, nst)
	}

	return testNodeSelector
}

func nodePodCIDRs(c clientset.Interface, name string) ([]string, error) {
	var node *v1.Node
	nodePollErr := wait.PollImmediate(time.Second, 5*time.Second, func() (bool, error) {
		var err error
		node, err = c.CoreV1().Nodes().Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return len(node.Spec.PodCIDRs) > 0, nil
	})

	return node.Spec.PodCIDRs, nodePollErr
}
