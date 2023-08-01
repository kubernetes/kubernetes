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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controller/nodeipam"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	netutils "k8s.io/utils/net"
)

func TestIPAMMultiCIDRRangeAllocatorCIDRAllocate(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// set the feature gate to enable MultiCIDRRangeAllocator
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MultiCIDRRangeAllocator, true)()

	_, kubeConfig, tearDownFn := framework.StartTestServer(ctx, t, framework.TestServerSetup{
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

	go ipamController.Run(ctx)
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
			clusterCIDR:      makeClusterCIDR("ipv4-cc", "10.0.0.0/16", "", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "singlestack": {"true"}})),
			node:             makeNode("ipv4-node", map[string]string{"ipv4": "true", "singlestack": "true"}),
			expectedPodCIDRs: []string{"10.0.0.0/24"},
		},
		{
			name:             "Single stack IPv6 Pod CIDR assigned to a node",
			clusterCIDR:      makeClusterCIDR("ipv6-cc", "", "fd00:20:100::/112", 8, nodeSelector(map[string][]string{"ipv6": {"true"}})),
			node:             makeNode("ipv6-node", map[string]string{"ipv6": "true"}),
			expectedPodCIDRs: []string{"fd00:20:100::/120"},
		},
		{
			name:             "DualStack Pod CIDRs assigned to a node",
			clusterCIDR:      makeClusterCIDR("dualstack-cc", "192.168.0.0/16", "fd00:30:100::/112", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
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

func TestIPAMMultiCIDRRangeAllocatorCIDRRelease(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// set the feature gate to enable MultiCIDRRangeAllocator
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MultiCIDRRangeAllocator, true)()

	_, kubeConfig, tearDownFn := framework.StartTestServer(ctx, t, framework.TestServerSetup{
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

	go ipamController.Run(ctx)
	sharedInformer.Start(ctx.Done())

	t.Run("Pod CIDR release after node delete", func(t *testing.T) {
		// Create the test ClusterCIDR.
		clusterCIDR := makeClusterCIDR("dualstack-cc", "192.168.0.0/23", "fd00:30:100::/119", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}}))
		if _, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Create(context.TODO(), clusterCIDR, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}

		// Sleep for one second to make sure the controller process the new created ClusterCIDR.
		time.Sleep(1 * time.Second)

		// Create 1st node and validate that Pod CIDRs are correctly assigned.
		node1 := makeNode("dualstack-node", map[string]string{"ipv4": "true", "ipv6": "true"})
		expectedPodCIDRs1 := []string{"192.168.0.0/24", "fd00:30:100::/120"}
		if _, err := clientSet.CoreV1().Nodes().Create(context.TODO(), node1, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
		if gotPodCIDRs, err := nodePodCIDRs(clientSet, node1.Name); err != nil {
			t.Fatal(err)
		} else if !reflect.DeepEqual(expectedPodCIDRs1, gotPodCIDRs) {
			t.Errorf("unexpected result, expected Pod CIDRs %v but got %v", expectedPodCIDRs1, gotPodCIDRs)
		}

		// Create 2nd node and validate that Pod CIDRs are correctly assigned.
		node2 := makeNode("dualstack-node-2", map[string]string{"ipv4": "true", "ipv6": "true"})
		expectedPodCIDRs2 := []string{"192.168.1.0/24", "fd00:30:100::100/120"}
		if _, err := clientSet.CoreV1().Nodes().Create(context.TODO(), node2, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
		if gotPodCIDRs, err := nodePodCIDRs(clientSet, node2.Name); err != nil {
			t.Fatal(err)
		} else if !reflect.DeepEqual(expectedPodCIDRs2, gotPodCIDRs) {
			t.Errorf("unexpected result, expected Pod CIDRs %v but got %v", expectedPodCIDRs2, gotPodCIDRs)
		}

		// Delete the 1st node, to validate that the PodCIDRs are released.
		if err := clientSet.CoreV1().Nodes().Delete(context.TODO(), node1.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatal(err)
		}

		// Create 3rd node, validate that it has Pod CIDRs assigned from the released CIDR.
		node3 := makeNode("dualstack-node-3", map[string]string{"ipv4": "true", "ipv6": "true"})
		expectedPodCIDRs3 := []string{"192.168.0.0/24", "fd00:30:100::/120"}
		if _, err := clientSet.CoreV1().Nodes().Create(context.TODO(), node3, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
		if gotPodCIDRs, err := nodePodCIDRs(clientSet, node3.Name); err != nil {
			t.Fatal(err)
		} else if !reflect.DeepEqual(expectedPodCIDRs3, gotPodCIDRs) {
			t.Errorf("unexpected result, expected Pod CIDRs %v but got %v", expectedPodCIDRs3, gotPodCIDRs)
		}
	})
}

func TestIPAMMultiCIDRRangeAllocatorClusterCIDRDelete(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// set the feature gate to enable MultiCIDRRangeAllocator.
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MultiCIDRRangeAllocator, true)()

	_, kubeConfig, tearDownFn := framework.StartTestServer(ctx, t, framework.TestServerSetup{
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

	go ipamController.Run(ctx)
	sharedInformer.Start(ctx.Done())

	t.Run("delete cc with node associated", func(t *testing.T) {

		// Create a ClusterCIDR.
		clusterCIDR := makeClusterCIDR("dualstack-cc-del", "192.168.0.0/23", "fd00:30:100::/119", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}}))
		if _, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Create(context.TODO(), clusterCIDR, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}

		// Sleep for one second to make sure the controller processes the newly created ClusterCIDR.
		time.Sleep(1 * time.Second)

		// Create a node, which gets pod CIDR from the clusterCIDR created above.
		node := makeNode("dualstack-node", map[string]string{"ipv4": "true", "ipv6": "true"})
		expectedPodCIDRs := []string{"192.168.0.0/24", "fd00:30:100::/120"}
		if _, err := clientSet.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
		if gotPodCIDRs, err := nodePodCIDRs(clientSet, node.Name); err != nil {
			t.Fatal(err)
		} else if !reflect.DeepEqual(expectedPodCIDRs, gotPodCIDRs) {
			t.Errorf("unexpected result, expected Pod CIDRs %v but got %v", expectedPodCIDRs, gotPodCIDRs)
		}

		// Delete the ClusterCIDR.
		if err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Delete(context.TODO(), clusterCIDR.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatal(err)
		}

		// Sleep for five seconds to make sure the ClusterCIDR exists with a deletion timestamp after marked for deletion.
		time.Sleep(5 * time.Second)

		// Make sure that the ClusterCIDR is not deleted, as there is a node associated with it.
		cc, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Get(context.TODO(), clusterCIDR.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if cc == nil {
			t.Fatalf("expected Cluster CIDR got nil")
		}
		if cc.DeletionTimestamp.IsZero() {
			t.Fatalf("expected Cluster CIDR to have set a deletion timestamp ")
		}

		//Delete the node.
		if err := clientSet.CoreV1().Nodes().Delete(context.TODO(), node.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatal(err)
		}

		// Poll to make sure that the Node is deleted.
		if err := wait.PollImmediate(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
			_, err := clientSet.CoreV1().Nodes().Get(context.TODO(), node.Name, metav1.GetOptions{})
			if err != nil && !apierrors.IsNotFound(err) {
				return false, err
			}
			return apierrors.IsNotFound(err), nil
		}); err != nil {
			t.Fatalf("failed while waiting for Node %q to be deleted: %v", node.Name, err)
		}

		// Poll to make sure that the ClusterCIDR is now deleted, as there is no node associated with it.
		if err := wait.PollImmediate(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
			_, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Get(context.TODO(), clusterCIDR.Name, metav1.GetOptions{})
			if err != nil && !apierrors.IsNotFound(err) {
				return false, err
			}
			return apierrors.IsNotFound(err), nil
		}); err != nil {
			t.Fatalf("failed while waiting for ClusterCIDR %q to be deleted: %v", clusterCIDR.Name, err)
		}
	})
}

func TestIPAMMultiCIDRRangeAllocatorClusterCIDRTerminate(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// set the feature gate to enable MultiCIDRRangeAllocator.
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MultiCIDRRangeAllocator, true)()

	_, kubeConfig, tearDownFn := framework.StartTestServer(ctx, t, framework.TestServerSetup{
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

	go ipamController.Run(ctx)
	sharedInformer.Start(ctx.Done())

	t.Run("Pod CIDRS must not be allocated from a terminating CC", func(t *testing.T) {

		// Create a ClusterCIDR which is the best match based on number of matching labels.
		clusterCIDR := makeClusterCIDR("dualstack-cc-del", "192.168.0.0/23", "fd00:30:100::/119", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}}))
		if _, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Create(context.TODO(), clusterCIDR, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}

		// Create a ClusterCIDR which has fewer matching labels than the previous ClusterCIDR.
		clusterCIDR2 := makeClusterCIDR("few-label-match-cc-del", "10.1.0.0/23", "fd12:30:100::/119", 8, nodeSelector(map[string][]string{"ipv4": {"true"}}))
		if _, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Create(context.TODO(), clusterCIDR2, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}

		// Sleep for one second to make sure the controller processes the newly created ClusterCIDR.
		time.Sleep(1 * time.Second)

		// Create a node, which gets pod CIDR from the clusterCIDR created above.
		node := makeNode("dualstack-node", map[string]string{"ipv4": "true", "ipv6": "true"})
		expectedPodCIDRs := []string{"192.168.0.0/24", "fd00:30:100::/120"}
		if _, err := clientSet.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
		if gotPodCIDRs, err := nodePodCIDRs(clientSet, node.Name); err != nil {
			t.Fatal(err)
		} else if !reflect.DeepEqual(expectedPodCIDRs, gotPodCIDRs) {
			t.Errorf("unexpected result, expected Pod CIDRs %v but got %v", expectedPodCIDRs, gotPodCIDRs)
		}

		// Delete the ClusterCIDR
		if err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Delete(context.TODO(), clusterCIDR.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatal(err)
		}

		// Make sure that the ClusterCIDR is not deleted, as there is a node associated with it.
		cc, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Get(context.TODO(), clusterCIDR.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if cc == nil {
			t.Fatalf("expected Cluster CIDR got nil")
		}
		if cc.DeletionTimestamp.IsZero() {
			t.Fatalf("expected Cluster CIDR to have set a deletion timestamp ")
		}

		// Create a node, which should get Pod CIDRs from the ClusterCIDR with fewer matching label Count,
		// as the best match ClusterCIDR is marked as terminating.
		node2 := makeNode("dualstack-node-2", map[string]string{"ipv4": "true", "ipv6": "true"})
		expectedPodCIDRs2 := []string{"10.1.0.0/24", "fd12:30:100::/120"}
		if _, err := clientSet.CoreV1().Nodes().Create(context.TODO(), node2, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
		if gotPodCIDRs2, err := nodePodCIDRs(clientSet, node2.Name); err != nil {
			t.Fatal(err)
		} else if !reflect.DeepEqual(expectedPodCIDRs2, gotPodCIDRs2) {
			t.Errorf("unexpected result, expected Pod CIDRs %v but got %v", expectedPodCIDRs2, gotPodCIDRs2)
		}
	})
}

func TestIPAMMultiCIDRRangeAllocatorClusterCIDRTieBreak(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// set the feature gate to enable MultiCIDRRangeAllocator
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MultiCIDRRangeAllocator, true)()

	_, kubeConfig, tearDownFn := framework.StartTestServer(ctx, t, framework.TestServerSetup{
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

	go ipamController.Run(ctx)
	sharedInformer.Start(ctx.Done())

	tests := []struct {
		name             string
		clusterCIDRs     []*networkingv1alpha1.ClusterCIDR
		node             *v1.Node
		expectedPodCIDRs []string
	}{
		{
			name: "ClusterCIDR with highest matching labels",
			clusterCIDRs: []*networkingv1alpha1.ClusterCIDR{
				makeClusterCIDR("single-label-match-cc", "192.168.0.0/23", "fd00:30:100::/119", 8, nodeSelector(map[string][]string{"match": {"single"}})),
				makeClusterCIDR("double-label-match-cc", "10.0.0.0/23", "fd12:30:200::/119", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
			},
			node:             makeNode("dualstack-node", map[string]string{"ipv4": "true", "ipv6": "true", "match": "single"}),
			expectedPodCIDRs: []string{"10.0.0.0/24", "fd12:30:200::/120"},
		},
		{
			name: "ClusterCIDR with fewer allocatable Pod CIDRs",
			clusterCIDRs: []*networkingv1alpha1.ClusterCIDR{
				makeClusterCIDR("single-label-match-cc", "192.168.0.0/23", "fd00:30:100::/119", 8, nodeSelector(map[string][]string{"match": {"single"}})),
				makeClusterCIDR("double-label-match-cc", "10.0.0.0/20", "fd12:30:200::/116", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
				makeClusterCIDR("few-alloc-cc", "172.16.0.0/23", "fd34:30:100::/119", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
			},
			node:             makeNode("dualstack-node", map[string]string{"ipv4": "true", "ipv6": "true", "match": "single"}),
			expectedPodCIDRs: []string{"172.16.0.0/24", "fd34:30:100::/120"},
		},
		{
			name: "ClusterCIDR with lower perNodeHostBits",
			clusterCIDRs: []*networkingv1alpha1.ClusterCIDR{
				makeClusterCIDR("single-label-match-cc", "192.168.0.0/23", "fd00:30:100::/119", 8, nodeSelector(map[string][]string{"match": {"single"}})),
				makeClusterCIDR("double-label-match-cc", "10.0.0.0/20", "fd12:30:200::/116", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
				makeClusterCIDR("few-alloc-cc", "172.16.0.0/23", "fd34:30:100::/119", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
				makeClusterCIDR("low-pernodehostbits-cc", "172.31.0.0/24", "fd35:30:100::/120", 7, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
			},
			node:             makeNode("dualstack-node", map[string]string{"ipv4": "true", "ipv6": "true", "match": "single"}),
			expectedPodCIDRs: []string{"172.31.0.0/25", "fd35:30:100::/121"},
		},
		{
			name: "ClusterCIDR with label having lower alphanumeric value",
			clusterCIDRs: []*networkingv1alpha1.ClusterCIDR{
				makeClusterCIDR("single-label-match-cc", "192.168.0.0/23", "fd00:30:100::/119", 8, nodeSelector(map[string][]string{"match": {"single"}})),
				makeClusterCIDR("double-label-match-cc", "10.0.0.0/20", "fd12:30:200::/116", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
				makeClusterCIDR("few-alloc-cc", "172.16.0.0/23", "fd34:30:100::/119", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
				makeClusterCIDR("low-pernodehostbits-cc", "172.31.0.0/24", "fd35:30:100::/120", 7, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
				makeClusterCIDR("low-alpha-cc", "192.169.0.0/24", "fd12:40:100::/120", 7, nodeSelector(map[string][]string{"apv4": {"true"}, "bpv6": {"true"}})),
			},
			node:             makeNode("dualstack-node", map[string]string{"apv4": "true", "bpv6": "true", "ipv4": "true", "ipv6": "true", "match": "single"}),
			expectedPodCIDRs: []string{"192.169.0.0/25", "fd12:40:100::/121"},
		},
		{
			name: "ClusterCIDR with alphanumerically smaller IP address",
			clusterCIDRs: []*networkingv1alpha1.ClusterCIDR{
				makeClusterCIDR("single-label-match-cc", "192.168.0.0/23", "fd00:30:100::/119", 8, nodeSelector(map[string][]string{"match": {"single"}})),
				makeClusterCIDR("double-label-match-cc", "10.0.0.0/20", "fd12:30:200::/116", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
				makeClusterCIDR("few-alloc-cc", "172.16.0.0/23", "fd34:30:100::/119", 8, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
				makeClusterCIDR("low-pernodehostbits-cc", "172.31.0.0/24", "fd35:30:100::/120", 7, nodeSelector(map[string][]string{"ipv4": {"true"}, "ipv6": {"true"}})),
				makeClusterCIDR("low-alpha-cc", "192.169.0.0/24", "fd12:40:100::/120", 7, nodeSelector(map[string][]string{"apv4": {"true"}, "bpv6": {"true"}})),
				makeClusterCIDR("low-ip-cc", "10.1.0.0/24", "fd00:10:100::/120", 7, nodeSelector(map[string][]string{"apv4": {"true"}, "bpv6": {"true"}})),
			},
			node:             makeNode("dualstack-node", map[string]string{"apv4": "true", "bpv6": "true", "ipv4": "true", "ipv6": "true", "match": "single"}),
			expectedPodCIDRs: []string{"10.1.0.0/25", "fd00:10:100::/121"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, clusterCIDR := range test.clusterCIDRs {
				// Create the test ClusterCIDR
				if _, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Create(context.TODO(), clusterCIDR, metav1.CreateOptions{}); err != nil {
					t.Fatal(err)
				}
			}

			// Sleep for one second to make sure the controller process the new created ClusterCIDR.
			time.Sleep(1 * time.Second)

			// Create a node and validate that Pod CIDRs are correctly assigned.
			if _, err := clientSet.CoreV1().Nodes().Create(context.TODO(), test.node, metav1.CreateOptions{}); err != nil {
				t.Fatal(err)
			}
			if gotPodCIDRs, err := nodePodCIDRs(clientSet, test.node.Name); err != nil {
				t.Fatal(err)
			} else if !reflect.DeepEqual(test.expectedPodCIDRs, gotPodCIDRs) {
				t.Errorf("unexpected result, expected Pod CIDRs %v but got %v", test.expectedPodCIDRs, gotPodCIDRs)
			}

			// Delete the node.
			if err := clientSet.CoreV1().Nodes().Delete(context.TODO(), test.node.Name, metav1.DeleteOptions{}); err != nil {
				t.Fatal(err)
			}

			// Wait till the Node is deleted.
			if err := wait.PollImmediate(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
				_, err := clientSet.CoreV1().Nodes().Get(context.TODO(), test.node.Name, metav1.GetOptions{})
				if err != nil && !apierrors.IsNotFound(err) {
					return false, err
				}
				return apierrors.IsNotFound(err), nil
			}); err != nil {
				t.Fatalf("failed while waiting for Node %q to be deleted: %v", test.node.Name, err)
			}

			// Delete the Cluster CIDRs.
			for _, clusterCIDR := range test.clusterCIDRs {
				// Delete the test ClusterCIDR.
				if err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Delete(context.TODO(), clusterCIDR.Name, metav1.DeleteOptions{}); err != nil {
					t.Fatal(err)
				}

				// Wait till the ClusterCIDR is deleted.
				if err := wait.PollImmediate(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
					_, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Get(context.TODO(), clusterCIDR.Name, metav1.GetOptions{})
					if err != nil && !apierrors.IsNotFound(err) {
						return false, err
					}
					return apierrors.IsNotFound(err), nil
				}); err != nil {
					t.Fatalf("failed while waiting for ClusterCIDR %q to be deleted: %v", clusterCIDR.Name, err)
				}
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

	// set the current state of the informer, we can pre-seed nodes and ClusterCIDRs, so that we
	// can simulate the bootstrap
	initialCC := makeClusterCIDR("initial-cc", "10.2.0.0/16", "fd00:20:96::/112", 8, nodeSelector(map[string][]string{"bootstrap": {"true"}}))
	if _, err := clientSet.NetworkingV1alpha1().ClusterCIDRs().Create(context.TODO(), initialCC, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	initialNode := makeNode("initial-node", map[string]string{"bootstrap": "true"})
	if _, err := clientSet.CoreV1().Nodes().Create(context.TODO(), initialNode, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	_, ctx := ktesting.NewTestContext(t)
	ipamController, err := nodeipam.NewNodeIpamController(
		ctx,
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

func makeClusterCIDR(name, ipv4CIDR, ipv6CIDR string, perNodeHostBits int32, nodeSelector *v1.NodeSelector) *networkingv1alpha1.ClusterCIDR {
	return &networkingv1alpha1.ClusterCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: networkingv1alpha1.ClusterCIDRSpec{
			PerNodeHostBits: perNodeHostBits,
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
	nodePollErr := wait.PollImmediate(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		var err error
		node, err = c.CoreV1().Nodes().Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return len(node.Spec.PodCIDRs) > 0, nil
	})

	return node.Spec.PodCIDRs, nodePollErr
}
