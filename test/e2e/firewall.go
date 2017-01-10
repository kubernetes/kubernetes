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

package e2e

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Firewall rule", func() {
	var firewall_test_name = "firewall-test"
	f := framework.NewDefaultFramework(firewall_test_name)

	var cs clientset.Interface
	var cloudConfig framework.CloudConfig
	var gceCloud *gcecloud.GCECloud

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")
		cs = f.ClientSet
		cloudConfig = framework.TestContext.CloudConfig
		gceCloud = cloudConfig.Provider.(*gcecloud.GCECloud)
	})

	// This test takes around 4 minutes to run
	It("[Slow] [Serial] should create valid firewall rules for LoadBalancer type service", func() {
		ns := f.Namespace.Name
		// This source ranges is just used to examine we have exact same things on LB firewall rules
		firewallTestSourceRanges := []string{"0.0.0.0/1", "128.0.0.0/1"}
		serviceName := "firewall-test-loadbalancer"

		jig := framework.NewServiceTestJig(cs, serviceName)
		nodesNames := jig.GetNodesNames(framework.MaxNodesForEndpointsTests)
		if len(nodesNames) <= 0 {
			framework.Failf("Expect at least 1 node, got: %v", nodesNames)
		}
		nodesSet := sets.NewString(nodesNames...)

		// OnlyLocal service is needed to examine which exact nodes the requests are being forwarded to by the Load Balancer on GCE
		By("Creating a LoadBalancer type service with onlyLocal annotation")
		svc := jig.CreateOnlyLocalLoadBalancerService(ns, serviceName,
			framework.LoadBalancerCreateTimeoutDefault, false, func(svc *v1.Service) {
				svc.Spec.Ports = []v1.ServicePort{{Protocol: "TCP", Port: framework.FirewallTestHttpPort}}
				svc.Spec.LoadBalancerSourceRanges = firewallTestSourceRanges
			})
		defer func() {
			jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeNodePort
				svc.Spec.LoadBalancerSourceRanges = nil
			})
			Expect(cs.Core().Services(svc.Namespace).Delete(svc.Name, nil)).NotTo(HaveOccurred())
		}()
		svcExternalIP := svc.Status.LoadBalancer.Ingress[0].IP

		By("Checking if service's firewall rules are correct")
		nodeTags := framework.GetInstanceTags(cloudConfig, nodesNames[0])
		expFw := framework.ConstructFirewallForLBService(svc, nodeTags.Items)
		fw, err := gceCloud.GetFirewall(expFw.Name)
		Expect(err).NotTo(HaveOccurred())
		Expect(framework.VerifyFirewallRule(fw, expFw, cloudConfig.Network, false)).NotTo(HaveOccurred())

		By(fmt.Sprintf("Creating netexec pods on at most %v nodes", framework.MaxNodesForEndpointsTests))
		for i, nodeName := range nodesNames {
			podName := fmt.Sprintf("netexec%v", i)
			jig.LaunchNetexecPodOnNode(f, nodeName, podName, framework.FirewallTestHttpPort, framework.FirewallTestUdpPort, true)
			defer func() {
				framework.Logf("Cleaning up the netexec pod: %v", podName)
				Expect(cs.Core().Pods(ns).Delete(podName, nil)).NotTo(HaveOccurred())
			}()
		}

		// Send requests from outside of the cluster because internal traffic is whitelisted
		By("Accessing the external service ip from outside, all non-master nodes should be reached")
		Expect(framework.TestHitNodesFromOutside(svcExternalIP, framework.FirewallTestHttpPort, framework.FirewallTimeoutDefault, nodesSet)).NotTo(HaveOccurred())

		// Check if there are overlapping tags on the firewall that extend beyond just the vms in our cluster
		// by removing the tag on one vm and make sure it doesn't get any traffic. This is an imperfect
		// simulation, we really want to check that traffic doesn't reach a vm outside the GKE cluster, but
		// that's much harder to do in the current e2e framework.
		By("Removing tags from one of the nodes")
		nodesSet.Delete(nodesNames[0])
		removedTags := framework.SetInstanceTags(cloudConfig, nodesNames[0], []string{})
		defer func() {
			By("Adding tags back to the node and wait till the traffic is recovered")
			nodesSet.Insert(nodesNames[0])
			framework.SetInstanceTags(cloudConfig, nodesNames[0], removedTags)
			// Make sure traffic is recovered before exit
			Expect(framework.TestHitNodesFromOutside(svcExternalIP, framework.FirewallTestHttpPort, framework.FirewallTimeoutDefault, nodesSet)).NotTo(HaveOccurred())
		}()

		By("Accessing serivce through the external ip and examine got no response from the node without tags")
		Expect(framework.TestHitNodesFromOutsideWithCount(svcExternalIP, framework.FirewallTestHttpPort, framework.FirewallTimeoutDefault, nodesSet, 15)).NotTo(HaveOccurred())
	})

	It("should have correct firewall rules for e2e cluster", func() {
		By("Gathering firewall related information")
		masterTags := framework.GetInstanceTags(cloudConfig, cloudConfig.MasterName)
		Expect(len(masterTags.Items)).Should(Equal(1))

		nodes := framework.GetReadySchedulableNodesOrDie(cs)
		if len(nodes.Items) <= 0 {
			framework.Failf("Expect at least 1 node, got: %v", len(nodes.Items))
		}
		nodeTags := framework.GetInstanceTags(cloudConfig, nodes.Items[0].Name)
		Expect(len(nodeTags.Items)).Should(Equal(1))

		By("Checking if e2e firewall rules are correct")
		for _, expFw := range framework.GetE2eFirewalls(cloudConfig.MasterName, masterTags.Items[0], nodeTags.Items[0], cloudConfig.Network) {
			fw, err := gceCloud.GetFirewall(expFw.Name)
			Expect(err).NotTo(HaveOccurred())
			Expect(framework.VerifyFirewallRule(fw, expFw, cloudConfig.Network, false)).NotTo(HaveOccurred())
		}

		By("Checking well known ports on master and nodes are not exposed externally")
		nodeAddrs := framework.NodeAddresses(nodes, v1.NodeExternalIP)
		Expect(len(nodeAddrs)).NotTo(BeZero())
		masterAddr := framework.GetMasterAddress(cs)
		flag, _ := framework.TestNotReachableHTTPTimeout(masterAddr, ports.ControllerManagerPort, framework.FirewallTestTcpTimeout)
		Expect(flag).To(BeTrue())
		flag, _ = framework.TestNotReachableHTTPTimeout(masterAddr, ports.SchedulerPort, framework.FirewallTestTcpTimeout)
		Expect(flag).To(BeTrue())
		flag, _ = framework.TestNotReachableHTTPTimeout(nodeAddrs[0], ports.KubeletPort, framework.FirewallTestTcpTimeout)
		Expect(flag).To(BeTrue())
		flag, _ = framework.TestNotReachableHTTPTimeout(nodeAddrs[0], ports.KubeletReadOnlyPort, framework.FirewallTestTcpTimeout)
		Expect(flag).To(BeTrue())
		flag, _ = framework.TestNotReachableHTTPTimeout(nodeAddrs[0], ports.ProxyStatusPort, framework.FirewallTestTcpTimeout)
		Expect(flag).To(BeTrue())
	})
})
