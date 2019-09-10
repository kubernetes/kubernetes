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

package network

import (
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	gcecloud "k8s.io/legacy-cloud-providers/gce"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	firewallTestTCPTimeout = time.Duration(1 * time.Second)
	// Set ports outside of 30000-32767, 80 and 8080 to avoid being whitelisted by the e2e cluster
	firewallTestHTTPPort = int32(29999)
	firewallTestUDPPort  = int32(29998)
)

var _ = SIGDescribe("Firewall rule", func() {
	var firewallTestName = "firewall-test"
	f := framework.NewDefaultFramework(firewallTestName)

	var cs clientset.Interface
	var cloudConfig framework.CloudConfig
	var gceCloud *gcecloud.Cloud

	ginkgo.BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")

		var err error
		cs = f.ClientSet
		cloudConfig = framework.TestContext.CloudConfig
		gceCloud, err = gce.GetGCECloud()
		framework.ExpectNoError(err)
	})

	// This test takes around 6 minutes to run
	ginkgo.It("[Slow] [Serial] should create valid firewall rules for LoadBalancer type service", func() {
		ns := f.Namespace.Name
		// This source ranges is just used to examine we have exact same things on LB firewall rules
		firewallTestSourceRanges := []string{"0.0.0.0/1", "128.0.0.0/1"}
		serviceName := "firewall-test-loadbalancer"

		ginkgo.By("Getting cluster ID")
		clusterID, err := gce.GetClusterID(cs)
		framework.ExpectNoError(err)
		e2elog.Logf("Got cluster ID: %v", clusterID)

		jig := e2eservice.NewTestJig(cs, serviceName)
		nodeList := jig.GetNodes(e2eservice.MaxNodesForEndpointsTests)
		gomega.Expect(nodeList).NotTo(gomega.BeNil())
		nodesNames := jig.GetNodesNames(e2eservice.MaxNodesForEndpointsTests)
		if len(nodesNames) <= 0 {
			e2elog.Failf("Expect at least 1 node, got: %v", nodesNames)
		}
		nodesSet := sets.NewString(nodesNames...)

		ginkgo.By("Creating a LoadBalancer type service with ExternalTrafficPolicy=Global")
		svc := jig.CreateLoadBalancerService(ns, serviceName, e2eservice.LoadBalancerCreateTimeoutDefault, func(svc *v1.Service) {
			svc.Spec.Ports = []v1.ServicePort{{Protocol: v1.ProtocolTCP, Port: firewallTestHTTPPort}}
			svc.Spec.LoadBalancerSourceRanges = firewallTestSourceRanges
		})
		defer func() {
			jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeNodePort
				svc.Spec.LoadBalancerSourceRanges = nil
			})
			err = cs.CoreV1().Services(svc.Namespace).Delete(svc.Name, nil)
			framework.ExpectNoError(err)
			ginkgo.By("Waiting for the local traffic health check firewall rule to be deleted")
			localHCFwName := gce.MakeHealthCheckFirewallNameForLBService(clusterID, cloudprovider.DefaultLoadBalancerName(svc), false)
			_, err := gce.WaitForFirewallRule(gceCloud, localHCFwName, false, e2eservice.LoadBalancerCleanupTimeout)
			framework.ExpectNoError(err)
		}()
		svcExternalIP := svc.Status.LoadBalancer.Ingress[0].IP

		ginkgo.By("Checking if service's firewall rule is correct")
		lbFw := gce.ConstructFirewallForLBService(svc, cloudConfig.NodeTag)
		fw, err := gceCloud.GetFirewall(lbFw.Name)
		framework.ExpectNoError(err)
		err = gce.VerifyFirewallRule(fw, lbFw, cloudConfig.Network, false)
		framework.ExpectNoError(err)

		ginkgo.By("Checking if service's nodes health check firewall rule is correct")
		nodesHCFw := gce.ConstructHealthCheckFirewallForLBService(clusterID, svc, cloudConfig.NodeTag, true)
		fw, err = gceCloud.GetFirewall(nodesHCFw.Name)
		framework.ExpectNoError(err)
		err = gce.VerifyFirewallRule(fw, nodesHCFw, cloudConfig.Network, false)
		framework.ExpectNoError(err)

		// OnlyLocal service is needed to examine which exact nodes the requests are being forwarded to by the Load Balancer on GCE
		ginkgo.By("Updating LoadBalancer service to ExternalTrafficPolicy=Local")
		svc = jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		})

		ginkgo.By("Waiting for the nodes health check firewall rule to be deleted")
		_, err = gce.WaitForFirewallRule(gceCloud, nodesHCFw.Name, false, e2eservice.LoadBalancerCleanupTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for the correct local traffic health check firewall rule to be created")
		localHCFw := gce.ConstructHealthCheckFirewallForLBService(clusterID, svc, cloudConfig.NodeTag, false)
		fw, err = gce.WaitForFirewallRule(gceCloud, localHCFw.Name, true, e2eservice.LoadBalancerCreateTimeoutDefault)
		framework.ExpectNoError(err)
		err = gce.VerifyFirewallRule(fw, localHCFw, cloudConfig.Network, false)
		framework.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("Creating netexec pods on at most %v nodes", e2eservice.MaxNodesForEndpointsTests))
		for i, nodeName := range nodesNames {
			podName := fmt.Sprintf("netexec%v", i)

			e2elog.Logf("Creating netexec pod %q on node %v in namespace %q", podName, nodeName, ns)
			pod := f.NewAgnhostPod(podName,
				"netexec",
				fmt.Sprintf("--http-port=%d", firewallTestHTTPPort),
				fmt.Sprintf("--udp-port=%d", firewallTestUDPPort))
			pod.ObjectMeta.Labels = jig.Labels
			pod.Spec.NodeName = nodeName
			pod.Spec.HostNetwork = true
			_, err := cs.CoreV1().Pods(ns).Create(pod)
			framework.ExpectNoError(err)
			framework.ExpectNoError(f.WaitForPodRunning(podName))
			e2elog.Logf("Netexec pod %q in namespace %q running", podName, ns)

			defer func() {
				e2elog.Logf("Cleaning up the netexec pod: %v", podName)
				err = cs.CoreV1().Pods(ns).Delete(podName, nil)
				framework.ExpectNoError(err)
			}()
		}

		// Send requests from outside of the cluster because internal traffic is whitelisted
		ginkgo.By("Accessing the external service ip from outside, all non-master nodes should be reached")
		err = framework.TestHitNodesFromOutside(svcExternalIP, firewallTestHTTPPort, e2eservice.LoadBalancerCreateTimeoutDefault, nodesSet)
		framework.ExpectNoError(err)

		// Check if there are overlapping tags on the firewall that extend beyond just the vms in our cluster
		// by removing the tag on one vm and make sure it doesn't get any traffic. This is an imperfect
		// simulation, we really want to check that traffic doesn't reach a vm outside the GKE cluster, but
		// that's much harder to do in the current e2e framework.
		ginkgo.By(fmt.Sprintf("Removing tags from one of the nodes: %v", nodesNames[0]))
		nodesSet.Delete(nodesNames[0])
		// Instance could run in a different zone in multi-zone test. Figure out which zone
		// it is in before proceeding.
		zone := cloudConfig.Zone
		if zoneInLabel, ok := nodeList.Items[0].Labels[v1.LabelZoneFailureDomain]; ok {
			zone = zoneInLabel
		}
		removedTags := gce.SetInstanceTags(cloudConfig, nodesNames[0], zone, []string{})
		defer func() {
			ginkgo.By("Adding tags back to the node and wait till the traffic is recovered")
			nodesSet.Insert(nodesNames[0])
			gce.SetInstanceTags(cloudConfig, nodesNames[0], zone, removedTags)
			// Make sure traffic is recovered before exit
			err = framework.TestHitNodesFromOutside(svcExternalIP, firewallTestHTTPPort, e2eservice.LoadBalancerCreateTimeoutDefault, nodesSet)
			framework.ExpectNoError(err)
		}()

		ginkgo.By("Accessing serivce through the external ip and examine got no response from the node without tags")
		err = framework.TestHitNodesFromOutsideWithCount(svcExternalIP, firewallTestHTTPPort, e2eservice.LoadBalancerCreateTimeoutDefault, nodesSet, 15)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should have correct firewall rules for e2e cluster", func() {
		nodes := framework.GetReadySchedulableNodesOrDie(cs)
		if len(nodes.Items) <= 0 {
			e2elog.Failf("Expect at least 1 node, got: %v", len(nodes.Items))
		}

		ginkgo.By("Checking if e2e firewall rules are correct")
		for _, expFw := range gce.GetE2eFirewalls(cloudConfig.MasterName, cloudConfig.MasterTag, cloudConfig.NodeTag, cloudConfig.Network, cloudConfig.ClusterIPRange) {
			fw, err := gceCloud.GetFirewall(expFw.Name)
			framework.ExpectNoError(err)
			err = gce.VerifyFirewallRule(fw, expFw, cloudConfig.Network, false)
			framework.ExpectNoError(err)
		}

		ginkgo.By("Checking well known ports on master and nodes are not exposed externally")
		nodeAddrs := e2enode.FirstAddress(nodes, v1.NodeExternalIP)
		if len(nodeAddrs) == 0 {
			e2elog.Failf("did not find any node addresses")
		}

		masterAddresses := framework.GetAllMasterAddresses(cs)
		for _, masterAddress := range masterAddresses {
			assertNotReachableHTTPTimeout(masterAddress, ports.InsecureKubeControllerManagerPort, firewallTestTCPTimeout)
			assertNotReachableHTTPTimeout(masterAddress, ports.InsecureSchedulerPort, firewallTestTCPTimeout)
		}
		assertNotReachableHTTPTimeout(nodeAddrs[0], ports.KubeletPort, firewallTestTCPTimeout)
		assertNotReachableHTTPTimeout(nodeAddrs[0], ports.KubeletReadOnlyPort, firewallTestTCPTimeout)
		assertNotReachableHTTPTimeout(nodeAddrs[0], ports.ProxyStatusPort, firewallTestTCPTimeout)
	})
})

func assertNotReachableHTTPTimeout(ip string, port int, timeout time.Duration) {
	result := framework.PokeHTTP(ip, port, "/", &framework.HTTPPokeParams{Timeout: timeout})
	if result.Status == framework.HTTPError {
		e2elog.Failf("Unexpected error checking for reachability of %s:%d: %v", ip, port, result.Error)
	}
	if result.Code != 0 {
		e2elog.Failf("Was unexpectedly able to reach %s:%d", ip, port)
	}
}
