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
	"context"
	"fmt"
	"strings"
	"time"

	e2econfig "k8s.io/kubernetes/test/e2e/framework/config"
	e2eutils "k8s.io/kubernetes/test/e2e/framework/utils"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/kubernetes/pkg/cluster/ports"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	gcecloud "k8s.io/legacy-cloud-providers/gce"

	"github.com/onsi/ginkgo"
)

const (
	firewallTestTCPTimeout = time.Duration(1 * time.Second)
	// Set ports outside of 30000-32767, 80 and 8080 to avoid being allowlisted by the e2e cluster
	firewallTestHTTPPort = int32(29999)
	firewallTestUDPPort  = int32(29998)
)

var _ = common.SIGDescribe("Firewall rule", func() {
	var firewallTestName = "firewall-test"
	f := framework.NewDefaultFramework(firewallTestName)

	var cs clientset.Interface
	var cloudConfig e2econfig.CloudConfig
	var gceCloud *gcecloud.Cloud

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce")

		var err error
		cs = f.ClientSet
		cloudConfig = e2econfig.TestContext.CloudConfig
		gceCloud, err = gce.GetGCECloud()
		e2eutils.ExpectNoError(err)
	})

	// This test takes around 6 minutes to run
	ginkgo.It("[Slow] [Serial] should create valid firewall rules for LoadBalancer type service", func() {
		ns := f.Namespace.Name
		// This source ranges is just used to examine we have exact same things on LB firewall rules
		firewallTestSourceRanges := []string{"0.0.0.0/1", "128.0.0.0/1"}
		serviceName := "firewall-test-loadbalancer"

		ginkgo.By("Getting cluster ID")
		clusterID, err := gce.GetClusterID(cs)
		e2eutils.ExpectNoError(err)
		e2eutils.Logf("Got cluster ID: %v", clusterID)

		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		nodeList, err := e2enode.GetBoundedReadySchedulableNodes(cs, e2eservice.MaxNodesForEndpointsTests)
		e2eutils.ExpectNoError(err)

		nodesNames := []string{}
		for _, node := range nodeList.Items {
			nodesNames = append(nodesNames, node.Name)
		}
		nodesSet := sets.NewString(nodesNames...)

		ginkgo.By("Creating a LoadBalancer type service with ExternalTrafficPolicy=Global")
		svc, err := jig.CreateLoadBalancerService(e2eservice.GetServiceLoadBalancerCreationTimeout(cs), func(svc *v1.Service) {
			svc.Spec.Ports = []v1.ServicePort{{Protocol: v1.ProtocolTCP, Port: firewallTestHTTPPort}}
			svc.Spec.LoadBalancerSourceRanges = firewallTestSourceRanges
		})
		e2eutils.ExpectNoError(err)
		defer func() {
			_, err = jig.UpdateService(func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeNodePort
				svc.Spec.LoadBalancerSourceRanges = nil
			})
			e2eutils.ExpectNoError(err)
			err = cs.CoreV1().Services(svc.Namespace).Delete(context.TODO(), svc.Name, metav1.DeleteOptions{})
			e2eutils.ExpectNoError(err)
			ginkgo.By("Waiting for the local traffic health check firewall rule to be deleted")
			localHCFwName := gce.MakeHealthCheckFirewallNameForLBService(clusterID, cloudprovider.DefaultLoadBalancerName(svc), false)
			_, err := gce.WaitForFirewallRule(gceCloud, localHCFwName, false, e2eservice.LoadBalancerCleanupTimeout)
			e2eutils.ExpectNoError(err)
		}()
		svcExternalIP := svc.Status.LoadBalancer.Ingress[0].IP

		ginkgo.By("Checking if service's firewall rule is correct")
		lbFw := gce.ConstructFirewallForLBService(svc, cloudConfig.NodeTag)
		fw, err := gceCloud.GetFirewall(lbFw.Name)
		e2eutils.ExpectNoError(err)
		err = gce.VerifyFirewallRule(fw, lbFw, cloudConfig.Network, false)
		e2eutils.ExpectNoError(err)

		ginkgo.By("Checking if service's nodes health check firewall rule is correct")
		nodesHCFw := gce.ConstructHealthCheckFirewallForLBService(clusterID, svc, cloudConfig.NodeTag, true)
		fw, err = gceCloud.GetFirewall(nodesHCFw.Name)
		e2eutils.ExpectNoError(err)
		err = gce.VerifyFirewallRule(fw, nodesHCFw, cloudConfig.Network, false)
		e2eutils.ExpectNoError(err)

		// OnlyLocal service is needed to examine which exact nodes the requests are being forwarded to by the Load Balancer on GCE
		ginkgo.By("Updating LoadBalancer service to ExternalTrafficPolicy=Local")
		svc, err = jig.UpdateService(func(svc *v1.Service) {
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		})
		e2eutils.ExpectNoError(err)

		ginkgo.By("Waiting for the nodes health check firewall rule to be deleted")
		_, err = gce.WaitForFirewallRule(gceCloud, nodesHCFw.Name, false, e2eservice.LoadBalancerCleanupTimeout)
		e2eutils.ExpectNoError(err)

		ginkgo.By("Waiting for the correct local traffic health check firewall rule to be created")
		localHCFw := gce.ConstructHealthCheckFirewallForLBService(clusterID, svc, cloudConfig.NodeTag, false)
		fw, err = gce.WaitForFirewallRule(gceCloud, localHCFw.Name, true, e2eservice.GetServiceLoadBalancerCreationTimeout(cs))
		e2eutils.ExpectNoError(err)
		err = gce.VerifyFirewallRule(fw, localHCFw, cloudConfig.Network, false)
		e2eutils.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("Creating netexec pods on at most %v nodes", e2eservice.MaxNodesForEndpointsTests))
		for i, nodeName := range nodesNames {
			podName := fmt.Sprintf("netexec%v", i)

			e2eutils.Logf("Creating netexec pod %q on node %v in namespace %q", podName, nodeName, ns)
			pod := e2epod.NewAgnhostPod(ns, podName, nil, nil, nil,
				"netexec",
				fmt.Sprintf("--http-port=%d", firewallTestHTTPPort),
				fmt.Sprintf("--udp-port=%d", firewallTestUDPPort))
			pod.ObjectMeta.Labels = jig.Labels
			nodeSelection := e2epod.NodeSelection{Name: nodeName}
			e2epod.SetNodeSelection(&pod.Spec, nodeSelection)
			pod.Spec.HostNetwork = true
			_, err := cs.CoreV1().Pods(ns).Create(context.TODO(), pod, metav1.CreateOptions{})
			e2eutils.ExpectNoError(err)
			e2eutils.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, podName, f.Namespace.Name, e2eutils.PodStartTimeout))
			e2eutils.Logf("Netexec pod %q in namespace %q running", podName, ns)

			defer func() {
				e2eutils.Logf("Cleaning up the netexec pod: %v", podName)
				err = cs.CoreV1().Pods(ns).Delete(context.TODO(), podName, metav1.DeleteOptions{})
				e2eutils.ExpectNoError(err)
			}()
		}

		// Send requests from outside of the cluster because internal traffic is allowlisted
		ginkgo.By("Accessing the external service ip from outside, all non-master nodes should be reached")
		err = testHitNodesFromOutside(svcExternalIP, firewallTestHTTPPort, e2eservice.GetServiceLoadBalancerPropagationTimeout(cs), nodesSet)
		e2eutils.ExpectNoError(err)

		// Check if there are overlapping tags on the firewall that extend beyond just the vms in our cluster
		// by removing the tag on one vm and make sure it doesn't get any traffic. This is an imperfect
		// simulation, we really want to check that traffic doesn't reach a vm outside the GKE cluster, but
		// that's much harder to do in the current e2e framework.
		ginkgo.By(fmt.Sprintf("Removing tags from one of the nodes: %v", nodesNames[0]))
		nodesSet.Delete(nodesNames[0])
		// Instance could run in a different zone in multi-zone test. Figure out which zone
		// it is in before proceeding.
		zone := cloudConfig.Zone
		if zoneInLabel, ok := nodeList.Items[0].Labels[v1.LabelFailureDomainBetaZone]; ok {
			zone = zoneInLabel
		} else if zoneInLabel, ok := nodeList.Items[0].Labels[v1.LabelTopologyZone]; ok {
			zone = zoneInLabel
		}
		removedTags := gce.SetInstanceTags(cloudConfig, nodesNames[0], zone, []string{})
		defer func() {
			ginkgo.By("Adding tags back to the node and wait till the traffic is recovered")
			nodesSet.Insert(nodesNames[0])
			gce.SetInstanceTags(cloudConfig, nodesNames[0], zone, removedTags)
			// Make sure traffic is recovered before exit
			err = testHitNodesFromOutside(svcExternalIP, firewallTestHTTPPort, e2eservice.GetServiceLoadBalancerPropagationTimeout(cs), nodesSet)
			e2eutils.ExpectNoError(err)
		}()

		ginkgo.By("Accessing serivce through the external ip and examine got no response from the node without tags")
		err = testHitNodesFromOutsideWithCount(svcExternalIP, firewallTestHTTPPort, e2eservice.GetServiceLoadBalancerPropagationTimeout(cs), nodesSet, 15)
		e2eutils.ExpectNoError(err)
	})

	ginkgo.It("should have correct firewall rules for e2e cluster", func() {
		ginkgo.By("Checking if e2e firewall rules are correct")
		for _, expFw := range gce.GetE2eFirewalls(cloudConfig.MasterName, cloudConfig.MasterTag, cloudConfig.NodeTag, cloudConfig.Network, cloudConfig.ClusterIPRange) {
			fw, err := gceCloud.GetFirewall(expFw.Name)
			e2eutils.ExpectNoError(err)
			err = gce.VerifyFirewallRule(fw, expFw, cloudConfig.Network, false)
			e2eutils.ExpectNoError(err)
		}
	})

	ginkgo.It("control plane should not expose well-known ports", func() {
		nodes, err := e2enode.GetReadySchedulableNodes(cs)
		e2eutils.ExpectNoError(err)

		ginkgo.By("Checking well known ports on master and nodes are not exposed externally")
		nodeAddr := e2enode.FirstAddress(nodes, v1.NodeExternalIP)
		if nodeAddr != "" {
			assertNotReachableHTTPTimeout(nodeAddr, "/", ports.KubeletPort, firewallTestTCPTimeout, false)
			assertNotReachableHTTPTimeout(nodeAddr, "/", ports.KubeletReadOnlyPort, firewallTestTCPTimeout, false)
			assertNotReachableHTTPTimeout(nodeAddr, "/", ports.ProxyStatusPort, firewallTestTCPTimeout, false)
		}

		controlPlaneAddresses := e2eutils.GetControlPlaneAddresses(cs)
		for _, instanceAddress := range controlPlaneAddresses {
			assertNotReachableHTTPTimeout(instanceAddress, "/healthz", ports.KubeControllerManagerPort, firewallTestTCPTimeout, true)
			assertNotReachableHTTPTimeout(instanceAddress, "/healthz", kubeschedulerconfig.DefaultKubeSchedulerPort, firewallTestTCPTimeout, true)
		}
	})
})

func assertNotReachableHTTPTimeout(ip, path string, port int, timeout time.Duration, enableHTTPS bool) {
	result := e2enetwork.PokeHTTP(ip, port, path, &e2enetwork.HTTPPokeParams{Timeout: timeout, EnableHTTPS: enableHTTPS})
	if result.Status == e2enetwork.HTTPError {
		e2eutils.Failf("Unexpected error checking for reachability of %s:%d: %v", ip, port, result.Error)
	}
	if result.Code != 0 {
		e2eutils.Failf("Was unexpectedly able to reach %s:%d", ip, port)
	}
}

// testHitNodesFromOutside checkes HTTP connectivity from outside.
func testHitNodesFromOutside(externalIP string, httpPort int32, timeout time.Duration, expectedHosts sets.String) error {
	return testHitNodesFromOutsideWithCount(externalIP, httpPort, timeout, expectedHosts, 1)
}

// testHitNodesFromOutsideWithCount checkes HTTP connectivity from outside with count.
func testHitNodesFromOutsideWithCount(externalIP string, httpPort int32, timeout time.Duration, expectedHosts sets.String,
	countToSucceed int) error {
	e2eutils.Logf("Waiting up to %v for satisfying expectedHosts for %v times", timeout, countToSucceed)
	hittedHosts := sets.NewString()
	count := 0
	condition := func() (bool, error) {
		result := e2enetwork.PokeHTTP(externalIP, int(httpPort), "/hostname", &e2enetwork.HTTPPokeParams{Timeout: 1 * time.Second})
		if result.Status != e2enetwork.HTTPSuccess {
			return false, nil
		}

		hittedHost := strings.TrimSpace(string(result.Body))
		if !expectedHosts.Has(hittedHost) {
			e2eutils.Logf("Error hitting unexpected host: %v, reset counter: %v", hittedHost, count)
			count = 0
			return false, nil
		}
		if !hittedHosts.Has(hittedHost) {
			hittedHosts.Insert(hittedHost)
			e2eutils.Logf("Missing %+v, got %+v", expectedHosts.Difference(hittedHosts), hittedHosts)
		}
		if hittedHosts.Equal(expectedHosts) {
			count++
			if count >= countToSucceed {
				return true, nil
			}
		}
		return false, nil
	}

	if err := wait.Poll(time.Second, timeout, condition); err != nil {
		return fmt.Errorf("error waiting for expectedHosts: %v, hittedHosts: %v, count: %v, expected count: %v",
			expectedHosts, hittedHosts, count, countToSucceed)
	}
	return nil
}
