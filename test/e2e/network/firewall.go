//go:build !providerless
// +build !providerless

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
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
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
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var cs clientset.Interface
	var cloudConfig framework.CloudConfig
	var gceCloud *gcecloud.Cloud

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce")

		var err error
		cs = f.ClientSet
		cloudConfig = framework.TestContext.CloudConfig
		gceCloud, err = gce.GetGCECloud()
		framework.ExpectNoError(err)
	})

	// This test takes around 6 minutes to run
	f.It(f.WithSlow(), f.WithSerial(), "should create valid firewall rules for LoadBalancer type service", func(ctx context.Context) {
		ns := f.Namespace.Name
		// This source ranges is just used to examine we have exact same things on LB firewall rules
		firewallTestSourceRanges := []string{"0.0.0.0/1", "128.0.0.0/1"}
		serviceName := "firewall-test-loadbalancer"

		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		nodeList, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, e2eservice.MaxNodesForEndpointsTests)
		framework.ExpectNoError(err)

		nodesNames := []string{}
		for _, node := range nodeList.Items {
			nodesNames = append(nodesNames, node.Name)
		}
		nodesSet := sets.NewString(nodesNames...)

		ginkgo.By("Creating a LoadBalancer type service with ExternalTrafficPolicy=Global")
		svc, err := jig.CreateLoadBalancerService(ctx, e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs), func(svc *v1.Service) {
			svc.Spec.Ports = []v1.ServicePort{{Protocol: v1.ProtocolTCP, Port: firewallTestHTTPPort}}
			svc.Spec.LoadBalancerSourceRanges = firewallTestSourceRanges
		})
		framework.ExpectNoError(err)

		// This configmap is guaranteed to exist after a Loadbalancer type service is created
		ginkgo.By("Getting cluster ID")
		clusterID, err := gce.GetClusterID(ctx, cs)
		framework.ExpectNoError(err)
		framework.Logf("Got cluster ID: %v", clusterID)

		defer func() {
			_, err = jig.UpdateService(ctx, func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeNodePort
				svc.Spec.LoadBalancerSourceRanges = nil
			})
			framework.ExpectNoError(err)
			err = cs.CoreV1().Services(svc.Namespace).Delete(ctx, svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
			ginkgo.By("Waiting for the local traffic health check firewall rule to be deleted")
			localHCFwName := gce.MakeHealthCheckFirewallNameForLBService(clusterID, cloudprovider.DefaultLoadBalancerName(svc), false)
			_, err := gce.WaitForFirewallRule(ctx, gceCloud, localHCFwName, false, e2eservice.LoadBalancerCleanupTimeout)
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
		svc, err = jig.UpdateService(ctx, func(svc *v1.Service) {
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
		})
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for the nodes health check firewall rule to be deleted")
		_, err = gce.WaitForFirewallRule(ctx, gceCloud, nodesHCFw.Name, false, e2eservice.LoadBalancerCleanupTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for the correct local traffic health check firewall rule to be created")
		localHCFw := gce.ConstructHealthCheckFirewallForLBService(clusterID, svc, cloudConfig.NodeTag, false)
		fw, err = gce.WaitForFirewallRule(ctx, gceCloud, localHCFw.Name, true, e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cs))
		framework.ExpectNoError(err)
		err = gce.VerifyFirewallRule(fw, localHCFw, cloudConfig.Network, false)
		framework.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("Creating netexec pods on at most %v nodes", e2eservice.MaxNodesForEndpointsTests))
		for i, nodeName := range nodesNames {
			podName := fmt.Sprintf("netexec%v", i)

			framework.Logf("Creating netexec pod %q on node %v in namespace %q", podName, nodeName, ns)
			pod := e2epod.NewAgnhostPod(ns, podName, nil, nil, nil,
				"netexec",
				fmt.Sprintf("--http-port=%d", firewallTestHTTPPort),
				fmt.Sprintf("--udp-port=%d", firewallTestUDPPort))
			pod.ObjectMeta.Labels = jig.Labels
			nodeSelection := e2epod.NodeSelection{Name: nodeName}
			e2epod.SetNodeSelection(&pod.Spec, nodeSelection)
			pod.Spec.HostNetwork = true
			_, err := cs.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name, framework.PodStartTimeout))
			framework.Logf("Netexec pod %q in namespace %q running", podName, ns)

			defer func() {
				framework.Logf("Cleaning up the netexec pod: %v", podName)
				err = cs.CoreV1().Pods(ns).Delete(ctx, podName, metav1.DeleteOptions{})
				framework.ExpectNoError(err)
			}()
		}

		// Send requests from outside of the cluster because internal traffic is allowlisted
		ginkgo.By("Accessing the external service ip from outside, all non-master nodes should be reached")
		err = testHitNodesFromOutside(svcExternalIP, firewallTestHTTPPort, e2eservice.GetServiceLoadBalancerPropagationTimeout(ctx, cs), nodesSet)
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
			err = testHitNodesFromOutside(svcExternalIP, firewallTestHTTPPort, e2eservice.GetServiceLoadBalancerPropagationTimeout(ctx, cs), nodesSet)
			framework.ExpectNoError(err)
		}()

		ginkgo.By("Accessing service through the external ip and examine got no response from the node without tags")
		err = testHitNodesFromOutsideWithCount(svcExternalIP, firewallTestHTTPPort, e2eservice.GetServiceLoadBalancerPropagationTimeout(ctx, cs), nodesSet, 15)
		framework.ExpectNoError(err)
	})

	ginkgo.It("control plane should not expose well-known ports", func(ctx context.Context) {
		nodes, err := e2enode.GetReadySchedulableNodes(ctx, cs)
		framework.ExpectNoError(err)

		ginkgo.By("Checking well known ports on master and nodes are not exposed externally")
		nodeAddr := e2enode.FirstAddress(nodes, v1.NodeExternalIP)
		if nodeAddr != "" {
			assertNotReachableHTTPTimeout(nodeAddr, "/", ports.KubeletPort, firewallTestTCPTimeout, false)
			assertNotReachableHTTPTimeout(nodeAddr, "/", ports.KubeletReadOnlyPort, firewallTestTCPTimeout, false)
			assertNotReachableHTTPTimeout(nodeAddr, "/", ports.ProxyStatusPort, firewallTestTCPTimeout, false)
		}

		controlPlaneAddresses := framework.GetControlPlaneAddresses(ctx, cs)
		for _, instanceAddress := range controlPlaneAddresses {
			assertNotReachableHTTPTimeout(instanceAddress, "/healthz", ports.KubeControllerManagerPort, firewallTestTCPTimeout, true)
			assertNotReachableHTTPTimeout(instanceAddress, "/healthz", kubeschedulerconfig.DefaultKubeSchedulerPort, firewallTestTCPTimeout, true)
		}
	})
})

func assertNotReachableHTTPTimeout(ip, path string, port int, timeout time.Duration, enableHTTPS bool) {
	result := e2enetwork.PokeHTTP(ip, port, path, &e2enetwork.HTTPPokeParams{Timeout: timeout, EnableHTTPS: enableHTTPS})
	if result.Status == e2enetwork.HTTPError {
		framework.Failf("Unexpected error checking for reachability of %s:%d: %v", ip, port, result.Error)
	}
	if result.Code != 0 {
		framework.Failf("Was unexpectedly able to reach %s:%d", ip, port)
	}
}

// testHitNodesFromOutside checks HTTP connectivity from outside.
func testHitNodesFromOutside(externalIP string, httpPort int32, timeout time.Duration, expectedHosts sets.String) error {
	return testHitNodesFromOutsideWithCount(externalIP, httpPort, timeout, expectedHosts, 1)
}

// testHitNodesFromOutsideWithCount checks HTTP connectivity from outside with count.
func testHitNodesFromOutsideWithCount(externalIP string, httpPort int32, timeout time.Duration, expectedHosts sets.String,
	countToSucceed int) error {
	framework.Logf("Waiting up to %v for satisfying expectedHosts for %v times", timeout, countToSucceed)
	hittedHosts := sets.NewString()
	count := 0
	condition := func() (bool, error) {
		result := e2enetwork.PokeHTTP(externalIP, int(httpPort), "/hostname", &e2enetwork.HTTPPokeParams{Timeout: 1 * time.Second})
		if result.Status != e2enetwork.HTTPSuccess {
			return false, nil
		}

		hittedHost := strings.TrimSpace(string(result.Body))
		if !expectedHosts.Has(hittedHost) {
			framework.Logf("Error hitting unexpected host: %v, reset counter: %v", hittedHost, count)
			count = 0
			return false, nil
		}
		if !hittedHosts.Has(hittedHost) {
			hittedHosts.Insert(hittedHost)
			framework.Logf("Missing %+v, got %+v", expectedHosts.Difference(hittedHosts), hittedHosts)
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
