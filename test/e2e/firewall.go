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
	"bytes"
	"fmt"
	"net"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/cloudprovider"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	compute "google.golang.org/api/compute/v1"
)

const (
	firewallTimeoutDefault = 3 * time.Minute
	firewallTestTcpTimeout = time.Duration(1 * time.Second)
	// Set ports outside of 30000-32767, 80 and 8080 to avoid being whitelisted by the e2e cluster
	firewallTestHttpPort = int32(29999)
	firewallTestUdpPort  = int32(29998)
)

var _ = framework.KubeDescribe("Firewall rule", func() {
	var firewall_test_name = "firewall-test"
	f := framework.NewDefaultFramework(firewall_test_name)

	// This test takes around 3 minutes to run
	It("[Slow] [Serial] should create valid firewall rules for LoadBalancer type service", func() {
		framework.SkipUnlessProviderIs("gce")
		c := f.ClientSet
		ns := f.Namespace.Name
		cloudConfig := framework.TestContext.CloudConfig
		// This source ranges is just used to examine we have the same things on LB firewall rules
		firewallTestSourceRanges := []string{"0.0.0.0/1", "128.0.0.0/1"}
		serviceName := "firewall-test-loadbalancer"

		jig := NewServiceTestJig(c, serviceName)
		nodesNames := jig.GetNodesNames(maxNodesForEndpointsTests)
		if len(nodesNames) <= 0 {
			framework.Failf("Expect at least 1 node, got: %v", nodesNames)
		}
		nodesSet := sets.NewString(nodesNames...)

		// OnlyLocal service is needed to examine which exact nodes the requests are being forwarded to by the Load Balancer on GCE
		By("Creating a LoadBalancer type service with onlyLocal annotation")
		svc := jig.CreateOnlyLocalLoadBalancerServicePortSourceRanges(ns, serviceName, firewallTestHttpPort,
			loadBalancerCreateTimeoutDefault, false, firewallTestSourceRanges)
		healthCheckNodePort := int(service.GetServiceHealthCheckNodePort(svc))
		if healthCheckNodePort == 0 {
			framework.Failf("Service HealthCheck NodePort was not allocated")
		}
		defer func() {
			jig.ChangeServiceType(svc.Namespace, svc.Name, api.ServiceTypeClusterIP, loadBalancerCreateTimeoutDefault)
			Expect(c.Core().Services(svc.Namespace).Delete(svc.Name, nil)).NotTo(HaveOccurred())
		}()
		svcExternalIP := svc.Status.LoadBalancer.Ingress[0].IP

		By("Checking if service's firewall rules are correct")
		nodeTags, err := getInstanceTags(cloudConfig, nodesNames[0])
		Expect(err).NotTo(HaveOccurred())
		Expect(examineSvcFirewallRules(cloudConfig, svc, nodeTags.Items)).NotTo(HaveOccurred())

		By(fmt.Sprintf("Creating netexec pods on at most %v nodes", maxNodesForEndpointsTests))
		for i, nodeName := range nodesNames {
			podName := fmt.Sprintf("netexec%v", i)
			jig.LaunchNetexecPodOnNode(f, nodeName, podName, firewallTestHttpPort, firewallTestUdpPort, true)
			defer func() {
				framework.Logf("Cleaning up the netexec pod: %v", podName)
				Expect(c.Core().Pods(ns).Delete(podName, nil)).NotTo(HaveOccurred())
			}()
		}

		// Send requests from outside of the cluster because internal traffic is whitelisted
		By("Accessing the external service ip from outside, all non-master nodes should be reached")
		Expect(examineHitNodesFromOutside(svcExternalIP, firewallTestHttpPort, firewallTimeoutDefault, nodesSet)).NotTo(HaveOccurred())

		// Check if there are overlapping tags on the firewall that extend beyond just the vms in our cluster
		// by removing the tag on one vm and make sure it doesn't get any traffic. This is an imperfect
		// simulation, we really want to check that traffic doesn't reach a vm outside the GKE cluster, but
		// that's much harder to do in the current e2e framework.
		By("Removing tags from one of the nodes")
		nodesSet.Delete(nodesNames[0])
		removedTags, err := setInstanceTags(cloudConfig, nodesNames[0], []string{})
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("Adding tags back to the node")
			nodesSet.Insert(nodesNames[0])
			_, err := setInstanceTags(cloudConfig, nodesNames[0], removedTags)
			Expect(err).NotTo(HaveOccurred())
			// Make sure traffic is recovered before exit
			Expect(examineHitNodesFromOutside(svcExternalIP, firewallTestHttpPort, firewallTimeoutDefault, nodesSet)).NotTo(HaveOccurred())
		}()

		By("Accessing serivce through the external ip and examine got no response from the node without tags")
		Expect(examineHitNodesFromOutsideWithCount(svcExternalIP, firewallTestHttpPort, firewallTimeoutDefault, nodesSet, 15)).NotTo(HaveOccurred())
	})

	It("should have correct firewall rules for e2e cluster", func() {
		framework.SkipUnlessProviderIs("gce")
		cloudConfig := framework.TestContext.CloudConfig
		gceCloud := cloudConfig.Provider.(*gcecloud.GCECloud)

		By("Gathering firewall related information")
		masterTags, err := getInstanceTags(cloudConfig, cloudConfig.MasterName)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(masterTags.Items)).Should(Equal(1))

		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		if len(nodes.Items) <= 0 {
			framework.Failf("Expect at least 1 node, got: %v", len(nodes.Items))
		}
		nodeTags, err := getInstanceTags(cloudConfig, nodes.Items[0].Name)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(nodeTags.Items)).Should(Equal(1))

		instancePrefix, err := getInstancePrefix(cloudConfig.MasterName)
		Expect(err).NotTo(HaveOccurred())

		clusterIpRange := getClusterIpRange()

		By("Checking if e2e firewall rules are correct")
		for _, expFw := range getE2eFirewalls(cloudConfig.MasterName, masterTags.Items[0], instancePrefix, nodeTags.Items[0], cloudConfig.Network, clusterIpRange) {
			fw, err := gceCloud.GetFirewall(expFw.Name)
			Expect(err).NotTo(HaveOccurred())
			Expect(verifyFirewall(fw, expFw, cloudConfig.Network)).NotTo(HaveOccurred())
		}

		By("Checking well known ports on master and nodes are not exposed externally")
		nodeAddrs := framework.NodeAddresses(nodes, api.NodeExternalIP)
		Expect(len(nodeAddrs)).NotTo(BeZero())
		masterAddr := framework.GetMasterAddress(f.ClientSet)
		examinePortNotReachable("tcp", masterAddr, ports.ControllerManagerPort)
		examinePortNotReachable("tcp", masterAddr, ports.SchedulerPort)
		examinePortNotReachable("tcp", nodeAddrs[0], ports.KubeletPort)
		examinePortNotReachable("tcp", nodeAddrs[0], ports.KubeletReadOnlyPort)
		examinePortNotReachable("tcp", nodeAddrs[0], ports.ProxyStatusPort)
	})
})

func examineSvcFirewallRules(cloudConfig framework.CloudConfig, svc *api.Service, nodesTags []string) error {
	gceCloud := cloudConfig.Provider.(*gcecloud.GCECloud)
	fw, err := gceCloud.GetFirewall(makeFirewallName(cloudprovider.GetLoadBalancerName(svc)))
	if err != nil {
		return err
	}

	if err := sameStringArray(fw.TargetTags, nodesTags); err != nil {
		return fmt.Errorf("incorrect target tags: %v", err)
	}

	if err := verifySvcProtocolsPorts(fw.Allowed, svc.Spec.Ports); err != nil {
		return err
	}

	if err := verifySvcSourceRanges(fw.SourceRanges, svc.Spec.LoadBalancerSourceRanges); err != nil {
		return err
	}

	return nil
}

// This should match the formatting of makeFirewallName() in pkg/cloudprovider/providers/gce/gce.go
func makeFirewallName(name string) string {
	return fmt.Sprintf("k8s-fw-%s", name)
}

// sameStringArray verifies whether two string arrays have the same strings, return error if not.
// Order of elements does not matter.
func sameStringArray(result, expected []string) error {
	res := sets.NewString(result...)
	exp := sets.NewString(expected...)
	diff := res.Difference(exp)
	if len(diff) != 0 {
		return fmt.Errorf("found differences: %v", diff)
	}
	return nil
}

func verifySvcProtocolsPorts(alloweds []*compute.FirewallAllowed, servicePorts []api.ServicePort) error {
	resultProtocolPorts := packFwProtocolsPorts(alloweds)
	expectedProtocolPorts := []string{}
	for _, sp := range servicePorts {
		expectedProtocolPorts = append(expectedProtocolPorts, strings.ToLower(string(sp.Protocol)+"/"+strconv.Itoa(int(sp.Port))))
	}
	return sameStringArray(resultProtocolPorts, expectedProtocolPorts)
}

func packFwProtocolsPorts(alloweds []*compute.FirewallAllowed) []string {
	protocolPorts := []string{}
	for _, allowed := range alloweds {
		for _, port := range allowed.Ports {
			protocolPorts = append(protocolPorts, strings.ToLower(allowed.IPProtocol+"/"+port))
		}
	}
	return protocolPorts
}

func verifySvcSourceRanges(fwSourceRanges, svcSourceRanges []string) error {
	if svcSourceRanges == nil {
		svcSourceRanges = append(svcSourceRanges, "0.0.0.0/0")
	}
	return sameStringArray(fwSourceRanges, svcSourceRanges)
}

func examineHitNodesFromOutside(externalIP string, httpPort int32, timeout time.Duration, expectedHosts sets.String) error {
	return examineHitNodesFromOutsideWithCount(externalIP, httpPort, timeout, expectedHosts, 1)
}

func examineHitNodesFromOutsideWithCount(externalIP string, httpPort int32, timeout time.Duration, expectedHosts sets.String, countToSucceed int) error {
	framework.Logf("Waiting up to %v for satisfying expectedHosts for %v times", timeout, countToSucceed)
	hittedHosts := sets.NewString()
	count := 0
	condition := func() (bool, error) {
		var respBody bytes.Buffer
		reached, err := testReachableHTTPWithContentTimeout(externalIP, int(httpPort), "/hostname", "", &respBody, firewallTestTcpTimeout)
		if err != nil || !reached {
			return false, nil
		}
		hittedHost := strings.TrimSpace(respBody.String())
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
		return fmt.Errorf("error waiting for expectedHosts: %v, hittedHosts: %v, count: %v, expected count: %v", expectedHosts, hittedHosts, count, countToSucceed)
	}
	return nil
}

func getInstanceTags(cloudConfig framework.CloudConfig, instanceName string) (*compute.Tags, error) {
	gceCloud := cloudConfig.Provider.(*gcecloud.GCECloud)
	res, err := gceCloud.GetComputeService().Instances.Get(cloudConfig.ProjectID, cloudConfig.Zone, instanceName).Do()
	if err != nil {
		return nil, err
	}
	return res.Tags, nil
}

func setInstanceTags(cloudConfig framework.CloudConfig, instanceName string, tags []string) ([]string, error) {
	gceCloud := cloudConfig.Provider.(*gcecloud.GCECloud)
	// Re-get instance everytime because we need the latest fingerprint for updating metadata
	resTags, err := getInstanceTags(cloudConfig, instanceName)
	if err != nil {
		return nil, fmt.Errorf("failed to get instance tags: %v", err)
	}

	_, err = gceCloud.GetComputeService().Instances.SetTags(cloudConfig.ProjectID, cloudConfig.Zone, instanceName, &compute.Tags{Fingerprint: resTags.Fingerprint, Items: tags}).Do()
	if err != nil {
		return nil, fmt.Errorf("failed to set instance tags: %v", err)
	}
	framework.Logf("Sent request to set tags %v on instance: %v", tags, instanceName)
	return resTags.Items, nil
}

// From cluster/gce/config-test.sh, master name is set up using below format:
// MASTER_NAME="${INSTANCE_PREFIX}-master"
func getInstancePrefix(masterName string) (string, error) {
	if !strings.HasSuffix(masterName, "-master") {
		return "", fmt.Errorf("unexpected master name format: %v", masterName)
	}
	return masterName[:len(masterName)-7], nil
}

// From cluster/gce/config-test.sh, cluster ip range is set up using below command:
// CLUSTER_IP_RANGE="${CLUSTER_IP_RANGE:-10.180.0.0/14}"
func getClusterIpRange() string {
	return "10.180.0.0/14"
}

// From cluster/gce/util.sh, all firewall rules are constructed in the same way as the startup scripts.
func getE2eFirewalls(masterName, masterTag, instancePrefix, nodeTag, network, clusterIpRange string) []*compute.Firewall {
	fws := []*compute.Firewall{}
	fws = append(fws, &compute.Firewall{
		Name:         network + "-default-internal-master",
		SourceRanges: []string{"10.0.0.0/8"},
		TargetTags:   []string{masterTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"1-2379"},
			},
			{
				IPProtocol: "tcp",
				Ports:      []string{"2382-65535"},
			},
			{
				IPProtocol: "udp",
				Ports:      []string{"1-65535"},
			},
			{
				IPProtocol: "icmp",
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:         network + "-default-internal-node",
		SourceRanges: []string{"10.0.0.0/8"},
		TargetTags:   []string{nodeTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"1-65535"},
			},
			{
				IPProtocol: "udp",
				Ports:      []string{"1-65535"},
			},
			{
				IPProtocol: "icmp",
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:         network + "-default-ssh",
		SourceRanges: []string{"0.0.0.0/0"},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"22"},
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:       masterName + "-etcd",
		SourceTags: []string{masterTag},
		TargetTags: []string{masterTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"2380"},
			},
			{
				IPProtocol: "tcp",
				Ports:      []string{"2381"},
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:         masterName + "-https",
		SourceRanges: []string{"0.0.0.0/0"},
		TargetTags:   []string{masterTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"443"},
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:         nodeTag + "-all",
		SourceRanges: []string{clusterIpRange},
		TargetTags:   []string{nodeTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
			},
			{
				IPProtocol: "udp",
			},
			{
				IPProtocol: "icmp",
			},
			{
				IPProtocol: "esp",
			},
			{
				IPProtocol: "ah",
			},
			{
				IPProtocol: "sctp",
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:         nodeTag + "-" + instancePrefix + "-http-alt",
		SourceRanges: []string{"0.0.0.0/0"},
		TargetTags:   []string{nodeTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"80"},
			},
			{
				IPProtocol: "tcp",
				Ports:      []string{"8080"},
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:         nodeTag + "-" + instancePrefix + "-nodeports",
		SourceRanges: []string{"0.0.0.0/0"},
		TargetTags:   []string{nodeTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"30000-32767"},
			},
			{
				IPProtocol: "udp",
				Ports:      []string{"30000-32767"},
			},
		},
	})
	return fws
}

func verifyFirewall(res, exp *compute.Firewall, network string) error {
	// Sample Network value: https://www.googleapis.com/compute/v1/projects/{project-id}/global/networks/e2e
	if !strings.HasSuffix(res.Network, "/"+network) {
		return fmt.Errorf("incorrect network: %v, expect ends with: %v", res.Network, "/"+network)
	}
	if err := sameStringArray(packFwProtocolsPorts(res.Allowed), packFwProtocolsPorts(exp.Allowed)); err != nil {
		return fmt.Errorf("incorrect allowed protocols ports: %v", err)
	}
	if err := sameStringArray(res.SourceRanges, exp.SourceRanges); err != nil {
		return fmt.Errorf("incorrect source ranges %v, expected %v: %v", res.SourceRanges, exp.SourceRanges, err)
	}
	if err := sameStringArray(res.SourceTags, exp.SourceTags); err != nil {
		return fmt.Errorf("incorrect source tags %v, expected %v: %v", res.SourceTags, exp.SourceTags, err)
	}
	if err := sameStringArray(res.TargetTags, exp.TargetTags); err != nil {
		return fmt.Errorf("incorrect target tags %v, expected %v: %v", res.TargetTags, exp.TargetTags, err)
	}
	return nil
}

func examinePortNotReachable(protocol, targetIp string, port int32) {
	address := fmt.Sprintf("%v:%v", targetIp, port)
	_, err := net.DialTimeout(protocol, address, firewallTestTcpTimeout)
	Expect(err).To(HaveOccurred())
}
