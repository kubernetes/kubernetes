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
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/v1/service"
	"k8s.io/kubernetes/pkg/cloudprovider"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/util/sets"
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

	// Below consts should be consistent with kubernetes/contrib/ingress/controllers/gce package.
	// Src range from which the GCE L7 performs health checks.
	l7SrcRange = "130.211.0.0/22"
	// Suffix used in the l7 firewall rule. There is currently only one.
	// Note that this name is used by the cloudprovider lib that inserts its
	// own k8s-fw prefix.
	ingressGlobalFirewallSuffix = "l7"
	// Prefix used in the l7 firewall rule on GCE.
	ingressGCEFirewallPrefix = "k8s-fw-"
)

var _ = framework.KubeDescribe("Firewall rule", func() {
	var firewall_test_name = "firewall-test"
	f := framework.NewDefaultFramework(firewall_test_name)

	// This test takes around 4 minutes to run
	It("[Slow] [Serial] should create valid firewall rules for LoadBalancer type service", func() {
		framework.SkipUnlessProviderIs("gce")
		c := f.ClientSet
		ns := f.Namespace.Name
		cloudConfig := framework.TestContext.CloudConfig
		gceCloud := cloudConfig.Provider.(*gcecloud.GCECloud)
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
			jig.UpdateServiceOrFail(svc.Namespace, svc.Name, func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeNodePort
				svc.Spec.LoadBalancerSourceRanges = nil
			})
			Expect(c.Core().Services(svc.Namespace).Delete(svc.Name, nil)).NotTo(HaveOccurred())
		}()
		svcExternalIP := svc.Status.LoadBalancer.Ingress[0].IP

		By("Checking if service's firewall rules are correct")
		nodeTags, err := getInstanceTags(cloudConfig, nodesNames[0])
		Expect(err).NotTo(HaveOccurred())
		expFw, err := constructFwForLBSvc(svc, nodeTags.Items)
		Expect(err).NotTo(HaveOccurred())
		fw, err := gceCloud.GetFirewall(expFw.Name)
		Expect(err).NotTo(HaveOccurred())
		Expect(verifyFirewallRule(fw, expFw, cloudConfig.Network, false)).NotTo(HaveOccurred())

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
		Expect(testHitNodesFromOutside(svcExternalIP, firewallTestHttpPort, firewallTimeoutDefault, nodesSet)).NotTo(HaveOccurred())

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
			Expect(testHitNodesFromOutside(svcExternalIP, firewallTestHttpPort, firewallTimeoutDefault, nodesSet)).NotTo(HaveOccurred())
		}()

		By("Accessing serivce through the external ip and examine got no response from the node without tags")
		Expect(testHitNodesFromOutsideWithCount(svcExternalIP, firewallTestHttpPort, firewallTimeoutDefault, nodesSet, 15)).NotTo(HaveOccurred())
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
			Expect(verifyFirewallRule(fw, expFw, cloudConfig.Network, false)).NotTo(HaveOccurred())
		}

		By("Checking well known ports on master and nodes are not exposed externally")
		nodeAddrs := framework.NodeAddresses(nodes, v1.NodeExternalIP)
		Expect(len(nodeAddrs)).NotTo(BeZero())
		masterAddr := framework.GetMasterAddress(f.ClientSet)
		flag, _ := testNotReachableHTTPTimeout(masterAddr, ports.ControllerManagerPort, firewallTestTcpTimeout)
		Expect(flag).To(BeTrue())
		flag, _ = testNotReachableHTTPTimeout(masterAddr, ports.SchedulerPort, firewallTestTcpTimeout)
		Expect(flag).To(BeTrue())
		flag, _ = testNotReachableHTTPTimeout(nodeAddrs[0], ports.KubeletPort, firewallTestTcpTimeout)
		Expect(flag).To(BeTrue())
		flag, _ = testNotReachableHTTPTimeout(nodeAddrs[0], ports.KubeletReadOnlyPort, firewallTestTcpTimeout)
		Expect(flag).To(BeTrue())
		flag, _ = testNotReachableHTTPTimeout(nodeAddrs[0], ports.ProxyStatusPort, firewallTestTcpTimeout)
		Expect(flag).To(BeTrue())
	})
})

// This should match the formatting of makeFirewallName() in pkg/cloudprovider/providers/gce/gce.go
func makeFirewallName(name string) string {
	return fmt.Sprintf("k8s-fw-%s", name)
}

// sameStringArray verifies whether two string arrays have the same strings, return error if not.
// Order does not matter.
// When `include` is set to true, verifies whether result includes all elements from expected.
func sameStringArray(result, expected []string, include bool) error {
	res := sets.NewString(result...)
	exp := sets.NewString(expected...)
	if !include {
		diff := res.Difference(exp)
		if len(diff) != 0 {
			return fmt.Errorf("found differences: %v", diff)
		}
	} else {
		if !res.IsSuperset(exp) {
			return fmt.Errorf("some elements are missing: expected %v, got %v", expected, result)
		}
	}
	return nil
}

// constructFwForLBSvc returns the expected GCE firewall rule for a loadbalancer type service
func constructFwForLBSvc(svc *v1.Service, nodesTags []string) (*compute.Firewall, error) {
	if svc.Spec.Type != v1.ServiceTypeLoadBalancer {
		return nil, fmt.Errorf("can not construct firewall rule for non-loadbalancer type service")
	}
	fw := compute.Firewall{}
	fw.Name = makeFirewallName(cloudprovider.GetLoadBalancerName(svc))
	fw.TargetTags = nodesTags
	if svc.Spec.LoadBalancerSourceRanges == nil {
		fw.SourceRanges = []string{"0.0.0.0/0"}
	} else {
		fw.SourceRanges = svc.Spec.LoadBalancerSourceRanges
	}
	for _, sp := range svc.Spec.Ports {
		fw.Allowed = append(fw.Allowed, &compute.FirewallAllowed{
			IPProtocol: strings.ToLower(string(sp.Protocol)),
			Ports:      []string{strconv.Itoa(int(sp.Port))},
		})
	}
	return &fw, nil
}

// constructFwForIngress returns the expected GCE firewall rule for the ingress resource
func constructFwForIngress(fwName string, ports, nodesTags []string) (*compute.Firewall, error) {
	fw := compute.Firewall{}
	fw.Name = fwName
	fw.SourceRanges = []string{l7SrcRange}
	fw.TargetTags = nodesTags
	fw.Allowed = []*compute.FirewallAllowed{
		&compute.FirewallAllowed{
			IPProtocol: "tcp",
			Ports:      ports,
		},
	}
	return &fw, nil
}

// getFwForIngress returns the firewall rule for ingress resource from GCE
func getFwForIngress(cloudConfig framework.CloudConfig) (*compute.Firewall, error) {
	gceCloud := cloudConfig.Provider.(*gcecloud.GCECloud)
	res, err := gceCloud.GetComputeService().Firewalls.List(cloudConfig.ProjectID).Filter(
		fmt.Sprintf("name eq %v%v.*", ingressGCEFirewallPrefix, ingressGlobalFirewallSuffix)).Do()
	if err != nil {
		return nil, err
	}
	if len(res.Items) != 1 {
		return nil, fmt.Errorf("get unexpected number of firewall rules, got %v", len(res.Items))
	}
	return res.Items[0], nil
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
// Warning: this need to be consistent with the CLUSTER_IP_RANGE in startup scripts,
// which is hardcoded currently
func getClusterIpRange() string {
	return "10.180.0.0/14"
}

// From cluster/gce/util.sh, all firewall rules should be consistent with the ones created by startup scripts
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

func packProtocolsPortsFromFw(alloweds []*compute.FirewallAllowed) []string {
	protocolPorts := []string{}
	for _, allowed := range alloweds {
		for _, port := range allowed.Ports {
			protocolPorts = append(protocolPorts, strings.ToLower(allowed.IPProtocol+"/"+port))
		}
	}
	return protocolPorts
}

func verifyFirewallRule(res, exp *compute.Firewall, network string, portsSubset bool) error {
	if res.Name != exp.Name {
		return fmt.Errorf("incorrect name: %v, expected %v", res.Name, exp.Name)
	}
	// Sample Network value: https://www.googleapis.com/compute/v1/projects/{project-id}/global/networks/e2e
	if !strings.HasSuffix(res.Network, "/"+network) {
		return fmt.Errorf("incorrect network: %v, expected ends with: %v", res.Network, "/"+network)
	}
	if err := sameStringArray(packProtocolsPortsFromFw(res.Allowed), packProtocolsPortsFromFw(exp.Allowed), portsSubset); err != nil {
		return fmt.Errorf("incorrect allowed protocols ports: %v", err)
	}
	if err := sameStringArray(res.SourceRanges, exp.SourceRanges, false); err != nil {
		return fmt.Errorf("incorrect source ranges %v, expected %v: %v", res.SourceRanges, exp.SourceRanges, err)
	}
	if err := sameStringArray(res.SourceTags, exp.SourceTags, false); err != nil {
		return fmt.Errorf("incorrect source tags %v, expected %v: %v", res.SourceTags, exp.SourceTags, err)
	}
	if err := sameStringArray(res.TargetTags, exp.TargetTags, false); err != nil {
		return fmt.Errorf("incorrect target tags %v, expected %v: %v", res.TargetTags, exp.TargetTags, err)
	}
	return nil
}
