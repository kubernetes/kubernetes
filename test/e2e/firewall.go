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
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	compute "google.golang.org/api/compute/v1"
)

const (
	firewallTimeoutDefault  = 3 * time.Minute
	firewallTestHttpTimeout = time.Duration(1 * time.Second)
	// Set ports outside of 30000-32767, 80 and 8080 to avoid being whitelisted by the e2e cluster
	firewallTestHttpPort = int32(29999)
	firewallTestUdpPort  = int32(29998)
)

var _ = framework.KubeDescribe("Firewall rule", func() {
	var firewall_test_name = "firewall-test"
	f := framework.NewDefaultFramework(firewall_test_name)

	// This test takes around 3 minutes to run
	It("[slow] [serial] should create valid firewall rules for LoadBalancer type service", func() {
		framework.SkipUnlessProviderIs("gce")
		c := f.ClientSet
		ns := f.Namespace.Name
		cloudConfig := framework.TestContext.CloudConfig
		gceCloud := framework.TestContext.CloudConfig.Provider.(*gcecloud.GCECloud)
		// This source ranges is just used to examine we have the same things on LB firewall rules
		firewallTestSourceRanges := []string{"0.0.0.0/1", "128.0.0.0/1"}

		// OnlyLocal service is needed to examine which exact nodes the requests are being forward to by the Load Balancer on GCE
		By("Creating a LoadBalancer type service with onlyLocal annotation")
		serviceName := "firewall-test-loadbalancer"
		jig := NewServiceTestJig(c, serviceName)
		svc := createOnlyLocalLoadBalancServiceWithPort(jig, ns, serviceName, firewallTestHttpPort, firewallTestSourceRanges)
		defer func() {
			Expect(c.Core().Services(svc.Namespace).Delete(svc.Name, nil)).NotTo(HaveOccurred())
		}()
		svcExternalIP := svc.Status.LoadBalancer.Ingress[0].IP

		By("Checking if service's firewall rules are correct")
		Expect(examineSvcFirewallRules(gceCloud, svc)).NotTo(HaveOccurred())

		By("Creating a daemon set of netexec pods ")
		dsName := "firewall-test-daemon-set"
		Expect(createNetexecDaemonSet(c, dsName, ns, jig.Labels, firewallTestHttpPort, firewallTestUdpPort, true)).NotTo(HaveOccurred())
		defer func() {
			deleteDaemonSet(f, dsName, ns, jig.Labels)
		}()

		By("Checking that daemon pods launch on every node of the cluster.")
		Expect(wait.Poll(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, jig.Labels))).NotTo(HaveOccurred(),
			"error waiting for daemon pod to start")
		Expect(checkDaemonStatus(f, dsName)).NotTo(HaveOccurred())

		// Send requests from outside of the cluster because internal traffic is whitelisted
		By("Accessing the external service ip from outside, all non-master nodes should be reached")
		nodeNames, err := getNonMaterNodeNames(c)
		Expect(err).NotTo(HaveOccurred())
		Expect(examineHitNodesFromOutside(svcExternalIP, firewallTestHttpPort, firewallTimeoutDefault, nodeNames, true)).NotTo(HaveOccurred())

		By("Removing tags from one of the nodes")
		nodeNamesList := nodeNames.UnsortedList()
		if len(nodeNamesList) <= 0 {
			framework.Failf("Expect %v have at least 1 node", nodeNamesList)
		}
		removedTags, err := setNodeTags(gceCloud, cloudConfig.ProjectID, cloudConfig.Zone, nodeNamesList[0], []string{})
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("Adding tags back to the node")
			_, err := setNodeTags(gceCloud, cloudConfig.ProjectID, cloudConfig.Zone, nodeNamesList[0], removedTags)
			Expect(err).NotTo(HaveOccurred())
			// Make sure the traffic is recover before exit
			Expect(examineHitNodesFromOutside(svcExternalIP, firewallTestHttpPort, firewallTimeoutDefault, nodeNames, true)).NotTo(HaveOccurred())
		}()

		By("Accessing serivce through the external ip and examine got no response from the node without tags")
		Expect(examineNotHitNodeFromOutside(svcExternalIP, firewallTestHttpPort, firewallTimeoutDefault, nodeNamesList[0], 15)).NotTo(HaveOccurred())
	})
})

func createOnlyLocalLoadBalancServiceWithPort(jig *ServiceTestJig, ns, serviceName string, port int32, sourceRanges []string) *api.Service {
	svc := jig.CreateTCPServiceOrFail(ns, func(svc *api.Service) {
		svc.Spec.Type = api.ServiceTypeLoadBalancer
		svc.Spec.Ports = []api.ServicePort{{Port: port}}
		svc.ObjectMeta.Annotations = map[string]string{
			service.BetaAnnotationExternalTraffic: service.AnnotationValueExternalTrafficLocal}
		svc.Spec.LoadBalancerSourceRanges = sourceRanges
	})
	svc = jig.WaitForLoadBalancerOrFail(ns, serviceName, loadBalancerCreateTimeoutDefault)
	jig.SanityCheckService(svc, api.ServiceTypeLoadBalancer)
	return svc
}

func examineSvcFirewallRules(gceCloud *gcecloud.GCECloud, svc *api.Service) error {
	nodesTag, err := getNodesTagFromGroupName(framework.TestContext.CloudConfig.NodeInstanceGroup)
	if err != nil {
		return err
	}
	framework.Logf("nodesTag: %v", nodesTag)

	fw, err := gceCloud.GetFirewall(makeFirewallName(cloudprovider.GetLoadBalancerName(svc)))
	if err != nil {
		return err
	}

	if len(fw.TargetTags) != 1 {
		return fmt.Errorf("error number of tags, got %v", fw.TargetTags)
	}
	if fw.TargetTags[0] != nodesTag {
		return fmt.Errorf("expect tag: %v, got: %v", nodesTag, fw.TargetTags[0])
	}
	framework.Logf("Tags are correct.")

	if err := verifyProtocolsPorts(fw.Allowed, svc.Spec.Ports); err != nil {
		return err
	}
	framework.Logf("Protocols and ports are correct.")

	if err := verifySourceRanges(fw.SourceRanges, svc.Spec.LoadBalancerSourceRanges); err != nil {
		return err
	}
	framework.Logf("Source ranges are correct.")

	return nil
}

func makeFirewallName(name string) string {
	return fmt.Sprintf("k8s-fw-%s", name)
}

func getNodesTagFromGroupName(name string) (string, error) {
	if !strings.HasSuffix(name, "-group") {
		return "", fmt.Errorf("node instance group name does not ends with '-group'")
	}
	return name[:len(name)-6], nil
}

func verifyProtocolsPorts(alloweds []*compute.FirewallAllowed, servicePorts []api.ServicePort) error {
	resultProtocolPorts := []string{}
	expectedProtocolPorts := []string{}

	for _, allowed := range alloweds {
		for _, port := range allowed.Ports {
			resultProtocolPorts = append(resultProtocolPorts, strings.ToLower(allowed.IPProtocol+"/"+port))
		}
	}
	for _, sp := range servicePorts {
		expectedProtocolPorts = append(expectedProtocolPorts, strings.ToLower(string(sp.Protocol)+"/"+strconv.Itoa(int(sp.Port))))
	}
	return sameStringElements(resultProtocolPorts, expectedProtocolPorts)
}

func verifySourceRanges(resultSourceRanges, expectedSourceRanges []string) error {
	if expectedSourceRanges == nil {
		expectedSourceRanges = append(expectedSourceRanges, "0.0.0.0/0")
	}
	return sameStringElements(resultSourceRanges, expectedSourceRanges)
}

func sameStringElements(result, expected []string) error {
	ex := make(map[string]bool)
	for _, s := range expected {
		_, ok := ex[s]
		if ok {
			return fmt.Errorf("duplicated element: %v", s)
		}
		ex[s] = true
	}
	for _, s := range result {
		_, ok := ex[s]
		if !ok {
			return fmt.Errorf("unexpected element: %v", s)
		}
		delete(ex, s)
	}
	if len(ex) != 0 {
		return fmt.Errorf("missing elements: %v", ex)
	}
	return nil
}

func createNetexecDaemonSet(c clientset.Interface, dsName, ns string, labels map[string]string, httpPort, udpPort int32, hostNetwork bool) error {
	_, err := c.Extensions().DaemonSets(ns).Create(&extensions.DaemonSet{
		ObjectMeta: api.ObjectMeta{
			Name:   dsName,
			Labels: labels,
		},
		Spec: extensions.DaemonSetSpec{
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: labels,
				},
				Spec: api.PodSpec{
					NodeSelector: nil,
					Containers: []api.Container{
						{
							Name:  dsName,
							Image: framework.NetexecImageName,
							Command: []string{
								"/netexec",
								fmt.Sprintf("--http-port=%d", httpPort),
								fmt.Sprintf("--udp-port=%d", udpPort),
							},
							Ports: []api.ContainerPort{
								{
									Name:          "http",
									ContainerPort: httpPort,
								},
								{
									Name:          "udp",
									ContainerPort: udpPort,
								},
							},
						},
					},
					SecurityContext: &api.PodSecurityContext{
						HostNetwork: hostNetwork,
					},
				},
			},
		},
	})
	return err
}

func deleteDaemonSet(f *framework.Framework, dsName, ns string, labels map[string]string) {
	framework.Logf("Check that reaper kills all daemon pods for %s", dsName)
	dsReaper, err := kubectl.ReaperFor(extensions.Kind("DaemonSet"), f.ClientSet)
	Expect(err).NotTo(HaveOccurred())
	err = dsReaper.Stop(ns, dsName, 0, nil)
	Expect(err).NotTo(HaveOccurred())
	err = wait.Poll(dsRetryPeriod, dsRetryTimeout, checkRunningOnNoNodes(f, labels))
	Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to be reaped")
}

func getNonMaterNodeNames(c clientset.Interface) (sets.String, error) {
	masterName := framework.TestContext.CloudConfig.MasterName
	nodeList, err := c.Core().Nodes().List(api.ListOptions{})
	if err != nil {
		return nil, err
	}
	nodeNames := sets.NewString()
	for _, node := range nodeList.Items {
		if node.Name != masterName {
			nodeNames.Insert(node.Name)
		}
	}
	return nodeNames, nil
}

func examineHitNodesFromOutside(externalIP string, httpPort int32, timeout time.Duration, expectedHosts sets.String, exitWhenSatisfy bool) error {
	hittedHosts := sets.NewString()

	framework.Logf("Waiting up to %v for satisfying expectedHosts", timeout)
	condition := func() (bool, error) {
		respBody, err := queryFromUrl(externalIP, "hostname", httpPort, firewallTestHttpTimeout)
		if err != nil {
			framework.Logf("%v", err)
			return false, nil
		}
		hittedHost := strings.TrimSpace(respBody)
		hittedHosts.Insert(hittedHost)
		if !expectedHosts.Has(hittedHost) {
			return false, fmt.Errorf("error hitting unexpected host: %v", hittedHost)
		}

		framework.Logf("Missing %+v, got %+v", expectedHosts.Difference(hittedHosts), hittedHosts)
		if exitWhenSatisfy && hittedHosts.Equal(expectedHosts) {
			return true, nil
		}
		return false, nil
	}

	if err := wait.Poll(time.Second, timeout, condition); err != nil {
		if !exitWhenSatisfy && hittedHosts.Equal(expectedHosts) {
			return nil
		}
		return fmt.Errorf("error waiting for expectedHosts: %v, hittedHosts: %v, err message: %v", expectedHosts, hittedHosts, err)
	}
	return nil
}

func examineNotHitNodeFromOutside(externalIP string, httpPort int32, timeout time.Duration, targetHost string, countToSucceed int) error {
	count := 0

	framework.Logf("Waiting up to %v for not hitting %v for %v times", timeout, targetHost, countToSucceed)
	condition := func() (bool, error) {
		respBody, err := queryFromUrl(externalIP, "hostname", httpPort, firewallTestHttpTimeout)
		if err != nil {
			framework.Logf("%v", err)
			return false, nil
		}
		hittedHost := strings.TrimSpace(respBody)
		if hittedHost == targetHost {
			framework.Logf("Error hitting unexpected host: %v, reset counter: %v", targetHost, count)
			count = 0
		} else {
			count++
		}

		if count >= countToSucceed {
			return true, nil
		}
		return false, nil
	}

	if err := wait.Poll(time.Second, timeout, condition); err != nil {
		return fmt.Errorf("error: not hitting %v for %v times, expect at least %v times, err message: %v", targetHost, count, countToSucceed, err)
	}
	return nil
}

func queryFromUrl(ip, path string, port int32, timeout time.Duration) (string, error) {
	// Create a new Transport for not re-using connnections
	client := &http.Client{
		Timeout:   timeout,
		Transport: &http.Transport{},
	}
	url := fmt.Sprintf("http://%v:%v/%v", ip, port, path)
	resp, err := client.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("Got non-ok status code: %v", resp.StatusCode)
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body), nil
}

func setNodeTags(gceCloud *gcecloud.GCECloud, projectID, zone string, nodeName string, tags []string) ([]string, error) {
	// Re-get instance everytime because we need the latest fingerprint for updating metadata
	res, err := gceCloud.GetComputeService().Instances.Get(projectID, zone, nodeName).Do()
	if err != nil {
		return nil, fmt.Errorf("failed to get instance: %v", err)
	}

	_, err = gceCloud.GetComputeService().Instances.SetTags(projectID, zone, nodeName, &compute.Tags{Fingerprint: res.Tags.Fingerprint, Items: tags}).Do()
	if err != nil {
		return nil, fmt.Errorf("failed to set instance tags: %v", err)
	}
	framework.Logf("Sent request to set tags %v on instance: %v", tags, nodeName)
	if err := waitForNodeTags(gceCloud, projectID, zone, nodeName, tags, firewallTimeoutDefault); err != nil {
		return res.Tags.Items, err
	}
	framework.Logf("Saw expected tags on instance.")
	return res.Tags.Items, nil
}

func waitForNodeTags(gceCloud *gcecloud.GCECloud, projectID, zone string, nodeName string, tags []string, timeout time.Duration) error {
	var tagsFromGCE []string
	condition := func() (bool, error) {
		res, err := gceCloud.GetComputeService().Instances.Get(projectID, zone, nodeName).Do()
		if err != nil {
			framework.Logf("Failed to get instance: %v", err)
			return false, nil
		}
		tagsFromGCE = res.Tags.Items
		if err := sameStringElements(tagsFromGCE, tags); err != nil {
			framework.Logf("Error expecting tags: %v, got: %v", tags, tagsFromGCE)
			return false, nil
		}
		return true, nil
	}

	if err := wait.Poll(time.Second, timeout, condition); err != nil {
		return fmt.Errorf("error waiting for expected tags: %v, got: %v\nerr message: %v", tags, tagsFromGCE, err)
	}
	return nil
}
