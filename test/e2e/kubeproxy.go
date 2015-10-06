/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	api "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	endpointHttpPort        = 8080
	endpointUdpPort         = 8081
	endpointHostPort        = 8082
	testContainerHttpPort   = 8080
	clusterHttpPort         = 80
	clusterUdpPort          = 90
	nodeHttpPort            = 32080
	nodeUdpPort             = 32081
	loadBalancerHttpPort    = 100
	netexecImageName        = "gcr.io/google_containers/netexec:1.0"
	testPodName             = "test-container-pod"
	nodePortServiceName     = "node-port-service"
	loadBalancerServiceName = "load-balancer-service"
	enableLoadBalancerTest  = false
)

type KubeProxyTestConfig struct {
	testContainerPod    *api.Pod
	testHostPod         *api.Pod
	endpointPods        []*api.Pod
	f                   *Framework
	nodePortService     *api.Service
	loadBalancerService *api.Service
	nodes               []string
}

var _ = Describe("KubeProxy", func() {
	f := NewFramework("e2e-kubeproxy")
	config := &KubeProxyTestConfig{
		f: f,
	}

	It("should test kube-proxy", func() {
		SkipUnlessProviderIs(providersWithSSH...)

		By("cleaning up any pre-existing namespaces used by this test")
		config.cleanup()

		By("Setting up for the tests")
		config.setup()

		//TODO Need to add hit externalIPs test
		By("TODO: Need to add hit externalIPs test")

		By("Hit Test with All Endpoints")
		config.hitAll()

		config.deleteNetProxyPod()
		By("Hit Test with Fewer Endpoints")
		config.hitAll()

		By("Deleting nodePortservice and ensuring that service cannot be hit")
		config.deleteNodePortService()
		config.hitNodePort(0) // expect 0 endpoints to be hit

		if enableLoadBalancerTest {
			By("Deleting loadBalancerService and ensuring that service cannot be hit")
			config.deleteLoadBalancerService()
			config.hitLoadBalancer(0) // expect 0 endpoints to be hit
		}
	})
})

func (config *KubeProxyTestConfig) hitAll() {
	By("Hitting endpoints from host and container")
	config.hitEndpoints()

	By("Hitting clusterIP from host and container")
	config.hitClusterIP(len(config.endpointPods))

	By("Hitting nodePort from host and container")
	config.hitNodePort(len(config.endpointPods))

	if enableLoadBalancerTest {
		By("Waiting for LoadBalancer Ingress Setup")
		config.waitForLoadBalancerIngressSetup()

		By("Hitting LoadBalancer")
		config.hitLoadBalancer(len(config.endpointPods))
	}
}

func (config *KubeProxyTestConfig) hitLoadBalancer(epCount int) {
	lbIP := config.loadBalancerService.Status.LoadBalancer.Ingress[0].IP
	hostNames := make(map[string]bool)
	tries := epCount*epCount + 5
	for i := 0; i < tries; i++ {
		transport := &http.Transport{}
		httpClient := createHTTPClient(transport)
		resp, err := httpClient.Get(fmt.Sprintf("http://%s:%d/hostName", lbIP, loadBalancerHttpPort))
		if err == nil {
			defer resp.Body.Close()
			hostName, err := ioutil.ReadAll(resp.Body)
			if err == nil {
				hostNames[string(hostName)] = true
			}
		}
		transport.CloseIdleConnections()
	}
	Expect(len(hostNames)).To(BeNumerically("==", epCount), "LoadBalancer did not hit all pods")
}

func createHTTPClient(transport *http.Transport) *http.Client {
	client := &http.Client{
		Transport: transport,
		Timeout:   5 * time.Second,
	}
	return client
}

func (config *KubeProxyTestConfig) hitClusterIP(epCount int) {
	clusterIP := config.nodePortService.Spec.ClusterIP
	tries := epCount*epCount + 5 // if epCount == 0
	By("dialing(udp) node1 --> clusterIP:clusterUdpPort")
	config.dialFromNode("udp", clusterIP, clusterUdpPort, tries, epCount)
	By("dialing(http) node1 --> clusterIP:clusterHttpPort")
	config.dialFromNode("http", clusterIP, clusterHttpPort, tries, epCount)

	By("dialing(udp) test container --> clusterIP:clusterUdpPort")
	config.dialFromTestContainer("udp", clusterIP, clusterUdpPort, tries, epCount)
	By("dialing(http) test container --> clusterIP:clusterHttpPort")
	config.dialFromTestContainer("http", clusterIP, clusterHttpPort, tries, epCount)

	By("dialing(udp) endpoint container --> clusterIP:clusterUdpPort")
	config.dialFromEndpointContainer("udp", clusterIP, clusterUdpPort, tries, epCount)
	By("dialing(http) endpoint container --> clusterIP:clusterHttpPort")
	config.dialFromEndpointContainer("http", clusterIP, clusterHttpPort, tries, epCount)
}

func (config *KubeProxyTestConfig) hitNodePort(epCount int) {
	node1_IP := config.nodes[0]
	tries := epCount*epCount + 5 // + 10 if epCount == 0
	By("dialing(udp) node1 --> node1:nodeUdpPort")
	config.dialFromNode("udp", node1_IP, nodeUdpPort, tries, epCount)
	By("dialing(http) node1  --> node1:nodeHttpPort")
	config.dialFromNode("http", node1_IP, nodeHttpPort, tries, epCount)

	By("dialing(udp) test container --> node1:nodeUdpPort")
	config.dialFromTestContainer("udp", node1_IP, nodeUdpPort, tries, epCount)
	By("dialing(http) container --> node1:nodeHttpPort")
	config.dialFromTestContainer("http", node1_IP, nodeHttpPort, tries, epCount)

	By("dialing(udp) endpoint container --> node1:nodeUdpPort")
	config.dialFromEndpointContainer("udp", node1_IP, nodeUdpPort, tries, epCount)
	By("dialing(http) endpoint container --> node1:nodeHttpPort")
	config.dialFromEndpointContainer("http", node1_IP, nodeHttpPort, tries, epCount)

	// TODO: doesnt work because masquerading is not done
	By("TODO: Test disabled. dialing(udp) node --> 127.0.0.1:nodeUdpPort")
	//config.dialFromNode("udp", "127.0.0.1", nodeUdpPort, tries, epCount)
	// TODO: doesnt work because masquerading is not done
	By("Test disabled. dialing(http) node --> 127.0.0.1:nodeHttpPort")
	//config.dialFromNode("http", "127.0.0.1", nodeHttpPort, tries, epCount)

	node2_IP := config.nodes[1]
	By("dialing(udp) node1 --> node2:nodeUdpPort")
	config.dialFromNode("udp", node2_IP, nodeUdpPort, tries, epCount)
	By("dialing(http) node1 --> node2:nodeHttpPort")
	config.dialFromNode("http", node2_IP, nodeHttpPort, tries, epCount)
}

func (config *KubeProxyTestConfig) hitEndpoints() {
	for _, endpointPod := range config.endpointPods {
		Expect(len(endpointPod.Status.PodIP)).To(BeNumerically(">", 0), "podIP is empty:%s", endpointPod.Status.PodIP)
		By("dialing(udp) endpointPodIP:endpointUdpPort from node1")
		config.dialFromNode("udp", endpointPod.Status.PodIP, endpointUdpPort, 5, 1)
		By("dialing(http) endpointPodIP:endpointHttpPort from node1")
		config.dialFromNode("http", endpointPod.Status.PodIP, endpointHttpPort, 5, 1)
		By("dialing(udp) endpointPodIP:endpointUdpPort from test container")
		config.dialFromTestContainer("udp", endpointPod.Status.PodIP, endpointUdpPort, 5, 1)
		By("dialing(http) endpointPodIP:endpointHttpPort from test container")
		config.dialFromTestContainer("http", endpointPod.Status.PodIP, endpointHttpPort, 5, 1)
	}
}

func (config *KubeProxyTestConfig) dialFromEndpointContainer(protocol, targetIP string, targetPort, tries, expectedCount int) {
	config.dialFromContainer(protocol, config.endpointPods[0].Status.PodIP, targetIP, endpointHttpPort, targetPort, tries, expectedCount)
}

func (config *KubeProxyTestConfig) dialFromTestContainer(protocol, targetIP string, targetPort, tries, expectedCount int) {
	config.dialFromContainer(protocol, config.testContainerPod.Status.PodIP, targetIP, testContainerHttpPort, targetPort, tries, expectedCount)
}

func (config *KubeProxyTestConfig) dialFromContainer(protocol, containerIP, targetIP string, containerHttpPort, targetPort, tries, expectedCount int) {
	cmd := fmt.Sprintf("curl -q 'http://%s:%d/dial?request=hostName&protocol=%s&host=%s&port=%d&tries=%d'",
		containerIP,
		containerHttpPort,
		protocol,
		targetIP,
		targetPort,
		tries)

	By(fmt.Sprintf("Dialing from container. Running command:%s", cmd))
	stdout := config.ssh(cmd)
	var output map[string][]string
	err := json.Unmarshal([]byte(stdout), &output)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Could not unmarshal curl response: %s", stdout))
	hostNamesMap := array2map(output["responses"])
	Expect(len(hostNamesMap)).To(BeNumerically("==", expectedCount), fmt.Sprintf("Response was:%v", output))
}

func (config *KubeProxyTestConfig) dialFromNode(protocol, targetIP string, targetPort, tries, expectedCount int) {
	var cmd string
	if protocol == "udp" {
		cmd = fmt.Sprintf("echo 'hostName' | nc -w 1 -u %s %d", targetIP, targetPort)
	} else {
		cmd = fmt.Sprintf("curl -s --connect-timeout 1 http://%s:%d/hostName", targetIP, targetPort)
	}
	forLoop := fmt.Sprintf("for i in $(seq 1 %d); do %s; echo; done | grep -v '^\\s*$' |sort | uniq -c | wc -l", tries, cmd)
	By(fmt.Sprintf("Dialing from node. command:%s", forLoop))
	stdout := config.ssh(forLoop)
	Expect(strconv.Atoi(strings.TrimSpace(stdout))).To(BeNumerically("==", expectedCount))
}

func (config *KubeProxyTestConfig) ssh(cmd string) string {
	stdout, _, code, err := SSH(cmd, config.nodes[0]+":22", testContext.Provider)
	Expect(err).NotTo(HaveOccurred(), "error while SSH-ing to node: %v (code %v)", err, code)
	Expect(code).Should(BeZero(), "command exited with non-zero code %v. cmd:%s", code, cmd)
	return stdout
}

func (config *KubeProxyTestConfig) createNetShellPodSpec(podName string) *api.Pod {
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: latest.GroupOrDie("").Version,
		},
		ObjectMeta: api.ObjectMeta{
			Name:      podName,
			Namespace: config.f.Namespace.Name,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            "webserver",
					Image:           netexecImageName,
					ImagePullPolicy: api.PullIfNotPresent,
					Command: []string{
						"/netexec",
						fmt.Sprintf("--http-port=%d", endpointHttpPort),
						fmt.Sprintf("--udp-port=%d", endpointUdpPort),
					},
					Ports: []api.ContainerPort{
						{
							Name:          "http",
							ContainerPort: endpointHttpPort,
						},
						{
							Name:          "udp",
							ContainerPort: endpointUdpPort,
						},
						{
							Name:          "host",
							ContainerPort: endpointHttpPort,
							HostPort:      endpointHostPort,
						},
					},
				},
			},
		},
	}
	return pod
}

func (config *KubeProxyTestConfig) createTestPodSpec() *api.Pod {
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: latest.GroupOrDie("").Version,
		},
		ObjectMeta: api.ObjectMeta{
			Name:      testPodName,
			Namespace: config.f.Namespace.Name,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            "webserver",
					Image:           netexecImageName,
					ImagePullPolicy: api.PullIfNotPresent,
					Command: []string{
						"/netexec",
						fmt.Sprintf("--http-port=%d", endpointHttpPort),
						fmt.Sprintf("--udp-port=%d", endpointUdpPort),
					},
					Ports: []api.ContainerPort{
						{
							Name:          "http",
							ContainerPort: testContainerHttpPort,
						},
					},
				},
			},
		},
	}
	return pod
}

func (config *KubeProxyTestConfig) createNodePortService(selector map[string]string) {
	serviceSpec := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: nodePortServiceName,
		},
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeNodePort,
			Ports: []api.ServicePort{
				{Port: clusterHttpPort, Name: "http", Protocol: "TCP", NodePort: nodeHttpPort, TargetPort: util.NewIntOrStringFromInt(endpointHttpPort)},
				{Port: clusterUdpPort, Name: "udp", Protocol: "UDP", NodePort: nodeUdpPort, TargetPort: util.NewIntOrStringFromInt(endpointUdpPort)},
			},
			Selector: selector,
		},
	}
	config.nodePortService = config.createService(serviceSpec)
}

func (config *KubeProxyTestConfig) deleteNodePortService() {
	err := config.getServiceClient().Delete(config.nodePortService.Name)
	Expect(err).NotTo(HaveOccurred(), "error while deleting NodePortService. err:%v)", err)
	time.Sleep(15 * time.Second) // wait for kube-proxy to catch up with the service being deleted.
}

func (config *KubeProxyTestConfig) createLoadBalancerService(selector map[string]string) {
	serviceSpec := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: loadBalancerServiceName,
		},
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{
				{Port: loadBalancerHttpPort, Name: "http", Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(endpointHttpPort)},
			},
			Selector: selector,
		},
	}
	config.createService(serviceSpec)
}

func (config *KubeProxyTestConfig) deleteLoadBalancerService() {
	go func() { config.getServiceClient().Delete(config.loadBalancerService.Name) }()
	time.Sleep(15 * time.Second) // wait for kube-proxy to catch up with the service being deleted.
}

func (config *KubeProxyTestConfig) waitForLoadBalancerIngressSetup() {
	err := wait.Poll(2*time.Second, 120*time.Second, func() (bool, error) {
		service, err := config.getServiceClient().Get(loadBalancerServiceName)
		if err != nil {
			return false, err
		} else {
			if len(service.Status.LoadBalancer.Ingress) > 0 {
				return true, nil
			} else {
				return false, fmt.Errorf("Service LoadBalancer Ingress was not setup.")
			}
		}
	})
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to setup Load Balancer Service. err:%v", err))
	config.loadBalancerService, _ = config.getServiceClient().Get(loadBalancerServiceName)
}

func (config *KubeProxyTestConfig) createTestPod() {
	testContainerPod := config.createTestPodSpec()
	config.testContainerPod = config.createPod(testContainerPod)
}

func (config *KubeProxyTestConfig) createService(serviceSpec *api.Service) *api.Service {
	_, err := config.getServiceClient().Create(serviceSpec)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create %s service: %v", serviceSpec.Name, err))

	err = waitForService(config.f.Client, config.f.Namespace.Name, serviceSpec.Name, true, 5*time.Second, 45*time.Second)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("error while waiting for service:%s err: %v", serviceSpec.Name, err))

	createdService, err := config.getServiceClient().Get(serviceSpec.Name)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create %s service: %v", serviceSpec.Name, err))

	return createdService
}

func (config *KubeProxyTestConfig) setup() {
	By("creating a selector")
	selectorName := "selector-" + string(util.NewUUID())
	serviceSelector := map[string]string{
		selectorName: "true",
	}

	By("Getting ssh-able hosts")
	hosts, err := NodeSSHHosts(config.f.Client)
	Expect(err).NotTo(HaveOccurred())
	config.nodes = make([]string, 0, len(hosts))
	for _, h := range hosts {
		config.nodes = append(config.nodes, strings.TrimSuffix(h, ":22"))
	}

	if enableLoadBalancerTest {
		By("Creating the LoadBalancer Service on top of the pods in kubernetes")
		config.createLoadBalancerService(serviceSelector)
	}

	By("Creating the service pods in kubernetes")
	podName := "netserver"
	config.endpointPods = config.createNetProxyPods(podName, serviceSelector, testContext.CloudConfig.NumNodes)

	By("Creating the service on top of the pods in kubernetes")
	config.createNodePortService(serviceSelector)

	By("Creating test pods")
	config.createTestPod()
}

func (config *KubeProxyTestConfig) cleanup() {
	nsClient := config.getNamespacesClient()
	nsList, err := nsClient.List(nil, nil)
	if err == nil {
		for _, ns := range nsList.Items {
			if strings.Contains(ns.Name, config.f.BaseName) && ns.Name != config.f.Namespace.Name {
				nsClient.Delete(ns.Name)
			}
		}
	}
}

func (config *KubeProxyTestConfig) createNetProxyPods(podName string, selector map[string]string, nodeCount int) []*api.Pod {
	//testContext.CloudConfig.NumNodes
	pods := make([]*api.Pod, 0)

	for i := 0; i < nodeCount; i++ {
		podName := fmt.Sprintf("%s-%d", podName, i)
		pod := config.createNetShellPodSpec(podName)
		pod.ObjectMeta.Labels = selector
		createdPod := config.createPod(pod)
		pods = append(pods, createdPod)
	}
	return pods
}

func (config *KubeProxyTestConfig) deleteNetProxyPod() {
	pod := config.endpointPods[0]
	config.getPodClient().Delete(pod.Name, nil)
	config.endpointPods = config.endpointPods[1:]
	// wait for pod being deleted.
	err := waitForPodToDisappear(config.f.Client, config.f.Namespace.Name, pod.Name, labels.Everything(), time.Second, util.ForeverTestTimeout)
	if err != nil {
		Failf("Failed to delete %s pod: %v", pod.Name, err)
	}
	// wait for endpoint being removed.
	err = waitForServiceEndpointsNum(config.f.Client, config.f.Namespace.Name, nodePortServiceName, len(config.endpointPods), time.Second, util.ForeverTestTimeout)
	if err != nil {
		Failf("Failed to remove endpoint from service: %s", nodePortServiceName)
	}
	// wait for kube-proxy to catch up with the pod being deleted.
	time.Sleep(5 * time.Second)
}

func (config *KubeProxyTestConfig) createPod(pod *api.Pod) *api.Pod {
	createdPod, err := config.getPodClient().Create(pod)
	if err != nil {
		Failf("Failed to create %s pod: %v", pod.Name, err)
	}
	expectNoError(config.f.WaitForPodRunning(pod.Name))
	createdPod, err = config.getPodClient().Get(pod.Name)
	if err != nil {
		Failf("Failed to retrieve %s pod: %v", pod.Name, err)
	}
	return createdPod
}

func (config *KubeProxyTestConfig) getPodClient() client.PodInterface {
	return config.f.Client.Pods(config.f.Namespace.Name)
}

func (config *KubeProxyTestConfig) getServiceClient() client.ServiceInterface {
	return config.f.Client.Services(config.f.Namespace.Name)
}

func (config *KubeProxyTestConfig) getNamespacesClient() client.NamespaceInterface {
	return config.f.Client.Namespaces()
}

func array2map(arr []string) map[string]bool {
	retval := make(map[string]bool)
	if len(arr) == 0 {
		return retval
	}
	for _, str := range arr {
		retval[str] = true
	}
	return retval
}
