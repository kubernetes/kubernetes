/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/intstr"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	endpointHttpPort        = 8080
	endpointUdpPort         = 8081
	testContainerHttpPort   = 8080
	clusterHttpPort         = 80
	clusterUdpPort          = 90
	nodeHttpPort            = 32080
	nodeUdpPort             = 32081
	loadBalancerHttpPort    = 100
	netexecImageName        = "gcr.io/google_containers/netexec:1.5"
	testPodName             = "test-container-pod"
	hostTestPodName         = "host-test-container-pod"
	nodePortServiceName     = "node-port-service"
	loadBalancerServiceName = "load-balancer-service"
	enableLoadBalancerTest  = false
	hitEndpointRetryDelay   = 1 * time.Second
	// Number of retries to hit a given set of endpoints. Needs to be high
	// because we verify iptables statistical rr loadbalancing.
	testTries = 30
)

type KubeProxyTestConfig struct {
	testContainerPod     *api.Pod
	hostTestContainerPod *api.Pod
	endpointPods         []*api.Pod
	f                    *framework.Framework
	nodePortService      *api.Service
	loadBalancerService  *api.Service
	externalAddrs        []string
	nodes                []api.Node
}

var _ = framework.KubeDescribe("KubeProxy", func() {
	f := framework.NewDefaultFramework("e2e-kubeproxy")
	config := &KubeProxyTestConfig{
		f: f,
	}

	// Slow issue #14204 (10 min)
	It("should test kube-proxy [Slow]", func() {
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
		transport := utilnet.SetTransportDefaults(&http.Transport{})
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
	tries := epCount*epCount + testTries // if epCount == 0
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
	node1_IP := config.externalAddrs[0]
	tries := epCount*epCount + testTries //  if epCount == 0
	By("dialing(udp) node1 --> node1:nodeUdpPort")
	config.dialFromNode("udp", node1_IP, nodeUdpPort, tries, epCount)
	By("dialing(http) node1  --> node1:nodeHttpPort")
	config.dialFromNode("http", node1_IP, nodeHttpPort, tries, epCount)

	By("dialing(udp) test container --> node1:nodeUdpPort")
	config.dialFromTestContainer("udp", node1_IP, nodeUdpPort, tries, epCount)
	By("dialing(http) test container --> node1:nodeHttpPort")
	config.dialFromTestContainer("http", node1_IP, nodeHttpPort, tries, epCount)

	By("dialing(udp) endpoint container --> node1:nodeUdpPort")
	config.dialFromEndpointContainer("udp", node1_IP, nodeUdpPort, tries, epCount)
	By("dialing(http) endpoint container --> node1:nodeHttpPort")
	config.dialFromEndpointContainer("http", node1_IP, nodeHttpPort, tries, epCount)

	By("dialing(udp) node --> 127.0.0.1:nodeUdpPort")
	config.dialFromNode("udp", "127.0.0.1", nodeUdpPort, tries, epCount)
	By("dialing(http) node --> 127.0.0.1:nodeHttpPort")
	config.dialFromNode("http", "127.0.0.1", nodeHttpPort, tries, epCount)

	node2_IP := config.externalAddrs[1]
	By("dialing(udp) node1 --> node2:nodeUdpPort")
	config.dialFromNode("udp", node2_IP, nodeUdpPort, tries, epCount)
	By("dialing(http) node1 --> node2:nodeHttpPort")
	config.dialFromNode("http", node2_IP, nodeHttpPort, tries, epCount)

	By("checking kube-proxy URLs")
	config.getSelfURL("/healthz", "ok")
	config.getSelfURL("/proxyMode", "iptables") // the default
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
	stdout := framework.RunHostCmdOrDie(config.f.Namespace.Name, config.hostTestContainerPod.Name, cmd)
	var output map[string][]string
	err := json.Unmarshal([]byte(stdout), &output)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Could not unmarshal curl response: %s", stdout))
	hostNamesMap := array2map(output["responses"])
	Expect(len(hostNamesMap)).To(BeNumerically("==", expectedCount), fmt.Sprintf("Response was:%v", output))
}

func (config *KubeProxyTestConfig) dialFromNode(protocol, targetIP string, targetPort, tries, expectedCount int) {
	var cmd string
	if protocol == "udp" {
		cmd = fmt.Sprintf("echo 'hostName' | timeout -t 3 nc -w 1 -u %s %d", targetIP, targetPort)
	} else {
		cmd = fmt.Sprintf("curl -s --connect-timeout 1 http://%s:%d/hostName", targetIP, targetPort)
	}
	// TODO: This simply tells us that we can reach the endpoints. Check that
	// the probability of hitting a specific endpoint is roughly the same as
	// hitting any other.
	forLoop := fmt.Sprintf("for i in $(seq 1 %d); do %s; echo; sleep %v; done | grep -v '^\\s*$' |sort | uniq -c | wc -l", tries, cmd, hitEndpointRetryDelay)
	By(fmt.Sprintf("Dialing from node. command:%s", forLoop))
	stdout := framework.RunHostCmdOrDie(config.f.Namespace.Name, config.hostTestContainerPod.Name, forLoop)
	Expect(strconv.Atoi(strings.TrimSpace(stdout))).To(BeNumerically("==", expectedCount))
}

func (config *KubeProxyTestConfig) getSelfURL(path string, expected string) {
	cmd := fmt.Sprintf("curl -s --connect-timeout 1 http://localhost:10249%s", path)
	By(fmt.Sprintf("Getting kube-proxy self URL %s", path))
	stdout := framework.RunHostCmdOrDie(config.f.Namespace.Name, config.hostTestContainerPod.Name, cmd)
	Expect(strings.Contains(stdout, expected)).To(BeTrue())
}

func (config *KubeProxyTestConfig) createNetShellPodSpec(podName string, node string) *api.Pod {
	probe := &api.Probe{
		InitialDelaySeconds: 10,
		TimeoutSeconds:      30,
		PeriodSeconds:       10,
		SuccessThreshold:    1,
		FailureThreshold:    3,
		Handler: api.Handler{
			HTTPGet: &api.HTTPGetAction{
				Path: "/healthz",
				Port: intstr.IntOrString{IntVal: endpointHttpPort},
			},
		},
	}
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: registered.GroupOrDie(api.GroupName).GroupVersion.String(),
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
							Protocol:      api.ProtocolUDP,
						},
					},
					LivenessProbe:  probe,
					ReadinessProbe: probe,
				},
			},
			NodeName: node,
		},
	}
	return pod
}

func (config *KubeProxyTestConfig) createTestPodSpec() *api.Pod {
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: registered.GroupOrDie(api.GroupName).GroupVersion.String(),
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
				{Port: clusterHttpPort, Name: "http", Protocol: api.ProtocolTCP, NodePort: nodeHttpPort, TargetPort: intstr.FromInt(endpointHttpPort)},
				{Port: clusterUdpPort, Name: "udp", Protocol: api.ProtocolUDP, NodePort: nodeUdpPort, TargetPort: intstr.FromInt(endpointUdpPort)},
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
				{Port: loadBalancerHttpPort, Name: "http", Protocol: "TCP", TargetPort: intstr.FromInt(endpointHttpPort)},
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

func (config *KubeProxyTestConfig) createTestPods() {
	testContainerPod := config.createTestPodSpec()
	hostTestContainerPod := framework.NewHostExecPodSpec(config.f.Namespace.Name, hostTestPodName)

	config.createPod(testContainerPod)
	config.createPod(hostTestContainerPod)

	framework.ExpectNoError(config.f.WaitForPodRunning(testContainerPod.Name))
	framework.ExpectNoError(config.f.WaitForPodRunning(hostTestContainerPod.Name))

	var err error
	config.testContainerPod, err = config.getPodClient().Get(testContainerPod.Name)
	if err != nil {
		framework.Failf("Failed to retrieve %s pod: %v", testContainerPod.Name, err)
	}

	config.hostTestContainerPod, err = config.getPodClient().Get(hostTestContainerPod.Name)
	if err != nil {
		framework.Failf("Failed to retrieve %s pod: %v", hostTestContainerPod.Name, err)
	}
}

func (config *KubeProxyTestConfig) createService(serviceSpec *api.Service) *api.Service {
	_, err := config.getServiceClient().Create(serviceSpec)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create %s service: %v", serviceSpec.Name, err))

	err = framework.WaitForService(config.f.Client, config.f.Namespace.Name, serviceSpec.Name, true, 5*time.Second, 45*time.Second)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("error while waiting for service:%s err: %v", serviceSpec.Name, err))

	createdService, err := config.getServiceClient().Get(serviceSpec.Name)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create %s service: %v", serviceSpec.Name, err))

	return createdService
}

func (config *KubeProxyTestConfig) setup() {
	By("creating a selector")
	selectorName := "selector-" + string(uuid.NewUUID())
	serviceSelector := map[string]string{
		selectorName: "true",
	}

	By("Getting node addresses")
	framework.ExpectNoError(framework.WaitForAllNodesSchedulable(config.f.Client))
	nodeList := framework.GetReadySchedulableNodesOrDie(config.f.Client)
	config.externalAddrs = framework.NodeAddresses(nodeList, api.NodeExternalIP)
	if len(config.externalAddrs) < 2 {
		// fall back to legacy IPs
		config.externalAddrs = framework.NodeAddresses(nodeList, api.NodeLegacyHostIP)
	}
	Expect(len(config.externalAddrs)).To(BeNumerically(">=", 2), fmt.Sprintf("At least two nodes necessary with an external or LegacyHostIP"))
	config.nodes = nodeList.Items

	if enableLoadBalancerTest {
		By("Creating the LoadBalancer Service on top of the pods in kubernetes")
		config.createLoadBalancerService(serviceSelector)
	}

	By("Creating the service pods in kubernetes")
	podName := "netserver"
	config.endpointPods = config.createNetProxyPods(podName, serviceSelector)

	By("Creating the service on top of the pods in kubernetes")
	config.createNodePortService(serviceSelector)

	By("Creating test pods")
	config.createTestPods()
}

func (config *KubeProxyTestConfig) cleanup() {
	nsClient := config.getNamespacesClient()
	nsList, err := nsClient.List(api.ListOptions{})
	if err == nil {
		for _, ns := range nsList.Items {
			if strings.Contains(ns.Name, config.f.BaseName) && ns.Name != config.f.Namespace.Name {
				nsClient.Delete(ns.Name)
			}
		}
	}
}

func (config *KubeProxyTestConfig) createNetProxyPods(podName string, selector map[string]string) []*api.Pod {
	framework.ExpectNoError(framework.WaitForAllNodesSchedulable(config.f.Client))
	nodes := framework.GetReadySchedulableNodesOrDie(config.f.Client)

	// create pods, one for each node
	createdPods := make([]*api.Pod, 0, len(nodes.Items))
	for i, n := range nodes.Items {
		podName := fmt.Sprintf("%s-%d", podName, i)
		pod := config.createNetShellPodSpec(podName, n.Name)
		pod.ObjectMeta.Labels = selector
		createdPod := config.createPod(pod)
		createdPods = append(createdPods, createdPod)
	}

	// wait that all of them are up
	runningPods := make([]*api.Pod, 0, len(nodes.Items))
	for _, p := range createdPods {
		framework.ExpectNoError(config.f.WaitForPodReady(p.Name))
		rp, err := config.getPodClient().Get(p.Name)
		framework.ExpectNoError(err)
		runningPods = append(runningPods, rp)
	}

	return runningPods
}

func (config *KubeProxyTestConfig) deleteNetProxyPod() {
	pod := config.endpointPods[0]
	config.getPodClient().Delete(pod.Name, api.NewDeleteOptions(0))
	config.endpointPods = config.endpointPods[1:]
	// wait for pod being deleted.
	err := framework.WaitForPodToDisappear(config.f.Client, config.f.Namespace.Name, pod.Name, labels.Everything(), time.Second, wait.ForeverTestTimeout)
	if err != nil {
		framework.Failf("Failed to delete %s pod: %v", pod.Name, err)
	}
	// wait for endpoint being removed.
	err = framework.WaitForServiceEndpointsNum(config.f.Client, config.f.Namespace.Name, nodePortServiceName, len(config.endpointPods), time.Second, wait.ForeverTestTimeout)
	if err != nil {
		framework.Failf("Failed to remove endpoint from service: %s", nodePortServiceName)
	}
	// wait for kube-proxy to catch up with the pod being deleted.
	time.Sleep(5 * time.Second)
}

func (config *KubeProxyTestConfig) createPod(pod *api.Pod) *api.Pod {
	createdPod, err := config.getPodClient().Create(pod)
	if err != nil {
		framework.Failf("Failed to create %s pod: %v", pod.Name, err)
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
