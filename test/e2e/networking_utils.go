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
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	endpointHttpPort      = 8080
	endpointUdpPort       = 8081
	testContainerHttpPort = 8080
	clusterHttpPort       = 80
	clusterUdpPort        = 90
	netexecImageName      = "gcr.io/google_containers/netexec:1.5"
	hostexecImageName     = "gcr.io/google_containers/hostexec:1.2"
	testPodName           = "test-container-pod"
	hostTestPodName       = "host-test-container-pod"
	nodePortServiceName   = "node-port-service"
	hitEndpointRetryDelay = 1 * time.Second
	// Number of retries to hit a given set of endpoints. Needs to be high
	// because we verify iptables statistical rr loadbalancing.
	testTries = 30
)

// NewNetworkingTestConfig creates and sets up a new test config helper.
func NewNetworkingTestConfig(f *framework.Framework) *NetworkingTestConfig {
	config := &NetworkingTestConfig{f: f, ns: f.Namespace.Name}
	By(fmt.Sprintf("Performing setup for networking test in namespace %v", config.ns))
	config.setup()
	return config
}

// NetworkingTestConfig is a convenience class around some utility methods
// for testing kubeproxy/networking/services/endpoints.
type NetworkingTestConfig struct {
	// testContaienrPod is a test pod running the netexec image. It is capable
	// of executing tcp/udp requests against ip:port.
	testContainerPod *api.Pod
	// hostTestContainerPod is a pod running with hostNetworking=true, and the
	// hostexec image.
	hostTestContainerPod *api.Pod
	// endpointPods are the pods belonging to the Service created by this
	// test config. Each invocation of `setup` creates a service with
	// 1 pod per node running the netexecImage.
	endpointPods []*api.Pod
	f            *framework.Framework
	// nodePortService is a Service with Type=NodePort spanning over all
	// endpointPods.
	nodePortService *api.Service
	// externalAddrs is a list of external IPs of nodes in the cluster.
	externalAddrs []string
	// nodes is a list of nodes in the cluster.
	nodes []api.Node
	// maxTries is the number of retries tolerated for tests run against
	// endpoints and services created by this config.
	maxTries int
	// The clusterIP of the Service reated by this test config.
	clusterIP string
	// External ip of first node for use in nodePort testing.
	nodeIP string
	// The http/udp nodePorts of the Service.
	nodeHttpPort int
	nodeUdpPort  int
	// The kubernetes namespace within which all resources for this
	// config are created
	ns string
}

func (config *NetworkingTestConfig) dialFromEndpointContainer(protocol, targetIP string, targetPort, maxTries, minTries int, expectedEps sets.String) {
	config.dialFromContainer(protocol, config.endpointPods[0].Status.PodIP, targetIP, endpointHttpPort, targetPort, maxTries, minTries, expectedEps)
}

func (config *NetworkingTestConfig) dialFromTestContainer(protocol, targetIP string, targetPort, maxTries, minTries int, expectedEps sets.String) {
	config.dialFromContainer(protocol, config.testContainerPod.Status.PodIP, targetIP, testContainerHttpPort, targetPort, maxTries, minTries, expectedEps)
}

// diagnoseMissingEndpoints prints debug information about the endpoints that
// are NOT in the given list of foundEndpoints. These are the endpoints we
// expected a response from.
func (config *NetworkingTestConfig) diagnoseMissingEndpoints(foundEndpoints sets.String) {
	for _, e := range config.endpointPods {
		if foundEndpoints.Has(e.Name) {
			continue
		}
		framework.Logf("\nOutput of kubectl describe pod %v/%v:\n", e.Namespace, e.Name)
		desc, _ := framework.RunKubectl(
			"describe", "pod", e.Name, fmt.Sprintf("--namespace=%v", e.Namespace))
		framework.Logf(desc)
	}
}

// endpointHostnames returns a set of hostnames for existing endpoints.
func (config *NetworkingTestConfig) endpointHostnames() sets.String {
	expectedEps := sets.NewString()
	for _, p := range config.endpointPods {
		expectedEps.Insert(p.Name)
	}
	return expectedEps
}

// dialFromContainers executes a curl via kubectl exec in a test container,
// which might then translate to a tcp or udp request based on the protocol
// argument in the url.
// - minTries is the minimum number of curl attempts required before declaring
//   success. Set to 0 if you'd like to return as soon as all endpoints respond
//   at least once.
// - maxTries is the maximum number of curl attempts. If this many attempts pass
//   and we don't see all expected endpoints, the test fails.
// - expectedEps is the set of endpointnames to wait for. Typically this is also
//   the hostname reported by each pod in the service through /hostName.
// maxTries == minTries will confirm that we see the expected endpoints and no
// more for maxTries. Use this if you want to eg: fail a readiness check on a
// pod and confirm it doesn't show up as an endpoint.
func (config *NetworkingTestConfig) dialFromContainer(protocol, containerIP, targetIP string, containerHttpPort, targetPort, maxTries, minTries int, expectedEps sets.String) {
	cmd := fmt.Sprintf("curl -q -s 'http://%s:%d/dial?request=hostName&protocol=%s&host=%s&port=%d&tries=1'",
		containerIP,
		containerHttpPort,
		protocol,
		targetIP,
		targetPort)

	eps := sets.NewString()

	for i := 0; i < maxTries; i++ {
		stdout, err := framework.RunHostCmd(config.ns, config.hostTestContainerPod.Name, cmd)
		if err != nil {
			// A failure to kubectl exec counts as a try, not a hard fail.
			// Also note that we will keep failing for maxTries in tests where
			// we confirm unreachability.
			framework.Logf("Failed to execute %v: %v", cmd, err)
		} else {
			var output map[string][]string
			if err := json.Unmarshal([]byte(stdout), &output); err != nil {
				framework.Logf("WARNING: Failed to unmarshal curl response. Cmd %v run in %v, output: %s, err: %v",
					cmd, config.hostTestContainerPod.Name, stdout, err)
				continue
			}
			for _, hostName := range output["responses"] {
				eps.Insert(hostName)
			}
		}
		framework.Logf("Waiting for endpoints: %v", expectedEps.Difference(eps))

		// Check against i+1 so we exit if minTries == maxTries.
		if (eps.Equal(expectedEps) || eps.Len() == 0 && expectedEps.Len() == 0) && i+1 >= minTries {
			return
		}
	}

	config.diagnoseMissingEndpoints(eps)
	framework.Failf("Failed to find expected endpoints:\nTries %d\nCommand %v\nretrieved %v\nexpected %v\n", minTries, cmd, eps, expectedEps)
}

// dialFromNode executes a tcp or udp request based on protocol via kubectl exec
// in a test container running with host networking.
// - minTries is the minimum number of curl attempts required before declaring
//   success. Set to 0 if you'd like to return as soon as all endpoints respond
//   at least once.
// - maxTries is the maximum number of curl attempts. If this many attempts pass
//   and we don't see all expected endpoints, the test fails.
// maxTries == minTries will confirm that we see the expected endpoints and no
// more for maxTries. Use this if you want to eg: fail a readiness check on a
// pod and confirm it doesn't show up as an endpoint.
func (config *NetworkingTestConfig) dialFromNode(protocol, targetIP string, targetPort, maxTries, minTries int, expectedEps sets.String) {
	var cmd string
	if protocol == "udp" {
		cmd = fmt.Sprintf("echo 'hostName' | timeout -t 3 nc -w 1 -u %s %d", targetIP, targetPort)
	} else {
		cmd = fmt.Sprintf("curl -q -s --connect-timeout 1 http://%s:%d/hostName", targetIP, targetPort)
	}

	// TODO: This simply tells us that we can reach the endpoints. Check that
	// the probability of hitting a specific endpoint is roughly the same as
	// hitting any other.
	eps := sets.NewString()

	filterCmd := fmt.Sprintf("%s | grep -v '^\\s*$'", cmd)
	for i := 0; i < maxTries; i++ {
		stdout, err := framework.RunHostCmd(config.ns, config.hostTestContainerPod.Name, filterCmd)
		if err != nil {
			// A failure to kubectl exec counts as a try, not a hard fail.
			// Also note that we will keep failing for maxTries in tests where
			// we confirm unreachability.
			framework.Logf("Failed to execute %v: %v", filterCmd, err)
		} else {
			eps.Insert(strings.TrimSpace(stdout))
		}
		framework.Logf("Waiting for %+v endpoints, got endpoints %+v", expectedEps.Difference(eps), eps)

		// Check against i+1 so we exit if minTries == maxTries.
		if (eps.Equal(expectedEps) || eps.Len() == 0 && expectedEps.Len() == 0) && i+1 >= minTries {
			return
		}
	}

	config.diagnoseMissingEndpoints(eps)
	framework.Failf("Failed to find expected endpoints:\nTries %d\nCommand %v\nretrieved %v\nexpected %v\n", minTries, cmd, eps, expectedEps)
}

// getSelfURL executes a curl against the given path via kubectl exec into a
// test container running with host networking, and fails if the output
// doesn't match the expected string.
func (config *NetworkingTestConfig) getSelfURL(path string, expected string) {
	cmd := fmt.Sprintf("curl -q -s --connect-timeout 1 http://localhost:10249%s", path)
	By(fmt.Sprintf("Getting kube-proxy self URL %s", path))
	stdout := framework.RunHostCmdOrDie(config.ns, config.hostTestContainerPod.Name, cmd)
	Expect(strings.Contains(stdout, expected)).To(BeTrue())
}

func (config *NetworkingTestConfig) createNetShellPodSpec(podName string, node string) *api.Pod {
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
			Namespace: config.ns,
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

func (config *NetworkingTestConfig) createTestPodSpec() *api.Pod {
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: registered.GroupOrDie(api.GroupName).GroupVersion.String(),
		},
		ObjectMeta: api.ObjectMeta{
			Name:      testPodName,
			Namespace: config.ns,
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

func (config *NetworkingTestConfig) createNodePortService(selector map[string]string) {
	serviceSpec := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: nodePortServiceName,
		},
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeNodePort,
			Ports: []api.ServicePort{
				{Port: clusterHttpPort, Name: "http", Protocol: api.ProtocolTCP, TargetPort: intstr.FromInt(endpointHttpPort)},
				{Port: clusterUdpPort, Name: "udp", Protocol: api.ProtocolUDP, TargetPort: intstr.FromInt(endpointUdpPort)},
			},
			Selector: selector,
		},
	}
	config.nodePortService = config.createService(serviceSpec)
}

func (config *NetworkingTestConfig) deleteNodePortService() {
	err := config.getServiceClient().Delete(config.nodePortService.Name)
	Expect(err).NotTo(HaveOccurred(), "error while deleting NodePortService. err:%v)", err)
	time.Sleep(15 * time.Second) // wait for kube-proxy to catch up with the service being deleted.
}

func (config *NetworkingTestConfig) createTestPods() {
	testContainerPod := config.createTestPodSpec()
	hostTestContainerPod := framework.NewHostExecPodSpec(config.ns, hostTestPodName)

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

func (config *NetworkingTestConfig) createService(serviceSpec *api.Service) *api.Service {
	_, err := config.getServiceClient().Create(serviceSpec)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create %s service: %v", serviceSpec.Name, err))

	err = framework.WaitForService(config.f.Client, config.ns, serviceSpec.Name, true, 5*time.Second, 45*time.Second)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("error while waiting for service:%s err: %v", serviceSpec.Name, err))

	createdService, err := config.getServiceClient().Get(serviceSpec.Name)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create %s service: %v", serviceSpec.Name, err))

	return createdService
}

func (config *NetworkingTestConfig) setup() {
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

	By("Creating the service pods in kubernetes")
	podName := "netserver"
	config.endpointPods = config.createNetProxyPods(podName, serviceSelector)

	By("Creating the service on top of the pods in kubernetes")
	config.createNodePortService(serviceSelector)

	By("Creating test pods")
	config.createTestPods()
	for _, p := range config.nodePortService.Spec.Ports {
		switch p.Protocol {
		case api.ProtocolUDP:
			config.nodeUdpPort = int(p.NodePort)
		case api.ProtocolTCP:
			config.nodeHttpPort = int(p.NodePort)
		default:
			continue
		}
	}

	epCount := len(config.endpointPods)
	config.maxTries = epCount*epCount + testTries
	config.clusterIP = config.nodePortService.Spec.ClusterIP
	config.nodeIP = config.externalAddrs[0]
}

func (config *NetworkingTestConfig) cleanup() {
	nsClient := config.getNamespacesClient()
	nsList, err := nsClient.List(api.ListOptions{})
	if err == nil {
		for _, ns := range nsList.Items {
			if strings.Contains(ns.Name, config.f.BaseName) && ns.Name != config.ns {
				nsClient.Delete(ns.Name)
			}
		}
	}
}

func (config *NetworkingTestConfig) createNetProxyPods(podName string, selector map[string]string) []*api.Pod {
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

func (config *NetworkingTestConfig) deleteNetProxyPod() {
	pod := config.endpointPods[0]
	config.getPodClient().Delete(pod.Name, api.NewDeleteOptions(0))
	config.endpointPods = config.endpointPods[1:]
	// wait for pod being deleted.
	err := framework.WaitForPodToDisappear(config.f.Client, config.ns, pod.Name, labels.Everything(), time.Second, wait.ForeverTestTimeout)
	if err != nil {
		framework.Failf("Failed to delete %s pod: %v", pod.Name, err)
	}
	// wait for endpoint being removed.
	err = framework.WaitForServiceEndpointsNum(config.f.Client, config.ns, nodePortServiceName, len(config.endpointPods), time.Second, wait.ForeverTestTimeout)
	if err != nil {
		framework.Failf("Failed to remove endpoint from service: %s", nodePortServiceName)
	}
	// wait for kube-proxy to catch up with the pod being deleted.
	time.Sleep(5 * time.Second)
}

func (config *NetworkingTestConfig) createPod(pod *api.Pod) *api.Pod {
	createdPod, err := config.getPodClient().Create(pod)
	if err != nil {
		framework.Failf("Failed to create %s pod: %v", pod.Name, err)
	}
	return createdPod
}

func (config *NetworkingTestConfig) getPodClient() client.PodInterface {
	return config.f.Client.Pods(config.ns)
}

func (config *NetworkingTestConfig) getServiceClient() client.ServiceInterface {
	return config.f.Client.Services(config.ns)
}

func (config *NetworkingTestConfig) getNamespacesClient() client.NamespaceInterface {
	return config.f.Client.Namespaces()
}
