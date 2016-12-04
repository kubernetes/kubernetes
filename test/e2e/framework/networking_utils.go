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

package framework

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	coreclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/rand"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	EndpointHttpPort      = 8080
	EndpointUdpPort       = 8081
	TestContainerHttpPort = 8080
	ClusterHttpPort       = 80
	ClusterUdpPort        = 90
	NetexecImageName      = "gcr.io/google_containers/netexec:1.7"
	HostexecImageName     = "gcr.io/google_containers/hostexec:1.2"
	testPodName           = "test-container-pod"
	hostTestPodName       = "host-test-container-pod"
	nodePortServiceName   = "node-port-service"
	// wait time between poll attempts of a Service vip and/or nodePort.
	// coupled with testTries to produce a net timeout value.
	hitEndpointRetryDelay = 2 * time.Second
	// Number of retries to hit a given set of endpoints. Needs to be high
	// because we verify iptables statistical rr loadbalancing.
	testTries = 30
	// Maximum number of pods in a test, to make test work in large clusters.
	maxNetProxyPodsCount = 10
)

// NewNetworkingTestConfig creates and sets up a new test config helper.
func NewNetworkingTestConfig(f *Framework) *NetworkingTestConfig {
	config := &NetworkingTestConfig{f: f, Namespace: f.Namespace.Name}
	By(fmt.Sprintf("Performing setup for networking test in namespace %v", config.Namespace))
	config.setup(getServiceSelector())
	return config
}

// NewNetworkingTestNodeE2EConfig creates and sets up a new test config helper for Node E2E.
func NewCoreNetworkingTestConfig(f *Framework) *NetworkingTestConfig {
	config := &NetworkingTestConfig{f: f, Namespace: f.Namespace.Name}
	By(fmt.Sprintf("Performing setup for networking test in namespace %v", config.Namespace))
	config.setupCore(getServiceSelector())
	return config
}

func getServiceSelector() map[string]string {
	By("creating a selector")
	selectorName := "selector-" + string(uuid.NewUUID())
	serviceSelector := map[string]string{
		selectorName: "true",
	}
	return serviceSelector
}

// NetworkingTestConfig is a convenience class around some utility methods
// for testing kubeproxy/networking/services/endpoints.
type NetworkingTestConfig struct {
	// TestContaienrPod is a test pod running the netexec image. It is capable
	// of executing tcp/udp requests against ip:port.
	TestContainerPod *v1.Pod
	// HostTestContainerPod is a pod running with hostNetworking=true, and the
	// hostexec image.
	HostTestContainerPod *v1.Pod
	// EndpointPods are the pods belonging to the Service created by this
	// test config. Each invocation of `setup` creates a service with
	// 1 pod per node running the netexecImage.
	EndpointPods []*v1.Pod
	f            *Framework
	podClient    *PodClient
	// NodePortService is a Service with Type=NodePort spanning over all
	// endpointPods.
	NodePortService *v1.Service
	// ExternalAddrs is a list of external IPs of nodes in the cluster.
	ExternalAddrs []string
	// Nodes is a list of nodes in the cluster.
	Nodes []v1.Node
	// MaxTries is the number of retries tolerated for tests run against
	// endpoints and services created by this config.
	MaxTries int
	// The ClusterIP of the Service reated by this test config.
	ClusterIP string
	// External ip of first node for use in nodePort testing.
	NodeIP string
	// The http/udp nodePorts of the Service.
	NodeHttpPort int
	NodeUdpPort  int
	// The kubernetes namespace within which all resources for this
	// config are created
	Namespace string
}

func (config *NetworkingTestConfig) DialFromEndpointContainer(protocol, targetIP string, targetPort, maxTries, minTries int, expectedEps sets.String) {
	config.DialFromContainer(protocol, config.EndpointPods[0].Status.PodIP, targetIP, EndpointHttpPort, targetPort, maxTries, minTries, expectedEps)
}

func (config *NetworkingTestConfig) DialFromTestContainer(protocol, targetIP string, targetPort, maxTries, minTries int, expectedEps sets.String) {
	config.DialFromContainer(protocol, config.TestContainerPod.Status.PodIP, targetIP, TestContainerHttpPort, targetPort, maxTries, minTries, expectedEps)
}

// diagnoseMissingEndpoints prints debug information about the endpoints that
// are NOT in the given list of foundEndpoints. These are the endpoints we
// expected a response from.
func (config *NetworkingTestConfig) diagnoseMissingEndpoints(foundEndpoints sets.String) {
	for _, e := range config.EndpointPods {
		if foundEndpoints.Has(e.Name) {
			continue
		}
		Logf("\nOutput of kubectl describe pod %v/%v:\n", e.Namespace, e.Name)
		desc, _ := RunKubectl(
			"describe", "pod", e.Name, fmt.Sprintf("--namespace=%v", e.Namespace))
		Logf(desc)
	}
}

// EndpointHostnames returns a set of hostnames for existing endpoints.
func (config *NetworkingTestConfig) EndpointHostnames() sets.String {
	expectedEps := sets.NewString()
	for _, p := range config.EndpointPods {
		expectedEps.Insert(p.Name)
	}
	return expectedEps
}

// DialFromContainers executes a curl via kubectl exec in a test container,
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
func (config *NetworkingTestConfig) DialFromContainer(protocol, containerIP, targetIP string, containerHttpPort, targetPort, maxTries, minTries int, expectedEps sets.String) {
	cmd := fmt.Sprintf("curl -q -s 'http://%s:%d/dial?request=hostName&protocol=%s&host=%s&port=%d&tries=1'",
		containerIP,
		containerHttpPort,
		protocol,
		targetIP,
		targetPort)

	eps := sets.NewString()

	for i := 0; i < maxTries; i++ {
		stdout, stderr, err := config.f.ExecShellInPodWithFullOutput(config.HostTestContainerPod.Name, cmd)
		if err != nil {
			// A failure to kubectl exec counts as a try, not a hard fail.
			// Also note that we will keep failing for maxTries in tests where
			// we confirm unreachability.
			Logf("Failed to execute %q: %v, stdout: %q, stderr %q", cmd, err, stdout, stderr)
		} else {
			var output map[string][]string
			if err := json.Unmarshal([]byte(stdout), &output); err != nil {
				Logf("WARNING: Failed to unmarshal curl response. Cmd %v run in %v, output: %s, err: %v",
					cmd, config.HostTestContainerPod.Name, stdout, err)
				continue
			}

			for _, hostName := range output["responses"] {
				trimmed := strings.TrimSpace(hostName)
				if trimmed != "" {
					eps.Insert(trimmed)
				}
			}
		}
		Logf("Waiting for endpoints: %v", expectedEps.Difference(eps))

		// Check against i+1 so we exit if minTries == maxTries.
		if (eps.Equal(expectedEps) || eps.Len() == 0 && expectedEps.Len() == 0) && i+1 >= minTries {
			return
		}
		// TODO: get rid of this delay #36281
		time.Sleep(hitEndpointRetryDelay)
	}

	config.diagnoseMissingEndpoints(eps)
	Failf("Failed to find expected endpoints:\nTries %d\nCommand %v\nretrieved %v\nexpected %v\n", minTries, cmd, eps, expectedEps)
}

// DialFromNode executes a tcp or udp request based on protocol via kubectl exec
// in a test container running with host networking.
// - minTries is the minimum number of curl attempts required before declaring
//   success. Set to 0 if you'd like to return as soon as all endpoints respond
//   at least once.
// - maxTries is the maximum number of curl attempts. If this many attempts pass
//   and we don't see all expected endpoints, the test fails.
// maxTries == minTries will confirm that we see the expected endpoints and no
// more for maxTries. Use this if you want to eg: fail a readiness check on a
// pod and confirm it doesn't show up as an endpoint.
func (config *NetworkingTestConfig) DialFromNode(protocol, targetIP string, targetPort, maxTries, minTries int, expectedEps sets.String) {
	var cmd string
	if protocol == "udp" {
		// TODO: It would be enough to pass 1s+epsilon to timeout, but unfortunately
		// busybox timeout doesn't support non-integer values.
		cmd = fmt.Sprintf("echo 'hostName' | timeout -t 2 nc -w 1 -u %s %d", targetIP, targetPort)
	} else {
		cmd = fmt.Sprintf("timeout -t 15 curl -q -s --connect-timeout 1 http://%s:%d/hostName", targetIP, targetPort)
	}

	// TODO: This simply tells us that we can reach the endpoints. Check that
	// the probability of hitting a specific endpoint is roughly the same as
	// hitting any other.
	eps := sets.NewString()

	filterCmd := fmt.Sprintf("%s | grep -v '^\\s*$'", cmd)
	for i := 0; i < maxTries; i++ {
		stdout, stderr, err := config.f.ExecShellInPodWithFullOutput(config.HostTestContainerPod.Name, filterCmd)
		if err != nil || len(stderr) > 0 {
			// A failure to exec command counts as a try, not a hard fail.
			// Also note that we will keep failing for maxTries in tests where
			// we confirm unreachability.
			Logf("Failed to execute %q: %v, stdout: %q, stderr: %q", filterCmd, err, stdout, stderr)
		} else {
			trimmed := strings.TrimSpace(stdout)
			if trimmed != "" {
				eps.Insert(trimmed)
			}
		}
		Logf("Waiting for %+v endpoints, got endpoints %+v", expectedEps.Difference(eps), eps)

		// Check against i+1 so we exit if minTries == maxTries.
		if (eps.Equal(expectedEps) || eps.Len() == 0 && expectedEps.Len() == 0) && i+1 >= minTries {
			return
		}
		// TODO: get rid of this delay #36281
		time.Sleep(hitEndpointRetryDelay)
	}

	config.diagnoseMissingEndpoints(eps)
	Failf("Failed to find expected endpoints:\nTries %d\nCommand %v\nretrieved %v\nexpected %v\n", minTries, cmd, eps, expectedEps)
}

// GetSelfURL executes a curl against the given path via kubectl exec into a
// test container running with host networking, and fails if the output
// doesn't match the expected string.
func (config *NetworkingTestConfig) GetSelfURL(path string, expected string) {
	cmd := fmt.Sprintf("curl -q -s --connect-timeout 1 http://localhost:10249%s", path)
	By(fmt.Sprintf("Getting kube-proxy self URL %s", path))

	// These are arbitrary timeouts. The curl command should pass on first try,
	// unless kubeproxy is starved/bootstrapping/restarting etc.
	const retryInterval = 1 * time.Second
	const retryTimeout = 30 * time.Second
	podName := config.HostTestContainerPod.Name
	var msg string
	if pollErr := wait.PollImmediate(retryInterval, retryTimeout, func() (bool, error) {
		stdout, err := RunHostCmd(config.Namespace, podName, cmd)
		if err != nil {
			msg = fmt.Sprintf("failed executing cmd %v in %v/%v: %v", cmd, config.Namespace, podName, err)
			Logf(msg)
			return false, nil
		}
		if !strings.Contains(stdout, expected) {
			msg = fmt.Sprintf("successfully executed %v in %v/%v, but output '%v' doesn't contain expected string '%v'", cmd, config.Namespace, podName, stdout, expected)
			Logf(msg)
			return false, nil
		}
		return true, nil
	}); pollErr != nil {
		Logf("\nOutput of kubectl describe pod %v/%v:\n", config.Namespace, podName)
		desc, _ := RunKubectl(
			"describe", "pod", podName, fmt.Sprintf("--namespace=%v", config.Namespace))
		Logf("%s", desc)
		Failf("Timed out in %v: %v", retryTimeout, msg)
	}
}

func (config *NetworkingTestConfig) createNetShellPodSpec(podName string, node string) *v1.Pod {
	probe := &v1.Probe{
		InitialDelaySeconds: 10,
		TimeoutSeconds:      30,
		PeriodSeconds:       10,
		SuccessThreshold:    1,
		FailureThreshold:    3,
		Handler: v1.Handler{
			HTTPGet: &v1.HTTPGetAction{
				Path: "/healthz",
				Port: intstr.IntOrString{IntVal: EndpointHttpPort},
			},
		},
	}
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: registered.GroupOrDie(v1.GroupName).GroupVersion.String(),
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      podName,
			Namespace: config.Namespace,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "webserver",
					Image:           NetexecImageName,
					ImagePullPolicy: v1.PullIfNotPresent,
					Command: []string{
						"/netexec",
						fmt.Sprintf("--http-port=%d", EndpointHttpPort),
						fmt.Sprintf("--udp-port=%d", EndpointUdpPort),
					},
					Ports: []v1.ContainerPort{
						{
							Name:          "http",
							ContainerPort: EndpointHttpPort,
						},
						{
							Name:          "udp",
							ContainerPort: EndpointUdpPort,
							Protocol:      v1.ProtocolUDP,
						},
					},
					LivenessProbe:  probe,
					ReadinessProbe: probe,
				},
			},
			NodeSelector: map[string]string{
				"kubernetes.io/hostname": node,
			},
		},
	}
	return pod
}

func (config *NetworkingTestConfig) createTestPodSpec() *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: registered.GroupOrDie(v1.GroupName).GroupVersion.String(),
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      testPodName,
			Namespace: config.Namespace,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "webserver",
					Image:           NetexecImageName,
					ImagePullPolicy: v1.PullIfNotPresent,
					Command: []string{
						"/netexec",
						fmt.Sprintf("--http-port=%d", EndpointHttpPort),
						fmt.Sprintf("--udp-port=%d", EndpointUdpPort),
					},
					Ports: []v1.ContainerPort{
						{
							Name:          "http",
							ContainerPort: TestContainerHttpPort,
						},
					},
				},
			},
		},
	}
	return pod
}

func (config *NetworkingTestConfig) createNodePortService(selector map[string]string) {
	serviceSpec := &v1.Service{
		ObjectMeta: v1.ObjectMeta{
			Name: nodePortServiceName,
		},
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeNodePort,
			Ports: []v1.ServicePort{
				{Port: ClusterHttpPort, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(EndpointHttpPort)},
				{Port: ClusterUdpPort, Name: "udp", Protocol: v1.ProtocolUDP, TargetPort: intstr.FromInt(EndpointUdpPort)},
			},
			Selector: selector,
		},
	}
	config.NodePortService = config.createService(serviceSpec)
}

func (config *NetworkingTestConfig) DeleteNodePortService() {
	err := config.getServiceClient().Delete(config.NodePortService.Name, nil)
	Expect(err).NotTo(HaveOccurred(), "error while deleting NodePortService. err:%v)", err)
	time.Sleep(15 * time.Second) // wait for kube-proxy to catch up with the service being deleted.
}

func (config *NetworkingTestConfig) createTestPods() {
	testContainerPod := config.createTestPodSpec()
	hostTestContainerPod := NewHostExecPodSpec(config.Namespace, hostTestPodName)

	config.createPod(testContainerPod)
	config.createPod(hostTestContainerPod)

	ExpectNoError(config.f.WaitForPodRunning(testContainerPod.Name))
	ExpectNoError(config.f.WaitForPodRunning(hostTestContainerPod.Name))

	var err error
	config.TestContainerPod, err = config.getPodClient().Get(testContainerPod.Name)
	if err != nil {
		Failf("Failed to retrieve %s pod: %v", testContainerPod.Name, err)
	}

	config.HostTestContainerPod, err = config.getPodClient().Get(hostTestContainerPod.Name)
	if err != nil {
		Failf("Failed to retrieve %s pod: %v", hostTestContainerPod.Name, err)
	}
}

func (config *NetworkingTestConfig) createService(serviceSpec *v1.Service) *v1.Service {
	_, err := config.getServiceClient().Create(serviceSpec)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create %s service: %v", serviceSpec.Name, err))

	err = WaitForService(config.f.ClientSet, config.Namespace, serviceSpec.Name, true, 5*time.Second, 45*time.Second)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("error while waiting for service:%s err: %v", serviceSpec.Name, err))

	createdService, err := config.getServiceClient().Get(serviceSpec.Name)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create %s service: %v", serviceSpec.Name, err))

	return createdService
}

// setupCore sets up the pods and core test config
// mainly for simplified node e2e setup
func (config *NetworkingTestConfig) setupCore(selector map[string]string) {
	By("Creating the service pods in kubernetes")
	podName := "netserver"
	config.EndpointPods = config.createNetProxyPods(podName, selector)

	By("Creating test pods")
	config.createTestPods()

	epCount := len(config.EndpointPods)
	config.MaxTries = epCount*epCount + testTries
}

// setup includes setupCore and also sets up services
func (config *NetworkingTestConfig) setup(selector map[string]string) {
	config.setupCore(selector)

	By("Getting node addresses")
	ExpectNoError(WaitForAllNodesSchedulable(config.f.ClientSet, 10*time.Minute))
	nodeList := GetReadySchedulableNodesOrDie(config.f.ClientSet)
	config.ExternalAddrs = NodeAddresses(nodeList, v1.NodeExternalIP)
	if len(config.ExternalAddrs) < 2 {
		// fall back to legacy IPs
		config.ExternalAddrs = NodeAddresses(nodeList, v1.NodeLegacyHostIP)
	}
	SkipUnlessNodeCountIsAtLeast(2)
	config.Nodes = nodeList.Items

	By("Creating the service on top of the pods in kubernetes")
	config.createNodePortService(selector)

	for _, p := range config.NodePortService.Spec.Ports {
		switch p.Protocol {
		case v1.ProtocolUDP:
			config.NodeUdpPort = int(p.NodePort)
		case v1.ProtocolTCP:
			config.NodeHttpPort = int(p.NodePort)
		default:
			continue
		}
	}
	config.ClusterIP = config.NodePortService.Spec.ClusterIP
	config.NodeIP = config.ExternalAddrs[0]
}

func (config *NetworkingTestConfig) cleanup() {
	nsClient := config.getNamespacesClient()
	nsList, err := nsClient.List(v1.ListOptions{})
	if err == nil {
		for _, ns := range nsList.Items {
			if strings.Contains(ns.Name, config.f.BaseName) && ns.Name != config.Namespace {
				nsClient.Delete(ns.Name, nil)
			}
		}
	}
}

// shuffleNodes copies nodes from the specified slice into a copy in random
// order. It returns a new slice.
func shuffleNodes(nodes []v1.Node) []v1.Node {
	shuffled := make([]v1.Node, len(nodes))
	perm := rand.Perm(len(nodes))
	for i, j := range perm {
		shuffled[j] = nodes[i]
	}
	return shuffled
}

func (config *NetworkingTestConfig) createNetProxyPods(podName string, selector map[string]string) []*v1.Pod {
	ExpectNoError(WaitForAllNodesSchedulable(config.f.ClientSet, 10*time.Minute))
	nodeList := GetReadySchedulableNodesOrDie(config.f.ClientSet)

	// To make this test work reasonably fast in large clusters,
	// we limit the number of NetProxyPods to no more than
	// maxNetProxyPodsCount on random nodes.
	nodes := shuffleNodes(nodeList.Items)
	if len(nodes) > maxNetProxyPodsCount {
		nodes = nodes[:maxNetProxyPodsCount]
	}

	// create pods, one for each node
	createdPods := make([]*v1.Pod, 0, len(nodes))
	for i, n := range nodes {
		podName := fmt.Sprintf("%s-%d", podName, i)
		pod := config.createNetShellPodSpec(podName, n.Name)
		pod.ObjectMeta.Labels = selector
		createdPod := config.createPod(pod)
		createdPods = append(createdPods, createdPod)
	}

	// wait that all of them are up
	runningPods := make([]*v1.Pod, 0, len(nodes))
	for _, p := range createdPods {
		ExpectNoError(config.f.WaitForPodReady(p.Name))
		rp, err := config.getPodClient().Get(p.Name)
		ExpectNoError(err)
		runningPods = append(runningPods, rp)
	}

	return runningPods
}

func (config *NetworkingTestConfig) DeleteNetProxyPod() {
	pod := config.EndpointPods[0]
	config.getPodClient().Delete(pod.Name, v1.NewDeleteOptions(0))
	config.EndpointPods = config.EndpointPods[1:]
	// wait for pod being deleted.
	err := WaitForPodToDisappear(config.f.ClientSet, config.Namespace, pod.Name, labels.Everything(), time.Second, wait.ForeverTestTimeout)
	if err != nil {
		Failf("Failed to delete %s pod: %v", pod.Name, err)
	}
	// wait for endpoint being removed.
	err = WaitForServiceEndpointsNum(config.f.ClientSet, config.Namespace, nodePortServiceName, len(config.EndpointPods), time.Second, wait.ForeverTestTimeout)
	if err != nil {
		Failf("Failed to remove endpoint from service: %s", nodePortServiceName)
	}
	// wait for kube-proxy to catch up with the pod being deleted.
	time.Sleep(5 * time.Second)
}

func (config *NetworkingTestConfig) createPod(pod *v1.Pod) *v1.Pod {
	return config.getPodClient().Create(pod)
}

func (config *NetworkingTestConfig) getPodClient() *PodClient {
	if config.podClient == nil {
		config.podClient = config.f.PodClient()
	}
	return config.podClient
}

func (config *NetworkingTestConfig) getServiceClient() coreclientset.ServiceInterface {
	return config.f.ClientSet.Core().Services(config.Namespace)
}

func (config *NetworkingTestConfig) getNamespacesClient() coreclientset.NamespaceInterface {
	return config.f.ClientSet.Core().Namespaces()
}
