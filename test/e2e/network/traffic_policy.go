/*
Copyright 2023 The Kubernetes Authors.

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
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edaemonset "k8s.io/kubernetes/test/e2e/framework/daemonset"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

// seenPodsCheckFunc is a function to be used by checkEndpointHost below.
// seenPodsCheckFunc should return true if all the handling pods are expected, and false if an invalid result is found.
type seenPodsCheckFunc func(tr *trafficPolicyTestResources, seenPods map[string]string) bool

// This fixture is used to contain resources used to check traffic policy behavior
// At a high level, traffic policy tests below will use a daemonset and NodePort service to verify expected pods handle requests
type trafficPolicyTestResources struct {
	// these maps are helpful when looking at response hostname to verify correct zone or node
	zonesByPod map[string]string
	nodesByPod map[string]string

	// endpointZone is the zone which is expected to respond to an external request
	endpointZone string
	// endpointNode is the node which is expected to respond to an external request
	endpointNode *v1.Node

	// internalClientPod resides on endpointNode and is used to check internal traffic policy behavior
	internalClientPod *v1.Pod
	// externalClientPod runs on a different node from the endpointNode and is used to validate external traffic policy behavior
	externalClientPod *v1.Pod
}

// createTrafficPolicyTestResources will setup a daemonset and 2 client pods and return deployment details in a trafficPolicyTestResources structure
//
// 1. Agnhost netexec DaemonSet with a NodePort Service. The nodeport service is configured according to topologyHints, etp, itp args.
// 2. An 'internal' and 'external' client pod repeatedly send 'hostname' requests to associated endpoints. Responses are used by tests to verify correct pods respond.
func setupAgnhostDaemonset(ctx context.Context, c clientset.Interface, f *framework.Framework, topologyHints bool, etp v1.ServiceExternalTrafficPolicyType, itp v1.ServiceInternalTrafficPolicyType) *trafficPolicyTestResources {
	portNum := 8080
	specName := "traffic-policy"
	tr := trafficPolicyTestResources{
		zonesByPod:        make(map[string]string),
		nodesByPod:        make(map[string]string),
		endpointZone:      "",
		endpointNode:      nil,
		internalClientPod: nil,
		externalClientPod: nil,
	}
	thLabels := map[string]string{labelKey: specName}
	img := imageutils.GetE2EImage(imageutils.Agnhost)
	ports := []v1.ContainerPort{{ContainerPort: int32(portNum)}}
	dsConf := e2edaemonset.NewDaemonSet(specName+"-ds", img, thLabels, nil, nil, ports, "netexec")
	ds, err := c.AppsV1().DaemonSets(f.Namespace.Name).Create(ctx, dsConf, metav1.CreateOptions{})
	framework.ExpectNoError(err, "error creating DaemonSet")

	topologyHintsValue := "Disabled"
	if topologyHints {
		topologyHintsValue = "Auto"
	}
	svc := createServiceReportErr(ctx, c, f.Namespace.Name, &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: specName,
			Annotations: map[string]string{
				v1.AnnotationTopologyAwareHints: topologyHintsValue,
			},
		},
		Spec: v1.ServiceSpec{
			Type:                  v1.ServiceTypeNodePort,
			Selector:              thLabels,
			ExternalTrafficPolicy: etp,
			InternalTrafficPolicy: &itp,
			Ports: []v1.ServicePort{{
				Name:       "agnhost",
				Port:       80,
				TargetPort: intstr.FromInt(portNum),
				Protocol:   v1.ProtocolTCP,
			}},
		},
	})

	err = wait.PollWithContext(ctx, 5*time.Second, framework.PodStartTimeout, func(ctx context.Context) (bool, error) {
		return e2edaemonset.CheckRunningOnAllNodes(ctx, f, ds)
	})
	framework.ExpectNoError(err, "timed out waiting for DaemonSets to be ready")

	// Build a map of nodes which may run the daemonset
	schedulableNodes := map[string]*v1.Node{}
	for _, nodeName := range e2edaemonset.SchedulableNodes(ctx, c, ds) {
		schedulableNodes[nodeName] = nil
	}

	// All Nodes should have same allocatable CPUs. If not, then skip the test.
	nodeList, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "Error when listing all Nodes")
	var lastNodeCPU resource.Quantity
	firstNode := true
	for i := range nodeList.Items {
		node := nodeList.Items[i]
		if _, ok := schedulableNodes[node.Name]; !ok {
			continue
		}
		schedulableNodes[node.Name] = &node

		nodeCPU, found := node.Status.Allocatable[v1.ResourceCPU]
		if !found {
			framework.Failf("Error when getting allocatable CPU of Node '%s'", node.Name)
		}
		if firstNode {
			lastNodeCPU = nodeCPU
			firstNode = false
		} else if !nodeCPU.Equal(lastNodeCPU) {
			e2eskipper.Skipf("Expected Nodes to have equivalent allocatable CPUs, but Node '%s' is different from the previous one. %d not equals %d",
				node.Name, nodeCPU.Value(), lastNodeCPU.Value())
		}
	}

	framework.Logf("Waiting for %d endpoints to be tracked in EndpointSlices", len(schedulableNodes))

	var finalSlices []discoveryv1.EndpointSlice
	err = wait.PollWithContext(ctx, 5*time.Second, svcReadyTimeout, func(ctx context.Context) (bool, error) {
		slices, listErr := c.DiscoveryV1().EndpointSlices(f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: fmt.Sprintf("%s=%s", discoveryv1.LabelServiceName, svc.Name)})
		if listErr != nil {
			return false, listErr
		}

		numEndpoints := 0
		for _, slice := range slices.Items {
			numEndpoints += len(slice.Endpoints)
		}
		if len(schedulableNodes) > numEndpoints {
			framework.Logf("Expected %d endpoints, got %d", len(schedulableNodes), numEndpoints)
			return false, nil
		}

		finalSlices = slices.Items
		return true, nil
	})
	framework.ExpectNoError(err, "timed out waiting for EndpointSlices to be ready")

	ginkgo.By("having correct hints set for each endpoint")
	for _, slice := range finalSlices {
		for _, ep := range slice.Endpoints {
			if ep.Zone == nil {
				framework.Failf("Expected endpoint in %s to have zone: %v", slice.Name, ep)
			}
			if topologyHints {
				if ep.Hints == nil || len(ep.Hints.ForZones) == 0 {
					framework.Failf("Expected endpoint in %s to have hints: %v", slice.Name, ep)
				}
				if len(ep.Hints.ForZones) > 1 {
					framework.Failf("Expected endpoint in %s to have exactly 1 zone hint, got %d: %v", slice.Name, len(ep.Hints.ForZones), ep)
				}
				if *ep.Zone != ep.Hints.ForZones[0].Name {
					framework.Failf("Expected endpoint in %s to have same zone hint, got %s: %v", slice.Name, *ep.Zone, ep)
				}
			} else {
				if ep.Hints != nil && len(ep.Hints.ForZones) > 0 {
					framework.Failf("Unexpected hints on endpoint in %s: %v", slice.Name, ep)
				}
			}

		}
	}

	nodesByZone := make(map[string]string)
	zonesWithNodes := make(map[string][]*v1.Node)
	for _, node := range schedulableNodes {
		if zone, ok := node.Labels[v1.LabelTopologyZone]; ok {
			nodesByZone[node.Name] = zone
			zonesWithNodes[zone] = append(zonesWithNodes[zone], node)
		}
	}

	podList, err := c.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err)
	for _, pod := range podList.Items {
		if zone, ok := nodesByZone[pod.Spec.NodeName]; ok {
			tr.zonesByPod[pod.Name] = zone
			tr.nodesByPod[pod.Name] = pod.Spec.NodeName
		}
	}

	// we want the destination node to have other nodes available in the zone so that load balancing can be expected
	// find the largest zone and use it for the endpoint
	largestSeen := 0
	for zone, nodes := range zonesWithNodes {
		if len(nodes) > largestSeen {
			largestSeen = len(nodes)
			tr.endpointZone = zone
		}
	}
	framework.ExpectNotEqual("", tr.endpointZone, "unable to find a zone with more than one node")
	if largestSeen == 1 {
		e2eskipper.Skipf("unable to find a zone with more than one node")
	}

	// always send to a node in the largest zone so balancing will result in more than 1 host name being seen
	tr.endpointNode = zonesWithNodes[tr.endpointZone][0]

	// if hints are disabled, using this other node in same zone should load balance outside the endpointZone
	peerNode := zonesWithNodes[tr.endpointZone][1]

	// setup the endpoints which are used as target of curl in configurations below
	// first determine the endpoint node IP (may be ip6)
	family := v1.IPv4Protocol
	if framework.TestContext.ClusterIsIPv6() {
		family = v1.IPv6Protocol
	}

	ips := e2enode.GetAddressesByTypeAndFamily(tr.endpointNode, v1.NodeInternalIP, family)
	framework.ExpectNotEqual(len(ips), 0)
	nodeIp := ips[0]
	nodePort := svc.Spec.Ports[0].NodePort

	// endpointNodeIP is used a destination for 'external' traffic requests
	endpointNodeIP := net.JoinHostPort(nodeIp, strconv.Itoa(int(nodePort)))
	// endpointClusterIP is used as a destination for 'internal' traffic requests
	endpointClusterIP := svc.Spec.ClusterIP

	var externalNode *v1.Node // node in a different zone from endpoint
	for zone, nodes := range zonesWithNodes {
		if zone != tr.endpointZone {
			externalNode = nodes[0]
			break
		}
	}

	var internalClientNodeName string
	var externalClientNodeName string

	// The test checks internal traffic and external traffic
	internalClientNodeName = tr.endpointNode.Name
	if topologyHints {
		externalClientNodeName = externalNode.Name
	} else {
		externalClientNodeName = peerNode.Name
	}

	// create client pods to make requests to the NodePort service backed by daemonset pod
	podClient := e2epod.NewPodClient(f)
	// internalClientPod is used to validate internal traffic policy
	tr.internalClientPod = createClientPod(ctx, podClient, endpointClusterIP, f.Namespace.Name, internalClientNodeName, "internal")
	// externalClientPod is used to validate external traffic policy
	tr.externalClientPod = createClientPod(ctx, podClient, endpointNodeIP, f.Namespace.Name, externalClientNodeName, "external")

	return &tr
}

// checkEndpointHost waits for several curls to an endpoint and calls the checkResponses function to validate the expected pods which handled the requests.
// If expectedZone is specified, each request pod will be verified to reside in this zone. Otherwise the zone of the pod is not checked.
func (tr *trafficPolicyTestResources) checkEndpointHost(ctx context.Context, c clientset.Interface, f *framework.Framework, expectedZone string, podName string, checkResponses seenPodsCheckFunc) {
	var logs string
	seenPods := make(map[string]string)
	// Wait for at least 10 responses as a more meaningful check in cases where load balancing is expected / more than one pod is expected to respond
	if pollErr := wait.PollWithContext(ctx, 5*time.Second, e2eservice.KubeProxyLagTimeout, func(ctx context.Context) (bool, error) {
		var err error
		logs, err = e2epod.GetPodLogs(ctx, c, f.Namespace.Name, podName, podName)
		framework.ExpectNoError(err)
		framework.Logf("Pod client logs: %s", logs)

		logLines := strings.Split(logs, "\n")
		if len(logLines) < 6 {
			framework.Logf("only %d log lines, waiting for at least 6", len(logLines))
			return false, nil
		}

		responses := 0

		for i := len(logLines) - 1; i > 0; i-- {
			if logLines[i] == "" || strings.HasPrefix(logLines[i], "Date:") {
				continue
			}
			destZone, ok := tr.zonesByPod[logLines[i]]
			if !ok {
				framework.Logf("could not determine dest zone from log line: %s", logLines[i])
				return false, nil
			}
			seenPods[logLines[i]] = destZone
			// if a expectedZone is provided, verify traffic stays within
			if expectedZone != "" && expectedZone != destZone {
				framework.Logf("expected request from %s to stay in %s zone, delivered to %s zone", podName, expectedZone, destZone)
				return false, nil
			}
			responses++
			if responses >= 10 {
				if checkResponses(tr, seenPods) {
					return true, nil
				} else {
					return false, fmt.Errorf("invalid responses")
				}
			}
		}
		return false, nil
	}); pollErr != nil {
		framework.Failf("expected 10 consecutive requests from %s to stay in zone %s within %v, stdout: %v", podName, expectedZone, e2eservice.KubeProxyLagTimeout, logs)
	}
}

func (tr *trafficPolicyTestResources) checkEndpointExternal(ctx context.Context, c clientset.Interface, f *framework.Framework, expectedZone string, checkResponses seenPodsCheckFunc) {
	tr.checkEndpointHost(ctx, c, f, expectedZone, tr.externalClientPod.Name, checkResponses)
}

func (tr *trafficPolicyTestResources) checkEndpointInternal(ctx context.Context, c clientset.Interface, f *framework.Framework, expectedZone string, checkResponses seenPodsCheckFunc) {
	tr.checkEndpointHost(ctx, c, f, expectedZone, tr.internalClientPod.Name, checkResponses)
}

// This is used with 'Cluster' traffic policies within a zone (topology hints: auto)
func sawMoreThanOnePod(_ *trafficPolicyTestResources, seenPods map[string]string) bool {
	return len(seenPods) > 1
}

// This is used when 'Local' traffic policy is specified to verify the expected node
func sawOnePodOnEndpointNode(tr *trafficPolicyTestResources, seenPods map[string]string) bool {
	if len(seenPods) != 1 {
		return false
	}
	for pod := range seenPods {
		node, ok := tr.nodesByPod[pod]
		return ok && node == tr.endpointNode.Name
	}
	return false
}

// This is used with 'Cluster' traffic policies with topology hints disabled
func sawMultipleZones(_ *trafficPolicyTestResources, seenPods map[string]string) bool {
	zones := make(map[string]bool)
	for _, zone := range seenPods {
		zones[zone] = true
	}
	return len(seenPods) > 1 && len(zones) > 1
}

func createClientPod(ctx context.Context, podClient *e2epod.PodClient, endpoint string, namespaceName string, nodeName string, podSuffix string) *v1.Pod {
	ginkgo.By("creating a client pod for probing the service from " + nodeName)
	// put a pod in the single node zone to use for curling
	podName := "curl-from-" + nodeName + podSuffix

	/// turn this into a function which takes endpoint, and expected reply details
	clientPod := e2epod.NewAgnhostPod(namespaceName, podName, nil, nil, nil, "serve-hostname")
	nodeSelection := e2epod.NodeSelection{Name: nodeName}
	e2epod.SetNodeSelection(&clientPod.Spec, nodeSelection)

	ginkgo.By("Checking access to endpoint " + endpoint)

	cmd := fmt.Sprintf(`date; for i in $(seq 1 3000); do sleep 0.5; echo "Date: $(date) Try: ${i}"; curl -q -s --connect-timeout 2 http://%s/hostname ; echo; done`, endpoint)
	clientPod.Spec.Containers[0].Command = []string{"/bin/sh", "-c", cmd}
	clientPod.Spec.Containers[0].Name = clientPod.Name
	return podClient.CreateSync(ctx, clientPod)
}

// Create a netexec daemonset with NodePort service use client pods in each zone to verify routing
// need at least 3 nodes, 2 in one zone and 1 in another
// 2 in a zone to verify Local vs Cluster and sep zones to verify topology hints
var _ = common.SIGDescribe("[Feature:Topology Hints][Feature:ServiceInternalTrafficPolicy]", func() {
	f := framework.NewDefaultFramework("traffic-policy")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	// filled in BeforeEach
	var c clientset.Interface

	ginkgo.BeforeEach(func(ctx context.Context) {
		c = f.ClientSet
		e2eskipper.SkipUnlessMultizone(ctx, c)
		e2eskipper.SkipUnlessNodeCountIsAtLeast(3)
	})

	ginkgo.Context("With hints enabled", func() {
		// With hints enabled, the endpointZone is always expected and checked
		// When local policies are configured, only one pod is expected to respond

		ginkgo.It("honors hints when ITP and ETP use Cluster policies", func(ctx context.Context) {
			tr := setupAgnhostDaemonset(ctx, c, f, true, v1.ServiceExternalTrafficPolicyTypeCluster, v1.ServiceInternalTrafficPolicyCluster)

			ginkgo.By("using the endpoint zone for external requests")
			tr.checkEndpointExternal(ctx, c, f, tr.endpointZone, sawMoreThanOnePod)

			ginkgo.By("using the endpoint zone for internal requests")
			tr.checkEndpointInternal(ctx, c, f, tr.endpointZone, sawMoreThanOnePod)
		})

		ginkgo.It("honors hints only for internal traffic when ETP is local", func(ctx context.Context) {
			tr := setupAgnhostDaemonset(ctx, c, f, true, v1.ServiceExternalTrafficPolicyTypeLocal, v1.ServiceInternalTrafficPolicyCluster)

			ginkgo.By("using the endpoint node for external requests")
			tr.checkEndpointExternal(ctx, c, f, tr.endpointZone, sawOnePodOnEndpointNode)

			ginkgo.By("using the endpoint zone for internal requests")
			tr.checkEndpointInternal(ctx, c, f, tr.endpointZone, sawMoreThanOnePod)
		})

		ginkgo.It("honors hints only for external requests when ITP is local", func(ctx context.Context) {
			tr := setupAgnhostDaemonset(ctx, c, f, true, v1.ServiceExternalTrafficPolicyTypeCluster, v1.ServiceInternalTrafficPolicyLocal)

			ginkgo.By("using the endpoint zone for external requests")
			tr.checkEndpointExternal(ctx, c, f, tr.endpointZone, sawMoreThanOnePod)

			ginkgo.By("using the endpoint node for internal requests")
			tr.checkEndpointInternal(ctx, c, f, tr.endpointZone, sawOnePodOnEndpointNode)
		})

		ginkgo.It("ignores hints when ITP and ETP are both local", func(ctx context.Context) {
			tr := setupAgnhostDaemonset(ctx, c, f, true, v1.ServiceExternalTrafficPolicyTypeLocal, v1.ServiceInternalTrafficPolicyLocal)

			ginkgo.By("using the endpoint node for external requests")
			tr.checkEndpointExternal(ctx, c, f, tr.endpointZone, sawOnePodOnEndpointNode)

			ginkgo.By("using the endpoint node for internal requests")
			tr.checkEndpointInternal(ctx, c, f, tr.endpointZone, sawOnePodOnEndpointNode)
		})

	})
	ginkgo.Context("With hints disabled", func() {
		// When cluster traffic policies are used, multiple zones are expected to respond
		// When local policies are used, the endpointZone and a specific node are expected to respond

		ginkgo.It("balances requests across the cluster when ITP and ETP are Cluster", func(ctx context.Context) {
			tr := setupAgnhostDaemonset(ctx, c, f, false, v1.ServiceExternalTrafficPolicyTypeCluster, v1.ServiceInternalTrafficPolicyCluster)

			ginkgo.By("using multiple zones for external requests")
			tr.checkEndpointExternal(ctx, c, f, "", sawMultipleZones)

			ginkgo.By("using multiple zones for internal requests")
			tr.checkEndpointInternal(ctx, c, f, "", sawMultipleZones)
		})

		ginkgo.It("balances internal requests only when ETP is Local and ITP is Cluster", func(ctx context.Context) {
			tr := setupAgnhostDaemonset(ctx, c, f, false, v1.ServiceExternalTrafficPolicyTypeLocal, v1.ServiceInternalTrafficPolicyCluster)

			ginkgo.By("using the endpoint node for external requests")
			tr.checkEndpointExternal(ctx, c, f, tr.endpointZone, sawOnePodOnEndpointNode)

			ginkgo.By("using multiple zones for internal requests")
			tr.checkEndpointInternal(ctx, c, f, "", sawMultipleZones)
		})

		ginkgo.It("balances external requests only when ETP is Cluster and ITP is Local", func(ctx context.Context) {
			tr := setupAgnhostDaemonset(ctx, c, f, false, v1.ServiceExternalTrafficPolicyTypeCluster, v1.ServiceInternalTrafficPolicyLocal)

			ginkgo.By("using multiple zones for external requests")
			tr.checkEndpointExternal(ctx, c, f, "", sawMultipleZones)

			ginkgo.By("using the endpoint node for internal requests")
			tr.checkEndpointInternal(ctx, c, f, tr.endpointZone, sawOnePodOnEndpointNode)
		})

		ginkgo.It("uses the endpoint node for all requests when ITP and ETP are Local", func(ctx context.Context) {
			tr := setupAgnhostDaemonset(ctx, c, f, false, v1.ServiceExternalTrafficPolicyTypeLocal, v1.ServiceInternalTrafficPolicyLocal)

			ginkgo.By("using the endpoint node for external requests")
			tr.checkEndpointExternal(ctx, c, f, tr.endpointZone, sawOnePodOnEndpointNode)

			ginkgo.By("using the endpoint node for internal requests")
			tr.checkEndpointInternal(ctx, c, f, tr.endpointZone, sawOnePodOnEndpointNode)
		})
	})
})
