/*
Copyright 2024 The Kubernetes Authors.

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
	"slices"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eendpointslice "k8s.io/kubernetes/test/e2e/framework/endpointslice"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	"k8s.io/kubernetes/test/utils/format"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = common.SIGDescribe("Traffic Distribution", func() {
	f := framework.NewDefaultFramework("traffic-distribution")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var c clientset.Interface

	ginkgo.BeforeEach(func(ctx context.Context) {
		c = f.ClientSet
		e2eskipper.SkipUnlessAtLeastNZones(ctx, c, 3)
	})

	////////////////////////////////////////////////////////////////////////////
	// Helper functions
	////////////////////////////////////////////////////////////////////////////

	// endpointSlicesForService returns a helper function to be used with
	// gomega.Eventually(...). It fetches the EndpointSlices for the given
	// serviceName.
	endpointSlicesForService := func(serviceName string) any {
		return func(ctx context.Context) ([]discoveryv1.EndpointSlice, error) {
			slices, err := c.DiscoveryV1().EndpointSlices(f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: fmt.Sprintf("%s=%s", discoveryv1.LabelServiceName, serviceName)})
			if err != nil {
				return nil, err
			}
			return slices.Items, nil
		}
	}

	// gomegaCustomError constructs a function that can be returned from a gomega
	// matcher to report an error.
	gomegaCustomError := func(format string, a ...any) func() string {
		return func() string {
			return fmt.Sprintf(format, a...)
		}
	}

	// requestsFromClient returns a helper function to be used with
	// gomega.Eventually(...). It fetches the logs from the clientPod and returns
	// them in reverse-chronological order.
	requestsFromClient := func(clientPod *v1.Pod) any {
		return func(ctx context.Context) (reverseChronologicalLogLines []string, err error) {
			logs, err := e2epod.GetPodLogs(ctx, c, f.Namespace.Name, clientPod.Name, clientPod.Spec.Containers[0].Name)
			if err != nil {
				return nil, err
			}
			logLines := strings.Split(logs, "\n")
			slices.Reverse(logLines)
			return logLines, nil
		}
	}

	// getNodesForMultiNode returns a set of nodes for a test case with 3 zones with 2
	// nodes each. If there are not suitable nodes/zones, the test is skipped.
	getNodesForMultiNode := func(ctx context.Context) ([]*v1.Node, []*v1.Node, []*v1.Node) {
		nodeList, err := e2enode.GetReadySchedulableNodes(ctx, c)
		framework.ExpectNoError(err)
		nodesForZone := make(map[string][]*v1.Node)
		for _, node := range nodeList.Items {
			zone := node.Labels[v1.LabelTopologyZone]
			nodesForZone[zone] = append(nodesForZone[zone], &node)
		}
		if len(nodesForZone) < 3 {
			e2eskipper.Skipf("need at least 3 zones, with at least 2 schedulable nodes each")
		}

		var multiNodeZones [][]*v1.Node
		for _, nodes := range nodesForZone {
			if len(nodes) > 1 {
				multiNodeZones = append(multiNodeZones, nodes)
			}
			if len(multiNodeZones) == 3 {
				break
			}
		}
		if len(multiNodeZones) < 3 {
			e2eskipper.Skipf("need at least 3 zones, with at least 2 schedulable nodes each")
		}

		return multiNodeZones[0], multiNodeZones[1], multiNodeZones[2]
	}

	// Data structures for tracking server and client pods
	type serverPod struct {
		node *v1.Node
		pod  *v1.Pod
	}

	type clientPod struct {
		node      *v1.Node
		endpoints []*serverPod
		pod       *v1.Pod
	}

	// allocateClientsAndServers figures out where to put clients and servers for
	// a simple "same-zone" traffic distribution test.
	allocateClientsAndServers := func(ctx context.Context) ([]*clientPod, []*serverPod) {
		ginkgo.By("finding 3 zones with schedulable nodes")
		nodeList, err := e2enode.GetReadySchedulableNodes(ctx, c)
		framework.ExpectNoError(err)
		nodeForZone := make(map[string]*v1.Node)
		for _, node := range nodeList.Items {
			zone := node.Labels[v1.LabelTopologyZone]
			if nodeForZone[zone] != nil {
				continue
			}
			nodeForZone[zone] = &node
			if len(nodeForZone) == 3 {
				break
			}
		}
		if len(nodeForZone) < 3 {
			e2eskipper.Skipf("got %d zones with schedulable nodes, need at least 3", len(nodeForZone))
		}

		var clientPods []*clientPod
		var serverPods []*serverPod

		// We want clients in all three zones
		for _, node := range nodeForZone {
			clientPods = append(clientPods, &clientPod{node: node})
		}

		// and endpoints in the first two zones
		serverPods = []*serverPod{
			{node: clientPods[0].node},
			{node: clientPods[1].node},
		}

		// The clients with an endpoint in the same zone should only connect to
		// that endpoint. The client with no endpoint in its zone should connect
		// to both endpoints.
		clientPods[0].endpoints = []*serverPod{serverPods[0]}
		clientPods[1].endpoints = []*serverPod{serverPods[1]}
		clientPods[2].endpoints = serverPods

		return clientPods, serverPods
	}

	// allocateMultiNodeClientsAndServers figures out where to put clients and servers
	// for a "same-zone" traffic distribution test with multiple nodes in each zone.
	allocateMultiNodeClientsAndServers := func(ctx context.Context) ([]*clientPod, []*serverPod) {
		ginkgo.By("finding a set of zones and nodes for the test")
		zone1Nodes, zone2Nodes, zone3Nodes := getNodesForMultiNode(ctx)

		var clientPods []*clientPod
		var serverPods []*serverPod

		// First zone: a client and an endpoint on each node, and both clients
		// should talk to both endpoints.
		endpointsForZone := []*serverPod{
			{node: zone1Nodes[0]},
			{node: zone1Nodes[1]},
		}

		clientPods = append(clientPods,
			&clientPod{
				node:      zone1Nodes[0],
				endpoints: endpointsForZone,
			},
			&clientPod{
				node:      zone1Nodes[1],
				endpoints: endpointsForZone,
			},
		)
		serverPods = append(serverPods, endpointsForZone...)

		// Second zone: a client on one node and a server on the other.
		endpointsForZone = []*serverPod{
			{node: zone2Nodes[1]},
		}

		clientPods = append(clientPods,
			&clientPod{
				node:      zone2Nodes[0],
				endpoints: endpointsForZone,
			},
		)
		serverPods = append(serverPods, endpointsForZone...)

		// Third zone: just a client, which should connect to the servers in the
		// other two zones.
		clientPods = append(clientPods,
			&clientPod{
				node:      zone3Nodes[0],
				endpoints: serverPods,
			},
		)

		return clientPods, serverPods
	}

	// createService creates the service for a traffic distribution test
	createService := func(ctx context.Context, trafficDist string) *v1.Service {
		serviceName := "traffic-dist-test-service"
		ginkgo.By(fmt.Sprintf("creating a service %q with trafficDistribution %q", serviceName, trafficDist))
		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: serviceName,
			},
			Spec: v1.ServiceSpec{
				Selector: map[string]string{
					"app": f.UniqueName,
				},
				TrafficDistribution: &trafficDist,
				Ports: []v1.ServicePort{{
					Port:       80,
					TargetPort: intstr.FromInt32(9376),
					Protocol:   v1.ProtocolTCP,
				}},
			},
		}
		svc, err := c.CoreV1().Services(f.Namespace.Name).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "error creating Service")
		return svc
	}

	// createPods creates endpoint pods for svc as described by serverPods, waits for
	// the EndpointSlices to be updated, and creates clientPods as described by
	// clientPods.
	createPods := func(ctx context.Context, svc *v1.Service, clientPods []*clientPod, serverPods []*serverPod) {
		var podsToCreate []*v1.Pod
		for i, sp := range serverPods {
			node := sp.node.Name
			zone := sp.node.Labels[v1.LabelTopologyZone]
			pod := e2epod.NewAgnhostPod(f.Namespace.Name, fmt.Sprintf("server-%d-%s", i, node), nil, nil, nil, "serve-hostname")
			ginkgo.By(fmt.Sprintf("creating a server pod %q on node %q in zone %q", pod.Name, node, zone))
			nodeSelection := e2epod.NodeSelection{Name: node}
			e2epod.SetNodeSelection(&pod.Spec, nodeSelection)
			pod.Labels = svc.Spec.Selector

			sp.pod = pod
			podsToCreate = append(podsToCreate, pod)
		}
		e2epod.NewPodClient(f).CreateBatch(ctx, podsToCreate)

		ginkgo.By("waiting for EndpointSlices to be created")
		err := e2eendpointslice.WaitForEndpointCount(ctx, c, svc.Namespace, svc.Name, len(serverPods))
		framework.ExpectNoError(err)
		slices := endpointSlicesForService(svc.Name)
		framework.Logf("got slices:\n%v", format.Object(slices, 1))

		podsToCreate = nil
		for i, cp := range clientPods {
			node := cp.node.Name
			zone := cp.node.Labels[v1.LabelTopologyZone]
			pod := e2epod.NewAgnhostPod(f.Namespace.Name, fmt.Sprintf("client-%d-%s", i, node), nil, nil, nil)
			ginkgo.By(fmt.Sprintf("creating a client pod %q on node %q in zone %q", pod.Name, node, zone))
			nodeSelection := e2epod.NodeSelection{Name: node}
			e2epod.SetNodeSelection(&pod.Spec, nodeSelection)
			cmd := fmt.Sprintf(`date; for i in $(seq 1 3000); do sleep 1; echo "Date: $(date) Try: ${i}"; curl -q -s --connect-timeout 2 http://%s:80/ ; echo; done`, svc.Name)
			pod.Spec.Containers[0].Command = []string{"/bin/sh", "-c", cmd}
			pod.Spec.Containers[0].Name = pod.Name

			cp.pod = pod
			podsToCreate = append(podsToCreate, pod)
		}
		e2epod.NewPodClient(f).CreateBatch(ctx, podsToCreate)
	}

	// checkTrafficDistribution checks that traffic from clientPods is distributed in
	// the expected way.
	checkTrafficDistribution := func(ctx context.Context, clientPods []*clientPod) {
		for _, cp := range clientPods {
			wantedEndpoints := sets.New[string]()
			for _, sp := range cp.endpoints {
				wantedEndpoints.Insert(sp.pod.Name)
			}
			unreachedEndpoints := wantedEndpoints.Clone()

			ginkgo.By(fmt.Sprintf("ensuring that requests from %s on %s go to the endpoint(s) %v", cp.pod.Name, cp.node.Name, wantedEndpoints.UnsortedList()))

			requestsSucceed := framework.MakeMatcher(func(reverseChronologicalLogLines []string) (func() string, error) {
				logLines := reverseChronologicalLogLines
				if len(logLines) < 20 {
					return gomegaCustomError("got %d log lines, waiting for at least 20\nreverseChronologicalLogLines=\n%v", len(logLines), strings.Join(reverseChronologicalLogLines, "\n")), nil
				}
				consecutiveSuccessfulRequests := 0

				for _, logLine := range logLines {
					if logLine == "" || strings.HasPrefix(logLine, "Date:") {
						continue
					}
					destEndpoint := logLine
					if !wantedEndpoints.Has(destEndpoint) {
						return gomegaCustomError("request from %s should not have reached %s\nreverseChronologicalLogLines=\n%v", cp.pod.Name, destEndpoint, strings.Join(reverseChronologicalLogLines, "\n")), nil
					}
					consecutiveSuccessfulRequests++
					unreachedEndpoints.Delete(destEndpoint)
					if consecutiveSuccessfulRequests >= 10 && len(unreachedEndpoints) == 0 {
						return nil, nil // Pass condition.
					}
				}
				// Ideally, the matcher would never reach this condition
				return gomegaCustomError("requests didn't meet the required criteria to reach all endpoints %v\nreverseChronologicalLogLines=\n%v", wantedEndpoints.UnsortedList(), strings.Join(reverseChronologicalLogLines, "\n")), nil
			})

			gomega.Eventually(ctx, requestsFromClient(cp.pod)).WithPolling(5 * time.Second).WithTimeout(e2eservice.KubeProxyLagTimeout).Should(requestsSucceed)
		}
	}

	////////////////////////////////////////////////////////////////////////////
	// Main test specifications.
	////////////////////////////////////////////////////////////////////////////

	framework.It("should route traffic to an endpoint in the same zone when using PreferClose", func(ctx context.Context) {
		clientPods, serverPods := allocateClientsAndServers(ctx)
		svc := createService(ctx, v1.ServiceTrafficDistributionPreferClose)
		createPods(ctx, svc, clientPods, serverPods)
		checkTrafficDistribution(ctx, clientPods)
	})

	framework.It("should route traffic correctly between pods on multiple nodes when using PreferClose", func(ctx context.Context) {
		clientPods, serverPods := allocateMultiNodeClientsAndServers(ctx)
		svc := createService(ctx, v1.ServiceTrafficDistributionPreferClose)
		createPods(ctx, svc, clientPods, serverPods)
		checkTrafficDistribution(ctx, clientPods)
	})

	framework.It("should route traffic to an endpoint in the same zone when using PreferSameZone", func(ctx context.Context) {
		clientPods, serverPods := allocateClientsAndServers(ctx)
		svc := createService(ctx, v1.ServiceTrafficDistributionPreferSameZone)
		createPods(ctx, svc, clientPods, serverPods)
		checkTrafficDistribution(ctx, clientPods)
	})

	framework.It("should route traffic correctly between pods on multiple nodes when using PreferSameZone", func(ctx context.Context) {
		clientPods, serverPods := allocateMultiNodeClientsAndServers(ctx)
		svc := createService(ctx, v1.ServiceTrafficDistributionPreferSameZone)
		createPods(ctx, svc, clientPods, serverPods)
		checkTrafficDistribution(ctx, clientPods)
	})

	framework.It("should route traffic to an endpoint on the same node or fall back to same zone when using PreferSameNode", func(ctx context.Context) {
		ginkgo.By("finding a set of nodes for the test")
		zone1Nodes, zone2Nodes, zone3Nodes := getNodesForMultiNode(ctx)

		var clientPods []*clientPod
		var serverPods []*serverPod

		// The first zone: a client and a server on each node. Each client only
		// talks to the server on the same node.
		endpointsForZone := []*serverPod{
			{node: zone1Nodes[0]},
			{node: zone1Nodes[1]},
		}
		clientPods = append(clientPods,
			&clientPod{
				node:      zone1Nodes[0],
				endpoints: []*serverPod{endpointsForZone[0]},
			},
			&clientPod{
				node:      zone1Nodes[1],
				endpoints: []*serverPod{endpointsForZone[1]},
			},
		)
		serverPods = append(serverPods, endpointsForZone...)

		// The second zone: a client on one node and a server on the other. The
		// client should fall back to connecting (only) to its same-zone endpoint.
		endpointsForZone = []*serverPod{
			{node: zone2Nodes[1]},
		}
		clientPods = append(clientPods,
			&clientPod{
				node:      zone2Nodes[0],
				endpoints: endpointsForZone,
			},
		)
		serverPods = append(serverPods, endpointsForZone...)

		// The third zone: just a client. Since it has neither a same-node nor a
		// same-zone endpoint, it should connect to all endpoints.
		clientPods = append(clientPods,
			&clientPod{
				node:      zone3Nodes[0],
				endpoints: serverPods,
			},
		)

		svc := createService(ctx, v1.ServiceTrafficDistributionPreferSameNode)
		createPods(ctx, svc, clientPods, serverPods)
		checkTrafficDistribution(ctx, clientPods)
	})

	framework.It("should route traffic to an endpoint on the same node when using PreferSameNode and fall back when the endpoint becomes unavailable", func(ctx context.Context) {
		ginkgo.By("finding a set of nodes for the test")
		nodeList, err := e2enode.GetReadySchedulableNodes(ctx, c)
		framework.ExpectNoError(err)
		if len(nodeList.Items) < 2 {
			e2eskipper.Skipf("have %d schedulable nodes, need at least 2", len(nodeList.Items))
		}
		nodes := nodeList.Items[:2]

		// One client and one server on each node
		serverPods := []*serverPod{
			{node: &nodes[0]},
			{node: &nodes[1]},
		}
		clientPods := []*clientPod{
			{
				node:      &nodes[0],
				endpoints: []*serverPod{serverPods[0]},
			},
			{
				node:      &nodes[1],
				endpoints: []*serverPod{serverPods[1]},
			},
		}

		svc := createService(ctx, v1.ServiceTrafficDistributionPreferSameNode)
		createPods(ctx, svc, clientPods, serverPods)

		ginkgo.By("ensuring that each client talks to its same-node endpoint when both endpoints exist")
		checkTrafficDistribution(ctx, clientPods)

		ginkgo.By("killing the server pod on the first node and waiting for the EndpointSlices to be updated")
		err = c.CoreV1().Pods(f.Namespace.Name).Delete(ctx, serverPods[0].pod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		err = e2eendpointslice.WaitForEndpointCount(ctx, c, svc.Namespace, svc.Name, 1)
		framework.ExpectNoError(err)

		ginkgo.By("ensuring that both clients talk to the remaining endpoint when only one endpoint exists")
		serverPods[0].pod = nil
		clientPods[0].endpoints = []*serverPod{serverPods[1]}
		checkTrafficDistribution(ctx, clientPods)

		ginkgo.By("recreating the missing server pod and waiting for the EndpointSlices to be updated")
		// We can't use createPods() here because if we only tell it about
		// serverPods[0] and not serverPods[1] it will expect there to be only one
		// endpoint.
		pod := e2epod.NewAgnhostPod(f.Namespace.Name, "server-0-new", nil, nil, nil, "serve-hostname")
		nodeSelection := e2epod.NodeSelection{Name: serverPods[0].node.Name}
		e2epod.SetNodeSelection(&pod.Spec, nodeSelection)
		pod.Labels = svc.Spec.Selector
		serverPods[0].pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
		err = e2eendpointslice.WaitForEndpointCount(ctx, c, svc.Namespace, svc.Name, 2)
		framework.ExpectNoError(err)

		ginkgo.By("ensuring that each client talks only to its same-node endpoint again")
		clientPods[0].endpoints = []*serverPod{serverPods[0]}
		checkTrafficDistribution(ctx, clientPods)
	})
})
