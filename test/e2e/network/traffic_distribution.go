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
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
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

	// endpointSlicesHaveSameZoneHints returns a matcher function to be used with
	// gomega.Eventually().Should(...). It checks that the passed EndpointSlices
	// have zone-hints which match the endpoint's zone.
	endpointSlicesHaveSameZoneHints := framework.MakeMatcher(func(slices []discoveryv1.EndpointSlice) (func() string, error) {
		if len(slices) == 0 {
			return nil, fmt.Errorf("no endpointslices found")
		}
		for _, slice := range slices {
			for _, endpoint := range slice.Endpoints {
				var ip string
				if len(endpoint.Addresses) > 0 {
					ip = endpoint.Addresses[0]
				}
				var zone string
				if endpoint.Zone != nil {
					zone = *endpoint.Zone
				}
				if endpoint.Hints == nil || len(endpoint.Hints.ForZones) != 1 || endpoint.Hints.ForZones[0].Name != zone {
					return gomegaCustomError("endpoint with ip %v does not have the correct hint, want hint for zone %q\nEndpointSlices=\n%v", ip, zone, format.Object(slices, 1 /* indent one level */)), nil
				}
			}
		}
		return nil, nil
	})

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

	// Data structures for representing client and endpoint pods
	type endpoint struct {
		node *v1.Node
		pod  *v1.Pod
	}

	type client struct {
		node      *v1.Node
		endpoints []*endpoint
		pod       *v1.Pod
	}

	////////////////////////////////////////////////////////////////////////////
	// Main test specifications.
	////////////////////////////////////////////////////////////////////////////

	// doTrafficDistributionTest runs a test of a service with the given trafficDist,
	// clients, and endpoints, ensuring that connections go to the expected endpoints.
	doTrafficDistributionTest := func(ctx context.Context, trafficDist string, clients []*client, endpoints []*endpoint) {
		var servingPods []*v1.Pod
		servingPodLabels := map[string]string{"app": f.UniqueName}
		for i, ep := range endpoints {
			node := ep.node.Name
			zone := ep.node.Labels[v1.LabelTopologyZone]
			pod := e2epod.NewAgnhostPod(f.Namespace.Name, fmt.Sprintf("endpoint-%d-%s", i, node), nil, nil, nil, "serve-hostname")
			ginkgo.By(fmt.Sprintf("creating a server pod %q on node %q in zone %q", pod.Name, node, zone))
			nodeSelection := e2epod.NodeSelection{Name: node}
			e2epod.SetNodeSelection(&pod.Spec, nodeSelection)
			pod.Labels = servingPodLabels

			ep.pod = pod
			servingPods = append(servingPods, pod)
		}
		e2epod.NewPodClient(f).CreateBatch(ctx, servingPods)

		svc := createServiceReportErr(ctx, c, f.Namespace.Name, &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "traffic-dist-test-service",
			},
			Spec: v1.ServiceSpec{
				Selector:            servingPodLabels,
				TrafficDistribution: &trafficDist,
				Ports: []v1.ServicePort{{
					Port:       80,
					TargetPort: intstr.FromInt32(9376),
					Protocol:   v1.ProtocolTCP,
				}},
			},
		})
		ginkgo.By(fmt.Sprintf("creating a service=%q with trafficDistribution=%v", svc.GetName(), *svc.Spec.TrafficDistribution))

		ginkgo.By("ensuring EndpointSlice for service have correct same-zone hints")
		gomega.Eventually(ctx, endpointSlicesForService(svc.GetName())).WithPolling(5 * time.Second).WithTimeout(e2eservice.ServiceEndpointsTimeout).Should(endpointSlicesHaveSameZoneHints)

		for i, cp := range clients {
			node := cp.node.Name
			zone := cp.node.Labels[v1.LabelTopologyZone]
			pod := e2epod.NewAgnhostPod(f.Namespace.Name, fmt.Sprintf("client-%d-%s", i, node), nil, nil, nil)
			ginkgo.By(fmt.Sprintf("creating a client pod %q on node %q in zone %q", pod.Name, node, zone))
			nodeSelection := e2epod.NodeSelection{Name: node}
			e2epod.SetNodeSelection(&pod.Spec, nodeSelection)
			cmd := fmt.Sprintf(`date; for i in $(seq 1 3000); do sleep 1; echo "Date: $(date) Try: ${i}"; curl -q -s --connect-timeout 2 http://%s:80/ ; echo; done`, svc.Name)
			pod.Spec.Containers[0].Command = []string{"/bin/sh", "-c", cmd}
			pod.Spec.Containers[0].Name = pod.Name

			cp.pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
		}

		for _, cp := range clients {
			wantedEndpoints := sets.New[string]()
			for _, ep := range cp.endpoints {
				wantedEndpoints.Insert(ep.pod.Name)
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

	doSameZoneTrafficDistributionTest := func(ctx context.Context, trafficDist string) {
		var clients []*client
		var endpoints []*endpoint

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
			e2eskipper.Skipf("have %d zones with schedulable nodes, need at least 3", len(clients))
		}

		// We want clients in all three zones
		for _, node := range nodeForZone {
			clients = append(clients, &client{node: node})
		}

		// and endpoints in the first two zones
		endpoints = []*endpoint{
			{node: clients[0].node},
			{node: clients[1].node},
		}

		// The clients with an endpoint in the same zone should only connect to
		// that endpoint. The client with no endpoint in its zone should connect
		// to both endpoints.
		clients[0].endpoints = []*endpoint{endpoints[0]}
		clients[1].endpoints = []*endpoint{endpoints[1]}
		clients[2].endpoints = endpoints

		doTrafficDistributionTest(ctx, trafficDist, clients, endpoints)
	}

	framework.It("should route traffic to an endpoint in the same zone when using PreferClose", func(ctx context.Context) {
		doSameZoneTrafficDistributionTest(ctx, v1.ServiceTrafficDistributionPreferClose)
	})

	framework.It("should route traffic to an endpoint in the same zone when using PreferSameZone", framework.WithFeatureGate(features.PreferSameTrafficDistribution), func(ctx context.Context) {
		doSameZoneTrafficDistributionTest(ctx, v1.ServiceTrafficDistributionPreferSameZone)
	})
})
