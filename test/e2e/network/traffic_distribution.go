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

	////////////////////////////////////////////////////////////////////////////
	// Main test specifications.
	////////////////////////////////////////////////////////////////////////////

	ginkgo.It("should route traffic to an endpoint in the same zone when using PreferClose", func(ctx context.Context) {

		ginkgo.By("finding 3 zones with schedulable nodes")
		allZonesSet, err := e2enode.GetSchedulableClusterZones(ctx, c)
		framework.ExpectNoError(err)
		if len(allZonesSet) < 3 {
			framework.Failf("got %d zones with schedulable nodes, want atleast 3 zones with schedulable nodes", len(allZonesSet))
		}
		zones := allZonesSet.UnsortedList()[:3]

		ginkgo.By(fmt.Sprintf("finding a node in each of the chosen 3 zones %v", zones))
		nodeList, err := e2enode.GetReadySchedulableNodes(ctx, c)
		framework.ExpectNoError(err)
		nodeForZone := make(map[string]*v1.Node)
		for _, zone := range zones {
			found := false
			for _, node := range nodeList.Items {
				if zone == node.Labels[v1.LabelTopologyZone] {
					found = true
					nodeForZone[zone] = &node
				}
			}
			if !found {
				framework.Failf("could not find a node in zone %q; nodes=\n%v", zone, format.Object(nodeList, 1 /* indent one level */))
			}
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

		var podsToCreate []*v1.Pod
		servingPodLabels := map[string]string{"app": f.UniqueName}
		for i, sp := range serverPods {
			node := sp.node.Name
			zone := sp.node.Labels[v1.LabelTopologyZone]
			pod := e2epod.NewAgnhostPod(f.Namespace.Name, fmt.Sprintf("server-%d-%s", i, node), nil, nil, nil, "serve-hostname")
			ginkgo.By(fmt.Sprintf("creating a server pod %q on node %q in zone %q", pod.Name, node, zone))
			nodeSelection := e2epod.NodeSelection{Name: node}
			e2epod.SetNodeSelection(&pod.Spec, nodeSelection)
			pod.Labels = servingPodLabels

			sp.pod = pod
			podsToCreate = append(podsToCreate, pod)
		}
		e2epod.NewPodClient(f).CreateBatch(ctx, podsToCreate)

		trafficDist := v1.ServiceTrafficDistributionPreferClose
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

		ginkgo.By("waiting for EndpointSlices to be created")
		err = framework.WaitForServiceEndpointsNum(ctx, c, svc.Namespace, svc.Name, len(serverPods), 1*time.Second, e2eservice.ServiceEndpointsTimeout)
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
	})
})
