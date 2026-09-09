/*
Copyright 2015 The Kubernetes Authors.

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
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = common.SIGDescribe("Topology Hints", func() {
	f := framework.NewDefaultFramework("topology-hints")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// filled in BeforeEach
	var c clientset.Interface

	ginkgo.BeforeEach(func(ctx context.Context) {
		c = f.ClientSet
		e2eskipper.SkipUnlessMultizone(ctx, c)
	})

	ginkgo.It("should distribute endpoints evenly", func(ctx context.Context) {
		portNum := int32(9376)
		thLabels := map[string]string{labelKey: clientLabelValue}
		img := imageutils.GetE2EImage(imageutils.Agnhost)
		ports := []v1.ContainerPort{{ContainerPort: int32(portNum)}}
		dsConf := e2edaemonset.NewDaemonSet("topology-serve-hostname", img, thLabels, nil, nil, ports, "serve-hostname")
		ds, err := c.AppsV1().DaemonSets(f.Namespace.Name).Create(ctx, dsConf, metav1.CreateOptions{})
		framework.ExpectNoError(err, "error creating DaemonSet")

		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "topology-hints",
				Annotations: map[string]string{
					v1.AnnotationTopologyMode: "Auto",
				},
			},
			Spec: v1.ServiceSpec{
				Selector:                 thLabels,
				PublishNotReadyAddresses: true,
				Ports: []v1.ServicePort{{
					Name:       "example",
					Port:       80,
					TargetPort: intstr.FromInt32(portNum),
					Protocol:   v1.ProtocolTCP,
				}},
			},
		}
		svc, err = c.CoreV1().Services(f.Namespace.Name).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "error creating Service")

		err = wait.PollUntilContextTimeout(ctx, 5*time.Second, framework.PodStartTimeout, false, func(ctx context.Context) (bool, error) {
			return e2edaemonset.CheckRunningOnAllNodes(ctx, f, ds)
		})
		framework.ExpectNoError(err, "timed out waiting for DaemonSets to be ready")

		// All Nodes should have same allocatable CPUs. If not, then skip the test.
		schedulableNodes := map[string]*v1.Node{}
		for _, nodeName := range e2edaemonset.SchedulableNodes(ctx, c, ds) {
			schedulableNodes[nodeName] = nil
		}

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
		err = wait.PollUntilContextTimeout(ctx, 5*time.Second, 3*time.Minute, false, func(ctx context.Context) (bool, error) {
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

		ginkgo.By("having hints set for each endpoint")
		for _, slice := range finalSlices {
			for _, ep := range slice.Endpoints {
				if ep.Zone == nil {
					framework.Failf("Expected endpoint in %s to have zone: %v", slice.Name, ep)
				}
				if ep.Hints == nil || len(ep.Hints.ForZones) == 0 {
					framework.Failf("Expected endpoint in %s to have hints: %v", slice.Name, ep)
				}
				if len(ep.Hints.ForZones) > 1 {
					framework.Failf("Expected endpoint in %s to have exactly 1 zone hint, got %d: %v", slice.Name, len(ep.Hints.ForZones), ep)
				}
				if *ep.Zone != ep.Hints.ForZones[0].Name {
					framework.Failf("Expected endpoint in %s to have same zone hint, got %s: %v", slice.Name, *ep.Zone, ep)
				}
			}
		}

		nodesByZone := map[string]string{}
		zonesWithNode := map[string]string{}
		for _, node := range schedulableNodes {
			if zone, ok := node.Labels[v1.LabelTopologyZone]; ok {
				nodesByZone[node.Name] = zone
				zonesWithNode[zone] = node.Name
			}
		}

		podList, err := c.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		podsByZone := map[string]string{}
		for _, pod := range podList.Items {
			if zone, ok := nodesByZone[pod.Spec.NodeName]; ok {
				podsByZone[pod.Name] = zone
			}
		}

		ginkgo.By("keeping requests in the same zone")
		for fromZone, nodeName := range zonesWithNode {
			ginkgo.By("creating a client pod for probing the service from " + fromZone)
			podName := "curl-from-" + fromZone
			clientPod := e2epod.NewAgnhostPod(f.Namespace.Name, podName, nil, nil, nil, "serve-hostname")
			nodeSelection := e2epod.NodeSelection{Name: nodeName}
			e2epod.SetNodeSelection(&clientPod.Spec, nodeSelection)
			cmd := fmt.Sprintf(`date; for i in $(seq 1 3000); do sleep 1; echo "Date: $(date) Try: ${i}"; curl -q -s --connect-timeout 2 http://%s:80/ ; echo; done`, svc.Name)
			clientPod.Spec.Containers[0].Command = []string{"/bin/sh", "-c", cmd}
			clientPod.Spec.Containers[0].Name = clientPod.Name
			e2epod.NewPodClient(f).CreateSync(ctx, clientPod)

			framework.Logf("Ensuring that requests from %s pod on %s node stay in %s zone", clientPod.Name, nodeName, fromZone)

			var logs string
			if pollErr := wait.PollUntilContextTimeout(ctx, 5*time.Second, e2eservice.KubeProxyLagTimeout, false, func(ctx context.Context) (bool, error) {
				var err error
				logs, err = e2epod.GetPodLogs(ctx, c, f.Namespace.Name, clientPod.Name, clientPod.Name)
				framework.ExpectNoError(err)
				framework.Logf("Pod client logs: %s", logs)

				logLines := strings.Split(logs, "\n")
				if len(logLines) < 6 {
					framework.Logf("only %d log lines, waiting for at least 6", len(logLines))
					return false, nil
				}

				consecutiveSameZone := 0

				for i := len(logLines) - 1; i > 0; i-- {
					if logLines[i] == "" || strings.HasPrefix(logLines[i], "Date:") {
						continue
					}
					destZone, ok := podsByZone[logLines[i]]
					if !ok {
						framework.Logf("could not determine dest zone from log line: %s", logLines[i])
						return false, nil
					}
					if fromZone != destZone {
						framework.Logf("expected request from %s to stay in %s zone, delivered to %s zone", clientPod.Name, fromZone, destZone)
						return false, nil
					}
					consecutiveSameZone++
					if consecutiveSameZone >= 5 {
						return true, nil
					}
				}

				return false, nil
			}); pollErr != nil {
				framework.Failf("expected 5 consecutive requests from %s to stay in zone %s within %v, stdout: %v", clientPod.Name, fromZone, e2eservice.KubeProxyLagTimeout, logs)
			}
		}
	})
})
