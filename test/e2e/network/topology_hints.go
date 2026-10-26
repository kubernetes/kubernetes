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
	"math"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
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

		schedulableNodes := map[string]*v1.Node{}
		for _, nodeName := range e2edaemonset.SchedulableNodes(ctx, c, ds) {
			schedulableNodes[nodeName] = nil
		}

		nodeList, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "Error when listing all Nodes")
		for i := range nodeList.Items {
			if _, ok := schedulableNodes[nodeList.Items[i].Name]; ok {
				schedulableNodes[nodeList.Items[i].Name] = &nodeList.Items[i]
			}
		}

		// Nodes may have different allocatable CPU, e.g. a large node for
		// monitoring add-ons. Hint allocations are derived from the same
		// per-zone CPU ratios the controller uses instead of assuming equal
		// CPU per node.
		cpuRatiosByZone, err := zoneCPURatios(nodeList.Items)
		if err != nil {
			e2eskipper.Skipf("EndpointSlice controller would not add topology hints in this cluster: %v", err)
		}
		if _, err := expectedHintAllocations(cpuRatiosByZone, len(schedulableNodes)); err != nil {
			e2eskipper.Skipf("EndpointSlice controller would not add topology hints for %d endpoints: %v", len(schedulableNodes), err)
		}
		framework.Logf("Allocatable CPU ratios by zone: %v", cpuRatiosByZone)

		framework.Logf("Waiting for %d endpoints to be tracked in EndpointSlices with valid hints", len(schedulableNodes))

		ginkgo.By("having hints set for each endpoint")
		var hintedZoneByPod map[string]string
		var lastHintErr error
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

			hintedZoneByPod, lastHintErr = validateTopologyHints(slices.Items, cpuRatiosByZone)
			if lastHintErr != nil {
				framework.Logf("EndpointSlice hints not valid yet: %v", lastHintErr)
				// Node readiness may have changed; the controller allocates from the current nodes.
				if nodes, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{}); err == nil {
					if ratios, err := zoneCPURatios(nodes.Items); err == nil {
						cpuRatiosByZone = ratios
					}
				}
				return false, nil
			}
			return true, nil
		})
		if lastHintErr != nil {
			framework.Failf("EndpointSlice hints for Service %s never became valid: %v", svc.Name, lastHintErr)
		}
		framework.ExpectNoError(err, "timed out waiting for EndpointSlices to be ready")

		zonesWithNode := map[string]string{}
		for _, node := range schedulableNodes {
			if zone, ok := node.Labels[v1.LabelTopologyZone]; ok {
				zonesWithNode[zone] = node.Name
			}
		}

		// kube-proxy routes by hint, not by the endpoint's own zone. They only
		// differ when the Service spans multiple EndpointSlices.
		ginkgo.By("keeping requests on endpoints hinted for the client's zone")
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
				// Hints can move while probing; compare against the current slices.
				slices, err := c.DiscoveryV1().EndpointSlices(f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: fmt.Sprintf("%s=%s", discoveryv1.LabelServiceName, svc.Name)})
				framework.ExpectNoError(err)
				if hintedZoneByPod, err = validateTopologyHints(slices.Items, cpuRatiosByZone); err != nil {
					framework.Logf("EndpointSlice hints not valid: %v", err)
					return false, nil
				}
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
					destZone, ok := hintedZoneByPod[logLines[i]]
					if !ok {
						framework.Logf("could not determine dest zone hint from log line: %s", logLines[i])
						return false, nil
					}
					if fromZone != destZone {
						framework.Logf("expected request from %s to reach an endpoint hinted for %s zone, delivered to one hinted for %s zone", clientPod.Name, fromZone, destZone)
						return false, nil
					}
					consecutiveSameZone++
					if consecutiveSameZone >= 5 {
						return true, nil
					}
				}

				return false, nil
			}); pollErr != nil {
				framework.Failf("expected 5 consecutive requests from %s to reach endpoints hinted for zone %s within %v, stdout: %v", clientPod.Name, fromZone, e2eservice.KubeProxyLagTimeout, logs)
			}
		}
	})
})

// The helpers below mirror k8s.io/endpointslice/topologycache. They are
// duplicated rather than imported so the test checks the controller's
// output against an independent copy of the intended behavior.

// topologyHintsOverloadThreshold mirrors overloadThreshold.
const topologyHintsOverloadThreshold = 0.2

// zoneHintAllocation mirrors the unexported allocation type.
type zoneHintAllocation struct {
	desired          float64
	minimum, maximum int
}

// zoneCPURatios mirrors TopologyCache.SetNodes: the share of allocatable CPU
// in each zone across ready nodes that are not control plane nodes. It returns
// an error when the controller would disable hints because node info is
// insufficient.
func zoneCPURatios(nodes []v1.Node) (map[string]float64, error) {
	cpuByZone := map[string]int64{}
	var totalCPU int64
	for i := range nodes {
		node := &nodes[i]
		if _, ok := node.Labels["node-role.kubernetes.io/control-plane"]; ok {
			continue
		}
		if _, ok := node.Labels["node-role.kubernetes.io/master"]; ok {
			continue
		}
		ready := false
		for _, cond := range node.Status.Conditions {
			if cond.Type == v1.NodeReady {
				ready = cond.Status == v1.ConditionTrue
				break
			}
		}
		if !ready {
			continue
		}
		zone := node.Labels[v1.LabelTopologyZone]
		cpu := node.Status.Allocatable.Cpu().MilliValue()
		if zone == "" || cpu == 0 {
			return nil, fmt.Errorf("ready Node %q has zone %q and allocatable CPU %dm, both are required", node.Name, zone, cpu)
		}
		cpuByZone[zone] += cpu
		totalCPU += cpu
	}
	if len(cpuByZone) < 2 {
		return nil, fmt.Errorf("ready Nodes span %d zones, at least 2 are required", len(cpuByZone))
	}

	ratios := map[string]float64{}
	for zone, cpu := range cpuByZone {
		ratios[zone] = float64(cpu) / float64(totalCPU)
	}
	return ratios, nil
}

// expectedHintAllocations mirrors TopologyCache.getAllocations.
func expectedHintAllocations(cpuRatiosByZone map[string]float64, numEndpoints int) (map[string]zoneHintAllocation, error) {
	if len(cpuRatiosByZone) > numEndpoints {
		return nil, fmt.Errorf("%d endpoints is fewer than %d zones", numEndpoints, len(cpuRatiosByZone))
	}
	allocations := map[string]zoneHintAllocation{}
	minTotal := 0
	for zone, ratio := range cpuRatiosByZone {
		desired := ratio * float64(numEndpoints)
		minimum := int(math.Ceil(desired * (1 / (1 + topologyHintsOverloadThreshold))))
		allocations[zone] = zoneHintAllocation{desired: math.Max(desired, float64(minimum)), minimum: minimum}
		minTotal += minimum
	}
	if minTotal > numEndpoints {
		return nil, fmt.Errorf("minimum allocations add up to %d, more than the %d endpoints available", minTotal, numEndpoints)
	}
	for zone, a := range allocations {
		a.maximum = a.minimum + numEndpoints - minTotal
		allocations[zone] = a
	}
	return allocations, nil
}

// validateTopologyHints checks that every ready endpoint has exactly one zone
// hint and that per-zone hint totals fall within the controller's allocation
// band. It returns the hinted zone keyed by backing pod name.
//
// Hints are only required to match the endpoint's own zone when the Service
// fits in a single EndpointSlice and every zone's endpoint count is within
// half an endpoint of its CPU-weighted target. Only then does the controller
// skip redistribution. With multiple slices, the controller keeps hints on
// unchanged slices as long as totals stay within the band, so cross-zone hints
// assigned while endpoints were still arriving can persist.
func validateTopologyHints(slices []discoveryv1.EndpointSlice, cpuRatiosByZone map[string]float64) (map[string]string, error) {
	hintedZoneByPod := map[string]string{}
	endpointsByZone := map[string]int{}
	hintsByZone := map[string]int{}
	crossZone := 0
	numReady := 0

	for _, slice := range slices {
		for _, ep := range slice.Endpoints {
			// The controller strips hints from endpoints that are not ready.
			if ep.Conditions.Ready == nil || !*ep.Conditions.Ready {
				continue
			}
			numReady++
			if ep.Zone == nil {
				return nil, fmt.Errorf("endpoint in %s has no zone: %v", slice.Name, ep)
			}
			if ep.Hints == nil || len(ep.Hints.ForZones) == 0 {
				return nil, fmt.Errorf("endpoint in %s has no hints: %v", slice.Name, ep)
			}
			if len(ep.Hints.ForZones) > 1 {
				return nil, fmt.Errorf("endpoint in %s has %d zone hints, expected exactly 1: %v", slice.Name, len(ep.Hints.ForZones), ep)
			}
			if ep.TargetRef == nil || ep.TargetRef.Kind != "Pod" {
				return nil, fmt.Errorf("endpoint in %s does not reference a Pod: %v", slice.Name, ep)
			}
			hint := ep.Hints.ForZones[0].Name
			if _, ok := cpuRatiosByZone[hint]; !ok {
				return nil, fmt.Errorf("endpoint in %s is hinted for zone %s, which has no ready Nodes: %v", slice.Name, hint, ep)
			}
			if *ep.Zone != hint {
				crossZone++
			}
			endpointsByZone[*ep.Zone]++
			hintsByZone[hint]++
			hintedZoneByPod[ep.TargetRef.Name] = hint
		}
	}

	allocations, err := expectedHintAllocations(cpuRatiosByZone, numReady)
	if err != nil {
		return nil, err
	}
	expectSameZone := len(slices) == 1
	for zone, a := range allocations {
		if hintsByZone[zone] < a.minimum || hintsByZone[zone] > a.maximum {
			return nil, fmt.Errorf("zone %s has %d hints, expected between %d and %d; endpoints by zone: %v, hints by zone: %v, CPU ratios by zone: %v",
				zone, hintsByZone[zone], a.minimum, a.maximum, endpointsByZone, hintsByZone, cpuRatiosByZone)
		}
		if math.Abs(float64(endpointsByZone[zone])-a.desired) >= 0.5 {
			expectSameZone = false
		}
	}
	if expectSameZone && crossZone > 0 {
		return nil, fmt.Errorf("%d endpoints in a single EndpointSlice are hinted for another zone although endpoints are already spread by CPU; endpoints by zone: %v, hints by zone: %v",
			crossZone, endpointsByZone, hintsByZone)
	}

	framework.Logf("Topology hints across %d EndpointSlices: endpoints by zone %v, hints by zone %v, %d cross-zone hints, same-zone required: %v",
		len(slices), endpointsByZone, hintsByZone, crossZone, expectSameZone)
	return hintedZoneByPod, nil
}
