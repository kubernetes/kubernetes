/*
Copyright 2021 The Kubernetes Authors.

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

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
)

var _ = common.SIGDescribe("[Feature:TopologyAwareHints] Topology Aware Hints [Experimental]", func() {

	f := framework.NewDefaultFramework("topology-hints")

	var cs clientset.Interface

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		zoneNames, err := e2enode.GetClusterZones(cs)
		framework.ExpectNoError(err)
		zoneCount := len(zoneNames)
		ginkgo.By(fmt.Sprintf("Checking for multi-zone cluster.  Zone count = %d", zoneCount))
		msg := fmt.Sprintf("Zone count is %d, only run for multi-zone clusters, skipping test", zoneCount)
		e2eskipper.SkipUnlessAtLeast(zoneCount, 2, msg)
	})

	ginkgo.It("Services with topology annotation should forward traffic to nodes on the same zone", func() {
		namespace := f.Namespace.Name
		serviceName := "svc-topology"
		port := 80

		jig := e2eservice.NewTestJig(cs, namespace, serviceName)
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, e2eservice.MaxNodesForEndpointsTests)
		framework.ExpectNoError(err)
		// select one node in a specific zone to create a pod to generate the traffic
		nodeName := nodes.Items[0].Name
		nodeZone, ok := nodes.Items[0].Labels[v1.LabelTopologyZone]
		if !ok || nodeZone == "" {
			framework.Failf("Fail to obtain zone information for node %s", nodeName)
		}

		ginkgo.By("creating an annotated service with no endpoints and topology aware annotation")
		svc, err := jig.CreateTCPServiceWithPort(func(svc *v1.Service) {
			svc.Annotations = map[string]string{v1.AnnotationTopologyAwareHints: "Auto"}
			svc.Spec.Ports = []v1.ServicePort{
				{Port: int32(port), Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(9376)},
			}
		}, int32(port))
		framework.ExpectNoError(err)
		svcIP := svc.Spec.ClusterIP

		ginkgo.By("creating backend pods for the service on each node" + serviceName)
		// hints are only generated if there is a reasonable distribution of endpoints
		// TODO(aojea): revisit, this number was obtained with KIND in a cluster with 3 nodes
		err = jig.CreateServicePods(3 * len(nodes.Items))
		framework.ExpectNoError(err)

		// check that topology hints are created
		opts := metav1.ListOptions{
			LabelSelector: "kubernetes.io/service-name=" + serviceName,
		}

		// map that contains the zone associated to each pod
		podsZones := map[string]string{}
		err = wait.PollImmediate(3*time.Second, 1*time.Minute, func() (bool, error) {
			es, err := cs.DiscoveryV1().EndpointSlices(namespace).List(context.TODO(), opts)
			if err != nil {
				framework.Logf("Failed to get list EndpointSlice objects: %v", err)
				// Retry the error
				return false, nil
			}
			for _, endpointSlice := range es.Items {
				for _, ep := range endpointSlice.Endpoints {
					if ep.Hints == nil {
						framework.Logf("Failed to get Topology hints on endpoint")
						return false, nil
					}
					podsZones[ep.TargetRef.Name] = *ep.Zone
					framework.Logf("pod %s on zone %s with hints %v", ep.TargetRef.Name, *ep.Zone, ep.Hints)
				}
			}
			return true, nil

		})
		framework.ExpectNoError(err)

		// create a pod in a specific zone to check the traffic distribution
		execPod := e2epod.CreateExecPodOrFail(cs, namespace, "execpod", func(pod *v1.Pod) {
			pod.Spec.NodeName = nodeName
		})
		defer func() {
			framework.Logf("Cleaning up the exec pod")
			err := cs.CoreV1().Pods(namespace).Delete(context.TODO(), execPod.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete pod: %s in namespace: %s", execPod.Name, namespace)
		}()
		err = jig.CheckServiceReachability(svc, execPod)
		framework.ExpectNoError(err)

		// poll the service 100 times and get a per zone traffic distribution
		zones := map[string]int{}
		maxConnections := 100
		nc := fmt.Sprintf(`echo hostName | nc -v -w 5 %s %d`, svcIP, port)
		cmd := fmt.Sprintf("for i in $(seq 0 %d); do echo; %s ; done", maxConnections, nc)
		stdout, err := framework.RunHostCmd(execPod.Namespace, execPod.Name, cmd)
		framework.ExpectNoError(err)

		hostnames := strings.Split(stdout, "\n")
		failed := 0
		// failed request return an empty string that we can not associate to a zone
		for _, h := range hostnames {
			z, ok := podsZones[h]
			if ok {
				zones[z]++
			} else {
				failed++
			}
		}
		framework.Logf("Connections from Zone %s distributed with topology hints enabled %v Failed: %d", nodeZone, zones, failed)
		// Fail if the traffic in the zone is not higher than the 65%
		// don't consider the failed connections for the stats
		// TODO(aojea): based on local testing with KIND revisit percentages.
		zoneTraffic := zones[nodeZone] * 100 / (maxConnections - failed)
		if zoneTraffic < 65 {
			framework.Failf("Traffic within the zone is lower than 65 per cent : %d", zoneTraffic)
		}

		// disable topology aware hints on the service
		ginkgo.By("disabled topology aware annotation on service")
		svc, err = jig.UpdateService(func(svc *v1.Service) {
			svc.Annotations["service.kubernetes.io/topology-aware-hints"] = "disabled"
		})
		framework.ExpectNoError(err)
		// check that hints are disabled
		err = wait.PollImmediate(3*time.Second, 1*time.Minute, func() (bool, error) {
			es, err := cs.DiscoveryV1().EndpointSlices(namespace).List(context.TODO(), opts)
			if err != nil {
				framework.Logf("Failed to get list EndpointSlice objects: %v", err)
				// Retry the error
				return false, nil
			}
			for _, endpointSlice := range es.Items {
				for _, ep := range endpointSlice.Endpoints {
					if ep.Hints != nil {
						framework.Logf("EndpointSlice still have topology hints")
						return false, nil
					}
				}
			}
			return true, nil

		})
		framework.ExpectNoError(err)

		err = jig.CheckServiceReachability(svc, execPod)
		framework.ExpectNoError(err)
		// get the traffic distribution without Topology Hints
		zones = map[string]int{}
		stdout, err = framework.RunHostCmd(execPod.Namespace, execPod.Name, cmd)
		framework.ExpectNoError(err)

		hostnames = strings.Split(stdout, "\n")
		failed = 0
		for _, h := range hostnames {
			z, ok := podsZones[h]
			if ok {
				zones[z]++
			} else {
				failed++
			}
		}
		framework.Logf("Connections per zone from %s without topology hints %v Failed %d", nodeZone, zones, failed)
		// Fail if traffic per zone is not equally distributed [40%,60%]
		// don't consider the failed connections for the stats
		// TODO(aojea): based on local testing with KIND revisit percentages.
		zoneTraffic = zones[nodeZone] * 100 / (100 - failed)
		if zoneTraffic < 40 || zoneTraffic > 60 {
			framework.Failf("Traffic within the zone %s is lower than 40 or greater than 60 per cent : %d", nodeZone, zoneTraffic)
		}

	})

})
