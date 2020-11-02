/*
Copyright 2020 The Kubernetes Authors.

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
)

const (
	serviceName = "svc-udp"
	podClient   = "pod-client"
	podBackend1 = "pod-server-1"
	podBackend2 = "pod-server-2"
	srcPort     = 12345
)

// Linux NAT uses conntrack to perform NAT, everytime a new
// flow is seen, a connection is created in the conntrack table, and it
// is being used by the NAT module.
// Each entry in the conntrack table has associated a timeout, that removes
// the connection once it expires.
// UDP is a connectionless protocol, so the conntrack module tracking functions
// are not very advanced.
// It uses a short timeout (30 sec by default) that is renewed if there are new flows
// matching the connection. Otherwise it expires the entry.
// This behaviour can cause issues in Kubernetes when one entry on the conntrack table
// is never expired because the sender does not stop sending traffic, but the pods or
// endpoints were deleted, blackholing the traffic
// In order to mitigate this problem, Kubernetes delete the stale entries:
// - when an endpoint is removed
// - when a service goes from no endpoints to new endpoint

// Ref: https://api.semanticscholar.org/CorpusID:198903401
// Boye, Magnus. "Netfilter Connection Tracking and NAT Implementation." (2012).

var _ = SIGDescribe("Conntrack", func() {

	fr := framework.NewDefaultFramework("conntrack")

	type nodeInfo struct {
		name   string
		nodeIP string
	}

	var (
		cs                             clientset.Interface
		ns                             string
		clientNodeInfo, serverNodeInfo nodeInfo
	)

	logContainsFn := func(text string) wait.ConditionFunc {
		return func() (bool, error) {
			logs, err := e2epod.GetPodLogs(cs, ns, podClient, podClient)
			if err != nil {
				// Retry the error next time.
				return false, nil
			}
			if !strings.Contains(string(logs), text) {
				return false, nil
			}
			return true, nil
		}
	}

	ginkgo.BeforeEach(func() {
		cs = fr.ClientSet
		ns = fr.Namespace.Name

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf(
				"Test requires >= 2 Ready nodes, but there are only %v nodes",
				len(nodes.Items))
		}

		ips := e2enode.CollectAddresses(nodes, v1.NodeInternalIP)

		clientNodeInfo = nodeInfo{
			name:   nodes.Items[0].Name,
			nodeIP: ips[0],
		}

		serverNodeInfo = nodeInfo{
			name:   nodes.Items[1].Name,
			nodeIP: ips[1],
		}
	})

	ginkgo.It("should be able to preserve UDP traffic when server pod cycles for a NodePort service", func() {
		// TODO(#91236): Remove once the test is debugged and fixed.
		// dump conntrack table for debugging
		defer dumpConntrack(cs)

		// Create a NodePort service
		udpJig := e2eservice.NewTestJig(cs, ns, serviceName)
		ginkgo.By("creating a UDP service " + serviceName + " with type=NodePort in " + ns)
		udpService, err := udpJig.CreateUDPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "udp", Protocol: v1.ProtocolUDP, TargetPort: intstr.FromInt(80)},
			}
		})
		framework.ExpectNoError(err)

		// Create a pod in one node to create the UDP traffic against the NodePort service every 5 seconds
		ginkgo.By("creating a client pod for probing the service " + serviceName)
		clientPod := e2epod.NewAgnhostPod(ns, podClient, nil, nil, nil)
		clientPod.Spec.NodeName = clientNodeInfo.name
		cmd := fmt.Sprintf(`date; for i in $(seq 1 3000); do echo "$(date) Try: ${i}"; echo hostname | nc -u -w 5 -p %d %s %d; echo; done`, srcPort, serverNodeInfo.nodeIP, udpService.Spec.Ports[0].NodePort)
		clientPod.Spec.Containers[0].Command = []string{"/bin/sh", "-c", cmd}
		clientPod.Spec.Containers[0].Name = podClient
		fr.PodClient().CreateSync(clientPod)

		// Read the client pod logs
		logs, err := e2epod.GetPodLogs(cs, ns, podClient, podClient)
		framework.ExpectNoError(err)
		framework.Logf("Pod client logs: %s", logs)

		// Add a backend pod to the service in the other node
		ginkgo.By("creating a backend pod " + podBackend1 + " for the service " + serviceName)
		serverPod1 := e2epod.NewAgnhostPod(ns, podBackend1, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod1.Labels = udpJig.Labels
		serverPod1.Spec.NodeName = serverNodeInfo.name
		fr.PodClient().CreateSync(serverPod1)

		// Waiting for service to expose endpoint.
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{podBackend1: {80}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		// Note that the fact that Endpoints object already exists, does NOT mean
		// that iptables (or whatever else is used) was already programmed.
		// Additionally take into account that UDP conntract entries timeout is
		// 30 seconds by default.
		// Based on the above check if the pod receives the traffic.
		ginkgo.By("checking client pod connected to the backend 1 on Node IP " + serverNodeInfo.nodeIP)
		if err := wait.PollImmediate(5*time.Second, time.Minute, logContainsFn(podBackend1)); err != nil {
			logs, err = e2epod.GetPodLogs(cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 1")
		}

		// Create a second pod
		ginkgo.By("creating a second backend pod " + podBackend2 + " for the service " + serviceName)
		serverPod2 := e2epod.NewAgnhostPod(ns, podBackend2, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod2.Labels = udpJig.Labels
		serverPod2.Spec.NodeName = serverNodeInfo.name
		fr.PodClient().CreateSync(serverPod2)

		// and delete the first pod
		framework.Logf("Cleaning up %s pod", podBackend1)
		fr.PodClient().DeleteSync(podBackend1, metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)

		// Waiting for service to expose endpoint.
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{podBackend2: {80}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		// Check that the second pod keeps receiving traffic
		// UDP conntrack entries timeout is 30 sec by default
		ginkgo.By("checking client pod connected to the backend 2 on Node IP " + serverNodeInfo.nodeIP)
		if err := wait.PollImmediate(5*time.Second, time.Minute, logContainsFn(podBackend2)); err != nil {
			logs, err = e2epod.GetPodLogs(cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 2")
		}
	})

	ginkgo.It("should be able to preserve UDP traffic when server pod cycles for a ClusterIP service", func() {
		// TODO(#91236): Remove once the test is debugged and fixed.
		// dump conntrack table for debugging
		defer dumpConntrack(cs)

		// Create a ClusterIP service
		udpJig := e2eservice.NewTestJig(cs, ns, serviceName)
		ginkgo.By("creating a UDP service " + serviceName + " with type=ClusterIP in " + ns)
		udpService, err := udpJig.CreateUDPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "udp", Protocol: v1.ProtocolUDP, TargetPort: intstr.FromInt(80)},
			}
		})
		framework.ExpectNoError(err)

		// Create a pod in one node to create the UDP traffic against the ClusterIP service every 5 seconds
		ginkgo.By("creating a client pod for probing the service " + serviceName)
		clientPod := e2epod.NewAgnhostPod(ns, podClient, nil, nil, nil)
		clientPod.Spec.NodeName = clientNodeInfo.name
		cmd := fmt.Sprintf(`date; for i in $(seq 1 3000); do echo "$(date) Try: ${i}"; echo hostname | nc -u -w 5 -p %d %s %d; echo; done`, srcPort, udpService.Spec.ClusterIP, udpService.Spec.Ports[0].Port)
		clientPod.Spec.Containers[0].Command = []string{"/bin/sh", "-c", cmd}
		clientPod.Spec.Containers[0].Name = podClient
		fr.PodClient().CreateSync(clientPod)

		// Read the client pod logs
		logs, err := e2epod.GetPodLogs(cs, ns, podClient, podClient)
		framework.ExpectNoError(err)
		framework.Logf("Pod client logs: %s", logs)

		// Add a backend pod to the service in the other node
		ginkgo.By("creating a backend pod " + podBackend1 + " for the service " + serviceName)
		serverPod1 := e2epod.NewAgnhostPod(ns, podBackend1, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod1.Labels = udpJig.Labels
		serverPod1.Spec.NodeName = serverNodeInfo.name
		fr.PodClient().CreateSync(serverPod1)

		// Waiting for service to expose endpoint.
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{podBackend1: {80}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		// Note that the fact that Endpoints object already exists, does NOT mean
		// that iptables (or whatever else is used) was already programmed.
		// Additionally take into account that UDP conntract entries timeout is
		// 30 seconds by default.
		// Based on the above check if the pod receives the traffic.
		ginkgo.By("checking client pod connected to the backend 1 on Node IP " + serverNodeInfo.nodeIP)
		if err := wait.PollImmediate(5*time.Second, time.Minute, logContainsFn(podBackend1)); err != nil {
			logs, err = e2epod.GetPodLogs(cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 1")
		}

		// Create a second pod
		ginkgo.By("creating a second backend pod " + podBackend2 + " for the service " + serviceName)
		serverPod2 := e2epod.NewAgnhostPod(ns, podBackend2, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod2.Labels = udpJig.Labels
		serverPod2.Spec.NodeName = serverNodeInfo.name
		fr.PodClient().CreateSync(serverPod2)

		// and delete the first pod
		framework.Logf("Cleaning up %s pod", podBackend1)
		fr.PodClient().DeleteSync(podBackend1, metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)

		// Waiting for service to expose endpoint.
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{podBackend2: {80}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		// Check that the second pod keeps receiving traffic
		// UDP conntrack entries timeout is 30 sec by default
		ginkgo.By("checking client pod connected to the backend 2 on Node IP " + serverNodeInfo.nodeIP)
		if err := wait.PollImmediate(5*time.Second, time.Minute, logContainsFn(podBackend2)); err != nil {
			logs, err = e2epod.GetPodLogs(cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 2")
		}
	})
})

func dumpConntrack(cs clientset.Interface) {
	// Dump conntrack table of each node for troubleshooting using the kube-proxy pods
	namespace := "kube-system"
	pods, err := cs.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{})
	if err != nil || len(pods.Items) == 0 {
		framework.Logf("failed to list kube-proxy pods in namespace: %s", namespace)
		return
	}
	cmd := "conntrack -L"
	for _, pod := range pods.Items {
		if strings.Contains(pod.Name, "kube-proxy") {
			stdout, err := framework.RunHostCmd(namespace, pod.Name, cmd)
			if err != nil {
				framework.Logf("Failed to dump conntrack table of node %s: %v", pod.Spec.NodeName, err)
				continue
			}
			framework.Logf("conntrack table of node %s: %s", pod.Spec.NodeName, stdout)
		}
	}
}
