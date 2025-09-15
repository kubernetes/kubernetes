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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
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
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	serviceName = "svc-udp"
	podClient   = "pod-client"
	podBackend1 = "pod-server-1"
	podBackend2 = "pod-server-2"
	podBackend3 = "pod-server-3"
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

var _ = common.SIGDescribe("Conntrack", func() {

	fr := framework.NewDefaultFramework("conntrack")
	fr.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	type nodeInfo struct {
		name   string
		nodeIP string
	}

	var (
		cs                             clientset.Interface
		ns                             string
		clientNodeInfo, serverNodeInfo nodeInfo
	)

	logContainsFn := func(text, podName string) wait.ConditionWithContextFunc {
		return func(ctx context.Context) (bool, error) {
			logs, err := e2epod.GetPodLogs(ctx, cs, ns, podName, podName)
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

	ginkgo.BeforeEach(func(ctx context.Context) {
		cs = fr.ClientSet
		ns = fr.Namespace.Name

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 2)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf(
				"Test requires >= 2 Ready nodes, but there are only %v nodes",
				len(nodes.Items))
		}

		family := v1.IPv4Protocol
		if framework.TestContext.ClusterIsIPv6() {
			family = v1.IPv6Protocol
		}

		ips := e2enode.GetAddressesByTypeAndFamily(&nodes.Items[0], v1.NodeInternalIP, family)
		gomega.Expect(ips).ToNot(gomega.BeEmpty())

		clientNodeInfo = nodeInfo{
			name:   nodes.Items[0].Name,
			nodeIP: ips[0],
		}

		ips = e2enode.GetAddressesByTypeAndFamily(&nodes.Items[1], v1.NodeInternalIP, family)
		gomega.Expect(ips).ToNot(gomega.BeEmpty())

		serverNodeInfo = nodeInfo{
			name:   nodes.Items[1].Name,
			nodeIP: ips[0],
		}
	})

	ginkgo.It("should be able to preserve UDP traffic when server pod cycles for a NodePort service", func(ctx context.Context) {

		// Create a NodePort service
		udpJig := e2eservice.NewTestJig(cs, ns, serviceName)
		ginkgo.By("creating a UDP service " + serviceName + " with type=NodePort in " + ns)
		udpService, err := udpJig.CreateUDPService(ctx, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "udp", Protocol: v1.ProtocolUDP, TargetPort: intstr.FromInt32(80)},
			}
		})
		framework.ExpectNoError(err)

		// Create a pod in one node to create the UDP traffic against the NodePort service every 5 seconds
		ginkgo.By("creating a client pod for probing the service " + serviceName)
		clientPod := e2epod.NewAgnhostPod(ns, podClient, nil, nil, nil)
		nodeSelection := e2epod.NodeSelection{Name: clientNodeInfo.name}
		e2epod.SetNodeSelection(&clientPod.Spec, nodeSelection)
		cmd := fmt.Sprintf(`date; for i in $(seq 1 3000); do echo "$(date) Try: ${i}"; echo hostname | nc -u -w 5 -p %d %s %d; echo; done`, srcPort, serverNodeInfo.nodeIP, udpService.Spec.Ports[0].NodePort)
		clientPod.Spec.Containers[0].Command = []string{"/bin/sh", "-c", cmd}
		clientPod.Spec.Containers[0].Name = podClient
		e2epod.NewPodClient(fr).CreateSync(ctx, clientPod)

		// Read the client pod logs
		logs, err := e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
		framework.ExpectNoError(err)
		framework.Logf("Pod client logs: %s", logs)

		// Add a backend pod to the service in the other node
		ginkgo.By("creating a backend pod " + podBackend1 + " for the service " + serviceName)
		serverPod1 := e2epod.NewAgnhostPod(ns, podBackend1, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod1.Labels = udpJig.Labels
		nodeSelection = e2epod.NodeSelection{Name: serverNodeInfo.name}
		e2epod.SetNodeSelection(&serverPod1.Spec, nodeSelection)
		e2epod.NewPodClient(fr).CreateSync(ctx, serverPod1)

		validateEndpointsPortsOrFail(ctx, cs, ns, serviceName, portsByPodName{podBackend1: {80}})

		// Note that the fact that Endpoints object already exists, does NOT mean
		// that iptables (or whatever else is used) was already programmed.
		// Additionally take into account that UDP conntract entries timeout is
		// 30 seconds by default.
		// Based on the above check if the pod receives the traffic.
		ginkgo.By("checking client pod connected to the backend 1 on Node IP " + serverNodeInfo.nodeIP)
		if err := wait.PollUntilContextTimeout(ctx, 5*time.Second, time.Minute, true, logContainsFn(podBackend1, podClient)); err != nil {
			logs, err = e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 1")
		}

		// Create a second pod
		ginkgo.By("creating a second backend pod " + podBackend2 + " for the service " + serviceName)
		serverPod2 := e2epod.NewAgnhostPod(ns, podBackend2, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod2.Labels = udpJig.Labels
		nodeSelection = e2epod.NodeSelection{Name: serverNodeInfo.name}
		e2epod.SetNodeSelection(&serverPod2.Spec, nodeSelection)
		e2epod.NewPodClient(fr).CreateSync(ctx, serverPod2)

		// and delete the first pod
		framework.Logf("Cleaning up %s pod", podBackend1)
		e2epod.NewPodClient(fr).DeleteSync(ctx, podBackend1, metav1.DeleteOptions{}, fr.Timeouts.PodDelete)

		validateEndpointsPortsOrFail(ctx, cs, ns, serviceName, portsByPodName{podBackend2: {80}})

		// Check that the second pod keeps receiving traffic
		// UDP conntrack entries timeout is 30 sec by default
		ginkgo.By("checking client pod connected to the backend 2 on Node IP " + serverNodeInfo.nodeIP)
		if err := wait.PollUntilContextTimeout(ctx, 5*time.Second, time.Minute, true, logContainsFn(podBackend2, podClient)); err != nil {
			logs, err = e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 2")
		}
	})

	ginkgo.It("should be able to preserve UDP traffic when server pod cycles for a ClusterIP service", func(ctx context.Context) {

		// Create a ClusterIP service
		udpJig := e2eservice.NewTestJig(cs, ns, serviceName)
		ginkgo.By("creating a UDP service " + serviceName + " with type=ClusterIP in " + ns)
		udpService, err := udpJig.CreateUDPService(ctx, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "udp", Protocol: v1.ProtocolUDP, TargetPort: intstr.FromInt32(80)},
			}
		})
		framework.ExpectNoError(err)

		// Create a pod in one node to create the UDP traffic against the ClusterIP service every 5 seconds
		ginkgo.By("creating a client pod for probing the service " + serviceName)
		clientPod := e2epod.NewAgnhostPod(ns, podClient, nil, nil, nil)
		nodeSelection := e2epod.NodeSelection{Name: clientNodeInfo.name}
		e2epod.SetNodeSelection(&clientPod.Spec, nodeSelection)
		cmd := fmt.Sprintf(`date; for i in $(seq 1 3000); do echo "$(date) Try: ${i}"; echo hostname | nc -u -w 5 -p %d %s %d; echo; done`, srcPort, udpService.Spec.ClusterIP, udpService.Spec.Ports[0].Port)
		clientPod.Spec.Containers[0].Command = []string{"/bin/sh", "-c", cmd}
		clientPod.Spec.Containers[0].Name = podClient
		e2epod.NewPodClient(fr).CreateSync(ctx, clientPod)

		// Read the client pod logs
		logs, err := e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
		framework.ExpectNoError(err)
		framework.Logf("Pod client logs: %s", logs)

		// Add a backend pod to the service in the other node
		ginkgo.By("creating a backend pod " + podBackend1 + " for the service " + serviceName)
		serverPod1 := e2epod.NewAgnhostPod(ns, podBackend1, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod1.Labels = udpJig.Labels
		nodeSelection = e2epod.NodeSelection{Name: serverNodeInfo.name}
		e2epod.SetNodeSelection(&serverPod1.Spec, nodeSelection)
		e2epod.NewPodClient(fr).CreateSync(ctx, serverPod1)

		validateEndpointsPortsOrFail(ctx, cs, ns, serviceName, portsByPodName{podBackend1: {80}})

		// Note that the fact that Endpoints object already exists, does NOT mean
		// that iptables (or whatever else is used) was already programmed.
		// Additionally take into account that UDP conntract entries timeout is
		// 30 seconds by default.
		// Based on the above check if the pod receives the traffic.
		ginkgo.By("checking client pod connected to the backend 1 on Node IP " + serverNodeInfo.nodeIP)
		if err := wait.PollUntilContextTimeout(ctx, 5*time.Second, time.Minute, true, logContainsFn(podBackend1, podClient)); err != nil {
			logs, err = e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 1")
		}

		// Create a second pod
		ginkgo.By("creating a second backend pod " + podBackend2 + " for the service " + serviceName)
		serverPod2 := e2epod.NewAgnhostPod(ns, podBackend2, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod2.Labels = udpJig.Labels
		nodeSelection = e2epod.NodeSelection{Name: serverNodeInfo.name}
		e2epod.SetNodeSelection(&serverPod2.Spec, nodeSelection)
		e2epod.NewPodClient(fr).CreateSync(ctx, serverPod2)

		// and delete the first pod
		framework.Logf("Cleaning up %s pod", podBackend1)
		e2epod.NewPodClient(fr).DeleteSync(ctx, podBackend1, metav1.DeleteOptions{}, fr.Timeouts.PodDelete)

		validateEndpointsPortsOrFail(ctx, cs, ns, serviceName, portsByPodName{podBackend2: {80}})

		// Check that the second pod keeps receiving traffic
		// UDP conntrack entries timeout is 30 sec by default
		ginkgo.By("checking client pod connected to the backend 2 on Node IP " + serverNodeInfo.nodeIP)
		if err := wait.PollUntilContextTimeout(ctx, 5*time.Second, time.Minute, true, logContainsFn(podBackend2, podClient)); err != nil {
			logs, err = e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 2")
		}
	})

	// Regression test for https://issues.k8s.io/126934
	ginkgo.It("should be able to preserve UDP traffic when server pod cycles for a ClusterIP service with InternalTrafficPolicy set to Local", func(ctx context.Context) {

		// Create a ClusterIP service with InternalTrafficPolicy local
		udpJig := e2eservice.NewTestJig(cs, ns, serviceName)
		ginkgo.By("creating a UDP service " + serviceName + " with type=ClusterIP in " + ns)
		udpService, err := udpJig.CreateUDPService(ctx, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "udp", Protocol: v1.ProtocolUDP, TargetPort: intstr.FromInt32(80)},
			}
			svc.Spec.InternalTrafficPolicy = ptr.To(v1.ServiceInternalTrafficPolicyLocal)
		})
		framework.ExpectNoError(err)

		// Create a pod in one node to create the UDP traffic against the ClusterIP service every second
		ginkgo.By("creating a client pod for probing the service " + serviceName)
		clientPod := e2epod.NewAgnhostPod(ns, podClient, nil, nil, nil)
		nodeSelection := e2epod.NodeSelection{Name: clientNodeInfo.name}
		e2epod.SetNodeSelection(&clientPod.Spec, nodeSelection)
		cmd := fmt.Sprintf(`date; for i in $(seq 1 3000); do echo "$(date) Try: ${i}"; echo hostname | nc -u -w 1 -p %d %s %d; echo; done`, srcPort, udpService.Spec.ClusterIP, udpService.Spec.Ports[0].Port)
		clientPod.Spec.Containers[0].Command = []string{"/bin/sh", "-c", cmd}
		clientPod.Spec.Containers[0].Name = podClient
		e2epod.NewPodClient(fr).CreateSync(ctx, clientPod)

		// Read the client pod logs
		logs, err := e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
		framework.ExpectNoError(err)
		framework.Logf("Pod client logs: %s", logs)

		// Create a daemonset for the Service with InternalTrafficPolicy Local so there is one Server in each Pod
		// and all pods in the node sends the traffic to the local server.
		ginkgo.By("creating a daemonset pod for the service " + serviceName)
		dsName := "ds-conntrack"
		ds := e2edaemonset.NewDaemonSet(dsName, imageutils.GetE2EImage(imageutils.Agnhost), udpJig.Labels, nil, nil, []v1.ContainerPort{{ContainerPort: 80}}, "netexec", "--udp-port=80", "--delay-shutdown=10")
		ginkgo.By("Creating a DaemonSet")
		ds, err = fr.ClientSet.AppsV1().DaemonSets(ns).Create(ctx, ds, metav1.CreateOptions{})
		if err != nil {
			framework.Failf("unable to create test DaemonSet %s: %v", dsName, err)
		}
		ginkgo.By("Waiting for DaemonSet pods to become ready")
		err = wait.PollUntilContextTimeout(ctx, framework.Poll, framework.PodStartTimeout, false, func(ctx context.Context) (bool, error) {
			return e2edaemonset.CheckRunningOnAllNodes(ctx, fr, ds)
		})
		framework.ExpectNoError(err)
		// Get name of the pod running on the same node than the client
		labelSelector := labels.SelectorFromSet(udpJig.Labels).String()
		fieldSelector := fields.OneTermEqualSelector("spec.nodeName", clientNodeInfo.name).String()
		listOpts := metav1.ListOptions{LabelSelector: labelSelector, FieldSelector: fieldSelector}
		pods, err := fr.ClientSet.CoreV1().Pods(ns).List(ctx, listOpts)
		framework.ExpectNoError(err)
		if len(pods.Items) != 1 {
			framework.Failf("expected 1 pod, got %d pods", len(pods.Items))
		}
		podBackend1 := pods.Items[0].Name
		// Note that the fact that Endpoints object already exists, does NOT mean
		// that iptables (or whatever else is used) was already programmed.
		// Additionally take into account that UDP conntract entries timeout is
		// 30 seconds by default.
		// Based on the above check if the pod receives the traffic.
		ginkgo.By("checking client pod connected to " + podBackend1 + " on Node " + clientNodeInfo.name)
		if err := wait.PollUntilContextTimeout(ctx, 5*time.Second, time.Minute, true, logContainsFn(podBackend1, podClient)); err != nil {
			logs, err = e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 1")
		}

		// Now recreate the first backend pod
		framework.Logf("Cleaning up %s pod", podBackend1)
		e2epod.NewPodClient(fr).DeleteSync(ctx, podBackend1, metav1.DeleteOptions{}, fr.Timeouts.PodDelete)

		ginkgo.By("Waiting for DaemonSet pods to become ready")
		err = wait.PollUntilContextTimeout(ctx, framework.Poll, framework.PodStartTimeout, false, func(ctx context.Context) (bool, error) {
			return e2edaemonset.CheckRunningOnAllNodes(ctx, fr, ds)
		})
		framework.ExpectNoError(err)

		pods, err = fr.ClientSet.CoreV1().Pods(ns).List(ctx, listOpts)
		framework.ExpectNoError(err)
		if len(pods.Items) != 1 {
			framework.Failf("expected 1 pod, got %d pods", len(pods.Items))
		}
		podBackend2 := pods.Items[0].Name
		// Check that the new pod keeps receiving traffic after is recreated
		// UDP conntrack entries timeout is 30 sec by default
		ginkgo.By("checking client pod connected to " + podBackend2 + " on Node " + clientNodeInfo.name)
		if err := wait.PollUntilContextTimeout(ctx, 5*time.Second, time.Minute, true, logContainsFn(podBackend2, podClient)); err != nil {
			logs, err = e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 3")
		}
	})

	ginkgo.It("should be able to preserve UDP traffic when server pod cycles for a ClusterIP service and client is hostNetwork", func(ctx context.Context) {

		// Create a ClusterIP service
		udpJig := e2eservice.NewTestJig(cs, ns, serviceName)
		ginkgo.By("creating a UDP service " + serviceName + " with type=ClusterIP in " + ns)
		udpService, err := udpJig.CreateUDPService(ctx, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "udp", Protocol: v1.ProtocolUDP, TargetPort: intstr.FromInt32(80)},
			}
		})
		framework.ExpectNoError(err)

		// Create a pod in one node to create the UDP traffic against the ClusterIP service every 5 seconds
		ginkgo.By("creating a client pod for probing the service " + serviceName)
		clientPod := e2epod.NewAgnhostPod(ns, podClient, nil, nil, nil)
		nodeSelection := e2epod.NodeSelection{Name: clientNodeInfo.name}
		e2epod.SetNodeSelection(&clientPod.Spec, nodeSelection)
		cmd := fmt.Sprintf(`date; for i in $(seq 1 3000); do echo "$(date) Try: ${i}"; echo hostname | nc -u -w 5 -p %d %s %d; echo; done`, srcPort, udpService.Spec.ClusterIP, udpService.Spec.Ports[0].Port)
		clientPod.Spec.Containers[0].Command = []string{"/bin/sh", "-c", cmd}
		clientPod.Spec.Containers[0].Name = podClient
		clientPod.Spec.HostNetwork = true
		e2epod.NewPodClient(fr).CreateSync(ctx, clientPod)

		// Read the client pod logs
		logs, err := e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
		framework.ExpectNoError(err)
		framework.Logf("Pod client logs: %s", logs)

		// Add a backend pod to the service in the other node
		ginkgo.By("creating a backend pod " + podBackend1 + " for the service " + serviceName)
		serverPod1 := e2epod.NewAgnhostPod(ns, podBackend1, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod1.Labels = udpJig.Labels
		nodeSelection = e2epod.NodeSelection{Name: serverNodeInfo.name}
		e2epod.SetNodeSelection(&serverPod1.Spec, nodeSelection)
		e2epod.NewPodClient(fr).CreateSync(ctx, serverPod1)

		validateEndpointsPortsOrFail(ctx, cs, ns, serviceName, portsByPodName{podBackend1: {80}})

		// Note that the fact that Endpoints object already exists, does NOT mean
		// that iptables (or whatever else is used) was already programmed.
		// Additionally take into account that UDP conntract entries timeout is
		// 30 seconds by default.
		// Based on the above check if the pod receives the traffic.
		ginkgo.By("checking client pod connected to the backend 1 on Node IP " + serverNodeInfo.nodeIP)
		if err := wait.PollUntilContextTimeout(ctx, 5*time.Second, time.Minute, true, logContainsFn(podBackend1, podClient)); err != nil {
			logs, err = e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 1")
		}

		// Create a second pod
		ginkgo.By("creating a second backend pod " + podBackend2 + " for the service " + serviceName)
		serverPod2 := e2epod.NewAgnhostPod(ns, podBackend2, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod2.Labels = udpJig.Labels
		nodeSelection = e2epod.NodeSelection{Name: serverNodeInfo.name}
		e2epod.SetNodeSelection(&serverPod2.Spec, nodeSelection)
		e2epod.NewPodClient(fr).CreateSync(ctx, serverPod2)

		// and delete the first pod
		framework.Logf("Cleaning up %s pod", podBackend1)
		e2epod.NewPodClient(fr).DeleteSync(ctx, podBackend1, metav1.DeleteOptions{}, fr.Timeouts.PodDelete)

		validateEndpointsPortsOrFail(ctx, cs, ns, serviceName, portsByPodName{podBackend2: {80}})

		// Check that the second pod keeps receiving traffic
		// UDP conntrack entries timeout is 30 sec by default
		ginkgo.By("checking client pod connected to the backend 2 on Node IP " + serverNodeInfo.nodeIP)
		if err := wait.PollUntilContextTimeout(ctx, 5*time.Second, time.Minute, true, logContainsFn(podBackend2, podClient)); err != nil {
			logs, err = e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 2")
		}
	})

	// Regression test for #105657
	// 1. Create an UDP Service
	// 2. Client Pod sending traffic to the UDP service
	// 3. Create an UDP server associated to the Service created in 1. with an init container that sleeps for some time
	// The init container makes that the server pod is not ready, however, the endpoint slices are created, it is just
	// that the Endpoint conditions Ready is false.
	// If the kube-proxy conntrack logic doesn't check readiness, it will delete the conntrack entries for the UDP server
	// when the endpoint slice has been created, however, the iptables rules will not installed until at least one
	// endpoint is ready. If some traffic arrives to since kube-proxy clear the entries (see the endpoint slice) and
	// installs the corresponding iptables rules (the endpoint is ready), a conntrack entry will be generated blackholing
	// subsequent traffic.
	ginkgo.It("should be able to preserve UDP traffic when initial unready endpoints get ready", func(ctx context.Context) {

		// Create a ClusterIP service
		udpJig := e2eservice.NewTestJig(cs, ns, serviceName)
		ginkgo.By("creating a UDP service " + serviceName + " with type=ClusterIP in " + ns)
		udpService, err := udpJig.CreateUDPService(ctx, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "udp", Protocol: v1.ProtocolUDP, TargetPort: intstr.FromInt32(80)},
			}
		})
		framework.ExpectNoError(err)

		// Create a pod in one node to create the UDP traffic against the ClusterIP service every 5 seconds
		ginkgo.By("creating a client pod for probing the service " + serviceName)
		clientPod := e2epod.NewAgnhostPod(ns, podClient, nil, nil, nil)
		nodeSelection := e2epod.NodeSelection{Name: clientNodeInfo.name}
		e2epod.SetNodeSelection(&clientPod.Spec, nodeSelection)
		cmd := fmt.Sprintf(`date; for i in $(seq 1 3000); do echo "$(date) Try: ${i}"; echo hostname | nc -u -w 5 -p %d %s %d; echo; done`, srcPort, udpService.Spec.ClusterIP, udpService.Spec.Ports[0].Port)
		clientPod.Spec.Containers[0].Command = []string{"/bin/sh", "-c", cmd}
		clientPod.Spec.Containers[0].Name = podClient
		e2epod.NewPodClient(fr).CreateSync(ctx, clientPod)

		// Read the client pod logs
		logs, err := e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
		framework.ExpectNoError(err)
		framework.Logf("Pod client logs: %s", logs)

		// Add a backend pod to the service in the other node
		ginkgo.By("creating a backend pod " + podBackend1 + " for the service " + serviceName)
		serverPod1 := e2epod.NewAgnhostPod(ns, podBackend1, nil, nil, nil, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod1.Labels = udpJig.Labels
		nodeSelection = e2epod.NodeSelection{Name: serverNodeInfo.name}
		// Add an init container to hold the pod to be ready for 15 seconds
		serverPod1.Spec.InitContainers = []v1.Container{
			{
				Name:    "init",
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: []string{"/bin/sh", "-c", "echo Pausing start. && sleep 15"},
			},
		}
		e2epod.SetNodeSelection(&serverPod1.Spec, nodeSelection)
		e2epod.NewPodClient(fr).CreateSync(ctx, serverPod1)

		// wait until the endpoints are ready
		validateEndpointsPortsOrFail(ctx, cs, ns, serviceName, portsByPodName{podBackend1: {80}})

		// Note that the fact that Endpoints object already exists, does NOT mean
		// that iptables (or whatever else is used) was already programmed.
		// Additionally take into account that UDP conntract entries timeout is
		// 30 seconds by default.
		// Based on the above check if the pod receives the traffic.
		ginkgo.By("checking client pod connected to the backend on Node IP " + serverNodeInfo.nodeIP)
		if err := wait.PollUntilContextTimeout(ctx, 5*time.Second, time.Minute, true, logContainsFn(podBackend1, podClient)); err != nil {
			logs, err = e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend pod")
		}

	})

	// Regression test for #74839
	// Packets considered INVALID by conntrack are not NATed, this can result
	// in a problem where spurious retransmits in a long-running TCP connection
	// to a service IP ends up with "Connection reset by peer" error.
	// Proxy implementations (which leverage conntrack) can either drop packets
	// marked INVALID by conntrack or enforce `nf_conntrack_tcp_be_liberal` to
	// overcome this.
	// xref: https://kubernetes.io/blog/2019/03/29/kube-proxy-subtleties-debugging-an-intermittent-connection-reset/
	ginkgo.It("proxy implementation should not be vulnerable to the invalid conntrack state bug [Privileged]", func(ctx context.Context) {
		serverLabel := map[string]string{
			"app": "boom-server",
		}

		serverPod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "boom-server",
				Labels: serverLabel,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "boom-server",
						Image: imageutils.GetE2EImage(imageutils.RegressionIssue74839),
						Ports: []v1.ContainerPort{
							{
								ContainerPort: 9000, // Default port exposed by boom-server
							},
						},
						Env: []v1.EnvVar{
							{
								Name: "POD_IP",
								ValueFrom: &v1.EnvVarSource{
									FieldRef: &v1.ObjectFieldSelector{
										APIVersion: "v1",
										FieldPath:  "status.podIP",
									},
								},
							},
							{
								Name: "POD_IPS",
								ValueFrom: &v1.EnvVarSource{
									FieldRef: &v1.ObjectFieldSelector{
										APIVersion: "v1",
										FieldPath:  "status.podIPs",
									},
								},
							},
						},
						SecurityContext: &v1.SecurityContext{
							Capabilities: &v1.Capabilities{
								Add: []v1.Capability{"NET_RAW"},
							},
						},
					},
				},
			},
		}
		nodeSelection := e2epod.NodeSelection{Name: serverNodeInfo.name}
		e2epod.SetNodeSelection(&serverPod.Spec, nodeSelection)
		e2epod.NewPodClient(fr).CreateSync(ctx, serverPod)
		ginkgo.By("Server pod created on node " + serverNodeInfo.name)

		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "boom-server",
			},
			Spec: v1.ServiceSpec{
				Selector: serverLabel,
				Ports: []v1.ServicePort{
					{
						Protocol: v1.ProtocolTCP,
						Port:     9000,
					},
				},
			},
		}
		_, err := fr.ClientSet.CoreV1().Services(fr.Namespace.Name).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Server service created")

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "startup-script",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "startup-script",
						Image: imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{
							"sh", "-c", "while true; do sleep 2; nc boom-server 9000& done",
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}
		nodeSelection = e2epod.NodeSelection{Name: clientNodeInfo.name}
		e2epod.SetNodeSelection(&pod.Spec, nodeSelection)

		e2epod.NewPodClient(fr).CreateSync(ctx, pod)
		ginkgo.By("Client pod created")

		// The client will open connections against the server.
		// The server will inject packets with out-of-window sequence numbers and
		// if these packets go without NAT client will receive an unexpected TCP
		// packet and RST the connection, the server will log ERROR if that happens.
		ginkgo.By("checking client pod does not RST the TCP connection because it receives an out-of-window packet")
		if err := wait.PollUntilContextTimeout(ctx, 5*time.Second, time.Minute, true, logContainsFn("ERROR", "boom-server")); err == nil {
			logs, err := e2epod.GetPodLogs(ctx, cs, ns, "boom-server", "boom-server")
			framework.ExpectNoError(err)
			framework.Logf("boom-server pod logs: %s", logs)
			framework.Failf("boom-server pod received a RST from the client, enabling nf_conntrack_tcp_be_liberal or dropping packets marked invalid by conntrack might help here.")
		}

		logs, err := e2epod.GetPodLogs(ctx, cs, ns, "boom-server", "boom-server")
		framework.ExpectNoError(err)
		if !strings.Contains(logs, "connection established") {
			framework.Logf("boom-server pod logs: %s", logs)
			framework.Failf("boom-server pod did not send any bad packet to the client")
		}
		framework.Logf("boom-server pod logs: %s", logs)
		framework.Logf("boom-server OK: did not receive any RST packet")
	})

	// This test checks that conntrack entries for old target ports are cleaned up when a UDP Service’s target port changes.
	// It also covers cases where it prints the serverport entries, to make sure cleanup works correctly.
	// This verifies the fix in https://pr.k8s.io/130542
	ginkgo.It("should be able to cleanup conntrack entries when UDP service target port changes for a NodePort service", func(ctx context.Context) {
		// Create a NodePort service
		udpJig := e2eservice.NewTestJig(cs, ns, serviceName)
		ginkgo.By("creating a UDP service " + serviceName + " with type=NodePort in " + ns)
		udpService, err := udpJig.CreateUDPService(ctx, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "udp", Protocol: v1.ProtocolUDP, TargetPort: intstr.FromInt32(8080)},
			}
		})
		framework.ExpectNoError(err)
		framework.Logf("Older Target Port: %v", udpService.Spec.Ports[0].TargetPort)

		// Create a pod in one node to create the UDP traffic against the NodePort service every 5 seconds
		ginkgo.By("creating a client pod for probing the service " + serviceName)
		clientPod := e2epod.NewAgnhostPod(ns, podClient, nil, nil, nil)
		nodeSelection := e2epod.NodeSelection{Name: clientNodeInfo.name}
		e2epod.SetNodeSelection(&clientPod.Spec, nodeSelection)
		cmd := fmt.Sprintf(`date; for i in $(seq 1 3000); do echo "$(date) Try: ${i}"; echo serverport | nc -u -w 5 -p %d %s %d; echo; done`, srcPort, serverNodeInfo.nodeIP, udpService.Spec.Ports[0].NodePort)
		clientPod.Spec.Containers[0].Command = []string{"/bin/sh", "-c", cmd}
		clientPod.Spec.Containers[0].Name = podClient
		e2epod.NewPodClient(fr).CreateSync(ctx, clientPod)

		// Add a backend pod to the service in the other node
		ginkgo.By("creating a backend pod " + podBackend1 + " for the service " + serviceName)

		// Add a backend pod with containers port
		port8080 := []v1.ContainerPort{
			{
				ContainerPort: 8080,
				Protocol:      v1.ProtocolUDP,
			},
		}
		port9090 := []v1.ContainerPort{
			{
				ContainerPort: 9090,
				Protocol:      v1.ProtocolUDP,
			},
		}
		serverPod := e2epod.NewAgnhostPodFromContainers(
			"", "conntrack-handle-target-ports", nil,
			e2epod.NewAgnhostContainer("container-handle-8080-request", nil, port8080, "netexec", "--http-port", "8080", "--udp-port", "8080"),
			e2epod.NewAgnhostContainer("container-handle-9090-request", nil, port9090, "netexec", "--http-port", "9090", "--udp-port", "9090"),
		)
		serverPod.Labels = udpJig.Labels
		nodeSelection = e2epod.NodeSelection{Name: serverNodeInfo.name}
		e2epod.SetNodeSelection(&serverPod.Spec, nodeSelection)
		e2epod.NewPodClient(fr).CreateSync(ctx, serverPod)

		validateEndpointsPortsOrFail(ctx, cs, ns, serviceName, portsByPodName{serverPod.Name: {8080}})

		// Check that the first container of server pod keeps receiving traffic
		// UDP conntrack entries timeout is 30 sec by default
		ginkgo.By("checking client pod connected to the container 1 of backend pod on Node IP " + serverNodeInfo.nodeIP)
		if err := wait.PollUntilContextTimeout(ctx, 5*time.Second, time.Minute, true, logContainsFn("8080", podClient)); err != nil {
			logs, err := e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 2")
		}

		// Update the NodePort Service TargetPort to 9090
		service, err := udpJig.UpdateService(ctx, func(svc *v1.Service) {
			svc.Spec.Ports[0].TargetPort = intstr.FromInt32(9090)
		})
		framework.ExpectNoError(err)
		framework.Logf("Updated Target Port: %v", service.Spec.Ports[0].TargetPort)

		validateEndpointsPortsOrFail(ctx, cs, ns, serviceName, portsByPodName{serverPod.Name: {9090}})

		// Check that the second container of server pod keeps receiving traffic after clearing up the conntrack entries
		// UDP conntrack entries timeout is 30 sec by default
		// After clearing entries it should validate it serverport on NodePort matches with the container port
		ginkgo.By("checking client pod connected to the container 2 of backend pod on Node IP " + serverNodeInfo.nodeIP)
		if err := wait.PollUntilContextTimeout(ctx, 5*time.Second, time.Minute, true, logContainsFn("9090", podClient)); err != nil {
			logs, err := e2epod.GetPodLogs(ctx, cs, ns, podClient, podClient)
			framework.ExpectNoError(err)
			framework.Logf("Pod client logs: %s", logs)
			framework.Failf("Failed to connect to backend 2")
		}
	})
})
