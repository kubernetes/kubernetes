/*
Copyright 2016 The Kubernetes Authors.

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
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eendpointslice "k8s.io/kubernetes/test/e2e/framework/endpointslice"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2eregistry "k8s.io/kubernetes/test/e2e/framework/registry"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	netutils "k8s.io/utils/net"
)

// expandIPv6ForConntrack expands an IPv6 address to the format used in /proc/net/nf_conntrack.
// e.g., "fc00:f853:ccd:e793::3" -> "fc00:f853:0ccd:e793:0000:0000:0000:0003"
func expandIPv6ForConntrack(ipStr string) string {
	ip := netutils.ParseIPSloppy(ipStr)
	if !netutils.IsIPv6(ip) {
		return ipStr
	}
	return fmt.Sprintf("%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x",
		ip[0], ip[1], ip[2], ip[3], ip[4], ip[5], ip[6], ip[7],
		ip[8], ip[9], ip[10], ip[11], ip[12], ip[13], ip[14], ip[15])
}

var kubeProxyE2eImage = imageutils.GetE2EImage(imageutils.Agnhost)

var _ = common.SIGDescribe("KubeProxy", func() {
	const (
		testDaemonTCPPort     = 11302
		postFinTimeoutSeconds = 30
	)

	fr := framework.NewDefaultFramework("kube-proxy")
	fr.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should set TCP CLOSE_WAIT timeout [Privileged]", func(ctx context.Context) {
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, fr.ClientSet, 2)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf(
				"Test requires >= 2 Ready nodes, but there are only %v nodes",
				len(nodes.Items))
		}

		type NodeInfo struct {
			node   *v1.Node
			name   string
			nodeIP string
		}

		var family v1.IPFamily
		if framework.TestContext.ClusterIsIPv6() {
			family = v1.IPv6Protocol
		} else {
			family = v1.IPv4Protocol
		}

		ips := e2enode.GetAddressesByTypeAndFamily(&nodes.Items[0], v1.NodeInternalIP, family)
		gomega.Expect(ips).ToNot(gomega.BeEmpty())

		clientNodeInfo := NodeInfo{
			node:   &nodes.Items[0],
			name:   nodes.Items[0].Name,
			nodeIP: ips[0],
		}

		ips = e2enode.GetAddressesByTypeAndFamily(&nodes.Items[1], v1.NodeInternalIP, family)
		gomega.Expect(ips).ToNot(gomega.BeEmpty())

		serverNodeInfo := NodeInfo{
			node:   &nodes.Items[1],
			name:   nodes.Items[1].Name,
			nodeIP: ips[0],
		}

		// Create a pod to check the conntrack entries on the host node
		privileged := true

		hostExecPod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "e2e-net-exec",
				Namespace: fr.Namespace.Name,
				Labels:    map[string]string{"app": "e2e-net-exec"},
			},
			Spec: v1.PodSpec{
				HostNetwork: true,
				NodeName:    clientNodeInfo.name,
				Containers: []v1.Container{
					{
						Name:            "e2e-net-exec",
						Image:           imageutils.GetE2EImage(imageutils.DistrolessIptables),
						ImagePullPolicy: v1.PullIfNotPresent,
						Command:         []string{"sleep", "600"},
						SecurityContext: &v1.SecurityContext{
							Privileged: &privileged,
						},
					},
				},
			},
		}
		e2epod.NewPodClient(fr).CreateSync(ctx, hostExecPod)

		// Create the client and server pods
		clientPodSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "e2e-net-client",
				Namespace: fr.Namespace.Name,
				Labels:    map[string]string{"app": "e2e-net-client"},
			},
			Spec: v1.PodSpec{
				NodeName: clientNodeInfo.name,
				Containers: []v1.Container{
					{
						Name:            "e2e-net-client",
						Image:           kubeProxyE2eImage,
						ImagePullPolicy: v1.PullIfNotPresent,
						Args: []string{
							"net",
							"--runner", "nat-closewait-client",
							"--options",
							fmt.Sprintf(`{"RemoteAddr":"%v", "PostFinTimeoutSeconds":%v, "TimeoutSeconds":%v, "LeakConnection":true}`,
								net.JoinHostPort(serverNodeInfo.nodeIP, strconv.Itoa(testDaemonTCPPort)),
								postFinTimeoutSeconds,
								0),
						},
					},
				},
			},
		}

		serverPodSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "e2e-net-server",
				Namespace: fr.Namespace.Name,
				Labels:    map[string]string{"app": "e2e-net-server"},
			},
			Spec: v1.PodSpec{
				NodeName: serverNodeInfo.name,
				Containers: []v1.Container{
					{
						Name:            "e2e-net-server",
						Image:           kubeProxyE2eImage,
						ImagePullPolicy: v1.PullIfNotPresent,
						Args: []string{
							"net",
							"--runner", "nat-closewait-server",
							"--options",
							fmt.Sprintf(`{"LocalAddr":":%v", "PostFinTimeoutSeconds":%v}`,
								testDaemonTCPPort,
								postFinTimeoutSeconds),
						},
						Ports: []v1.ContainerPort{
							{
								Name:          "tcp",
								ContainerPort: testDaemonTCPPort,
								HostPort:      testDaemonTCPPort,
							},
						},
					},
				},
			},
		}

		ginkgo.By(fmt.Sprintf(
			"Launching a server daemon on node %v (node ip: %v, image: %v)",
			serverNodeInfo.name,
			serverNodeInfo.nodeIP,
			kubeProxyE2eImage))
		e2epod.NewPodClient(fr).CreateSync(ctx, serverPodSpec)

		// The server should be listening before spawning the client pod
		if readyErr := e2epod.WaitTimeoutForPodReadyInNamespace(ctx, fr.ClientSet, serverPodSpec.Name, fr.Namespace.Name, framework.PodStartTimeout); readyErr != nil {
			framework.Failf("error waiting for server pod %s to be ready: %v", serverPodSpec.Name, readyErr)
		}
		// Connect to the server and leak the connection
		ginkgo.By(fmt.Sprintf(
			"Launching a client connection on node %v (node ip: %v, image: %v)",
			clientNodeInfo.name,
			clientNodeInfo.nodeIP,
			kubeProxyE2eImage))
		e2epod.NewPodClient(fr).CreateSync(ctx, clientPodSpec)

		ginkgo.By("Checking conntrack entries for the timeout")
		const epsilonSeconds = 60
		const expectedTimeoutSeconds = 60 * 60

		// Detect conntrack method and build command
		ip := serverNodeInfo.nodeIP
		ipFamily := "ipv4"
		if netutils.IsIPv6String(ip) {
			ipFamily = "ipv6"
		}

		var cmd, dumpCmd string
		var timeoutIdx int
		if _, err := e2epodoutput.RunHostCmd(fr.Namespace.Name, "e2e-net-exec", "test -f /proc/net/nf_conntrack"); err == nil {
			procIP := ip
			if ipFamily == "ipv6" {
				procIP = expandIPv6ForConntrack(ip)
			}
			cmd = fmt.Sprintf("cat /proc/net/nf_conntrack | grep -m 1 -E 'CLOSE_WAIT.*dst=%s.*dport=%d'", procIP, testDaemonTCPPort)
			dumpCmd = "cat /proc/net/nf_conntrack"
			timeoutIdx = 4 // ipv4 2 tcp 6 <timeout> CLOSE_WAIT ...
		} else if _, err := e2epodoutput.RunHostCmd(fr.Namespace.Name, "e2e-net-exec", "which conntrack"); err == nil {
			cmd = fmt.Sprintf("conntrack -L -f %s -d %s 2>/dev/null | grep -m 1 'CLOSE_WAIT.*dport=%d'", ipFamily, ip, testDaemonTCPPort)
			dumpCmd = "conntrack -L 2>/dev/null"
			timeoutIdx = 2 // tcp 6 <timeout> CLOSE_WAIT ...
		} else {
			e2eskipper.Skipf("Neither /proc/net/nf_conntrack nor conntrack binary available")
		}

		if err := wait.PollImmediate(2*time.Second, epsilonSeconds*time.Second, func() (bool, error) {
			result, err := e2epodoutput.RunHostCmd(fr.Namespace.Name, "e2e-net-exec", cmd)
			if err != nil {
				framework.Logf("failed to obtain conntrack entry: %v", err)
				return false, nil
			}
			fields := strings.Fields(result)
			if len(fields) <= timeoutIdx {
				return false, nil
			}
			timeoutSeconds, err := strconv.Atoi(fields[timeoutIdx])
			if err != nil {
				return false, nil
			}
			framework.Logf("conntrack timeout for %v:%v = %v", serverNodeInfo.nodeIP, testDaemonTCPPort, timeoutSeconds)
			if math.Abs(float64(timeoutSeconds-expectedTimeoutSeconds)) < epsilonSeconds {
				return true, nil
			}
			return false, fmt.Errorf("wrong TCP CLOSE_WAIT timeout: %v expected: %v", timeoutSeconds, expectedTimeoutSeconds)
		}); err != nil {
			result, _ := e2epodoutput.RunHostCmd(fr.Namespace.Name, "e2e-net-exec", dumpCmd)
			framework.Logf("conntrack entries: %v", result)
			framework.Failf("no valid conntrack entry for port %d on node %s: %v", testDaemonTCPPort, serverNodeInfo.nodeIP, err)
		}
	})

	framework.It("should update metric for tracking accepted packets destined for localhost nodeports", feature.KubeProxyNFAcct, func(ctx context.Context) {
		if framework.TestContext.ClusterIsIPv6() {
			e2eskipper.Skipf("test requires IPv4 cluster")
		}

		cs := fr.ClientSet
		ns := fr.Namespace.Name

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 1)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 1 {
			e2eskipper.Skipf(
				"Test requires >= 1 Ready nodes, but there are only %v nodes",
				len(nodes.Items))
		}
		nodeName := nodes.Items[0].Name

		metricName := "kubeproxy_iptables_localhost_nodeports_accepted_packets_total"
		metricsGrabber, err := e2emetrics.NewMetricsGrabber(ctx, fr.ClientSet, nil, fr.ClientConfig(), false, false, false, false, false, false)
		framework.ExpectNoError(err)

		// create a pod with host-network for execing
		hostExecPodName := "host-exec-pod"
		hostExecPod := e2epod.NewExecPodSpec(fr.Namespace.Name, hostExecPodName, true)
		nodeSelection := e2epod.NodeSelection{Name: nodeName}
		e2epod.SetNodeSelection(&hostExecPod.Spec, nodeSelection)
		e2epod.NewPodClient(fr).CreateSync(ctx, hostExecPod)

		// get proxyMode
		stdout, err := e2epodoutput.RunHostCmd(fr.Namespace.Name, hostExecPodName, fmt.Sprintf("curl --silent 127.0.0.1:%d/proxyMode", ports.ProxyStatusPort))
		if err != nil {
			framework.Failf("failed to get proxy mode: err: %v; stdout: %s", stdout, err)
		}
		proxyMode := strings.TrimSpace(stdout)

		// get value of route_localnet
		stdout, err = e2epodoutput.RunHostCmd(fr.Namespace.Name, hostExecPodName, "cat /proc/sys/net/ipv4/conf/all/route_localnet")
		framework.ExpectNoError(err)
		routeLocalnet := strings.TrimSpace(stdout)

		if !(proxyMode == string(config.ProxyModeIPTables) && routeLocalnet == "1") {
			e2eskipper.Skipf("test requires iptables proxy mode with route_localnet set")
		}

		// get value of target metric before accessing localhost nodeports
		metrics, err := metricsGrabber.GrabFromKubeProxy(ctx, nodeName)
		framework.ExpectNoError(err)
		targetMetricBefore, err := metrics.GetCounterMetricValue(metricName)
		framework.ExpectNoError(err)

		// create pod
		ginkgo.By("creating test pod")
		label := map[string]string{
			"app": "agnhost-localhost-nodeports",
		}
		httpPort := []v1.ContainerPort{
			{
				ContainerPort: 8080,
				Protocol:      v1.ProtocolTCP,
			},
		}
		pod := e2epod.NewAgnhostPod(ns, "agnhost-localhost-nodeports", nil, nil, httpPort, "netexec")
		pod.Labels = label
		e2epod.NewPodClient(fr).CreateSync(ctx, pod)

		// create nodeport service
		ginkgo.By("creating test nodeport service")
		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "agnhost-localhost-nodeports",
			},
			Spec: v1.ServiceSpec{
				Type:     v1.ServiceTypeNodePort,
				Selector: label,
				Ports: []v1.ServicePort{
					{
						Protocol:   v1.ProtocolTCP,
						Port:       9000,
						TargetPort: intstr.IntOrString{Type: 0, IntVal: 8080},
					},
				},
			},
		}
		svc, err = fr.ClientSet.CoreV1().Services(fr.Namespace.Name).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		// wait for endpoints update
		ginkgo.By("waiting for endpoints to be updated")
		err = e2eendpointslice.WaitForEndpointCount(ctx, fr.ClientSet, ns, svc.Name, 1)
		framework.ExpectNoError(err)

		ginkgo.By("accessing endpoint via localhost nodeports 10 times")
		for range 10 {
			if err := wait.PollUntilContextTimeout(ctx, 1*time.Second, 10*time.Second, true, func(_ context.Context) (bool, error) {
				_, err = e2epodoutput.RunHostCmd(fr.Namespace.Name, hostExecPodName, fmt.Sprintf("curl --silent http://localhost:%d/hostname", svc.Spec.Ports[0].NodePort))
				if err != nil {
					return false, nil
				}
				return true, nil
			}); err != nil {
				framework.ExpectNoError(err, "failed to access nodeport service on localhost")
			}
		}

		// our target metric should be updated by now
		if err := wait.PollUntilContextTimeout(ctx, 10*time.Second, 2*time.Minute, true, func(_ context.Context) (bool, error) {
			metrics, err := metricsGrabber.GrabFromKubeProxy(ctx, nodeName)
			if err != nil {
				return false, fmt.Errorf("failed to fetch metrics: %w", err)
			}
			targetMetricAfter, err := metrics.GetCounterMetricValue(metricName)
			if err != nil {
				return false, fmt.Errorf("failed to fetch metric: %w", err)
			}
			return targetMetricAfter > targetMetricBefore, nil
		}); err != nil {
			if wait.Interrupted(err) {
				framework.Failf("expected %s metric to be updated after accessing endpoints via localhost nodeports", metricName)
			}
			framework.ExpectNoError(err)
		}
	})

	loopbackURL := func(port int32, path string) string {
		if framework.TestContext.ClusterIsIPv6() {
			return fmt.Sprintf("http://[::1]:%d%s", port, path)
		}
		return fmt.Sprintf("http://127.0.0.1:%d%s", port, path)
	}

	framework.It("should proxy localhost NodePort traffic to backends across nodes", feature.LocalhostNodePorts, func(ctx context.Context) {
		cs := fr.ClientSet
		ns := fr.Namespace.Name

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 2)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf("Test requires >= 2 Ready nodes, but there are only %v nodes", len(nodes.Items))
		}
		clientNodeName := nodes.Items[0].Name
		remoteNodeName := nodes.Items[1].Name

		hostExecPodName := "host-exec-pod"
		hostExecPod := e2epod.NewExecPodSpec(ns, hostExecPodName, true)
		e2epod.SetNodeSelection(&hostExecPod.Spec, e2epod.NodeSelection{Name: clientNodeName})
		e2epod.NewPodClient(fr).CreateSync(ctx, hostExecPod)

		ginkgo.By("creating one backend pod on the client node and one on a different node")
		label := map[string]string{"app": "agnhost-localhost-nodeport"}
		httpPort := []v1.ContainerPort{{ContainerPort: 8080, Protocol: v1.ProtocolTCP}}
		backendNames := sets.New[string]()
		for _, bp := range []struct {
			name string
			node string
		}{
			{"agnhost-localhost-nodeport-local", clientNodeName},
			{"agnhost-localhost-nodeport-remote", remoteNodeName},
		} {
			backendNames.Insert(bp.name)
			pod := e2epod.NewAgnhostPod(ns, bp.name, nil, nil, httpPort, "netexec")
			pod.Labels = label
			e2epod.SetNodeSelection(&pod.Spec, e2epod.NodeSelection{Name: bp.node})
			e2epod.NewPodClient(fr).CreateSync(ctx, pod)
		}

		ginkgo.By("creating NodePort service")
		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{Name: "agnhost-localhost-nodeport"},
			Spec: v1.ServiceSpec{
				Type:     v1.ServiceTypeNodePort,
				Selector: label,
				Ports: []v1.ServicePort{{
					Protocol:   v1.ProtocolTCP,
					Port:       9000,
					TargetPort: intstr.FromInt32(8080),
				}},
			},
		}
		svc, err = cs.CoreV1().Services(ns).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("waiting for endpoints")
		framework.ExpectNoError(e2eendpointslice.WaitForEndpointCount(ctx, cs, ns, svc.Name, 2))

		ginkgo.By("curling localhost NodePort from the node's netns until both backends have responded")
		nodePort := svc.Spec.Ports[0].NodePort
		cmd := fmt.Sprintf("curl --silent --max-time 5 %s", loopbackURL(nodePort, "/hostname"))
		seen := sets.New[string]()
		if err := wait.PollUntilContextTimeout(ctx, 1*time.Second, 60*time.Second, true, func(_ context.Context) (bool, error) {
			out, err := e2epodoutput.RunHostCmd(ns, hostExecPodName, cmd)
			if err != nil {
				return false, nil
			}
			hostname := strings.TrimSpace(out)
			if hostname == "" {
				return false, nil
			}
			if !backendNames.Has(hostname) {
				return false, fmt.Errorf("response %q is not one of the backend pods %v", hostname, backendNames.UnsortedList())
			}
			seen.Insert(hostname)
			return seen.Equal(backendNames), nil
		}); err != nil {
			framework.Failf("expected localhost NodePort %d to reach both backends %v, only saw %v: %v", nodePort, backendNames.UnsortedList(), seen.UnsortedList(), err)
		}
	})

	framework.It("should honor sessionAffinity=ClientIP for localhost NodePort", feature.LocalhostNodePorts, func(ctx context.Context) {
		cs := fr.ClientSet
		ns := fr.Namespace.Name

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 1)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 1 {
			e2eskipper.Skipf("Test requires >= 1 Ready nodes, but there are only %v nodes", len(nodes.Items))
		}
		nodeName := nodes.Items[0].Name

		hostExecPodName := "host-exec-pod"
		hostExecPod := e2epod.NewExecPodSpec(ns, hostExecPodName, true)
		e2epod.SetNodeSelection(&hostExecPod.Spec, e2epod.NodeSelection{Name: nodeName})
		e2epod.NewPodClient(fr).CreateSync(ctx, hostExecPod)

		ginkgo.By("creating 3 backend pods pinned to the same node")
		label := map[string]string{"app": "agnhost-sticky-localhost"}
		httpPort := []v1.ContainerPort{{ContainerPort: 8080, Protocol: v1.ProtocolTCP}}
		backendNames := sets.New[string]()
		for i := range 3 {
			name := fmt.Sprintf("agnhost-sticky-localhost-%d", i)
			backendNames.Insert(name)
			pod := e2epod.NewAgnhostPod(ns, name, nil, nil, httpPort, "netexec")
			pod.Labels = label
			e2epod.SetNodeSelection(&pod.Spec, e2epod.NodeSelection{Name: nodeName})
			e2epod.NewPodClient(fr).CreateSync(ctx, pod)
		}

		ginkgo.By("creating NodePort service with sessionAffinity=ClientIP")
		timeoutSeconds := int32(10800)
		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{Name: "agnhost-sticky-localhost"},
			Spec: v1.ServiceSpec{
				Type:            v1.ServiceTypeNodePort,
				Selector:        label,
				SessionAffinity: v1.ServiceAffinityClientIP,
				SessionAffinityConfig: &v1.SessionAffinityConfig{
					ClientIP: &v1.ClientIPConfig{TimeoutSeconds: &timeoutSeconds},
				},
				Ports: []v1.ServicePort{{
					Protocol:   v1.ProtocolTCP,
					Port:       9000,
					TargetPort: intstr.FromInt32(8080),
				}},
			},
		}
		svc, err = cs.CoreV1().Services(ns).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("waiting for all 3 endpoints to be ready")
		framework.ExpectNoError(e2eendpointslice.WaitForEndpointCount(ctx, cs, ns, svc.Name, 3))

		ginkgo.By("curling localhost NodePort 10 times; all must hit the same pod")
		nodePort := svc.Spec.Ports[0].NodePort
		cmd := fmt.Sprintf("curl --silent --max-time 5 %s", loopbackURL(nodePort, "/hostname"))

		// Retry the first curl until the listener is reachable, then pin.
		var first string
		if err := wait.PollUntilContextTimeout(ctx, 1*time.Second, 10*time.Second, true, func(_ context.Context) (bool, error) {
			out, err := e2epodoutput.RunHostCmd(ns, hostExecPodName, cmd)
			if err != nil {
				return false, nil
			}
			first = strings.TrimSpace(out)
			return first != "", nil
		}); err != nil {
			framework.Failf("initial curl to localhost:%d failed: %v", nodePort, err)
		}
		if !backendNames.Has(first) {
			framework.Failf("first response %q is not one of the backend pods %v", first, backendNames.UnsortedList())
		}

		for i := range 9 {
			out, err := e2epodoutput.RunHostCmd(ns, hostExecPodName, cmd)
			framework.ExpectNoError(err, "curl %d failed", i+2)
			got := strings.TrimSpace(out)
			gomega.Expect(got).To(gomega.Equal(first),
				"sessionAffinity=ClientIP must pin localhost traffic to a single backend (request %d got %q, first was %q)", i+2, got, first)
		}
	})

	framework.It("should let a node pull an image from a registry exposed via localhost NodePort", feature.LocalhostNodePorts, func(ctx context.Context) {
		cs := fr.ClientSet
		ns := fr.Namespace.Name

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 2)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf("Test requires >= 2 Ready nodes, but there are only %v nodes", len(nodes.Items))
		}
		clientNodeName := nodes.Items[0].Name
		registryNodeName := nodes.Items[1].Name

		ginkgo.By("running a private fake registry on a different node, behind a NodePort service")
		label := map[string]string{"app": "localhost-nodeport-registry"}
		registryPorts := []v1.ContainerPort{{ContainerPort: 5000, Protocol: v1.ProtocolTCP}}
		registryPod := e2epod.NewAgnhostPod(ns, "localhost-nodeport-registry", nil, nil, registryPorts, "fake-registry-server", "--private")
		registryPod.Labels = label
		e2epod.SetNodeSelection(&registryPod.Spec, e2epod.NodeSelection{Name: registryNodeName})
		e2epod.NewPodClient(fr).CreateSync(ctx, registryPod)

		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{Name: "localhost-nodeport-registry"},
			Spec: v1.ServiceSpec{
				Type:     v1.ServiceTypeNodePort,
				Selector: label,
				Ports: []v1.ServicePort{{
					Protocol:   v1.ProtocolTCP,
					Port:       5000,
					TargetPort: intstr.FromInt32(5000),
				}},
			},
		}
		svc, err = cs.CoreV1().Services(ns).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("waiting for the registry endpoint")
		framework.ExpectNoError(e2eendpointslice.WaitForEndpointCount(ctx, cs, ns, svc.Name, 1))

		nodePort := svc.Spec.Ports[0].NodePort
		registryAddress := fmt.Sprintf("localhost:%d", nodePort)

		ginkgo.By("creating the image pull secret for the localhost registry")
		secret := e2eregistry.User1DockerSecret(registryAddress)
		secret.Name = "localhost-nodeport-registry-creds-" + string(uuid.NewUUID())
		_, err = cs.CoreV1().Secrets(ns).Create(ctx, secret, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("pulling an image through the localhost NodePort and running it")
		pullPod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "localhost-nodeport-registry-client"},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{
					Name:            "client",
					Image:           registryAddress + "/pause:testing",
					ImagePullPolicy: v1.PullAlways,
				}},
				ImagePullSecrets: []v1.LocalObjectReference{{Name: secret.Name}},
				RestartPolicy:    v1.RestartPolicyNever,
			},
		}
		e2epod.SetNodeSelection(&pullPod.Spec, e2epod.NodeSelection{Name: clientNodeName})
		pullPod = e2epod.NewPodClient(fr).Create(ctx, pullPod)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, pullPod))
	})
})
