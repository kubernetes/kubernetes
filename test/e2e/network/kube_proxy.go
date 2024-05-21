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
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	netutils "k8s.io/utils/net"
)

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
		// These must be synchronized from the default values set in
		// pkg/apis/../defaults.go ConntrackTCPCloseWaitTimeout. The
		// current defaults are hidden in the initialization code.
		const epsilonSeconds = 60
		const expectedTimeoutSeconds = 60 * 60
		// the conntrack file uses the IPv6 expanded format
		ip := serverNodeInfo.nodeIP
		ipFamily := "ipv4"
		if netutils.IsIPv6String(ip) {
			ipFamily = "ipv6"
		}
		// Obtain the corresponding conntrack entry on the host checking
		// the nf_conntrack file from the pod e2e-net-exec.
		// It retries in a loop if the entry is not found.
		cmd := fmt.Sprintf("conntrack -L -f %s -d %v "+
			"| grep -m 1 'CLOSE_WAIT.*dport=%v' ",
			ipFamily, ip, testDaemonTCPPort)
		if err := wait.PollImmediate(2*time.Second, epsilonSeconds*time.Second, func() (bool, error) {
			result, err := e2epodoutput.RunHostCmd(fr.Namespace.Name, "e2e-net-exec", cmd)
			// retry if we can't obtain the conntrack entry
			if err != nil {
				framework.Logf("failed to obtain conntrack entry: %v %v", result, err)
				return false, nil
			}
			framework.Logf("conntrack entry for node %v and port %v:  %v", serverNodeInfo.nodeIP, testDaemonTCPPort, result)
			// Timeout in seconds is available as the third column of the matched entry
			line := strings.Fields(result)
			if len(line) < 3 {
				return false, fmt.Errorf("conntrack entry does not have a timeout field: %v", line)
			}
			timeoutSeconds, err := strconv.Atoi(line[2])
			if err != nil {
				return false, fmt.Errorf("failed to convert matched timeout %s to integer: %w", line[2], err)
			}
			if math.Abs(float64(timeoutSeconds-expectedTimeoutSeconds)) < epsilonSeconds {
				return true, nil
			}
			return false, fmt.Errorf("wrong TCP CLOSE_WAIT timeout: %v expected: %v", timeoutSeconds, expectedTimeoutSeconds)
		}); err != nil {
			// Dump all conntrack entries for debugging
			result, err2 := e2epodoutput.RunHostCmd(fr.Namespace.Name, "e2e-net-exec", "conntrack -L")
			if err2 != nil {
				framework.Logf("failed to obtain conntrack entry: %v %v", result, err2)
			}
			framework.Logf("conntrack entries for node %v:  %v", serverNodeInfo.nodeIP, result)
			framework.Failf("no valid conntrack entry for port %d on node %s: %v", testDaemonTCPPort, serverNodeInfo.nodeIP, err)
		}
	})

	ginkgo.It("should update metric for tracking accepted packets destined for localhost nodeports", func(ctx context.Context) {
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
		framework.ExpectNoError(err)
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
		targetMetricBefore := metrics.GetCounterMetricValue(metricName)

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
		err = framework.WaitForServiceEndpointsNum(ctx, fr.ClientSet, ns, svc.Name, 1, time.Second, wait.ForeverTestTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("accessing endpoint via localhost nodeports 10 times")
		for i := 0; i < 10; i++ {
			if err := wait.PollUntilContextTimeout(ctx, 1*time.Second, 10*time.Second, true, func(_ context.Context) (bool, error) {
				_, err = e2epodoutput.RunHostCmd(fr.Namespace.Name, hostExecPodName, fmt.Sprintf("curl --silent http://localhost:%d/hostname", svc.Spec.Ports[0].NodePort))
				if err != nil {
					return false, nil
				}
				return true, nil
			}); err != nil {
				e2eskipper.Skipf("skipping test as localhost nodeports are not acceesible in this environment")
			}
		}

		// our target metric should be updated by now
		if err := wait.PollUntilContextTimeout(ctx, 10*time.Second, 2*time.Minute, true, func(_ context.Context) (bool, error) {
			metrics, err := metricsGrabber.GrabFromKubeProxy(ctx, nodeName)
			framework.ExpectNoError(err)
			targetMetricAfter := metrics.GetCounterMetricValue(metricName)
			return targetMetricAfter > targetMetricBefore, nil
		}); err != nil {
			framework.Failf("expected %s metric to be updated after accessing endpoints via localhost nodeports", metricName)
		}
	})
})
