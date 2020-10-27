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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"

	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	netutils "k8s.io/utils/net"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var kubeProxyE2eImage = imageutils.GetE2EImage(imageutils.Agnhost)

var _ = SIGDescribe("Network", func() {
	const (
		testDaemonHTTPPort    = 11301
		testDaemonTCPPort     = 11302
		postFinTimeoutSeconds = 30
	)

	fr := framework.NewDefaultFramework("network")

	ginkgo.It("should set TCP CLOSE_WAIT timeout [Privileged]", func() {
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(fr.ClientSet, 2)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf(
				"Test requires >= 2 Ready nodes, but there are only %v nodes",
				len(nodes.Items))
		}

		ips := e2enode.CollectAddresses(nodes, v1.NodeInternalIP)

		type NodeInfo struct {
			node   *v1.Node
			name   string
			nodeIP string
		}

		clientNodeInfo := NodeInfo{
			node:   &nodes.Items[0],
			name:   nodes.Items[0].Name,
			nodeIP: ips[0],
		}

		serverNodeInfo := NodeInfo{
			node:   &nodes.Items[1],
			name:   nodes.Items[1].Name,
			nodeIP: ips[1],
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
						Image:           imageutils.GetE2EImage(imageutils.DebianIptables),
						ImagePullPolicy: v1.PullIfNotPresent,
						Command:         []string{"sleep", "600"},
						SecurityContext: &v1.SecurityContext{
							Privileged: &privileged,
						},
					},
				},
			},
		}
		fr.PodClient().CreateSync(hostExecPod)

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
		fr.PodClient().CreateSync(serverPodSpec)

		// The server should be listening before spawning the client pod
		if readyErr := e2epod.WaitForPodsReady(fr.ClientSet, fr.Namespace.Name, serverPodSpec.Name, 0); readyErr != nil {
			framework.Failf("error waiting for server pod %s to be ready: %v", serverPodSpec.Name, readyErr)
		}
		// Connect to the server and leak the connection
		ginkgo.By(fmt.Sprintf(
			"Launching a client connection on node %v (node ip: %v, image: %v)",
			clientNodeInfo.name,
			clientNodeInfo.nodeIP,
			kubeProxyE2eImage))
		fr.PodClient().CreateSync(clientPodSpec)

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
		cmd := fmt.Sprintf("conntrack -L -f %s -d %v"+
			"| grep -m 1 'CLOSE_WAIT.*dport=%v' ",
			ipFamily, ip, testDaemonTCPPort)
		if err := wait.PollImmediate(1*time.Second, postFinTimeoutSeconds, func() (bool, error) {
			result, err := framework.RunHostCmd(fr.Namespace.Name, "e2e-net-exec", cmd)
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
				return false, fmt.Errorf("failed to convert matched timeout %s to integer: %v", line[2], err)
			}
			if math.Abs(float64(timeoutSeconds-expectedTimeoutSeconds)) < epsilonSeconds {
				return true, nil
			}
			return false, fmt.Errorf("wrong TCP CLOSE_WAIT timeout: %v expected: %v", timeoutSeconds, expectedTimeoutSeconds)
		}); err != nil {
			framework.Failf("no conntrack entry for port %d on node %s", testDaemonTCPPort, serverNodeInfo.nodeIP)
		}
	})

	// Regression test for #74839, where:
	// Packets considered INVALID by conntrack are now dropped. In particular, this fixes
	// a problem where spurious retransmits in a long-running TCP connection to a service
	// IP could result in the connection being closed with the error "Connection reset by
	// peer"
	ginkgo.It("should resolve connection reset issue #74839 [Slow]", func() {
		serverLabel := map[string]string{
			"app": "boom-server",
		}
		clientLabel := map[string]string{
			"app": "client",
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
					},
				},
				Affinity: &v1.Affinity{
					PodAntiAffinity: &v1.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchLabels: clientLabel,
								},
								TopologyKey: "kubernetes.io/hostname",
							},
						},
					},
				},
			},
		}
		_, err := fr.ClientSet.CoreV1().Pods(fr.Namespace.Name).Create(context.TODO(), serverPod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = e2epod.WaitForPodsRunningReady(fr.ClientSet, fr.Namespace.Name, 1, 0, framework.PodReadyBeforeTimeout, map[string]string{})
		framework.ExpectNoError(err)

		ginkgo.By("Server pod created")

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
		_, err = fr.ClientSet.CoreV1().Services(fr.Namespace.Name).Create(context.TODO(), svc, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Server service created")

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "startup-script",
				Labels: clientLabel,
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
				Affinity: &v1.Affinity{
					PodAntiAffinity: &v1.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchLabels: serverLabel,
								},
								TopologyKey: "kubernetes.io/hostname",
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}
		_, err = fr.ClientSet.CoreV1().Pods(fr.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Client pod created")

		for i := 0; i < 20; i++ {
			time.Sleep(3 * time.Second)
			resultPod, err := fr.ClientSet.CoreV1().Pods(fr.Namespace.Name).Get(context.TODO(), serverPod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(resultPod.Status.ContainerStatuses[0].LastTerminationState.Terminated).Should(gomega.BeNil())
		}
	})
})
