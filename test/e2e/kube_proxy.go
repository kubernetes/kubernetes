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

package e2e

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"

	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/images/net/nat"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const kubeProxyE2eImage = "gcr.io/google_containers/e2e-net-amd64:1.0"

var _ = framework.KubeDescribe("Network", func() {
	const (
		testDaemonHttpPort    = 11301
		testDaemonTcpPort     = 11302
		timeoutSeconds        = 10
		postFinTimeoutSeconds = 5
	)

	fr := framework.NewDefaultFramework("network")

	It("should set TCP CLOSE_WAIT timeout", func() {
		nodes := framework.GetReadySchedulableNodesOrDie(fr.ClientSet)
		ips := framework.CollectAddresses(nodes, v1.NodeInternalIP)

		if len(nodes.Items) < 2 {
			framework.Skipf(
				"Test requires >= 2 Ready nodes, but there are only %v nodes",
				len(nodes.Items))
		}

		type NodeInfo struct {
			node   *v1.Node
			name   string
			nodeIp string
		}

		clientNodeInfo := NodeInfo{
			node:   &nodes.Items[0],
			name:   nodes.Items[0].Name,
			nodeIp: ips[0],
		}

		serverNodeInfo := NodeInfo{
			node:   &nodes.Items[1],
			name:   nodes.Items[1].Name,
			nodeIp: ips[1],
		}

		zero := int64(0)

		clientPodSpec := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
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
						ImagePullPolicy: "Always",
						Command: []string{
							"/net", "-serve", fmt.Sprintf("0.0.0.0:%d", testDaemonHttpPort),
						},
					},
				},
				TerminationGracePeriodSeconds: &zero,
			},
		}

		serverPodSpec := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
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
						ImagePullPolicy: "Always",
						Command: []string{
							"/net",
							"-runner", "nat-closewait-server",
							"-options",
							fmt.Sprintf(`{"LocalAddr":"0.0.0.0:%v", "PostFindTimeoutSeconds":%v}`,
								testDaemonTcpPort,
								postFinTimeoutSeconds),
						},
						Ports: []v1.ContainerPort{
							{
								Name:          "tcp",
								ContainerPort: testDaemonTcpPort,
								HostPort:      testDaemonTcpPort,
							},
						},
					},
				},
				TerminationGracePeriodSeconds: &zero,
			},
		}

		By(fmt.Sprintf(
			"Launching a server daemon on node %v (node ip: %v, image: %v)",
			serverNodeInfo.name,
			serverNodeInfo.nodeIp,
			kubeProxyE2eImage))
		fr.PodClient().CreateSync(serverPodSpec)

		By(fmt.Sprintf(
			"Launching a client daemon on node %v (node ip: %v, image: %v)",
			clientNodeInfo.name,
			clientNodeInfo.nodeIp,
			kubeProxyE2eImage))
		fr.PodClient().CreateSync(clientPodSpec)

		By("Make client connect")

		options := nat.CloseWaitClientOptions{
			RemoteAddr: fmt.Sprintf("%v:%v",
				serverNodeInfo.nodeIp, testDaemonTcpPort),
			TimeoutSeconds:        timeoutSeconds,
			PostFinTimeoutSeconds: 0,
			LeakConnection:        true,
		}

		jsonBytes, err := json.Marshal(options)
		cmd := fmt.Sprintf(
			`curl -X POST http://localhost:%v/run/nat-closewait-client -d `+
				`'%v' 2>/dev/null`,
			testDaemonHttpPort,
			string(jsonBytes))
		framework.RunHostCmdOrDie(fr.Namespace.Name, "e2e-net-client", cmd)

		<-time.After(time.Duration(1) * time.Second)

		By("Checking /proc/net/nf_conntrack for the timeout")
		// If test flakes occur here, then this check should be performed
		// in a loop as there may be a race with the client connecting.
		framework.IssueSSHCommandWithResult(
			fmt.Sprintf("sudo cat /proc/net/ip_conntrack | grep 'dport=%v'",
				testDaemonTcpPort),
			framework.TestContext.Provider,
			clientNodeInfo.node)

		// Timeout in seconds is available as the third column from
		// /proc/net/ip_conntrack.
		result, err := framework.IssueSSHCommandWithResult(
			fmt.Sprintf(
				"sudo cat /proc/net/ip_conntrack "+
					"| grep 'CLOSE_WAIT.*dst=%v.*dport=%v' "+
					"| tail -n 1"+
					"| awk '{print $3}' ",
				serverNodeInfo.nodeIp,
				testDaemonTcpPort),
			framework.TestContext.Provider,
			clientNodeInfo.node)
		framework.ExpectNoError(err)

		timeoutSeconds, err := strconv.Atoi(strings.TrimSpace(result.Stdout))
		framework.ExpectNoError(err)

		// These must be synchronized from the default values set in
		// pkg/apis/../defaults.go ConntrackTCPCloseWaitTimeout. The
		// current defaults are hidden in the initialization code.
		const epsilonSeconds = 60
		const expectedTimeoutSeconds = 60 * 60

		framework.Logf("conntrack entry timeout was: %v, expected: %v",
			timeoutSeconds, expectedTimeoutSeconds)

		Expect(math.Abs(float64(timeoutSeconds - expectedTimeoutSeconds))).Should(
			BeNumerically("<", (epsilonSeconds)))
	})
})
