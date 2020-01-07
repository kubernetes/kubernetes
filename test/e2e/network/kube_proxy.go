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
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"

	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/images/agnhost/net/nat"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var kubeProxyE2eImage = imageutils.GetE2EImage(imageutils.Agnhost)

var _ = SIGDescribe("Network", func() {
	const (
		testDaemonHTTPPort     = 11301
		testDaemonTCPPort      = 11302
		deadlineTimeoutSeconds = 10
		postFinTimeoutSeconds  = 30
	)

	fr := framework.NewDefaultFramework("network")

	ginkgo.It("should set TCP CLOSE_WAIT timeout", func() {
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(fr.ClientSet, 2)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			framework.Skipf(
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

		zero := int64(0)

		// Create a pod to check the conntrack entries on the host node
		// It mounts the host /proc/net folder to be able to access
		// the nf_conntrack file with the host conntrack entries
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
						Image:           kubeProxyE2eImage,
						ImagePullPolicy: "Always",
						Args:            []string{"pause"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "proc-net",
								MountPath: "/rootfs/proc/net",
								ReadOnly:  true,
							},
						},
						SecurityContext: &v1.SecurityContext{
							Privileged: &privileged,
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "proc-net",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/proc/net",
							},
						},
					},
				},
				TerminationGracePeriodSeconds: &zero,
			},
		}
		fr.PodClient().CreateSync(hostExecPod)

		// Some distributions (Ubuntu 16.04 etc.) don't support the proc file.
		_, err = framework.RunHostCmd(fr.Namespace.Name, "e2e-net-exec",
			"ls /rootfs/proc/net/nf_conntrack")
		if err != nil && strings.Contains(err.Error(), "No such file or directory") {
			framework.Skipf("The node %s does not support /proc/net/nf_conntrack",
				clientNodeInfo.name)
		}
		framework.ExpectNoError(err)

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
						ImagePullPolicy: "Always",
						Args: []string{
							"net", "--serve", fmt.Sprintf(":%d", testDaemonHTTPPort),
						},
					},
				},
				TerminationGracePeriodSeconds: &zero,
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
						ImagePullPolicy: "Always",
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
				TerminationGracePeriodSeconds: &zero,
			},
		}

		ginkgo.By(fmt.Sprintf(
			"Launching a server daemon on node %v (node ip: %v, image: %v)",
			serverNodeInfo.name,
			serverNodeInfo.nodeIP,
			kubeProxyE2eImage))
		fr.PodClient().CreateSync(serverPodSpec)

		ginkgo.By(fmt.Sprintf(
			"Launching a client daemon on node %v (node ip: %v, image: %v)",
			clientNodeInfo.name,
			clientNodeInfo.nodeIP,
			kubeProxyE2eImage))
		fr.PodClient().CreateSync(clientPodSpec)

		ginkgo.By("Make client connect")

		options := nat.CloseWaitClientOptions{
			RemoteAddr: fmt.Sprintf("%v:%v",
				serverNodeInfo.nodeIP, testDaemonTCPPort),
			TimeoutSeconds:        deadlineTimeoutSeconds,
			PostFinTimeoutSeconds: postFinTimeoutSeconds,
			LeakConnection:        true,
		}

		jsonBytes, err := json.Marshal(options)
		framework.ExpectNoError(err, "could not marshal")

		cmd := fmt.Sprintf(
			`curl -X POST http://localhost:%v/run/nat-closewait-client -d `+
				`'%v' 2>/dev/null`,
			testDaemonHTTPPort,
			string(jsonBytes))
		// Run the closewait command in a subroutine so it keeps waiting during postFinTimeoutSeconds
		// otherwise the pod is deleted and the connection is closed loosing the conntrack entry
		go func() {
			framework.RunHostCmdOrDie(fr.Namespace.Name, "e2e-net-client", cmd)
		}()

		<-time.After(time.Duration(1) * time.Second)

		ginkgo.By("Checking /proc/net/nf_conntrack for the timeout")
		// These must be synchronized from the default values set in
		// pkg/apis/../defaults.go ConntrackTCPCloseWaitTimeout. The
		// current defaults are hidden in the initialization code.
		const epsilonSeconds = 60
		const expectedTimeoutSeconds = 60 * 60
		// Obtain the corresponding conntrack entry on the host checking
		// the nf_conntrack file from the pod e2e-net-exec.
		// It retries in a loop if the entry is not found.
		cmd = fmt.Sprintf("cat /rootfs/proc/net/nf_conntrack "+
			"| grep -m 1 'CLOSE_WAIT.*dst=%v.*dport=%v' ",
			serverNodeInfo.nodeIP,
			testDaemonTCPPort)
		if err := wait.PollImmediate(5*time.Second, 30*time.Second, func() (bool, error) {
			result, err := framework.RunHostCmd(fr.Namespace.Name, "e2e-net-exec", cmd)
			// retry if we can't obtain the conntrack entry
			if err != nil {
				framework.Logf("failed to obtain conntrack entry: %v %v", result, err)
				return false, nil
			}
			framework.Logf("conntrack entry for node %v and port %v:  %v", serverNodeInfo.nodeIP, testDaemonTCPPort, result)
			// Timeout in seconds is available as the fifth column of
			// the matched entry in /proc/net/nf_conntrack.
			line := strings.Fields(result)
			if len(line) < 5 {
				return false, fmt.Errorf("conntrack entry does not have a timeout field: %v", line)
			}
			timeoutSeconds, err := strconv.Atoi(line[4])
			if err != nil {
				return false, fmt.Errorf("failed to convert matched timeout %s to integer: %v", line[4], err)
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
	ginkgo.It("should resolve connrection reset issue #74839 [Slow]", func() {
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
		_, err := fr.ClientSet.CoreV1().Pods(fr.Namespace.Name).Create(serverPod)
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
		_, err = fr.ClientSet.CoreV1().Services(fr.Namespace.Name).Create(svc)
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
						Image: imageutils.GetE2EImage(imageutils.StartupScript),
						Command: []string{
							"bash", "-c", "while true; do sleep 2; nc boom-server 9000& done",
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
		_, err = fr.ClientSet.CoreV1().Pods(fr.Namespace.Name).Create(pod)
		framework.ExpectNoError(err)

		ginkgo.By("Client pod created")

		for i := 0; i < 20; i++ {
			time.Sleep(3 * time.Second)
			resultPod, err := fr.ClientSet.CoreV1().Pods(fr.Namespace.Name).Get(serverPod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(resultPod.Status.ContainerStatuses[0].LastTerminationState.Terminated).Should(gomega.BeNil())
		}
	})
})
