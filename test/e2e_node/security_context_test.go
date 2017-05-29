/*
Copyright 2017 The Kubernetes Authors.

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

package e2e_node

import (
	"fmt"
	"net"
	"os/exec"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("Security Context", func() {
	f := framework.NewDefaultFramework("security-context-test")
	var podClient *framework.PodClient
	BeforeEach(func() {
		podClient = f.PodClient()
	})

	Context("when creating a pod in the host PID namespace", func() {
		makeHostPidPod := func(podName, image string, command []string, hostPID bool) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					HostPID:       hostPID,
					Containers: []v1.Container{
						{
							Image:   image,
							Name:    podName,
							Command: command,
						},
					},
				},
			}
		}
		createAndWaitHostPidPod := func(podName string, hostPID bool) {
			podClient.Create(makeHostPidPod(podName,
				"gcr.io/google_containers/busybox:1.24",
				[]string{"sh", "-c", "pidof nginx || true"},
				hostPID,
			))

			podClient.WaitForSuccess(podName, framework.PodStartTimeout)
		}

		nginxPid := ""
		BeforeEach(func() {
			nginxPodName := "nginx-hostpid-" + string(uuid.NewUUID())
			podClient.CreateSync(makeHostPidPod(nginxPodName,
				"gcr.io/google_containers/nginx-slim:0.7",
				nil,
				true,
			))

			output := f.ExecShellInContainer(nginxPodName, nginxPodName,
				"cat /var/run/nginx.pid")
			nginxPid = strings.TrimSpace(output)
		})

		It("should show its pid in the host PID namespace", func() {
			busyboxPodName := "busybox-hostpid-" + string(uuid.NewUUID())
			createAndWaitHostPidPod(busyboxPodName, true)
			logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, busyboxPodName, busyboxPodName)
			if err != nil {
				framework.Failf("GetPodLogs for pod %q failed: %v", busyboxPodName, err)
			}

			pids := strings.TrimSpace(logs)
			framework.Logf("Got nginx's pid %q from pod %q", pids, busyboxPodName)
			if pids == "" {
				framework.Failf("nginx's pid should be seen by hostpid containers")
			}

			pidSets := sets.NewString(strings.Split(pids, " ")...)
			if !pidSets.Has(nginxPid) {
				framework.Failf("nginx's pid should be seen by hostpid containers")
			}
		})

		It("should not show its pid in the non-hostpid containers", func() {
			busyboxPodName := "busybox-non-hostpid-" + string(uuid.NewUUID())
			createAndWaitHostPidPod(busyboxPodName, false)
			logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, busyboxPodName, busyboxPodName)
			if err != nil {
				framework.Failf("GetPodLogs for pod %q failed: %v", busyboxPodName, err)
			}

			pids := strings.TrimSpace(logs)
			framework.Logf("Got nginx's pid %q from pod %q", pids, busyboxPodName)
			pidSets := sets.NewString(strings.Split(pids, " ")...)
			if pidSets.Has(nginxPid) {
				framework.Failf("nginx's pid should not be seen by non-hostpid containers")
			}
		})
	})

	Context("when creating a pod in the host IPC namespace", func() {
		makeHostIPCPod := func(podName, image string, command []string, hostIPC bool) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					HostIPC:       hostIPC,
					Containers: []v1.Container{
						{
							Image:   image,
							Name:    podName,
							Command: command,
						},
					},
				},
			}
		}
		createAndWaitHostIPCPod := func(podName string, hostNetwork bool) {
			podClient.Create(makeHostIPCPod(podName,
				"gcr.io/google_containers/busybox:1.24",
				[]string{"sh", "-c", "ipcs -m | awk '{print $2}'"},
				hostNetwork,
			))

			podClient.WaitForSuccess(podName, framework.PodStartTimeout)
		}

		hostSharedMemoryID := ""
		BeforeEach(func() {
			output, err := exec.Command("sh", "-c", "ipcmk -M 1M | awk '{print $NF}'").Output()
			if err != nil {
				framework.Failf("Failed to create the shared memory on the host: %v", err)
			}
			hostSharedMemoryID = strings.TrimSpace(string(output))
			framework.Logf("Got host shared memory ID %q", hostSharedMemoryID)
		})

		It("should show the shared memory ID in the host IPC containers", func() {
			busyboxPodName := "busybox-hostipc-" + string(uuid.NewUUID())
			createAndWaitHostIPCPod(busyboxPodName, true)
			logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, busyboxPodName, busyboxPodName)
			if err != nil {
				framework.Failf("GetPodLogs for pod %q failed: %v", busyboxPodName, err)
			}

			podSharedMemoryIDs := strings.TrimSpace(logs)
			framework.Logf("Got shared memory IDs %q from pod %q", podSharedMemoryIDs, busyboxPodName)
			if !strings.Contains(podSharedMemoryIDs, hostSharedMemoryID) {
				framework.Failf("hostIPC container should show shared memory IDs on host")
			}
		})

		It("should not show the shared memory ID in the non-hostIPC containers", func() {
			busyboxPodName := "busybox-non-hostipc-" + string(uuid.NewUUID())
			createAndWaitHostIPCPod(busyboxPodName, false)
			logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, busyboxPodName, busyboxPodName)
			if err != nil {
				framework.Failf("GetPodLogs for pod %q failed: %v", busyboxPodName, err)
			}

			podSharedMemoryIDs := strings.TrimSpace(logs)
			framework.Logf("Got shared memory IDs %q from pod %q", podSharedMemoryIDs, busyboxPodName)
			if strings.Contains(podSharedMemoryIDs, hostSharedMemoryID) {
				framework.Failf("non-hostIPC container should not show shared memory IDs on host")
			}
		})

		AfterEach(func() {
			if hostSharedMemoryID != "" {
				_, err := exec.Command("sh", "-c", fmt.Sprintf("ipcrm -m %q", hostSharedMemoryID)).Output()
				if err != nil {
					framework.Failf("Failed to remove shared memory %q on the host: %v", hostSharedMemoryID, err)
				}
			}
		})
	})

	Context("when creating a pod in the host network namespace", func() {
		makeHostNetworkPod := func(podName, image string, command []string, hostNetwork bool) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					HostNetwork:   hostNetwork,
					Containers: []v1.Container{
						{
							Image:   image,
							Name:    podName,
							Command: command,
						},
					},
				},
			}
		}
		listListeningPortsCommand := []string{"sh", "-c", "netstat -ln"}
		createAndWaitHostNetworkPod := func(podName string, hostNetwork bool) {
			podClient.Create(makeHostNetworkPod(podName,
				"gcr.io/google_containers/busybox:1.24",
				listListeningPortsCommand,
				hostNetwork,
			))

			podClient.WaitForSuccess(podName, framework.PodStartTimeout)
		}

		listeningPort := ""
		var l net.Listener
		BeforeEach(func() {
			l, err := net.Listen("tcp", ":0")
			if err != nil {
				framework.Failf("Failed to open a new tcp port: %v", err)
			}
			addr := strings.Split(l.Addr().String(), ":")
			listeningPort = addr[len(addr)-1]
			framework.Logf("Opened a new tcp port %q", listeningPort)
		})

		It("should listen on same port in the host network containers", func() {
			busyboxPodName := "busybox-hostnetwork-" + string(uuid.NewUUID())
			createAndWaitHostNetworkPod(busyboxPodName, true)
			logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, busyboxPodName, busyboxPodName)
			if err != nil {
				framework.Failf("GetPodLogs for pod %q failed: %v", busyboxPodName, err)
			}

			framework.Logf("Got logs for pod %q: %q", busyboxPodName, logs)
			if !strings.Contains(logs, listeningPort) {
				framework.Failf("host-networked container should listening on same port as host")
			}
		})

		It("shouldn't show the same port in the non-hostnetwork containers", func() {
			busyboxPodName := "busybox-non-hostnetwork-" + string(uuid.NewUUID())
			createAndWaitHostNetworkPod(busyboxPodName, false)
			logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, busyboxPodName, busyboxPodName)
			if err != nil {
				framework.Failf("GetPodLogs for pod %q failed: %v", busyboxPodName, err)
			}

			framework.Logf("Got logs for pod %q: %q", busyboxPodName, logs)
			if strings.Contains(logs, listeningPort) {
				framework.Failf("non-hostnetworked container shouldn't show the same port as host")
			}
		})

		AfterEach(func() {
			if l != nil {
				l.Close()
			}
		})
	})

})
