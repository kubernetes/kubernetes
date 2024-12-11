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

package e2enode

import (
	"context"
	"fmt"
	"net"
	"os/exec"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/nodefeature"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("Security Context", func() {
	f := framework.NewDefaultFramework("security-context-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var podClient *e2epod.PodClient
	ginkgo.BeforeEach(func() {
		podClient = e2epod.NewPodClient(f)
	})

	f.Context(framework.WithNodeConformance(), "[LinuxOnly] Container PID namespace sharing", func() {
		ginkgo.It("containers in pods using isolated PID namespaces should all receive PID 1", func(ctx context.Context) {
			ginkgo.By("Create a pod with isolated PID namespaces.")
			e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "isolated-pid-ns-test-pod"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "test-container-1",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/top"},
						},
						{
							Name:    "test-container-2",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sleep"},
							Args:    []string{"10000"},
						},
					},
				},
			})

			ginkgo.By("Check if both containers receive PID 1.")
			pid1 := e2epod.ExecCommandInContainer(f, "isolated-pid-ns-test-pod", "test-container-1", "/bin/pidof", "top")
			pid2 := e2epod.ExecCommandInContainer(f, "isolated-pid-ns-test-pod", "test-container-2", "/bin/pidof", "sleep")
			if pid1 != "1" || pid2 != "1" {
				framework.Failf("PIDs of different containers are not all 1: test-container-1=%v, test-container-2=%v", pid1, pid2)
			}
		})

		ginkgo.It("processes in containers sharing a pod namespace should be able to see each other", func(ctx context.Context) {
			ginkgo.By("Create a pod with shared PID namespace.")
			e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "shared-pid-ns-test-pod"},
				Spec: v1.PodSpec{
					ShareProcessNamespace: &[]bool{true}[0],
					Containers: []v1.Container{
						{
							Name:    "test-container-1",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/top"},
						},
						{
							Name:    "test-container-2",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sleep"},
							Args:    []string{"10000"},
						},
					},
				},
			})

			ginkgo.By("Check if the process in one container is visible to the process in the other.")
			pid1 := e2epod.ExecCommandInContainer(f, "shared-pid-ns-test-pod", "test-container-1", "/bin/pidof", "top")
			pid2 := e2epod.ExecCommandInContainer(f, "shared-pid-ns-test-pod", "test-container-2", "/bin/pidof", "top")
			if pid1 != pid2 {
				framework.Failf("PIDs are not the same in different containers: test-container-1=%v, test-container-2=%v", pid1, pid2)
			}
		})
	})

	ginkgo.Context("when creating a pod in the host PID namespace", func() {
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
		createAndWaitHostPidPod := func(ctx context.Context, podName string, hostPID bool) {
			podClient.Create(ctx, makeHostPidPod(podName,
				busyboxImage,
				[]string{"sh", "-c", "pidof nginx || true"},
				hostPID,
			))

			podClient.WaitForSuccess(ctx, podName, framework.PodStartTimeout)
		}

		nginxPid := ""
		ginkgo.BeforeEach(func(ctx context.Context) {
			nginxPodName := "nginx-hostpid-" + string(uuid.NewUUID())
			podClient.CreateSync(ctx, makeHostPidPod(nginxPodName,
				imageutils.GetE2EImage(imageutils.Nginx),
				nil,
				true,
			))

			output := e2epod.ExecShellInContainer(f, nginxPodName, nginxPodName,
				"cat /var/run/nginx.pid")
			nginxPid = strings.TrimSpace(output)
		})

		f.It("should show its pid in the host PID namespace", nodefeature.HostAccess, feature.HostAccess, func(ctx context.Context) {
			busyboxPodName := "busybox-hostpid-" + string(uuid.NewUUID())
			createAndWaitHostPidPod(ctx, busyboxPodName, true)
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, busyboxPodName, busyboxPodName)
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

		f.It("should not show its pid in the non-hostpid containers", nodefeature.HostAccess, feature.HostAccess, func(ctx context.Context) {
			busyboxPodName := "busybox-non-hostpid-" + string(uuid.NewUUID())
			createAndWaitHostPidPod(ctx, busyboxPodName, false)
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, busyboxPodName, busyboxPodName)
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

	ginkgo.Context("when creating a pod in the host IPC namespace", func() {
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
		createAndWaitHostIPCPod := func(ctx context.Context, podName string, hostNetwork bool) {
			podClient.Create(ctx, makeHostIPCPod(podName,
				imageutils.GetE2EImage(imageutils.IpcUtils),
				[]string{"sh", "-c", "ipcs -m | awk '{print $2}'"},
				hostNetwork,
			))

			podClient.WaitForSuccess(ctx, podName, framework.PodStartTimeout)
		}

		hostSharedMemoryID := ""
		ginkgo.BeforeEach(func() {
			output, err := exec.Command("sh", "-c", "ipcmk -M 1048576 | awk '{print $NF}'").Output()
			if err != nil {
				framework.Failf("Failed to create the shared memory on the host: %v", err)
			}
			hostSharedMemoryID = strings.TrimSpace(string(output))
			framework.Logf("Got host shared memory ID %q", hostSharedMemoryID)
		})

		f.It("should show the shared memory ID in the host IPC containers", nodefeature.HostAccess, feature.HostAccess, func(ctx context.Context) {
			ipcutilsPodName := "ipcutils-hostipc-" + string(uuid.NewUUID())
			createAndWaitHostIPCPod(ctx, ipcutilsPodName, true)
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, ipcutilsPodName, ipcutilsPodName)
			if err != nil {
				framework.Failf("GetPodLogs for pod %q failed: %v", ipcutilsPodName, err)
			}

			podSharedMemoryIDs := strings.TrimSpace(logs)
			framework.Logf("Got shared memory IDs %q from pod %q", podSharedMemoryIDs, ipcutilsPodName)
			if !strings.Contains(podSharedMemoryIDs, hostSharedMemoryID) {
				framework.Failf("hostIPC container should show shared memory IDs on host")
			}
		})

		f.It("should not show the shared memory ID in the non-hostIPC containers", nodefeature.HostAccess, feature.HostAccess, func(ctx context.Context) {
			ipcutilsPodName := "ipcutils-non-hostipc-" + string(uuid.NewUUID())
			createAndWaitHostIPCPod(ctx, ipcutilsPodName, false)
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, ipcutilsPodName, ipcutilsPodName)
			if err != nil {
				framework.Failf("GetPodLogs for pod %q failed: %v", ipcutilsPodName, err)
			}

			podSharedMemoryIDs := strings.TrimSpace(logs)
			framework.Logf("Got shared memory IDs %q from pod %q", podSharedMemoryIDs, ipcutilsPodName)
			if strings.Contains(podSharedMemoryIDs, hostSharedMemoryID) {
				framework.Failf("non-hostIPC container should not show shared memory IDs on host")
			}
		})

		ginkgo.AfterEach(func() {
			if hostSharedMemoryID != "" {
				_, err := exec.Command("sh", "-c", fmt.Sprintf("ipcrm -m %q", hostSharedMemoryID)).Output()
				if err != nil {
					framework.Failf("Failed to remove shared memory %q on the host: %v", hostSharedMemoryID, err)
				}
			}
		})
	})

	ginkgo.Context("when creating a pod in the host network namespace", func() {
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
		createAndWaitHostNetworkPod := func(ctx context.Context, podName string, hostNetwork bool) {
			podClient.Create(ctx, makeHostNetworkPod(podName,
				busyboxImage,
				listListeningPortsCommand,
				hostNetwork,
			))

			podClient.WaitForSuccess(ctx, podName, framework.PodStartTimeout)
		}

		listeningPort := ""
		var l net.Listener
		var err error
		ginkgo.BeforeEach(func() {
			l, err = net.Listen("tcp", ":0")
			if err != nil {
				framework.Failf("Failed to open a new tcp port: %v", err)
			}
			addr := strings.Split(l.Addr().String(), ":")
			listeningPort = addr[len(addr)-1]
			framework.Logf("Opened a new tcp port %q", listeningPort)
		})

		f.It("should listen on same port in the host network containers", nodefeature.HostAccess, feature.HostAccess, func(ctx context.Context) {
			busyboxPodName := "busybox-hostnetwork-" + string(uuid.NewUUID())
			createAndWaitHostNetworkPod(ctx, busyboxPodName, true)
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, busyboxPodName, busyboxPodName)
			if err != nil {
				framework.Failf("GetPodLogs for pod %q failed: %v", busyboxPodName, err)
			}

			framework.Logf("Got logs for pod %q: %q", busyboxPodName, logs)
			if !strings.Contains(logs, listeningPort) {
				framework.Failf("host-networked container should listening on same port as host")
			}
		})

		f.It("shouldn't show the same port in the non-hostnetwork containers", nodefeature.HostAccess, feature.HostAccess, func(ctx context.Context) {
			busyboxPodName := "busybox-non-hostnetwork-" + string(uuid.NewUUID())
			createAndWaitHostNetworkPod(ctx, busyboxPodName, false)
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, busyboxPodName, busyboxPodName)
			if err != nil {
				framework.Failf("GetPodLogs for pod %q failed: %v", busyboxPodName, err)
			}

			framework.Logf("Got logs for pod %q: %q", busyboxPodName, logs)
			if strings.Contains(logs, listeningPort) {
				framework.Failf("non-hostnetworked container shouldn't show the same port as host")
			}
		})

		ginkgo.AfterEach(func() {
			if l != nil {
				l.Close()
			}
		})
	})
})
