/*
Copyright 2022 The Kubernetes Authors.

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

package windows

import (
	"context"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/apimachinery/pkg/util/uuid"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = sigDescribe(feature.Windows, "[Excluded:WindowsDocker] [MinimumKubeletVersion:1.22] RebootHost containers", framework.WithSerial(), framework.WithDisruptive(), framework.WithSlow(), skipUnlessWindows(func() {
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
	})

	f := framework.NewDefaultFramework("reboot-host-test-windows")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should run as a reboot process on the host/node", func(ctx context.Context) {

		ginkgo.By("selecting a Windows node")
		targetNode, err := findWindowsNode(ctx, f)
		framework.ExpectNoError(err, "Error finding Windows node")
		framework.Logf("Using node: %v", targetNode.Name)

		bootID := targetNode.Status.NodeInfo.BootID
		windowsImage := imageutils.GetE2EImage(imageutils.Agnhost)

		// Create Windows pod on the selected Windows node Using Agnhost
		podName := "pod-" + string(uuid.NewUUID())
		agnPod := &v1.Pod{
			TypeMeta: metav1.TypeMeta{
				Kind:       "Pod",
				APIVersion: "v1",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "windows-container",
						Image: windowsImage,
						Ports: []v1.ContainerPort{{ContainerPort: 80}},
					},
				},
				RestartPolicy: v1.RestartPolicyAlways,
				NodeName:      targetNode.Name,
			},
		}
		agnPod.Spec.Containers[0].Args = []string{"test-webserver"}
		ginkgo.By("creating a windows pod and waiting for it to be running")
		agnPod = e2epod.NewPodClient(f).CreateSync(ctx, agnPod)

		// Create Linux pod to ping the windows pod
		linuxBusyBoxImage := imageutils.GetE2EImage(imageutils.Nginx)
		podName = "pod-" + string(uuid.NewUUID())
		nginxPod := &v1.Pod{
			TypeMeta: metav1.TypeMeta{
				Kind:       "Pod",
				APIVersion: "v1",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "linux-container",
						Image: linuxBusyBoxImage,
						Ports: []v1.ContainerPort{{ContainerPort: 80}},
					},
				},
				NodeSelector: map[string]string{
					"kubernetes.io/os": "linux",
				},
				Tolerations: []v1.Toleration{
					{
						Operator: v1.TolerationOpExists,
						Effect:   v1.TaintEffectNoSchedule,
					},
				},
			},
		}
		ginkgo.By("Waiting for the Linux pod to run")
		nginxPod = e2epod.NewPodClient(f).CreateSync(ctx, nginxPod)

		ginkgo.By("checking connectivity to 8.8.8.8 53 (google.com) from Linux")
		assertConsistentConnectivity(ctx, f, nginxPod.ObjectMeta.Name, "linux", linuxCheck("8.8.8.8", 53), externalMaxTries)

		ginkgo.By("checking connectivity to www.google.com from Windows")
		assertConsistentConnectivity(ctx, f, agnPod.ObjectMeta.Name, "windows", windowsCheck("www.google.com"), externalMaxTries)

		ginkgo.By("checking connectivity from Linux to Windows for the first time")
		assertConsistentConnectivity(ctx, f, nginxPod.ObjectMeta.Name, "linux", linuxCheck(agnPod.Status.PodIP, 80), internalMaxTries)

		initialRestartCount := podutil.GetExistingContainerStatus(agnPod.Status.ContainerStatuses, "windows-container").RestartCount

		ginkgo.By("scheduling a pod with a container that verifies reboot selected node works as well")

		trueVar := true
		podName = "reboot-host-test-pod"
		user := "NT AUTHORITY\\SYSTEM"
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess:   &trueVar,
						RunAsUserName: &user,
					},
				},
				HostNetwork: true,
				Containers: []v1.Container{
					{
						Image: windowsImage,
						Name:  "reboot-computer-test",
						Command: []string{
							"powershell.exe",
							"-Command",
							"$os = Get-WmiObject -Class win32_operatingsystem;",
							"[Environment]::SetEnvironmentVariable(\"TMP_BOOT_DATE\", $os.LastBootUpTime, \"Machine\");",
							"[Environment]::SetEnvironmentVariable(\"TMP_INSTALL_DATE\", $os.InstallDate, \"Machine\");",
							"shutdown.exe -r -t 30",
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				NodeName:      targetNode.Name,
			},
		}

		e2epod.NewPodClient(f).Create(ctx, pod)

		ginkgo.By("Waiting for pod to run")
		e2epod.NewPodClient(f).WaitForFinish(ctx, podName, 3*time.Minute)

		ginkgo.By("Then ensuring pod finished running successfully")
		p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(
			ctx,
			podName,
			metav1.GetOptions{})

		framework.ExpectNoError(err, "Error retrieving pod")
		gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodSucceeded))

		ginkgo.By("Waiting for Windows worker rebooting")

		restartCount := 0

		ginkgo.By("Waiting for nodes to be rebooted")
		gomega.Eventually(ctx, func(ctx context.Context) string {
			refreshNode, err := f.ClientSet.CoreV1().Nodes().Get(ctx, targetNode.Name, metav1.GetOptions{})
			if err != nil {
				return ""
			}
			return refreshNode.Status.NodeInfo.BootID
		}).WithPolling(time.Second*30).WithTimeout(time.Minute*10).
			Should(gomega.BeNumerically(">", bootID), "node was not rebooted")

		ginkgo.By("Then checking existed agn-test-pod is running on the rebooted host")
		agnPodOut, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, agnPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "getting pod info after reboot")

		lastRestartCount := podutil.GetExistingContainerStatus(agnPodOut.Status.ContainerStatuses, "windows-container").RestartCount
		restartCount = int(lastRestartCount - initialRestartCount)
		gomega.Expect(restartCount).To(gomega.Equal(1), "restart count of agn-test-pod is 1")
		gomega.Expect(agnPodOut.Status.Phase).To(gomega.Equal(v1.PodRunning))
		assertConsistentConnectivity(ctx, f, nginxPod.ObjectMeta.Name, "linux", linuxCheck(agnPodOut.Status.PodIP, 80), internalMaxTries)

		// create another host process pod to check system boot time
		checkPod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "check-reboot-pod",
			},
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess:   &trueVar,
						RunAsUserName: &user,
					},
				},
				HostNetwork: true,
				Containers: []v1.Container{
					{
						Image: windowsImage,
						Name:  "reboot-computer-check",
						Command: []string{
							"powershell.exe",
							"-Command",
							"$os = Get-WmiObject -Class win32_operatingsystem;",
							"$lastBootTime = [Environment]::GetEnvironmentVariable(\"TMP_BOOT_DATE\", \"Machine\");",
							"$lastInstallTime = [Environment]::GetEnvironmentVariable(\"TMP_INSTALL_DATE\", \"Machine\");",
							"$timeInterval = $os.ConvertToDateTime($os.LastBootUpTime) -  $os.ConvertToDateTime($lastBootTime);",
							"$installInterval = $os.ConvertToDateTime($os.InstallDate) -  $os.ConvertToDateTime($lastInstallTime);",
							"if ( $timeInterval.TotalSeconds -le 0 ) {exit -1};",
							"if ( $installInterval.TotalSeconds -ne 0 ) {exit -1};",
							"[Environment]::SetEnvironmentVariable(\"TMP_BOOT_DATE\", $null, \"Machine\");",
							"[Environment]::SetEnvironmentVariable(\"TMP_INSTALL_DATE\", $null, \"Machine\");",
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				NodeName:      targetNode.Name,
			},
		}

		e2epod.NewPodClient(f).Create(ctx, checkPod)

		ginkgo.By("Waiting for pod to run")
		e2epod.NewPodClient(f).WaitForFinish(ctx, "check-reboot-pod", 3*time.Minute)

		ginkgo.By("Then ensuring pod finished running successfully")
		p, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(
			ctx,
			"check-reboot-pod",
			metav1.GetOptions{})

		framework.ExpectNoError(err, "Error retrieving pod")
		gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodSucceeded))
	})
}))
