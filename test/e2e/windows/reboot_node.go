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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/apimachinery/pkg/util/uuid"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("[Feature:Windows] [Excluded:WindowsDocker] [MinimumKubeletVersion:1.22] RebootHost containers [Serial] [Disruptive] [Slow]", func() {
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
	})

	f := framework.NewDefaultFramework("reboot-host-test-windows")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	ginkgo.It("should run as a reboot process on the host/node", func() {

		ginkgo.By("selecting a Windows node")
		targetNode, err := findWindowsNode(f)
		framework.ExpectNoError(err, "Error finding Windows node")
		framework.Logf("Using node: %v", targetNode.Name)

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
		agnPod = f.PodClient().CreateSync(agnPod)

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
		nginxPod = f.PodClient().CreateSync(nginxPod)

		ginkgo.By("checking connectivity to 8.8.8.8 53 (google.com) from Linux")
		assertConsistentConnectivity(f, nginxPod.ObjectMeta.Name, "linux", linuxCheck("8.8.8.8", 53))

		ginkgo.By("checking connectivity to www.google.com from Windows")
		assertConsistentConnectivity(f, agnPod.ObjectMeta.Name, "windows", windowsCheck("www.google.com"))

		ginkgo.By("checking connectivity from Linux to Windows for the first time")
		assertConsistentConnectivity(f, nginxPod.ObjectMeta.Name, "linux", linuxCheck(agnPod.Status.PodIP, 80))

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

		f.PodClient().Create(pod)

		ginkgo.By("Waiting for pod to run")
		f.PodClient().WaitForFinish(podName, 3*time.Minute)

		ginkgo.By("Then ensuring pod finished running successfully")
		p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(
			context.TODO(),
			podName,
			metav1.GetOptions{})

		framework.ExpectNoError(err, "Error retrieving pod")
		framework.ExpectEqual(p.Status.Phase, v1.PodSucceeded)

		ginkgo.By("Waiting for Windows worker rebooting")

		restartCount := 0

		timeout := time.After(time.Minute * 10)
	FOR:
		for {
			select {
			case <-timeout:
				break FOR
			default:
				if restartCount > 0 {
					break FOR
				}
				ginkgo.By("Then checking existed agn-test-pod is running on the rebooted host")
				agnPodOut, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), agnPod.Name, metav1.GetOptions{})
				if err == nil {
					lastRestartCount := podutil.GetExistingContainerStatus(agnPodOut.Status.ContainerStatuses, "windows-container").RestartCount
					restartCount = int(lastRestartCount - initialRestartCount)
				}
				time.Sleep(time.Second * 30)
			}
		}

		ginkgo.By("Checking whether agn-test-pod is rebooted")
		framework.ExpectEqual(restartCount, 1)

		agnPodOut, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), agnPod.Name, metav1.GetOptions{})
		framework.ExpectEqual(agnPodOut.Status.Phase, v1.PodRunning)
		framework.ExpectNoError(err, "getting pod info after reboot")
		assertConsistentConnectivity(f, nginxPod.ObjectMeta.Name, "linux", linuxCheck(agnPodOut.Status.PodIP, 80))

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

		f.PodClient().Create(checkPod)

		ginkgo.By("Waiting for pod to run")
		f.PodClient().WaitForFinish("check-reboot-pod", 3*time.Minute)

		ginkgo.By("Then ensuring pod finished running successfully")
		p, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(
			context.TODO(),
			"check-reboot-pod",
			metav1.GetOptions{})

		framework.ExpectNoError(err, "Error retrieving pod")
		framework.ExpectEqual(p.Status.Phase, v1.PodSucceeded)
	})
})
