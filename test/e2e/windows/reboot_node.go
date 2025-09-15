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
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"bytes"
	"encoding/base64"
	"encoding/binary"
	"unicode/utf16"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
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
		nodes, err := findWindowsNodes(ctx, f)
		framework.ExpectNoError(err, "Error finding Windows nodes")

		if len(nodes) == 0 {
			e2eskipper.Skipf("Could not find and ready and schedulable Windows nodes")
		}

		targetNode := nodes[0]
		// reuse the node if find label contains "test/reboot-used" first
		for _, node := range nodes {
			if _, ok := node.Labels["test/reboot-used"]; ok {
				framework.Logf("Reusing node %s", node.Name)
				targetNode = node
				break
			}
		}

		framework.Logf("Using node: %v", targetNode.Name)

		ginkgo.DeferCleanup(cleanupContainers, f, targetNode.Name, []string{f.Namespace.Name})
		ginkgo.DeferCleanup(patchWindowsNodeIfNeeded, f, targetNode.Name)

		bootID, err := strconv.Atoi(targetNode.Status.NodeInfo.BootID)
		framework.ExpectNoError(err, "Error converting bootID to int")
		framework.Logf("Initial BootID: %d", bootID)

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
				agnPodOut, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, agnPod.Name, metav1.GetOptions{})
				if err == nil {
					lastRestartCount := podutil.GetExistingContainerStatus(agnPodOut.Status.ContainerStatuses, "windows-container").RestartCount
					restartCount = int(lastRestartCount - initialRestartCount)
				}
				time.Sleep(time.Second * 30)
			}
		}

		ginkgo.By("Checking whether the node is rebooted")
		refreshNode, err := f.ClientSet.CoreV1().Nodes().Get(ctx, targetNode.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Error getting node info after reboot")
		currentbootID, err := strconv.Atoi(refreshNode.Status.NodeInfo.BootID)
		framework.ExpectNoError(err, "Error converting bootID to int")
		framework.Logf("current BootID: %d", currentbootID)
		gomega.Expect(currentbootID).To(gomega.Equal(bootID+1), "BootID should be incremented by 1 after reboot")

		ginkgo.By("Checking whether agn-test-pod is rebooted")
		gomega.Expect(restartCount).To(gomega.Equal(1), "restart count of agn-test-pod is 1")

		agnPodOut, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, agnPod.Name, metav1.GetOptions{})
		gomega.Expect(agnPodOut.Status.Phase).To(gomega.Equal(v1.PodRunning))
		framework.ExpectNoError(err, "getting pod info after reboot")
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

// cleanupContainers removes all containers that inside the specified namespace
func cleanupContainers(ctx context.Context, f *framework.Framework, nodeName string, namespaces []string) {
	if len(namespaces) == 0 {
		framework.Logf("No namespaces provided for cleanup")
		return
	}

	var quoted []string
	for _, item := range namespaces {
		if item == f.Namespace.Name {
			continue
		}

		quoted = append(quoted, strconv.Quote(item))
	}

	namespace := strings.Join(quoted, ", ")

	framework.Logf("Deleting containers in namespace %s on node %s", namespace, nodeName)

	cmd := `Write-Host "Running cleanup script";
	$ErrorActionPreference = "Stop";
	$lines = & "$Env:ProgramFiles\\containerd\\ctr.exe" -n k8s.io containers list  | Select-String "io.containerd.runhcs.v1";
	$namespaces = @($NAMESPACES$);
	foreach ($line in $lines) {
		$columns = $line.ToString().Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries);
		$containerId = $columns[0].ToLower();
		$json = & "$Env:ProgramFiles\\containerd\\ctr.exe" -n k8s.io containers info $containerId | ConvertFrom-Json
		$containerName = $json.Labels.'io.kubernetes.container.name';
		$podName = $json.Labels.'io.kubernetes.pod.name';
		$nsName = $json.Labels.'io.kubernetes.pod.namespace';

		if (-not ($namespaces -Contains $nsName)) {
			Write-Host "Skipping container $containerId, $containerName, $podName, $nsName";
			continue;
		}

		if ($containerName -eq "$CONTAINERNAME$") {
			Write-Host "Skipping container $containerId, $containerName, $podName, $nsName";
			continue;
		}

		Write-Host "Deleting container $containerId, $containerName, $podName, $nsName";
		try {
			& "$Env:ProgramFiles\\containerd\\ctr.exe" -n k8s.io containers delete $containerId;
		} catch {
			Write-Host "Failed to delete container $containerId, $containerName, $podName, $nsName";
			Write-Host $_.Exception.Message;
		}
	}

	Write-Host "Rebooting host";
	shutdown.exe -r -t 5;
	Write-Host "Cleanup script finished";
	`

	containerName := fmt.Sprintf("%s-%08x", "cleanup-container", rand.Int31())
	cleanupPodName := "cleanup-pod"
	trueVar := true
	user := "NT AUTHORITY\\SYSTEM"
	windowsImage := imageutils.GetE2EImage(imageutils.Agnhost)

	// replace the container name and namespace in the command
	cmd = strings.ReplaceAll(cmd, "$CONTAINERNAME$", containerName)
	cmd = strings.ReplaceAll(cmd, "$NAMESPACES$", namespace)

	cleanupPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: cleanupPodName,
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
					Name:  containerName,
					Image: windowsImage, // Or another HostProcess-compatible image
					Command: []string{
						"powershell.exe",
						"-encodedCommand",
						encodePowerShellCommand(cmd),
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			NodeName:      nodeName,
		},
	}

	// Launch the pod
	e2epod.NewPodClient(f).Create(ctx, cleanupPod)
	e2epod.NewPodClient(f).WaitForFinish(ctx, cleanupPodName, 2*time.Minute)

	p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, cleanupPodName, metav1.GetOptions{})
	if err != nil {
		framework.Logf("Error retrieving cleanup pod: %v", err)
	} else {
		framework.Logf("Cleanup pod status: %s", p.Status.Phase)
	}

	// Delete the cleanup pod
	err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(ctx, cleanupPodName, metav1.DeleteOptions{})
	if err != nil {
		framework.Logf("Error deleting cleanup pod: %v", err)
	} else {
		framework.Logf("Cleanup pod deleted")
	}
}

func encodePowerShellCommand(cmd string) string {
	// Convert string to UTF-16 (code units)
	utf16Cmd := utf16.Encode([]rune(cmd))

	// Convert to bytes in little endian (UTF-16LE)
	buf := new(bytes.Buffer)
	for _, r := range utf16Cmd {
		_ = binary.Write(buf, binary.LittleEndian, r)
	}

	// Base64 encode
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func patchWindowsNodeIfNeeded(ctx context.Context, f *framework.Framework, nodeName string) {
	patch := []byte(`{"metadata":{"labels":{"test/reboot-used":"true"}}}`)

	p, error := f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if error != nil {
		framework.Logf("Error getting node %s: %v", nodeName, error)
		return
	}

	if _, ok := p.Labels["test/reboot-used"]; ok {
		framework.Logf("Node %s already patched", nodeName)
		return
	}

	// Patch the node with a label to indicate it has been used for testing
	framework.Logf("Patching node %s with label test/reboot-used=true", nodeName)
	_, err := f.ClientSet.CoreV1().Nodes().Patch(ctx, nodeName, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
	if err != nil {
		framework.Logf("Error patching node %s: %v", nodeName, err)
		return
	}

	// Wait for the node to be patched
	timeout := time.After(time.Minute * 5)
	for {
		select {
		case <-timeout:
			framework.Logf("Timeout waiting for node %s to be patched", nodeName)
			return
		default:
			node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
			if err != nil {
				framework.Logf("Error getting node %s: %v", nodeName, err)
				return
			}
			if _, ok := node.Labels["test/reboot-used"]; ok {
				framework.Logf("Node %s patched successfully", nodeName)
				return
			}
			time.Sleep(time.Second * 5)
		}
	}
}
