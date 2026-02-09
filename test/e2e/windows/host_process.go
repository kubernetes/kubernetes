/*
Copyright 2021 The Kubernetes Authors.

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
	"strings"
	"time"

	semver "github.com/blang/semver/v4"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	validation_script = `if (-not(Test-Path $env:CONTAINER_SANDBOX_MOUNT_POINT\etc\emptydir)) {
		throw "Cannot find emptydir volume"
	}
	if (-not(Test-Path $env:CONTAINER_SANDBOX_MOUNT_POINT\etc\configmap\text.txt)) {
		throw "Cannot find text.txt in configmap-volume"
	}
	$c = Get-Content -Path $env:CONTAINER_SANDBOX_MOUNT_POINT\etc\configmap\text.txt
	if ($c -ne "Lorem ipsum dolor sit amet") {
		throw "Contents of /etc/configmap/text.txt are not as expected"
	}
	if (-not(Test-Path $env:CONTAINER_SANDBOX_MOUNT_POINT\etc\hostpath)) {
		throw "Cannot find hostpath volume" 
	}
	if (-not(Test-Path $env:CONTAINER_SANDBOX_MOUNT_POINT\etc\downwardapi\podname)) {
		throw "Cannot find podname file in downward-api volume" 
	}
	$c = Get-Content -Path $env:CONTAINER_SANDBOX_MOUNT_POINT\etc\downwardapi\podname
	if ($c -ne "host-process-volume-mounts") {
		throw "Contents of /etc/downward-api/podname are not as expected"
	}
	if (-not(Test-Path $env:CONTAINER_SANDBOX_MOUNT_POINT\etc\secret\foo.txt)) {
		throw "Cannot find file foo.txt in secret volume"
	}
	$c = Get-Content $env:CONTAINER_SANDBOX_MOUNT_POINT\etc\secret\foo.txt
	if ($c -ne "bar") {
		Write-Output $c
		throw "Contents of /etc/secret/foo.txt are not as expected"
	}
	# Windows ComputerNames cannot exceed 15 characters, which means that the $env:COMPUTERNAME
	# can only be a substring of $env:NODE_NAME_TEST. We compare it with the hostname instead.
	# The following comparison is case insensitive.
	if ($env:NODE_NAME_TEST -ine "$(hostname)") {
		throw "NODE_NAME_TEST env var ($env:NODE_NAME_TEST) does not equal hostname"
	}
	Write-Output "SUCCESS"`
)

var (
	trueVar = true

	User_NTAuthorityLocalService = "NT AUTHORITY\\Local Service"
	User_NTAuthoritySystem       = "NT AUTHORITY\\SYSTEM"
)

var _ = sigDescribe(feature.WindowsHostProcessContainers, "HostProcess containers", skipUnlessWindows(func() {
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
	})

	f := framework.NewDefaultFramework("host-process-test-windows")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should run as a process on the host/node", func(ctx context.Context) {

		ginkgo.By("selecting a Windows node")
		targetNode, err := findWindowsNode(ctx, f)
		framework.ExpectNoError(err, "Error finding Windows node")
		framework.Logf("Using node: %v", targetNode.Name)

		ginkgo.By("scheduling a pod with a container that verifies %COMPUTERNAME% matches selected node name")
		image := imageutils.GetConfig(imageutils.BusyBox)
		podName := "host-process-test-pod"
		// We're passing this to powershell.exe -Command. Inside, we must use apostrophes for strings.
		command := fmt.Sprintf(`& {if ('%s' -ine "$(hostname)") { exit -1 }}`, targetNode.Name)
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess:   &trueVar,
						RunAsUserName: &User_NTAuthoritySystem,
					},
				},
				HostNetwork: true,
				Containers: []v1.Container{
					{
						Image:   image.GetE2EImage(),
						Name:    "computer-name-test",
						Command: []string{"powershell.exe", "-Command", command},
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
	})

	ginkgo.It("should support init containers", func(ctx context.Context) {
		ginkgo.By("scheduling a pod with a container that verifies init container can configure the node")
		podName := "host-process-init-pods"
		filename := fmt.Sprintf("/testfile%s.txt", string(uuid.NewUUID()))
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess:   &trueVar,
						RunAsUserName: &User_NTAuthoritySystem,
					},
				},
				HostNetwork: true,
				InitContainers: []v1.Container{
					{
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Name:    "configure-node",
						Command: []string{"powershell", "-c", "Set-content", "-Path", filename, "-V", "test"},
					},
				},
				Containers: []v1.Container{
					{
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Name:    "read-configuration",
						Command: []string{"powershell", "-c", "ls", filename},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				NodeSelector: map[string]string{
					"kubernetes.io/os": "windows",
				},
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

		if p.Status.Phase != v1.PodSucceeded {
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, podName, "read-configuration")
			if err != nil {
				framework.Logf("Error pulling logs: %v", err)
			}
			framework.Logf("Pod phase: %v\nlogs:\n%s", p.Status.Phase, logs)
		}
		gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodSucceeded))
	})

	ginkgo.It("container command path validation", func(ctx context.Context) {

		// The way hostprocess containers are created is being updated in container
		// v1.7 to better support volume mounts and part of these changes include
		// updates how the container's starting process is invoked.
		// These test cases are only valid for containerd v1.6.
		// See https://github.com/kubernetes/enhancements/blob/master/keps/sig-windows/1981-windows-privileged-container-support/README.md
		// for more details.
		ginkgo.By("Ensuring Windows nodes are running containerd v1.6.x")
		windowsNode, err := findWindowsNode(ctx, f)
		framework.ExpectNoError(err, "error finding Windows node")
		r, v, err := getNodeContainerRuntimeAndVersion(windowsNode)
		framework.ExpectNoError(err, "error getting node container runtime and version")
		framework.Logf("Got runtime: %s, version %v, node: %s", r, v, windowsNode.Name)

		if !strings.EqualFold(r, "containerd") {
			e2eskipper.Skipf("container runtime is not containerd")
		}

		v1dot7 := semver.MustParse("1.7.0")
		if v.GTE(v1dot7) {
			e2eskipper.Skipf("container runtime is >= 1.7.0")
		}

		// The following test cases are broken into batches to speed up the test.
		// Each batch will be scheduled as a single pod with a container for each test case.
		// Pods will be scheduled sequentially since the start-up cost of containers is high
		// on Windows and ginkgo may also schedule test cases in parallel.
		tests := [][]struct {
			command    []string
			args       []string
			workingDir string
		}{
			{
				{
					command: []string{"cmd.exe", "/c", "ver"},
				},
				{
					command:    []string{"System32\\cmd.exe", "/c", "ver"},
					workingDir: "c:\\Windows",
				},
				{
					command:    []string{"System32\\cmd.exe", "/c", "ver"},
					workingDir: "c:\\Windows\\",
				},
				{
					command: []string{"%CONTAINER_SANDBOX_MOUNT_POINT%\\bin\\uname.exe", "-o"},
				},
			},
			{
				{
					command: []string{"%CONTAINER_SANDBOX_MOUNT_POINT%/bin/uname.exe", "-o"},
				},
				{
					command: []string{"%CONTAINER_SANDBOX_MOUNT_POINT%\\bin/uname.exe", "-o"},
				},
				{
					command: []string{"bin/uname.exe", "-o"},
				},
				{
					command:    []string{"bin/uname.exe", "-o"},
					workingDir: "%CONTAINER_SANDBOX_MOUNT_POINT%",
				},
			},
			{
				{
					command:    []string{"bin\\uname.exe", "-o"},
					workingDir: "%CONTAINER_SANDBOX_MOUNT_POINT%",
				},
				{
					command:    []string{"uname.exe", "-o"},
					workingDir: "%CONTAINER_SANDBOX_MOUNT_POINT%/bin",
				},
				{
					command:    []string{"uname.exe", "-o"},
					workingDir: "%CONTAINER_SANDBOX_MOUNT_POINT%/bin/",
				},
				{
					command:    []string{"uname.exe", "-o"},
					workingDir: "%CONTAINER_SANDBOX_MOUNT_POINT%\\bin\\",
				},
			},
			{
				{
					command: []string{"powershell", "cmd.exe", "/ver"},
				},
				{
					command: []string{"powershell", "c:/Windows/System32/cmd.exe", "/c", "ver"},
				},
				{
					command: []string{"powershell", "c:\\Windows\\System32/cmd.exe", "/c", "ver"},
				},
				{
					command: []string{"powershell", "%CONTAINER_SANDBOX_MOUNT_POINT%\\bin\\uname.exe", "-o"},
				},
			},
			{
				{
					command: []string{"powershell", "$env:CONTAINER_SANDBOX_MOUNT_POINT\\bin\\uname.exe", "-o"},
				},
				{
					command: []string{"powershell", "%CONTAINER_SANDBOX_MOUNT_POINT%/bin/uname.exe", "-o"},
				},
				{
					command: []string{"powershell", "$env:CONTAINER_SANDBOX_MOUNT_POINT/bin/uname.exe", "-o"},
				},
				{
					command:    []string{"powershell", "bin/uname.exe", "-o"},
					workingDir: "%CONTAINER_SANDBOX_MOUNT_POINT%",
				},
			},
			{
				{
					command:    []string{"powershell", "bin/uname.exe", "-o"},
					workingDir: "$env:CONTAINER_SANDBOX_MOUNT_POINT",
				},
				{
					command:    []string{"powershell", "bin\\uname.exe", "-o"},
					workingDir: "$env:CONTAINER_SANDBOX_MOUNT_POINT",
				},
				{
					command:    []string{"powershell", ".\\uname.exe", "-o"},
					workingDir: "%CONTAINER_SANDBOX_MOUNT_POINT%/bin",
				},
				{
					command:    []string{"powershell", "./uname.exe", "-o"},
					workingDir: "%CONTAINER_SANDBOX_MOUNT_POINT%/bin",
				},
			},
			{
				{
					command:    []string{"powershell", "./uname.exe", "-o"},
					workingDir: "$env:CONTAINER_SANDBOX_MOUNT_POINT\\bin\\",
				},
				{
					command: []string{"%CONTAINER_SANDBOX_MOUNT_POINT%\\bin\\uname.exe"},
					args:    []string{"-o"},
				},
				{
					command:    []string{"bin\\uname.exe"},
					args:       []string{"-o"},
					workingDir: "%CONTAINER_SANDBOX_MOUNT_POINT%",
				},
				{
					command:    []string{"uname.exe"},
					args:       []string{"-o"},
					workingDir: "%CONTAINER_SANDBOX_MOUNT_POINT%\\bin",
				},
			},
			{
				{
					command: []string{"cmd.exe"},
					args:    []string{"/c", "dir", "%CONTAINER_SANDBOX_MOUNT_POINT%\\bin\\uname.exe"},
				},
				{
					command:    []string{"cmd.exe"},
					args:       []string{"/c", "dir", "bin\\uname.exe"},
					workingDir: "%CONTAINER_SANDBOX_MOUNT_POINT%",
				},
				{
					command: []string{"powershell"},
					args:    []string{"Get-ChildItem", "-Path", "$env:CONTAINER_SANDBOX_MOUNT_POINT\\bin\\uname.exe"},
				},
				{
					command:    []string{"powershell"},
					args:       []string{"Get-ChildItem", "-Path", "bin\\uname.exe"},
					workingDir: "$env:CONTAINER_SANDBOX_MOUNT_POINT",
				},
			},
			{
				{
					command:    []string{"powershell"},
					args:       []string{"Get-ChildItem", "-Path", "uname.exe"},
					workingDir: "$env:CONTAINER_SANDBOX_MOUNT_POINT/bin",
				},
			},
		}

		for podIndex, testCaseBatch := range tests {
			image := imageutils.GetConfig(imageutils.BusyBox)
			podName := fmt.Sprintf("host-process-command-%d", podIndex)
			containers := []v1.Container{}
			for containerIndex, testCase := range testCaseBatch {
				containerName := fmt.Sprintf("host-process-command-%d-%d", podIndex, containerIndex)
				ginkgo.By(fmt.Sprintf("Adding a container '%s' to pod '%s' with command: %s, args: %s, workingDir: %s", containerName, podName, strings.Join(testCase.command, " "), strings.Join(testCase.args, " "), testCase.workingDir))

				container := v1.Container{
					Image:      image.GetE2EImage(),
					Name:       containerName,
					Command:    testCase.command,
					Args:       testCase.args,
					WorkingDir: testCase.workingDir,
				}
				containers = append(containers, container)
			}

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{
						WindowsOptions: &v1.WindowsSecurityContextOptions{
							HostProcess:   &trueVar,
							RunAsUserName: &User_NTAuthorityLocalService,
						},
					},
					HostNetwork:   true,
					Containers:    containers,
					RestartPolicy: v1.RestartPolicyNever,
					NodeSelector: map[string]string{
						"kubernetes.io/os": "windows",
					},
				},
			}
			e2epod.NewPodClient(f).Create(ctx, pod)

			ginkgo.By(fmt.Sprintf("Waiting for pod '%s' to run", podName))
			e2epod.NewPodClient(f).WaitForFinish(ctx, podName, 3*time.Minute)

			ginkgo.By("Then ensuring pod finished running successfully")
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(
				ctx,
				podName,
				metav1.GetOptions{})

			framework.ExpectNoError(err, "Error retrieving pod")

			if p.Status.Phase != v1.PodSucceeded {
				framework.Logf("Getting pod events")
				options := metav1.ListOptions{
					FieldSelector: fields.Set{
						"involvedObject.kind":      "Pod",
						"involvedObject.name":      podName,
						"involvedObject.namespace": f.Namespace.Name,
					}.AsSelector().String(),
				}
				events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, options)
				framework.ExpectNoError(err, "Error getting events for failed pod")
				for _, event := range events.Items {
					framework.Logf("%s: %s", event.Reason, event.Message)
				}
				framework.Failf("Pod '%s' did failed.", p.Name)
			}
		}

	})

	ginkgo.It("should support various volume mount types", func(ctx context.Context) {
		ns := f.Namespace

		ginkgo.By("Creating a configmap containing test data and a validation script")
		configMap := &v1.ConfigMap{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "v1",
				Kind:       "ConfigMap",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: "sample-config-map",
			},
			Data: map[string]string{
				"text":              "Lorem ipsum dolor sit amet",
				"validation-script": validation_script,
			},
		}
		_, err := f.ClientSet.CoreV1().ConfigMaps(ns.Name).Create(ctx, configMap, metav1.CreateOptions{})
		framework.ExpectNoError(err, "unable to create create configmap")

		ginkgo.By("Creating a secret containing test data")
		secret := &v1.Secret{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "v1",
				Kind:       "Secret",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: "sample-secret",
			},
			Type: v1.SecretTypeOpaque,
			Data: map[string][]byte{
				"foo": []byte("bar"),
			},
		}
		_, err = f.ClientSet.CoreV1().Secrets(ns.Name).Create(ctx, secret, metav1.CreateOptions{})
		framework.ExpectNoError(err, "unable to create secret")

		ginkgo.By("Creating a pod with a HostProcess container that uses various types of volume mounts")

		podAndContainerName := "host-process-volume-mounts"
		pod := makeTestPodWithVolumeMounts(podAndContainerName)

		e2epod.NewPodClient(f).Create(ctx, pod)

		ginkgo.By("Waiting for pod to run")
		e2epod.NewPodClient(f).WaitForFinish(ctx, podAndContainerName, 3*time.Minute)

		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, ns.Name, podAndContainerName, podAndContainerName)
		framework.ExpectNoError(err, "Error getting pod logs")
		framework.Logf("Container logs: %s", logs)

		ginkgo.By("Then ensuring pod finished running successfully")
		p, err := f.ClientSet.CoreV1().Pods(ns.Name).Get(
			ctx,
			podAndContainerName,
			metav1.GetOptions{})

		framework.ExpectNoError(err, "Error retrieving pod")
		gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodSucceeded))
	})

	ginkgo.It("metrics should report count of started and failed to start HostProcess containers", func(ctx context.Context) {
		ginkgo.By("Selecting a Windows node")
		targetNode, err := findWindowsNode(ctx, f)
		framework.ExpectNoError(err, "Error finding Windows node")
		framework.Logf("Using node: %v", targetNode.Name)

		ginkgo.By("Getting initial kubelet metrics values")
		beforeMetrics, err := getCurrentHostProcessMetrics(ctx, f, targetNode.Name)
		framework.ExpectNoError(err, "Error getting initial kubelet metrics for node")
		framework.Logf("Initial HostProcess container metrics -- StartedContainers: %v, StartedContainersErrors: %v, StartedInitContainers: %v, StartedInitContainersErrors: %v",
			beforeMetrics.StartedContainersCount, beforeMetrics.StartedContainersErrorCount, beforeMetrics.StartedInitContainersCount, beforeMetrics.StartedInitContainersErrorCount)

		ginkgo.By("Scheduling a pod with a HostProcess init container that will fail")

		badUserName := "bad-user-name"

		podName := "host-process-metrics-pod-failing-init-container"
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess:   &trueVar,
						RunAsUserName: &User_NTAuthoritySystem,
					},
				},
				HostNetwork: true,
				InitContainers: []v1.Container{
					{
						SecurityContext: &v1.SecurityContext{
							WindowsOptions: &v1.WindowsSecurityContextOptions{
								RunAsUserName: &badUserName,
							},
						},
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Name:    "failing-init-container",
						Command: []string{"foobar.exe"},
					},
				},
				Containers: []v1.Container{
					{
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Name:    "container",
						Command: []string{"cmd.exe", "/c", "exit", "/b", "0"},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				NodeName:      targetNode.Name,
			},
		}

		e2epod.NewPodClient(f).Create(ctx, pod)
		e2epod.NewPodClient(f).WaitForFinish(ctx, podName, 3*time.Minute)

		ginkgo.By("Scheduling a pod with a HostProcess container that will fail")
		podName = "host-process-metrics-pod-failing-container"
		pod = &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess:   &trueVar,
						RunAsUserName: &User_NTAuthoritySystem,
					},
				},
				HostNetwork: true,
				Containers: []v1.Container{
					{
						SecurityContext: &v1.SecurityContext{
							WindowsOptions: &v1.WindowsSecurityContextOptions{
								RunAsUserName: &badUserName,
							},
						},
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Name:    "failing-container",
						Command: []string{"foobar.exe"},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				NodeName:      targetNode.Name,
			},
		}

		e2epod.NewPodClient(f).Create(ctx, pod)
		e2epod.NewPodClient(f).WaitForFinish(ctx, podName, 3*time.Minute)

		ginkgo.By("Getting subsequent kubelet metrics values")

		afterMetrics, err := getCurrentHostProcessMetrics(ctx, f, targetNode.Name)
		framework.ExpectNoError(err, "Error getting subsequent kubelet metrics for node")
		framework.Logf("Subsequent HostProcess container metrics -- StartedContainers: %v, StartedContainersErrors: %v, StartedInitContainers: %v, StartedInitContainersErrors: %v",
			afterMetrics.StartedContainersCount, afterMetrics.StartedContainersErrorCount, afterMetrics.StartedInitContainersCount, afterMetrics.StartedInitContainersErrorCount)

		// Note: This test performs relative comparisons to ensure metrics values were logged and does not validate specific values.
		// This done so the test can be run in parallel with other tests which may start HostProcess containers on the same node.
		ginkgo.By("Ensuring metrics were updated")
		gomega.Expect(beforeMetrics.StartedContainersCount).To(gomega.BeNumerically("<", afterMetrics.StartedContainersCount), "Count of started HostProcess containers should increase")
		gomega.Expect(beforeMetrics.StartedContainersErrorCount).To(gomega.BeNumerically("<", afterMetrics.StartedContainersErrorCount), "Count of started HostProcess errors containers should increase")
		gomega.Expect(beforeMetrics.StartedInitContainersCount).To(gomega.BeNumerically("<", afterMetrics.StartedInitContainersCount), "Count of started HostProcess init containers should increase")
		gomega.Expect(beforeMetrics.StartedInitContainersErrorCount).To(gomega.BeNumerically("<", afterMetrics.StartedInitContainersErrorCount), "Count of started HostProcess errors init containers should increase")
	})

	ginkgo.It("container stats validation", func(ctx context.Context) {
		ginkgo.By("selecting a Windows node")
		targetNode, err := findWindowsNode(ctx, f)
		framework.ExpectNoError(err, "Error finding Windows node")
		framework.Logf("Using node: %v", targetNode.Name)

		ginkgo.By("schedule a pod with a HostProcess container")
		image := imageutils.GetConfig(imageutils.BusyBox)
		podName := "host-process-stats-pod"
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess:   &trueVar,
						RunAsUserName: &User_NTAuthorityLocalService,
					},
				},
				HostNetwork: true,
				Containers: []v1.Container{
					{
						Image:   image.GetE2EImage(),
						Name:    "host-process-stats",
						Command: []string{"powershell.exe", "-Command", "Write-Host 'Hello'; sleep -Seconds 600"},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				NodeName:      targetNode.Name,
			},
		}

		e2epod.NewPodClient(f).Create(ctx, pod)

		ginkgo.By("Waiting for the pod to start running")
		timeout := 3 * time.Minute
		err = e2epod.WaitForPodsRunningReady(ctx, f.ClientSet, f.Namespace.Name, 1, timeout)
		framework.ExpectNoError(err)

		ginkgo.By("Getting container stats for pod")
		statsChecked := false

		wait.Poll(2*time.Second, timeout, func() (bool, error) {
			return ensurePodsStatsOnNode(ctx, f.ClientSet, f.Namespace.Name, targetNode.Name, &statsChecked)
		})

		if !statsChecked {
			framework.Failf("Failed to get stats for pod %s/%s", f.Namespace.Name, podName)
		}
	})

	ginkgo.It("should support querying api-server using in-cluster config", func(ctx context.Context) {
		// This functionality is only support on containerd  v1.7+
		skipUnlessContainerdOneSevenOrGreater(ctx, f)

		ginkgo.By("Scheduling a pod that runs agnhost inclusterclient")
		podName := "host-process-agnhost-icc"
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess:   &trueVar,
						RunAsUserName: &User_NTAuthoritySystem,
					},
				},
				HostNetwork: true,
				Containers: []v1.Container{
					{
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Name:  "hpc-agnhost",
						// TODO: Figure out why we need to copy agnhost as agnhost.exe here
						// Possibly related to https://github.com/microsoft/hcsshim/issues/1128 and
						// https://github.com/microsoft/hcsshim/pull/1174 updating PATHEXT here doesn't
						// seem address the issue.
						Command: []string{"cmd", "/C", "copy", "c:\\hpc\\agnhost", "c:\\hpc\\agnhost.exe", "&&", "c:\\hpc\\agnhost.exe", "inclusterclient"},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				NodeSelector: map[string]string{
					"kubernetes.io/os": "windows",
				},
			},
		}

		pc := e2epod.NewPodClient(f)
		pc.Create(ctx, pod)

		ginkgo.By("Waiting for pod to run")
		err := e2epod.WaitForPodsRunningReady(ctx, f.ClientSet, f.Namespace.Name, 1, 3*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for 60 seconds")
		// We wait an additional 60 seconds after the pod is Running because the
		// `InClusterClient` app in the agnhost image waits 30 seconds before
		// starting to make queries - https://github.com/kubernetes/kubernetes/blob/ceee3783ed963497db669504241af2ca6a11610f/test/images/agnhost/inclusterclient/main.go#L51
		time.Sleep(60 * time.Second)

		ginkgo.By("Ensuring the test app was able to successfully query the api-server")
		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, podName, "hpc-agnhost")
		framework.ExpectNoError(err, "Error getting pod logs")

		framework.Logf("Logs: %s\n", logs)

		gomega.Expect(logs).Should(gomega.ContainSubstring("calling /healthz"), "app logs should contain 'calling /healthz'")

		gomega.Expect(logs).ShouldNot(gomega.ContainSubstring("status=failed"), "app logs should not contain 'status=failed'")
	})

	ginkgo.It("should run as localgroup accounts", func(ctx context.Context) {
		// This functionality is only supported on containerd v1.7+
		skipUnlessContainerdOneSevenOrGreater(ctx, f)

		ginkgo.By("Scheduling a pod that creates a localgroup from an init container then starts a container using that group")
		localGroupName := getRandomUserGrounName()
		podName := "host-process-localgroup-pod"
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess:   &trueVar,
						RunAsUserName: &User_NTAuthoritySystem,
					},
				},
				HostNetwork: true,
				InitContainers: []v1.Container{
					{
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Name:    "setup",
						Command: []string{"cmd", "/C", "net", "localgroup", localGroupName, "/add"},
					},
				},
				Containers: []v1.Container{
					{
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Name:    "localgroup-container",
						Command: []string{"cmd", "/C", "whoami"},
						SecurityContext: &v1.SecurityContext{
							WindowsOptions: &v1.WindowsSecurityContextOptions{
								RunAsUserName: &localGroupName,
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				NodeSelector: map[string]string{
					"kubernetes.io/os": "windows",
				},
			},
		}

		e2epod.NewPodClient(f).Create(ctx, pod)

		ginkgo.By("Waiting for pod to run")
		e2epod.NewPodClient(f).WaitForFinish(ctx, podName, 3*time.Minute)

		ginkgo.By("Then ensuring pod finished running successfully")
		p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(
			context.TODO(),
			podName,
			metav1.GetOptions{})

		framework.ExpectNoError(err, "error retrieving pod")
		gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodSucceeded))

		// whoami will output %COMPUTER_NAME%/{randomly generated username} here.
		// It is sufficient to just check that the logs do not container `nt authority`
		// because all of the 'built-in' accounts that can be used with HostProcess
		// are prefixed with this.
		ginkgo.By("Then ensuring pod was not running as a system account")
		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, podName, "localgroup-container")
		framework.ExpectNoError(err, "error retrieving container logs")
		framework.Logf("Pod logs: %s", logs)
		gomega.Expect(strings.ToLower(logs)).ShouldNot(gomega.ContainSubstring("nt authority"), "Container runs 'whoami' and logs should not contain 'nt authority'")
	})

}))

func makeTestPodWithVolumeMounts(name string) *v1.Pod {
	hostPathDirectoryOrCreate := v1.HostPathDirectoryOrCreate
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{
				WindowsOptions: &v1.WindowsSecurityContextOptions{
					HostProcess:   &trueVar,
					RunAsUserName: &User_NTAuthoritySystem,
				},
			},
			HostNetwork: true,
			Containers: []v1.Container{
				{
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Name:    name,
					Command: []string{"powershell.exe", "./etc/configmap/validationscript.ps1"},
					Env: []v1.EnvVar{
						{
							Name: "NODE_NAME_TEST",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{
									FieldPath: "spec.nodeName",
								},
							},
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "emptydir-volume",
							MountPath: "/etc/emptydir",
						},
						{
							Name:      "configmap-volume",
							MountPath: "/etc/configmap",
						},
						{
							Name:      "hostpath-volume",
							MountPath: "/etc/hostpath",
						},
						{
							Name:      "downwardapi-volume",
							MountPath: "/etc/downwardapi",
						},
						{
							Name:      "secret-volume",
							MountPath: "/etc/secret",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			NodeSelector: map[string]string{
				"kubernetes.io/os": "windows",
			},
			Volumes: []v1.Volume{
				{
					Name: "emptydir-volume",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{
							Medium: v1.StorageMediumDefault,
						},
					},
				},
				{
					Name: "configmap-volume",
					VolumeSource: v1.VolumeSource{
						ConfigMap: &v1.ConfigMapVolumeSource{
							LocalObjectReference: v1.LocalObjectReference{
								Name: "sample-config-map",
							},
							Items: []v1.KeyToPath{
								{
									Key:  "text",
									Path: "text.txt",
								},
								{
									Key:  "validation-script",
									Path: "validationscript.ps1",
								},
							},
						},
					},
				},
				{
					Name: "hostpath-volume",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/var/hostpath",
							Type: &hostPathDirectoryOrCreate,
						},
					},
				},
				{
					Name: "downwardapi-volume",
					VolumeSource: v1.VolumeSource{
						DownwardAPI: &v1.DownwardAPIVolumeSource{
							Items: []v1.DownwardAPIVolumeFile{
								{
									Path: "podname",
									FieldRef: &v1.ObjectFieldSelector{
										FieldPath: "metadata.name",
									},
								},
							},
						},
					},
				},
				{
					Name: "secret-volume",
					VolumeSource: v1.VolumeSource{
						Secret: &v1.SecretVolumeSource{
							SecretName: "sample-secret",
							Items: []v1.KeyToPath{
								{
									Key:  "foo",
									Path: "foo.txt",
								},
							},
						},
					},
				},
			},
		},
	}
}

type HostProcessContainersMetrics struct {
	StartedContainersCount          int64
	StartedContainersErrorCount     int64
	StartedInitContainersCount      int64
	StartedInitContainersErrorCount int64
}

// ensurePodsStatsOnNode returns a boolean after checking Pods statistics on a particular node
func ensurePodsStatsOnNode(ctx context.Context, c clientset.Interface, namespace, nodeName string, statsChecked *bool) (bool, error) {
	nodeStats, err := e2ekubelet.GetStatsSummary(ctx, c, nodeName)
	framework.ExpectNoError(err, "Error getting node stats")

	for _, podStats := range nodeStats.Pods {
		if podStats.PodRef.Namespace != namespace {
			continue
		}

		// check various pod stats
		if *podStats.CPU.UsageCoreNanoSeconds <= 0 {
			framework.Logf("Pod %s/%s stats report cpu usage equal to %v but should be greater than 0", podStats.PodRef.Namespace, podStats.PodRef.Name, *podStats.CPU.UsageCoreNanoSeconds)
			return false, nil
		}
		if *podStats.Memory.WorkingSetBytes <= 0 {
			framework.Logf("Pod %s/%s stats report memory usage equal to %v but should be greater than 0", podStats.PodRef.Namespace, podStats.PodRef.Name, *podStats.CPU.UsageCoreNanoSeconds)
			return false, nil
		}

		for _, containerStats := range podStats.Containers {
			*statsChecked = true

			// check various container stats
			if *containerStats.CPU.UsageCoreNanoSeconds <= 0 {
				framework.Logf("Container %s stats report cpu usage equal to %v but should be greater than 0", containerStats.Name, *containerStats.CPU.UsageCoreNanoSeconds)
				return false, nil
			}
			if *containerStats.Memory.WorkingSetBytes <= 0 {
				framework.Logf("Container %s stats report memory usage equal to %v but should be greater than 0", containerStats.Name, *containerStats.Memory.WorkingSetBytes)
				return false, nil
			}
			if *containerStats.Logs.UsedBytes <= 0 {
				framework.Logf("Container %s stats log usage equal to %v but should be greater than 0", containerStats.Name, *containerStats.Logs.UsedBytes)
				return false, nil
			}
		}
	}
	return true, nil
}

// getCurrentHostProcessMetrics returns a HostPRocessContainersMetrics object. Any metrics that do not have any
// values reported will be set to 0.
func getCurrentHostProcessMetrics(ctx context.Context, f *framework.Framework, nodeName string) (HostProcessContainersMetrics, error) {
	var result HostProcessContainersMetrics

	metrics, err := e2emetrics.GetKubeletMetrics(ctx, f.ClientSet, nodeName)
	if err != nil {
		return result, err
	}

	for _, sample := range metrics["started_host_process_containers_total"] {
		switch sample.Metric["container_type"] {
		case "container":
			result.StartedContainersCount = int64(sample.Value)
		case "init_container":
			result.StartedInitContainersCount = int64(sample.Value)
		}
	}

	// note: accumulate failures of all types (ErrImagePull, RunContainerError, etc)
	// for each container type here.
	for _, sample := range metrics["started_host_process_containers_errors_total"] {
		switch sample.Metric["container_type"] {
		case "container":
			result.StartedContainersErrorCount += int64(sample.Value)
		case "init_container":
			result.StartedInitContainersErrorCount += int64(sample.Value)
		}
	}

	return result, nil
}
