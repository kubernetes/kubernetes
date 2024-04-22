/*
Copyright 2023 The Kubernetes Authors.

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
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var (
	WindowsHyperVContainerRuntimeClass = "runhcs-wcow-hypervisor"
)

var _ = sigDescribe(feature.WindowsHyperVContainers, "HyperV containers", skipUnlessWindows(func() {
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
	})

	f := framework.NewDefaultFramework("windows-hyperv-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should start a hyperv isolated container", func(ctx context.Context) {

		// HyperV isolated containers are only supported on containerd 1.7+
		skipUnlessContainerdOneSevenOrGreater(ctx, f)

		// check if hyperv runtime class is on node and skip otherwise
		// Note: the runtime class is expected to be added to a cluster before running this test.
		// see https://github.com/kubernetes-sigs/windows-testing/tree/master/helpers/hyper-v-mutating-webhook/hyperv-runtimeclass.yaml
		// for an example.
		_, err := f.ClientSet.NodeV1().RuntimeClasses().Get(ctx, WindowsHyperVContainerRuntimeClass, metav1.GetOptions{})
		if err != nil {
			framework.Logf("error getting runtime class: %v", err)
			e2eskipper.Skipf("skipping test because runhcs-wcow-hypervisor runtime class is not present")
		}

		ginkgo.By("selecting a Windows node")
		targetNode, err := findWindowsNode(ctx, f)
		framework.ExpectNoError(err, "error finding Windows node")
		framework.Logf("Using node: %v", targetNode.Name)

		ginkgo.By("schedule a pod to that node")
		image := imageutils.GetE2EImage(imageutils.BusyBox)
		hypervPodName := "hyperv-test-pod"
		hypervPod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: hypervPodName,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Image:   image,
						Name:    "busybox-1",
						Command: []string{"powershell.exe", "-Command", "Write-Host 'Hello'; sleep -Seconds 600"},
					},
					{
						Image:   image,
						Name:    "busybox-2",
						Command: []string{"powershell.exe", "-Command", "Write-Host 'Hello'; sleep -Seconds 600"},
					},
				},
				RestartPolicy:    v1.RestartPolicyNever,
				RuntimeClassName: &WindowsHyperVContainerRuntimeClass,
				NodeName:         targetNode.Name,
			},
		}

		pc := e2epod.NewPodClient(f)

		pc.Create(ctx, hypervPod)
		ginkgo.By("waiting for the pod to be running")
		timeout := 3 * time.Minute
		err = e2epod.WaitForPodsRunningReady(ctx, f.ClientSet, f.Namespace.Name, 1, timeout)
		framework.ExpectNoError(err)

		ginkgo.By("creating a host process container in another pod to verify the pod is running hyperv isolated containers")

		// Note: each pod runs in a separate UVM so even though we are scheduling 2 containers in the test pod
		// we should only expect a single UVM to be running on the host.
		podName := "validation-pod"
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
						Image:   image,
						Name:    "container",
						Command: []string{"powershell.exe", "-Command", "$vms = Get-ComputeProcess | Where-Object { ($_.Type -EQ 'VirtualMachine') -and ($_.Owner -EQ 'containerd-shim-runhcs-v1.exe') } ; if ($vms.Length -le 0) { throw 'error' }"},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				NodeName:      targetNode.Name,
			},
		}

		pc.Create(ctx, pod)
		ginkgo.By("waiting for the pod to be run")
		pc.WaitForFinish(ctx, podName, timeout)

		ginkgo.By("then ensuring pod finished running successfully")
		p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, podName, metav1.GetOptions{})
		framework.ExpectNoError(err, "error getting pod")

		if p.Status.Phase != v1.PodSucceeded {
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, podName, "container")
			if err != nil {
				framework.Logf("Error pulling logs: %v", err)
			}
			framework.Logf("Pod phase: %v\nlogs:\n%s", p.Status.Phase, logs)
		}

		gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodSucceeded), "pod should have succeeded")
	})
}))
