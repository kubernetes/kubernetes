/*
Copyright 2014 The Kubernetes Authors.

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

package node

import (
	"context"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/environment"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Sysctls [LinuxOnly]", framework.WithNodeConformance(), func() {

	ginkgo.BeforeEach(func() {
		// sysctl is not supported on Windows.
		e2eskipper.SkipIfNodeOSDistroIs("windows")
	})

	f := framework.NewDefaultFramework("sysctl")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var podClient *e2epod.PodClient

	testPod := func() *v1.Pod {
		podName := "sysctl-" + string(uuid.NewUUID())
		pod := v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:        podName,
				Annotations: map[string]string{},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "test-container",
						Image: imageutils.GetE2EImage(imageutils.BusyBox),
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		return &pod
	}

	ginkgo.BeforeEach(func() {
		podClient = e2epod.NewPodClient(f)
	})

	/*
	  Release: v1.21
	  Testname: Sysctl, test sysctls
	  Description: Pod is created with kernel.shm_rmid_forced sysctl. Kernel.shm_rmid_forced must be set to 1
	  [LinuxOnly]: This test is marked as LinuxOnly since Windows does not support sysctls
	  [Environment:NotInUserNS]: The test fails in UserNS (as expected): `open /proc/sys/kernel/shm_rmid_forced: permission denied`
	*/
	framework.ConformanceIt("should support sysctls", environment.NotInUserNS, func(ctx context.Context) {
		pod := testPod()
		pod.Spec.SecurityContext = &v1.PodSecurityContext{
			Sysctls: []v1.Sysctl{
				{
					Name:  "kernel.shm_rmid_forced",
					Value: "1",
				},
			},
		}
		pod.Spec.Containers[0].Command = []string{"/bin/sysctl", "kernel.shm_rmid_forced"}

		ginkgo.By("Creating a pod with the kernel.shm_rmid_forced sysctl")
		pod = podClient.Create(ctx, pod)

		ginkgo.By("Watching for error events or started pod")
		// watch for events instead of termination of pod because the kubelet deletes
		// failed pods without running containers. This would create a race as the pod
		// might have already been deleted here.
		ev, err := e2epod.NewPodClient(f).WaitForErrorEventOrSuccess(ctx, pod)
		framework.ExpectNoError(err)
		gomega.Expect(ev).To(gomega.BeNil())

		ginkgo.By("Waiting for pod completion")
		err = e2epod.WaitForPodNoLongerRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
		pod, err = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Checking that the pod succeeded")
		gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodSucceeded))

		ginkgo.By("Getting logs from the pod")
		log, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
		framework.ExpectNoError(err)

		ginkgo.By("Checking that the sysctl is actually updated")
		gomega.Expect(log).To(gomega.ContainSubstring("kernel.shm_rmid_forced = 1"))
	})

	/*
	  Release: v1.21
	  Testname: Sysctls, reject invalid sysctls
	  Description: Pod is created with one valid and two invalid sysctls. Pod should not apply invalid sysctls.
	  [LinuxOnly]: This test is marked as LinuxOnly since Windows does not support sysctls
	*/
	framework.ConformanceIt("should reject invalid sysctls", func(ctx context.Context) {
		pod := testPod()
		pod.Spec.SecurityContext = &v1.PodSecurityContext{
			Sysctls: []v1.Sysctl{
				// Safe parameters
				{
					Name:  "foo-",
					Value: "bar",
				},
				{
					Name:  "kernel.shmmax",
					Value: "100000000",
				},
				{
					Name:  "safe-and-unsafe",
					Value: "100000000",
				},
				{
					Name:  "bar..",
					Value: "42",
				},
			},
		}

		ginkgo.By("Creating a pod with one valid and two invalid sysctls")
		client := f.ClientSet.CoreV1().Pods(f.Namespace.Name)
		_, err := client.Create(ctx, pod, metav1.CreateOptions{})

		gomega.Expect(err).To(gomega.MatchError(gomega.SatisfyAll(
			gomega.ContainSubstring(`Invalid value: "foo-"`),
			gomega.ContainSubstring(`Invalid value: "bar.."`),
			gomega.Not(gomega.ContainSubstring(`safe-and-unsafe`)),
			gomega.Not(gomega.ContainSubstring("kernel.shmmax")),
		)))
	})

	// Pod is created with kernel.msgmax, an unsafe sysctl.
	ginkgo.It("should not launch unsafe, but not explicitly enabled sysctls on the node", func(ctx context.Context) {
		pod := testPod()
		pod.Spec.SecurityContext = &v1.PodSecurityContext{
			Sysctls: []v1.Sysctl{
				{
					Name:  "kernel.msgmax",
					Value: "10000000000",
				},
			},
		}

		ginkgo.By("Creating a pod with an ignorelisted, but not allowlisted sysctl on the node")
		pod = podClient.Create(ctx, pod)

		ginkgo.By("Wait for pod failed reason")
		// watch for pod failed reason instead of termination of pod
		err := e2epod.WaitForPodFailedReason(ctx, f.ClientSet, pod, "SysctlForbidden", f.Timeouts.PodStart)
		framework.ExpectNoError(err)
	})

	/*
	  Release: v1.23
	  Testname: Sysctl, test sysctls supports slashes
	  Description: Pod is created with kernel/shm_rmid_forced sysctl. Support slashes as sysctl separator. The '/' separator is also accepted in place of a '.'
	  [LinuxOnly]: This test is marked as LinuxOnly since Windows does not support sysctls
	  [Environment:NotInUserNS]: The test fails in UserNS (as expected): `open /proc/sys/kernel/shm_rmid_forced: permission denied`
	*/
	f.It("should support sysctls with slashes as separator", environment.NotInUserNS, func(ctx context.Context) {
		pod := testPod()
		pod.Spec.SecurityContext = &v1.PodSecurityContext{
			Sysctls: []v1.Sysctl{
				{
					Name:  "kernel/shm_rmid_forced",
					Value: "1",
				},
			},
		}
		pod.Spec.Containers[0].Command = []string{"/bin/sysctl", "kernel/shm_rmid_forced"}

		ginkgo.By("Creating a pod with the kernel/shm_rmid_forced sysctl")
		pod = podClient.Create(ctx, pod)

		ginkgo.By("Watching for error events or started pod")
		// watch for events instead of termination of pod because the kubelet deletes
		// failed pods without running containers. This would create a race as the pod
		// might have already been deleted here.
		ev, err := e2epod.NewPodClient(f).WaitForErrorEventOrSuccess(ctx, pod)
		framework.ExpectNoError(err)
		gomega.Expect(ev).To(gomega.BeNil())

		ginkgo.By("Waiting for pod completion")
		err = e2epod.WaitForPodNoLongerRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
		pod, err = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Checking that the pod succeeded")
		gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodSucceeded))

		ginkgo.By("Getting logs from the pod")
		log, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
		framework.ExpectNoError(err)

		ginkgo.By("Checking that the sysctl is actually updated")
		// Note that either "/" or "."  may be used as separators within sysctl variable names.
		// "kernel.shm_rmid_forced=1" and "kernel/shm_rmid_forced=1" are equivalent.
		// Run "/bin/sysctl kernel/shm_rmid_forced" command on Linux system
		// The displayed result is "kernel.shm_rmid_forced=1"
		// Therefore, the substring that needs to be checked for the obtained pod log is
		// "kernel.shm_rmid_forced=1" instead of "kernel/shm_rmid_forced=1".
		gomega.Expect(log).To(gomega.ContainSubstring("kernel.shm_rmid_forced = 1"))
	})
})
