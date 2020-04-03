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

package common

import (
	"context"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Sysctls [LinuxOnly] [NodeFeature:Sysctls]", func() {

	ginkgo.BeforeEach(func() {
		// sysctl is not supported on Windows.
		e2eskipper.SkipIfNodeOSDistroIs("windows")
	})

	f := framework.NewDefaultFramework("sysctl")
	var podClient *framework.PodClient

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
		podClient = f.PodClient()
	})

	ginkgo.It("should support sysctls", func() {
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
		pod = podClient.Create(pod)

		ginkgo.By("Watching for error events or started pod")
		// watch for events instead of termination of pod because the kubelet deletes
		// failed pods without running containers. This would create a race as the pod
		// might have already been deleted here.
		ev, err := f.PodClient().WaitForErrorEventOrSuccess(pod)
		framework.ExpectNoError(err)
		gomega.Expect(ev).To(gomega.BeNil())

		ginkgo.By("Waiting for pod completion")
		err = e2epod.WaitForPodNoLongerRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
		pod, err = podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Checking that the pod succeeded")
		framework.ExpectEqual(pod.Status.Phase, v1.PodSucceeded)

		ginkgo.By("Getting logs from the pod")
		log, err := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
		framework.ExpectNoError(err)

		ginkgo.By("Checking that the sysctl is actually updated")
		gomega.Expect(log).To(gomega.ContainSubstring("kernel.shm_rmid_forced = 1"))
	})

	ginkgo.It("should support unsafe sysctls which are actually whitelisted", func() {
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
		pod = podClient.Create(pod)

		ginkgo.By("Watching for error events or started pod")
		// watch for events instead of termination of pod because the kubelet deletes
		// failed pods without running containers. This would create a race as the pod
		// might have already been deleted here.
		ev, err := f.PodClient().WaitForErrorEventOrSuccess(pod)
		framework.ExpectNoError(err)
		gomega.Expect(ev).To(gomega.BeNil())

		ginkgo.By("Waiting for pod completion")
		err = e2epod.WaitForPodNoLongerRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err)
		pod, err = podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Checking that the pod succeeded")
		framework.ExpectEqual(pod.Status.Phase, v1.PodSucceeded)

		ginkgo.By("Getting logs from the pod")
		log, err := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
		framework.ExpectNoError(err)

		ginkgo.By("Checking that the sysctl is actually updated")
		gomega.Expect(log).To(gomega.ContainSubstring("kernel.shm_rmid_forced = 1"))
	})

	ginkgo.It("should reject invalid sysctls", func() {
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
		_, err := client.Create(context.TODO(), pod, metav1.CreateOptions{})

		gomega.Expect(err).NotTo(gomega.BeNil())
		gomega.Expect(err.Error()).To(gomega.ContainSubstring(`Invalid value: "foo-"`))
		gomega.Expect(err.Error()).To(gomega.ContainSubstring(`Invalid value: "bar.."`))
		gomega.Expect(err.Error()).NotTo(gomega.ContainSubstring(`safe-and-unsafe`))
		gomega.Expect(err.Error()).NotTo(gomega.ContainSubstring("kernel.shmmax"))
	})

	ginkgo.It("should not launch unsafe, but not explicitly enabled sysctls on the node", func() {
		pod := testPod()
		pod.Spec.SecurityContext = &v1.PodSecurityContext{
			Sysctls: []v1.Sysctl{
				{
					Name:  "kernel.msgmax",
					Value: "10000000000",
				},
			},
		}

		ginkgo.By("Creating a pod with a greylisted, but not whitelisted sysctl on the node")
		pod = podClient.Create(pod)

		ginkgo.By("Watching for error events or started pod")
		// watch for events instead of termination of pod because the kubelet deletes
		// failed pods without running containers. This would create a race as the pod
		// might have already been deleted here.
		ev, err := f.PodClient().WaitForErrorEventOrSuccess(pod)
		framework.ExpectNoError(err)

		ginkgo.By("Checking that the pod was rejected")
		gomega.Expect(ev).ToNot(gomega.BeNil())
		framework.ExpectEqual(ev.Reason, "SysctlForbidden")
	})
})
