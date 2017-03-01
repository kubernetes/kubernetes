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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/sysctl"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Sysctls", func() {
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
						Image: "gcr.io/google_containers/busybox:1.24",
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		return &pod
	}

	BeforeEach(func() {
		podClient = f.PodClient()
	})

	It("should support sysctls", func() {
		pod := testPod()
		pod.Annotations[v1.SysctlsPodAnnotationKey] = v1.PodAnnotationsFromSysctls([]v1.Sysctl{
			{
				Name:  "kernel.shm_rmid_forced",
				Value: "1",
			},
		})
		pod.Spec.Containers[0].Command = []string{"/bin/sysctl", "kernel.shm_rmid_forced"}

		By("Creating a pod with the kernel.shm_rmid_forced sysctl")
		pod = podClient.Create(pod)

		By("Watching for error events or started pod")
		// watch for events instead of termination of pod because the kubelet deletes
		// failed pods without running containers. This would create a race as the pod
		// might have already been deleted here.
		ev, err := f.PodClient().WaitForErrorEventOrSuccess(pod)
		Expect(err).NotTo(HaveOccurred())
		if ev != nil && ev.Reason == sysctl.UnsupportedReason {
			framework.Skipf("No sysctl support in Docker <1.12")
		}
		Expect(ev).To(BeNil())

		By("Waiting for pod completion")
		err = f.WaitForPodNoLongerRunning(pod.Name)
		Expect(err).NotTo(HaveOccurred())
		pod, err = podClient.Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		By("Checking that the pod succeeded")
		Expect(pod.Status.Phase).To(Equal(v1.PodSucceeded))

		By("Getting logs from the pod")
		log, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
		Expect(err).NotTo(HaveOccurred())

		By("Checking that the sysctl is actually updated")
		Expect(log).To(ContainSubstring("kernel.shm_rmid_forced = 1"))
	})

	It("should support unsafe sysctls which are actually whitelisted", func() {
		pod := testPod()
		pod.Annotations[v1.UnsafeSysctlsPodAnnotationKey] = v1.PodAnnotationsFromSysctls([]v1.Sysctl{
			{
				Name:  "kernel.shm_rmid_forced",
				Value: "1",
			},
		})
		pod.Spec.Containers[0].Command = []string{"/bin/sysctl", "kernel.shm_rmid_forced"}

		By("Creating a pod with the kernel.shm_rmid_forced sysctl")
		pod = podClient.Create(pod)

		By("Watching for error events or started pod")
		// watch for events instead of termination of pod because the kubelet deletes
		// failed pods without running containers. This would create a race as the pod
		// might have already been deleted here.
		ev, err := f.PodClient().WaitForErrorEventOrSuccess(pod)
		Expect(err).NotTo(HaveOccurred())
		if ev != nil && ev.Reason == sysctl.UnsupportedReason {
			framework.Skipf("No sysctl support in Docker <1.12")
		}
		Expect(ev).To(BeNil())

		By("Waiting for pod completion")
		err = f.WaitForPodNoLongerRunning(pod.Name)
		Expect(err).NotTo(HaveOccurred())
		pod, err = podClient.Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		By("Checking that the pod succeeded")
		Expect(pod.Status.Phase).To(Equal(v1.PodSucceeded))

		By("Getting logs from the pod")
		log, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
		Expect(err).NotTo(HaveOccurred())

		By("Checking that the sysctl is actually updated")
		Expect(log).To(ContainSubstring("kernel.shm_rmid_forced = 1"))
	})

	It("should reject invalid sysctls", func() {
		pod := testPod()
		pod.Annotations[v1.SysctlsPodAnnotationKey] = v1.PodAnnotationsFromSysctls([]v1.Sysctl{
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
		})
		pod.Annotations[v1.UnsafeSysctlsPodAnnotationKey] = v1.PodAnnotationsFromSysctls([]v1.Sysctl{
			{
				Name:  "kernel.shmall",
				Value: "100000000",
			},
			{
				Name:  "bar..",
				Value: "42",
			},
			{
				Name:  "safe-and-unsafe",
				Value: "100000000",
			},
		})

		By("Creating a pod with one valid and two invalid sysctls")
		client := f.ClientSet.Core().Pods(f.Namespace.Name)
		_, err := client.Create(pod)

		Expect(err).NotTo(BeNil())
		Expect(err.Error()).To(ContainSubstring(`Invalid value: "foo-"`))
		Expect(err.Error()).To(ContainSubstring(`Invalid value: "bar.."`))
		Expect(err.Error()).To(ContainSubstring(`safe-and-unsafe`))
		Expect(err.Error()).NotTo(ContainSubstring("kernel.shmmax"))
	})

	It("should not launch unsafe, but not explicitly enabled sysctls on the node", func() {
		pod := testPod()
		pod.Annotations[v1.SysctlsPodAnnotationKey] = v1.PodAnnotationsFromSysctls([]v1.Sysctl{
			{
				Name:  "kernel.msgmax",
				Value: "10000000000",
			},
		})

		By("Creating a pod with a greylisted, but not whitelisted sysctl on the node")
		pod = podClient.Create(pod)

		By("Watching for error events or started pod")
		// watch for events instead of termination of pod because the kubelet deletes
		// failed pods without running containers. This would create a race as the pod
		// might have already been deleted here.
		ev, err := f.PodClient().WaitForErrorEventOrSuccess(pod)
		Expect(err).NotTo(HaveOccurred())
		if ev != nil && ev.Reason == sysctl.UnsupportedReason {
			framework.Skipf("No sysctl support in Docker <1.12")
		}

		By("Checking that the pod was rejected")
		Expect(ev).ToNot(BeNil())
		Expect(ev.Reason).To(Equal("SysctlForbidden"))
	})
})
