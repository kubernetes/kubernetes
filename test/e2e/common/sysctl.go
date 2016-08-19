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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Sysctls", func() {
	f := framework.NewDefaultFramework("sysctl")
	var podClient *framework.PodClient

	testPod := func() *api.Pod {
		podName := "sysctl-" + string(uuid.NewUUID())
		pod := api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:        podName,
				Annotations: map[string]string{},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "test-container",
						Image: "gcr.io/google_containers/busybox:1.24",
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		return &pod
	}

	BeforeEach(func() {
		podClient = f.PodClient()
	})

	It("should support sysctls", func() {
		pod := testPod()
		pod.Annotations[api.SysctlsPodAnnotationKey] = api.PodAnnotationsFromSysctls([]api.Sysctl{
			{
				Name:  "kernel.shm_rmid_forced",
				Value: "1",
			},
		})
		pod.Spec.Containers[0].Command = []string{"/bin/sysctl", "kernel.shm_rmid_forced"}

		By("Creating a pod with the kernel.shm_rmid_forced sysctl")
		pod = podClient.Create(pod)

		By("Wait for pod no longer running")
		err := f.WaitForPodNoLongerRunning(pod.Name)
		Expect(err).NotTo(HaveOccurred())
		pod, err = podClient.Get(pod.Name)
		Expect(err).NotTo(HaveOccurred())

		if pod.Status.Phase == api.PodFailed && pod.Status.Reason == "SysctlUnsupported" {
			framework.Skipf("No sysctl support in Docker <1.12")
		}

		By("Checking that the pod succeeded")
		Expect(pod.Status.Phase).To(Equal(api.PodSucceeded))

		By("Getting logs from the pod")
		log, err := framework.GetPodLogs(f.Client, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
		Expect(err).NotTo(HaveOccurred())

		By("Checking that the sysctl is actually updated")
		Expect(log).To(ContainSubstring("kernel.shm_rmid_forced = 1"))
	})

	It("should reject invalid sysctls", func() {
		pod := testPod()
		pod.Annotations[api.SysctlsPodAnnotationKey] = api.PodAnnotationsFromSysctls([]api.Sysctl{
			{
				Name:  "foo-",
				Value: "bar",
			},
			{
				Name:  "kernel.shmmax",
				Value: "100000000",
			},
			{
				Name:  "bar..",
				Value: "42",
			},
		})

		By("Creating a pod with one valid and two invalid sysctls")
		client := f.Client.Pods(f.Namespace.Name)
		_, err := client.Create(pod)
		defer client.Delete(pod.Name, nil)

		Expect(err).NotTo(BeNil())
		Expect(err.Error()).To(ContainSubstring(`Invalid value: "foo-"`))
		Expect(err.Error()).To(ContainSubstring(`Invalid value: "bar.."`))
		Expect(err.Error()).NotTo(ContainSubstring("kernel.shmmax"))
	})

	It("should not launch greylisted, but not whitelisted sysctls on the node", func() {
		sysctl := "kernel.msgmax"
		pod := testPod()
		pod.Annotations[api.SysctlsPodAnnotationKey] = api.PodAnnotationsFromSysctls([]api.Sysctl{
			{
				Name:  sysctl,
				Value: "10000000000",
			},
		})

		By("Creating a pod with a greylisted, but not whitelisted sysctl on the node")
		pod = podClient.Create(pod)

		By("Wait for pod no longer running")
		err := f.WaitForPodNoLongerRunning(pod.Name)
		Expect(err).NotTo(HaveOccurred())
		pod, err = podClient.Get(pod.Name)
		Expect(err).NotTo(HaveOccurred())

		if pod.Status.Phase == api.PodFailed && pod.Status.Reason == "SysctlUnsupported" {
			framework.Skipf("No sysctl support in Docker <1.12")
		}

		By("Checking that the pod was rejected")
		Expect(pod.Status.Phase).To(Equal(api.PodFailed))
		Expect(pod.Status.Reason).To(Equal("SysctlForbidden"))
	})
})
