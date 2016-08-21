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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func testPod(hostIPC bool, hostPID bool) *api.Pod {
	podName := "sysctl-" + string(uuid.NewUUID())
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:        podName,
			Annotations: map[string]string{},
		},
		Spec: api.PodSpec{
			SecurityContext: &api.PodSecurityContext{
				HostIPC: hostIPC,
				HostPID: hostPID,
			},
			Containers: []api.Container{
				{
					Name:  "test-container",
					Image: "gcr.io/google_containers/busybox:1.24",
				},
			},
			RestartPolicy: api.RestartPolicyNever,
		},
	}

	return pod
}

var _ = framework.KubeDescribe("Sysctls", func() {
	f := framework.NewDefaultFramework("sysctl")

	It("should support sysctls", func() {
		pod := testPod(false, false)
		pod.Annotations[api.SysctlsPodAnnotationKey] = api.PodAnnotationsFromSysctls([]api.Sysctl{
			{
				Name:  "kernel.shm_rmid_forced",
				Value: "1",
			},
		})
		pod.Spec.Containers[0].Command = []string{"/bin/sysctl", "kernel.shm_rmid_forced"}

		f.TestContainerOutput(fmt.Sprintf("pod.Annotations[%s]", api.SysctlsPodAnnotationKey), pod, 0, []string{
			"kernel.shm_rmid_forced = 1",
		})
	})

	It("should reject invalid sysctls", func() {
		pod := testPod(false, false)
		pod.Annotations[api.SysctlsPodAnnotationKey] = api.PodAnnotationsFromSysctls([]api.Sysctl{
			{
				Name:  "net.foo-bar",
				Value: "bar",
			},
			{
				Name:  "vm.swappiness",
				Value: "42",
			},
		})

		By("Creating a pod with net.foo-bar sysctl")
		client := f.Client.Pods(f.Namespace.Name)
		_, err := client.Create(pod)
		defer client.Delete(pod.Name, nil)

		Expect(err).NotTo(BeNil())
		Expect(err.Error()).To(ContainSubstring(`Invalid value: "net.foo-bar"`))
		Expect(err.Error()).To(ContainSubstring(`Forbidden: sysctl "vm.swappiness" cannot be set in a pod`))
	})

	It("should not launch greylisted, but not whitelisted sysctls on the node", func() {
		sysctl := "kernel.msgmax"
		pod := testPod(false, false)
		pod.Annotations[api.SysctlsPodAnnotationKey] = api.PodAnnotationsFromSysctls([]api.Sysctl{
			{
				Name:  sysctl,
				Value: "10000000000",
			},
		})

		By("Creating a pod with a greylisted, but not whitelisted sysctl on the node")
		client := f.Client.Pods(f.Namespace.Name)
		pod, err := client.Create(pod)
		Expect(err).To(BeNil())
		defer client.Delete(pod.Name, nil)

		By("Watching for error events")
		var failEv api.Event
		err = wait.Poll(framework.Poll, framework.PodStartTimeout, func() (bool, error) {
			es, err := f.Client.Events(f.Namespace.Name).Search(pod)
			if err != nil {
				return false, fmt.Errorf("error in listing events: %s", err)
			}
			for _, e := range es.Items {
				if e.Reason == events.FailedSync {
					failEv = e
					return true, nil
				}
			}
			return false, nil
		})
		Expect(err).NotTo(HaveOccurred())

		// the exact error message might depend on the container runtime. But
		// at least it should say something about the non-namespaces sysctl.
		Expect(failEv.Message).Should(ContainSubstring(sysctl))
	})
})
