/*
Copyright 2017 The Kubernetes Authors.

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

package e2e_node

import (
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("Docker features [Feature:Docker]", func() {
	f := framework.NewDefaultFramework("docker-feature-test")

	BeforeEach(func() {
		framework.RunIfContainerRuntimeIs("docker")
	})

	Context("when shared PID namespace is enabled", func() {
		It("processes in different containers of the same pod should be able to see each other", func() {
			// TODO(yguo0905): Change this test to run unless the runtime is
			// Docker and its version is <1.13.
			By("Check whether shared PID namespace is enabled.")
			isEnabled, err := isSharedPIDNamespaceEnabled()
			framework.ExpectNoError(err)
			if !isEnabled {
				framework.Skipf("Skipped because shared PID namespace is not enabled.")
			}

			By("Create a pod with two containers.")
			f.PodClient().CreateSync(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "shared-pid-ns-test-pod"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "test-container-1",
							Image:   "gcr.io/google_containers/busybox:1.24",
							Command: []string{"/bin/top"},
						},
						{
							Name:    "test-container-2",
							Image:   "gcr.io/google_containers/busybox:1.24",
							Command: []string{"/bin/sleep"},
							Args:    []string{"10000"},
						},
					},
				},
			})

			By("Check if the process in one container is visible to the process in the other.")
			pid1 := f.ExecCommandInContainer("shared-pid-ns-test-pod", "test-container-1", "/bin/pidof", "top")
			pid2 := f.ExecCommandInContainer("shared-pid-ns-test-pod", "test-container-2", "/bin/pidof", "top")
			if pid1 != pid2 {
				framework.Failf("PIDs are not the same in different containers: test-container-1=%v, test-container-2=%v", pid1, pid2)
			}
		})
	})
})
