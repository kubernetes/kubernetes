/*
Copyright 2016 The Kubernetes Authors.

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

package e2e_cri

import (
	internalapi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/test/e2e_cri/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Test CRI container", func() {
	f := framework.NewDefaultCRIFramework("CRI-container-test")

	var c internalapi.RuntimeService

	BeforeEach(func() {
		c = f.CRIClient.CRIRuntimeClient
	})
	Context("test basic operations on container", func() {
		var podID string
		var podConfig *runtimeapi.PodSandboxConfig

		BeforeEach(func() {
			podID, podConfig = createPodSandboxForContainer(c)
		})

		AfterEach(func() {
			By("stop PodSandbox")
			c.StopPodSandbox(podID)
			By("delete PodSandbox")
			c.RemovePodSandbox(podID)
		})

		It("test create container", func() {
			By("test create container")
			containerID := testCreateContainer(c, podID, podConfig)

			By("test list container")
			containers := listContainerForIDOrFail(c, containerID)
			Expect(containerFound(containers, containerID)).To(BeTrue(), "container should be created")
		})

		It("test start container", func() {
			By("create container")
			containerID := createContainerOrFail(c, "container-for-create-test-", podID, podConfig)

			By("test start container")
			testStartContainer(c, containerID)
		})

		It("test stop container", func() {
			By("create container")
			containerID := createContainerOrFail(c, "container-for-create-test-", podID, podConfig)

			By("start container")
			startContainerOrFail(c, containerID)

			By("test stop container")
			testStopContainer(c, containerID)
		})

		It("test remove container", func() {
			By("create container")
			containerID := createContainerOrFail(c, "container-for-create-test-", podID, podConfig)

			By("test remove container")
			removeContainerOrFail(c, containerID)
			containers := listContainerForIDOrFail(c, containerID)
			Expect(containerFound(containers, containerID)).To(BeFalse(), "container should be removed")
		})
	})
})
