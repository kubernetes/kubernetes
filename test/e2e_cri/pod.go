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
	"k8s.io/kubernetes/test/e2e_cri/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Test CRI PodSandbox", func() {
	f := framework.NewDefaultCRIFramework("CRI-PodSandbox-test")

	var c internalapi.RuntimeService

	BeforeEach(func() {
		c = f.CRIClient.CRIRuntimeClient
	})

	It("test get version info", func() {
		testGetVersion(c)
	})

	Context("test basic operations on PodSandbox", func() {
		var podID string

		AfterEach(func() {
			By("stop PodSandbox")
			c.StopPodSandbox(podID)
			By("delete PodSandbox")
			c.RemovePodSandbox(podID)
		})

		It("test run PodSandbox", func() {
			By("test run PodSandbox")
			podID = testRunPodSandbox(c)

			By("test list PodSandbox")
			pods := listPodSanboxForIDOrFail(c, podID)
			Expect(podSandboxFound(pods, podID)).To(BeTrue(), "PodSandbox should be listed")
		})

		It("test stop PodSandbox", func() {
			By("run PodSandbox")
			podID = runPodSandboxOrFail(c, "PodSandbox-for-create-test-")

			By("test stop PodSandbox")
			testStopPodSandbox(c, podID)
		})

		It("test remove PodSandbox", func() {
			By("test run PodSandbox")
			podID = runPodSandboxOrFail(c, "PodSandbox-for-create-test-")

			By("stop PodSandbox")
			stopPodSandboxOrFail(c, podID)

			By("test remove PodSandbox")
			removePodSandboxOrFail(c, podID)
			pods := listPodSanboxForIDOrFail(c, podID)
			Expect(podSandboxFound(pods, podID)).To(BeFalse(), "PodSandbox should be removed")
		})
	})
})
