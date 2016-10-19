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
	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func testCreatePodSandbox(c internalApi.RuntimeService) string {
	By("create a PodSandbox with name")
	name := "PodSandbox-for-create-test-" + string(uuid.NewUUID())
	config := &runtimeApi.PodSandboxConfig{
		Metadata: &runtimeApi.PodSandboxMetadata{
			Name: &name,
		},
	}

	podID, err := c.RunPodSandbox(config)
	framework.ExpectNoError(err, "Failed to create PodSandbox: %v", err)
	framework.Logf("Created PodSandbox %s\n", podID)

	By("get PodSandbox status")
	status, err := c.PodSandboxStatus(podID)
	framework.ExpectNoError(err, "Failed to get PodSandbox %s status: %v", podID, err)
	Expect(*status.State).To(Equal(runtimeApi.PodSandBoxState_READY), "PodSandbox state should be ready")
	return podID
}

var _ = framework.KubeDescribe("Test CRI PodSandbox", func() {
	f := framework.NewDefaultCRIFramework("CRI-PodSandbox-test")

	var c internalApi.RuntimeService

	BeforeEach(func() {
		c = f.CRIClient.CRIRuntimeClient
	})

	It("test version", func() {
		By("Get CRI runtime name")
		version, err := c.Version("test")
		framework.ExpectNoError(err, "Failed to get version: %v", err)
		framework.Logf("get version runtime name " + *version.RuntimeName)
	})

	It("test create PodSandbox", func() {
		podID := testCreatePodSandbox(c)
		defer func() {
			By("delete PodSandbox")
			c.StopPodSandbox(podID)
			c.RemovePodSandbox(podID)
		}()
	})

	It("test stop PodSandbox", func() {
		podID := testCreatePodSandbox(c)
		defer func() {
			By("delete pod PodSandbox")
			c.StopPodSandbox(podID)
			c.RemovePodSandbox(podID)
		}()

		By("stop PodSandbox with poId")
		err := c.StopPodSandbox(podID)
		framework.ExpectNoError(err, "Failed to stop PodSandbox: %v", err)
		framework.Logf("Stopped PodSandbox %s\n", podID)

		By("get PodSandbox status")
		status, err := c.PodSandboxStatus(podID)
		framework.ExpectNoError(err, "Failed to get PodSandbox %s status: %v", podID, err)
		Expect(*status.State).To(Equal(runtimeApi.PodSandBoxState_NOTREADY), "PodSandbox state should be not ready")
	})

	It("test remove PodSandbox", func() {
		podID := testCreatePodSandbox(c)

		By("stop PodSandbox with poId")
		err := c.StopPodSandbox(podID)
		framework.ExpectNoError(err, "Failed to stop PodSandbox: %v", err)
		framework.Logf("Stopped PodSandbox %s\n", podID)

		By("remove PodSandbox with podID")
		err = c.RemovePodSandbox(podID)
		framework.ExpectNoError(err, "Failed to remove PodSandbox: %v", err)
		framework.Logf("Removed PodSandbox %s\n", podID)

		By("list PodSandbox with podID")
		filter := &runtimeApi.PodSandboxFilter{
			Id: &podID,
		}
		pods, err := c.ListPodSandbox(filter)
		framework.ExpectNoError(err, "Failed to list PodSandbox %s status: %v", podID, err)
		Expect(framework.PodFound(pods, podID)).To(BeFalse(), "PodSandbox should be removed")
	})

	It("test list podPodSandbox", func() {
		podID := testCreatePodSandbox(c)
		defer func() {
			By("delete PodSandbox")
			c.StopPodSandbox(podID)
			c.RemovePodSandbox(podID)
		}()

		By("list PodSandbox with podID")
		filter := &runtimeApi.PodSandboxFilter{
			Id: &podID,
		}
		pods, err := c.ListPodSandbox(filter)
		framework.ExpectNoError(err, "Failed to list PodSandbox %s status: %v", podID, err)
		Expect(framework.PodFound(pods, podID)).To(BeTrue(), "PodSandbox should be listed")
	})
})
