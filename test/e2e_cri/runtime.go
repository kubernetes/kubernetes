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

var _ = framework.KubeDescribe("Test CRI runtime", func() {
	f := framework.NewDefaultCRIFramework("cri-runtime-test")

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

	It("test create simple podsandbox", func() {
		name := "create-simple-sandbox-" + string(uuid.NewUUID())
		By("create a podSandbox with name")
		config := &runtimeApi.PodSandboxConfig{
			Metadata: &runtimeApi.PodSandboxMetadata{
				Name: &name,
			},
		}
		podId, err := c.RunPodSandbox(config)
		framework.ExpectNoError(err, "Failed to create podsandbox: %v", err)
		framework.Logf("Created Podsanbox %s\n", podId)
		defer func() {
			By("delete pod sandbox")
			c.RemovePodSandbox(podId)
		}()
		By("get podSandbox status")
		status, err := c.PodSandboxStatus(podId)
		framework.ExpectNoError(err, "Failed to get podsandbox %s status: %v", podId, err)
		Expect(framework.PodReady(status)).To(BeTrue(), "pod state shoud be ready")
	})

	It("test stop PodSandbox", func() {
		name := "create-simple-sandbox-for-stop" + string(uuid.NewUUID())

		By("create a podSandbox with name")
		config := &runtimeApi.PodSandboxConfig{
			Metadata: &runtimeApi.PodSandboxMetadata{
				Name: &name,
			},
		}
		podId, err := c.RunPodSandbox(config)
		framework.ExpectNoError(err, "Failed to create podsandbox: %v", err)
		framework.Logf("Created Podsanbox %s\n", podId)
		defer func() {
			By("delete pod sandbox")
			c.RemovePodSandbox(podId)
		}()

		By("get podSandbox status")
		status, err := c.PodSandboxStatus(podId)
		framework.ExpectNoError(err, "Failed to get podsandbox %s status: %v", podId, err)
		Expect(framework.PodReady(status)).To(BeTrue(), "pod state should be ready")

		By("stop podSandbox with poId")
		err = c.StopPodSandbox(podId)
		framework.ExpectNoError(err, "Failed to stop podsandbox: %v", err)
		framework.Logf("Stoped Podsanbox %s\n", podId)

		By("get podSandbox status")
		status, err = c.PodSandboxStatus(podId)
		framework.ExpectNoError(err, "Failed to get podsandbox %s status: %v", podId, err)
		Expect(framework.PodReady(status)).To(BeFalse(), "pod state should be not ready")
	})

	It("test remove podsandbox", func() {
		name := "create-simple-sandbox-for-remove" + string(uuid.NewUUID())
		By("create a podSandbox with name")
		config := &runtimeApi.PodSandboxConfig{
			Metadata: &runtimeApi.PodSandboxMetadata{
				Name: &name,
			},
		}
		podId, err := c.RunPodSandbox(config)
		framework.ExpectNoError(err, "Failed to create podsandbox: %v", err)
		framework.Logf("Created Podsanbox %s\n", podId)

		By("get podSandbox status")
		status, err := c.PodSandboxStatus(podId)
		framework.ExpectNoError(err, "Failed to get podsandbox %s status: %v", podId, err)
		Expect(framework.PodReady(status)).To(BeTrue(), "pod state should be ready")

		By("remove podSandbox with podId")
		err = c.RemovePodSandbox(podId)
		framework.ExpectNoError(err, "Failed to remove podsandbox: %v", err)
		framework.Logf("Removed Podsanbox %s\n", podId)

		By("list podSandbox with podId")
		filter := &runtimeApi.PodSandboxFilter{
			Id: &podId,
		}
		podsandboxs, err := c.ListPodSandbox(filter)
		framework.ExpectNoError(err, "Failed to list podsandbox %s status: %v", podId, err)
		Expect(framework.PodFound(podsandboxs, podId)).To(BeFalse(), "pod should be removed")
	})

	It("test list podsandbox", func() {
		name := "create-simple-sandbox-for-list" + string(uuid.NewUUID())
		By("create a podSandbox with name")
		config := &runtimeApi.PodSandboxConfig{
			Metadata: &runtimeApi.PodSandboxMetadata{
				Name: &name,
			},
		}
		podId, err := c.RunPodSandbox(config)
		framework.ExpectNoError(err, "Failed to create podsandbox: %v", err)
		framework.Logf("Created Podsanbox %s\n", podId)
		defer func() {
			By("delete pod sandbox")
			c.RemovePodSandbox(podId)
		}()
		By("get podSandbox status")
		status, err := c.PodSandboxStatus(podId)
		framework.ExpectNoError(err, "Failed to get podsandbox %s status: %v", podId, err)
		Expect(framework.PodReady(status)).To(BeTrue(), "pod state should be ready")

		By("list podSandbox with podId")
		filter := &runtimeApi.PodSandboxFilter{
			Id: &podId,
		}
		podsandboxs, err := c.ListPodSandbox(filter)
		framework.ExpectNoError(err, "Failed to list podsandbox %s status: %v", podId, err)
		Expect(framework.PodFound(podsandboxs, podId)).To(BeTrue(), "pod should be listed")
	})
})
