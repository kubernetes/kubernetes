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

package e2enode

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/cpuset"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
)

func runAutomaticallyRemoveInactivePodsFromCPUManagerStateFile(ctx context.Context, f *framework.Framework) {
	var cpu1 int
	var ctnAttrs []ctnAttribute
	var pod *v1.Pod
	var cpuList []int
	var expAllowedCPUsListRegex string
	var err error
	// First running a Gu Pod,
	// second disable cpu manager in kubelet,
	// then delete the Gu Pod,
	// then enable cpu manager in kubelet,
	// at last wait for the reconcile process cleaned up the state file, if the assignments map is empty,
	// it proves that the automatic cleanup in the reconcile process is in effect.
	ginkgo.By("running a Gu pod for test remove")
	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "gu-container-testremove",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
	}
	pod = makeCPUManagerPod("gu-pod-testremove", ctnAttrs)
	pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpu1 = 1
	if isHTEnabled() {
		cpuList = mustParseCPUSet(getCPUSiblingList(0)).List()
		cpu1 = cpuList[1]
	} else if isMultiNUMA() {
		cpuList = mustParseCPUSet(getCoreSiblingList(0)).List()
		if len(cpuList) > 1 {
			cpu1 = cpuList[1]
		}
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu1)
	err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[0].Name, pod.Name)

	deletePodSyncByName(ctx, f, pod.Name)
	// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
	// this is in turn needed because we will have an unavoidable (in the current framework) race with the
	// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
	waitForAllContainerRemoval(ctx, pod.Name, pod.Namespace)

}

func runCPUManagerTests(f *framework.Framework) {
	var cpuCap, cpuAlloc int64
	var oldCfg *kubeletconfig.KubeletConfiguration

	ginkgo.BeforeEach(func(ctx context.Context) {
		var err error
		if oldCfg == nil {
			oldCfg, err = getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)
		}
	})

	ginkgo.It("should assign CPUs as expected based on the Pod spec", func(ctx context.Context) {
		cpuCap, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)

		// Skip CPU Manager tests altogether if the CPU capacity < minCPUCapacity.
		if cpuCap < minCPUCapacity {
			e2eskipper.Skipf("Skipping CPU Manager tests since the CPU capacity < %d", minCPUCapacity)
		}

		// Enable CPU Manager in the kubelet.
		newCfg := configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
			policyName:         string(cpumanager.PolicyStatic),
			reservedSystemCPUs: cpuset.CPUSet{},
		})
		updateKubeletConfig(ctx, f, newCfg, true)

		ginkgo.By("running a non-Gu pod")
		runNonGuPodTest(ctx, f, cpuCap, cpuset.New())

		ginkgo.By("running a Gu pod")
		runGuPodTest(ctx, f, 1, cpuset.New())

		ginkgo.By("running multiple Gu and non-Gu pods")
		runMultipleGuNonGuPods(ctx, f, cpuCap, cpuAlloc)

		// Skip rest of the tests if CPU capacity < 3.
		if cpuCap < 3 {
			e2eskipper.Skipf("Skipping rest of the CPU Manager tests since CPU capacity < 3")
		}

		ginkgo.By("running a Gu pod requesting multiple CPUs")
		runMultipleCPUGuPod(ctx, f)

		ginkgo.By("running a Gu pod with multiple containers requesting integer CPUs")
		runMultipleCPUContainersGuPod(ctx, f)

		ginkgo.By("running multiple Gu pods")
		runMultipleGuPods(ctx, f)

		ginkgo.By("test for automatically remove inactive pods from cpumanager state file.")
		runAutomaticallyRemoveInactivePodsFromCPUManagerStateFile(ctx, f)
	})

	ginkgo.It("reservedSystemCPUs are excluded only for Gu pods (strict-cpu-reservation option not enabled by default)", func(ctx context.Context) {
		cpuCap, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)

		// Skip CPU Manager tests altogether if the CPU capacity < 2.
		if cpuCap < 2 {
			e2eskipper.Skipf("Skipping CPU Manager tests since the CPU capacity < 2")
		}

		reservedSystemCPUs := cpuset.New(0)
		newCfg := configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
			policyName:         string(cpumanager.PolicyStatic),
			reservedSystemCPUs: reservedSystemCPUs,
		})
		updateKubeletConfig(ctx, f, newCfg, true)

		ginkgo.By("running a Gu pod - it shouldn't use reserved system CPUs")
		runGuPodTest(ctx, f, 1, reservedSystemCPUs)

		ginkgo.By("running a non-Gu pod - it can use reserved system CPUs")
		runNonGuPodTest(ctx, f, cpuCap, cpuset.New())

	})

	ginkgo.It("reservedSystemCPUs are excluded for both Gu and non-Gu pods (strict-cpu-reservation option enabled)", func(ctx context.Context) {
		cpuCap, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)

		// Skip CPU Manager tests altogether if the CPU capacity < 2.
		if cpuCap < 2 {
			e2eskipper.Skipf("Skipping CPU Manager tests since the CPU capacity < 2")
		}

		reservedSystemCPUs := cpuset.New(0)
		cpuPolicyOptions := map[string]string{
			cpumanager.StrictCPUReservationOption: "true",
		}
		newCfg := configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
			policyName:              string(cpumanager.PolicyStatic),
			reservedSystemCPUs:      reservedSystemCPUs,
			enableCPUManagerOptions: true,
			options:                 cpuPolicyOptions,
		})
		updateKubeletConfig(ctx, f, newCfg, true)

		ginkgo.By("running a Gu pod - it shouldn't use reserved system CPUs")
		runGuPodTest(ctx, f, 1, reservedSystemCPUs)

		ginkgo.By("running a non-Gu pod - it shouldn't use reserved system CPUs with strict-cpu-reservation option enabled")
		runNonGuPodTest(ctx, f, cpuCap, reservedSystemCPUs)
	})

	f.It("should not reuse CPUs of restartable init containers", feature.SidecarContainers, func(ctx context.Context) {
		cpuCap, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)

		// Skip rest of the tests if CPU capacity < 3.
		if cpuCap < 3 {
			e2eskipper.Skipf("Skipping rest of the CPU Manager tests since CPU capacity < 3, got %d", cpuCap)
		}

		// Enable CPU Manager in the kubelet.
		newCfg := configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
			policyName:         string(cpumanager.PolicyStatic),
			reservedSystemCPUs: cpuset.CPUSet{},
		})
		updateKubeletConfig(ctx, f, newCfg, true)

		ginkgo.By("running a Gu pod with a regular init container and a restartable init container")
		ctrAttrs := []ctnAttribute{
			{
				ctnName:    "gu-init-container1",
				cpuRequest: "1000m",
				cpuLimit:   "1000m",
			},
			{
				ctnName:       "gu-restartable-init-container2",
				cpuRequest:    "1000m",
				cpuLimit:      "1000m",
				restartPolicy: &containerRestartPolicyAlways,
			},
		}
		pod := makeCPUManagerInitContainersPod("gu-pod", ctrAttrs)
		pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

		ginkgo.By("checking if the expected cpuset was assigned")
		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.InitContainers[0].Name)
		framework.ExpectNoError(err, "expected log not found in init container [%s] of pod [%s]", pod.Spec.InitContainers[0].Name, pod.Name)

		reusableCPUs := getContainerAllowedCPUsFromLogs(pod.Name, pod.Spec.InitContainers[0].Name, logs)

		gomega.Expect(reusableCPUs.Size()).To(gomega.Equal(1), "expected cpu set size == 1, got %q", reusableCPUs.String())

		logs, err = e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.InitContainers[1].Name)
		framework.ExpectNoError(err, "expected log not found in init container [%s] of pod [%s]", pod.Spec.InitContainers[1].Name, pod.Name)

		nonReusableCPUs := getContainerAllowedCPUsFromLogs(pod.Name, pod.Spec.InitContainers[1].Name, logs)

		gomega.Expect(nonReusableCPUs.Size()).To(gomega.Equal(1), "expected cpu set size == 1, got %q", nonReusableCPUs.String())

		logs, err = e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]", pod.Spec.Containers[0].Name, pod.Name)

		cpus := getContainerAllowedCPUsFromLogs(pod.Name, pod.Spec.Containers[0].Name, logs)

		gomega.Expect(cpus.Size()).To(gomega.Equal(1), "expected cpu set size == 1, got %q", cpus.String())

		gomega.Expect(reusableCPUs.Equals(nonReusableCPUs)).To(gomega.BeTrueBecause("expected reusable cpuset [%s] to be equal to non-reusable cpuset [%s]", reusableCPUs.String(), nonReusableCPUs.String()))
		gomega.Expect(nonReusableCPUs.Intersection(cpus).IsEmpty()).To(gomega.BeTrueBecause("expected non-reusable cpuset [%s] to be disjoint from cpuset [%s]", nonReusableCPUs.String(), cpus.String()))

		ginkgo.By("by deleting the pods and waiting for container removal")
		deletePods(ctx, f, []string{pod.Name})
		waitForContainerRemoval(ctx, pod.Spec.InitContainers[0].Name, pod.Name, pod.Namespace)
		waitForContainerRemoval(ctx, pod.Spec.InitContainers[1].Name, pod.Name, pod.Namespace)
		waitForContainerRemoval(ctx, pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		updateKubeletConfig(ctx, f, oldCfg, true)
	})
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("CPU Manager", framework.WithSerial(), feature.CPUManager, func() {
	f := framework.NewDefaultFramework("cpu-manager-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("With kubeconfig updated with static CPU Manager policy run the CPU Manager tests", func() {
		runCPUManagerTests(f)
	})
})
