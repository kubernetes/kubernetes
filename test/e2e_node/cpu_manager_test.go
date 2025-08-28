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
	"regexp"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

func runCfsQuotaGuPods(ctx context.Context, f *framework.Framework, disabledCPUQuotaWithExclusiveCPUs bool, cpuAlloc int64) {
	var err error
	var ctnAttrs []ctnAttribute
	var pod1, pod2, pod3 *v1.Pod
	podsToClean := make(map[string]*v1.Pod) // pod.UID -> pod

	framework.Logf("runCfsQuotaGuPods: disableQuota=%v, CPU Allocatable=%v", disabledCPUQuotaWithExclusiveCPUs, cpuAlloc)

	deleteTestPod := func(pod *v1.Pod) {
		// waitForContainerRemoval takes "long" to complete; if we use the parent ctx we get a
		// 'deadline expired' message and the cleanup aborts, which we don't want.
		// So let's use a separate and more generous timeout (determined by trial and error)
		ctx2, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()
		deletePodSyncAndWait(ctx2, f, pod.Namespace, pod.Name)
		delete(podsToClean, string(pod.UID))
	}

	// cleanup leftovers on test failure. The happy path is covered by `deleteTestPod` calls
	ginkgo.DeferCleanup(func() {
		ginkgo.By("by deleting the pods and waiting for container removal")
		// waitForContainerRemoval takes "long" to complete; if we use the parent ctx we get a
		// 'deadline expired' message and the cleanup aborts, which we don't want.
		// So let's use a separate and more generous timeout (determined by trial and error)
		ctx2, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()
		deletePodsAsync(ctx2, f, podsToClean)
	})

	podCFSCheckCommand := []string{"sh", "-c", `cat $(find /sysfscgroup | grep -E "($(cat /podinfo/uid)|$(cat /podinfo/uid | sed 's/-/_/g'))(/|\.slice/)cpu.max$") && sleep 1d`}
	cfsCheckCommand := []string{"sh", "-c", "cat /sys/fs/cgroup/cpu.max && sleep 1d"}
	defaultPeriod := "100000"

	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "gu-container-cfsquota-disabled",
			cpuRequest: "1",
			cpuLimit:   "1",
		},
	}
	pod1 = makeCPUManagerPod("gu-pod1", ctnAttrs)
	pod1.Spec.Containers[0].Command = cfsCheckCommand
	pod1 = e2epod.NewPodClient(f).CreateSync(ctx, pod1)
	podsToClean[string(pod1.UID)] = pod1

	ginkgo.By("checking if the expected cfs quota was assigned (GU pod, exclusive CPUs, unlimited)")

	expectedQuota := "100000"
	if disabledCPUQuotaWithExclusiveCPUs {
		expectedQuota = "max"
	}
	expCFSQuotaRegex := fmt.Sprintf("^%s %s\n$", expectedQuota, defaultPeriod)
	err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod1.Name, pod1.Spec.Containers[0].Name, expCFSQuotaRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod1.Spec.Containers[0].Name, pod1.Name)
	deleteTestPod(pod1)

	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "gu-container-cfsquota-enabled",
			cpuRequest: "500m",
			cpuLimit:   "500m",
		},
	}
	pod2 = makeCPUManagerPod("gu-pod2", ctnAttrs)
	pod2.Spec.Containers[0].Command = cfsCheckCommand
	pod2 = e2epod.NewPodClient(f).CreateSync(ctx, pod2)
	podsToClean[string(pod2.UID)] = pod2

	ginkgo.By("checking if the expected cfs quota was assigned (GU pod, limited)")

	expectedQuota = "50000"
	expCFSQuotaRegex = fmt.Sprintf("^%s %s\n$", expectedQuota, defaultPeriod)
	err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod2.Name, pod2.Spec.Containers[0].Name, expCFSQuotaRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod2.Spec.Containers[0].Name, pod2.Name)
	deleteTestPod(pod2)

	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "non-gu-container",
			cpuRequest: "100m",
			cpuLimit:   "500m",
		},
	}
	pod3 = makeCPUManagerPod("non-gu-pod3", ctnAttrs)
	pod3.Spec.Containers[0].Command = cfsCheckCommand
	pod3 = e2epod.NewPodClient(f).CreateSync(ctx, pod3)
	podsToClean[string(pod3.UID)] = pod3

	ginkgo.By("checking if the expected cfs quota was assigned (BU pod, limited)")

	expectedQuota = "50000"
	expCFSQuotaRegex = fmt.Sprintf("^%s %s\n$", expectedQuota, defaultPeriod)
	err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod3.Name, pod3.Spec.Containers[0].Name, expCFSQuotaRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod3.Spec.Containers[0].Name, pod3.Name)
	deleteTestPod(pod3)

	if cpuAlloc >= 2 {
		ctnAttrs = []ctnAttribute{
			{
				ctnName:    "gu-container-non-int-values",
				cpuRequest: "500m",
				cpuLimit:   "500m",
			},
			{
				ctnName:    "gu-container-int-values",
				cpuRequest: "1",
				cpuLimit:   "1",
			},
		}
		pod4 := makeCPUManagerPod("gu-pod4", ctnAttrs)
		pod4.Spec.Containers[0].Command = cfsCheckCommand
		pod4.Spec.Containers[1].Command = cfsCheckCommand
		pod4 = e2epod.NewPodClient(f).CreateSync(ctx, pod4)
		podsToClean[string(pod4.UID)] = pod4

		ginkgo.By("checking if the expected cfs quota was assigned (GU pod, container 0 exclusive CPUs unlimited, container 1 limited)")

		expectedQuota = "50000"
		expCFSQuotaRegex = fmt.Sprintf("^%s %s\n$", expectedQuota, defaultPeriod)
		err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod4.Name, pod4.Spec.Containers[0].Name, expCFSQuotaRegex)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
			pod4.Spec.Containers[0].Name, pod4.Name)
		expectedQuota = "100000"
		if disabledCPUQuotaWithExclusiveCPUs {
			expectedQuota = "max"
		}
		expCFSQuotaRegex = fmt.Sprintf("^%s %s\n$", expectedQuota, defaultPeriod)
		err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod4.Name, pod4.Spec.Containers[1].Name, expCFSQuotaRegex)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
			pod4.Spec.Containers[1].Name, pod4.Name)
		deleteTestPod(pod4)

		ctnAttrs = []ctnAttribute{
			{
				ctnName:    "gu-container-non-int-values",
				cpuRequest: "500m",
				cpuLimit:   "500m",
			},
			{
				ctnName:    "gu-container-int-values",
				cpuRequest: "1",
				cpuLimit:   "1",
			},
		}

		pod5 := makeCPUManagerPod("gu-pod5", ctnAttrs)
		pod5.Spec.Containers[0].Command = podCFSCheckCommand
		pod5 = e2epod.NewPodClient(f).CreateSync(ctx, pod5)
		podsToClean[string(pod5.UID)] = pod5

		ginkgo.By("checking if the expected cfs quota was assigned to pod (GU pod, unlimited)")

		expectedQuota = "150000"

		if disabledCPUQuotaWithExclusiveCPUs {
			expectedQuota = "max"
		}

		expCFSQuotaRegex = fmt.Sprintf("^%s %s\n$", expectedQuota, defaultPeriod)

		err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod5.Name, pod5.Spec.Containers[0].Name, expCFSQuotaRegex)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]", pod5.Spec.Containers[0].Name, pod5.Name)
		deleteTestPod(pod5)
	} else {
		ginkgo.By(fmt.Sprintf("some cases SKIPPED - requests at least %d allocatable cores, got %d", 2, cpuAlloc))
	}

	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "gu-container",
			cpuRequest: "100m",
			cpuLimit:   "100m",
		},
	}

	pod6 := makeCPUManagerPod("gu-pod6", ctnAttrs)
	pod6.Spec.Containers[0].Command = podCFSCheckCommand
	pod6 = e2epod.NewPodClient(f).CreateSync(ctx, pod6)
	podsToClean[string(pod6.UID)] = pod6

	ginkgo.By("checking if the expected cfs quota was assigned to pod (GU pod, limited)")

	expectedQuota = "10000"
	expCFSQuotaRegex = fmt.Sprintf("^%s %s\n$", expectedQuota, defaultPeriod)
	err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod6.Name, pod6.Spec.Containers[0].Name, expCFSQuotaRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]", pod6.Spec.Containers[0].Name, pod6.Name)
	deleteTestPod(pod6)
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

	ginkgo.It("should assign CPUs as expected with enhanced policy based on strict SMT alignment", func(ctx context.Context) {
		fullCPUsOnlyOpt := fmt.Sprintf("option=%s", cpumanager.FullPCPUsOnlyOption)
		_, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)
		smtLevel := getSMTLevel()

		// strict SMT alignment is trivially verified and granted on non-SMT systems
		if smtLevel < minSMTLevel {
			e2eskipper.Skipf("Skipping CPU Manager %s tests since SMT disabled", fullCPUsOnlyOpt)
		}

		// our tests want to allocate a full core, so we need at least 2*2=4 virtual cpus
		minCPUCount := int64(smtLevel * minCPUCapacity)
		if cpuAlloc < minCPUCount {
			e2eskipper.Skipf("Skipping CPU Manager %s tests since the CPU capacity < %d", fullCPUsOnlyOpt, minCPUCount)
		}

		framework.Logf("SMT level %d", smtLevel)

		// TODO: we assume the first available CPUID is 0, which is pretty fair, but we should probably
		// check what we do have in the node.
		cpuPolicyOptions := map[string]string{
			cpumanager.FullPCPUsOnlyOption: "true",
		}
		newCfg := configureCPUManagerInKubelet(oldCfg,
			&cpuManagerKubeletArguments{
				policyName:              string(cpumanager.PolicyStatic),
				reservedSystemCPUs:      cpuset.New(0),
				enableCPUManagerOptions: true,
				options:                 cpuPolicyOptions,
			},
		)
		updateKubeletConfig(ctx, f, newCfg, true)

		// the order between negative and positive doesn't really matter
		runSMTAlignmentNegativeTests(ctx, f)
		runSMTAlignmentPositiveTests(ctx, f, smtLevel, cpuset.New())
	})

	ginkgo.It("should assign CPUs as expected based on strict SMT alignment, reservedSystemCPUs should be excluded (both strict-cpu-reservation and full-pcpus-only options enabled)", func(ctx context.Context) {
		fullCPUsOnlyOpt := fmt.Sprintf("option=%s", cpumanager.FullPCPUsOnlyOption)
		_, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)
		smtLevel := getSMTLevel()

		// strict SMT alignment is trivially verified and granted on non-SMT systems
		if smtLevel < 2 {
			e2eskipper.Skipf("Skipping CPU Manager %s tests since SMT disabled", fullCPUsOnlyOpt)
		}

		// our tests want to allocate a full core, so we need at last smtLevel*2 virtual cpus
		if cpuAlloc < int64(smtLevel*2) {
			e2eskipper.Skipf("Skipping CPU Manager %s tests since the CPU capacity < %d", fullCPUsOnlyOpt, smtLevel*2)
		}

		framework.Logf("SMT level %d", smtLevel)

		reservedSystemCPUs := cpuset.New(0)
		cpuPolicyOptions := map[string]string{
			cpumanager.FullPCPUsOnlyOption:        "true",
			cpumanager.StrictCPUReservationOption: "true",
		}
		newCfg := configureCPUManagerInKubelet(oldCfg,
			&cpuManagerKubeletArguments{
				policyName:              string(cpumanager.PolicyStatic),
				reservedSystemCPUs:      reservedSystemCPUs,
				enableCPUManagerOptions: true,
				options:                 cpuPolicyOptions,
			},
		)
		updateKubeletConfig(ctx, f, newCfg, true)

		// the order between negative and positive doesn't really matter
		runSMTAlignmentNegativeTests(ctx, f)
		runSMTAlignmentPositiveTests(ctx, f, smtLevel, reservedSystemCPUs)
	})

	ginkgo.It("should not enforce CFS quota for containers with static CPUs assigned", func(ctx context.Context) {
		if !IsCgroup2UnifiedMode() {
			e2eskipper.Skipf("Skipping since CgroupV2 not used")
		}
		_, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)
		if cpuAlloc < 1 { // save expensive kubelet restart
			e2eskipper.Skipf("Skipping since not enough allocatable CPU got %d required 1", cpuAlloc)
		}
		newCfg := configureCPUManagerInKubelet(oldCfg,
			&cpuManagerKubeletArguments{
				policyName:                       string(cpumanager.PolicyStatic),
				reservedSystemCPUs:               cpuset.New(0),
				disableCPUQuotaWithExclusiveCPUs: true,
			},
		)
		updateKubeletConfig(ctx, f, newCfg, true)

		_, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f) // check again after we reserved 1 full CPU. Some tests require > 1 exclusive CPU
		runCfsQuotaGuPods(ctx, f, true, cpuAlloc)
	})

	ginkgo.It("should keep enforcing the CFS quota for containers with static CPUs assigned and feature gate disabled", func(ctx context.Context) {
		if !IsCgroup2UnifiedMode() {
			e2eskipper.Skipf("Skipping since CgroupV2 not used")
		}
		_, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)
		if cpuAlloc < 1 { // save expensive kubelet restart
			e2eskipper.Skipf("Skipping since not enough allocatable CPU got %d required 1", cpuAlloc)
		}
		newCfg := configureCPUManagerInKubelet(oldCfg,
			&cpuManagerKubeletArguments{
				policyName:                       string(cpumanager.PolicyStatic),
				reservedSystemCPUs:               cpuset.New(0),
				disableCPUQuotaWithExclusiveCPUs: false,
			},
		)

		updateKubeletConfig(ctx, f, newCfg, true)

		_, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f) // check again after we reserved 1 full CPU. Some tests require > 1 exclusive CPU
		runCfsQuotaGuPods(ctx, f, false, cpuAlloc)
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

	ginkgo.It("should assign packed CPUs with distribute-cpus-across-numa disabled and pcpu-only policy options enabled", func(ctx context.Context) {
		fullCPUsOnlyOpt := fmt.Sprintf("option=%s", cpumanager.FullPCPUsOnlyOption)
		_, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)
		smtLevel := getSMTLevel()

		// strict SMT alignment is trivially verified and granted on non-SMT systems
		if smtLevel < minSMTLevel {
			e2eskipper.Skipf("Skipping CPU Manager %s tests since SMT disabled", fullCPUsOnlyOpt)
		}

		// our tests want to allocate a full core, so we need at least 2*2=4 virtual cpus
		minCPUCount := int64(smtLevel * minCPUCapacity)
		if cpuAlloc < minCPUCount {
			e2eskipper.Skipf("Skipping CPU Manager %s tests since the CPU capacity < %d", fullCPUsOnlyOpt, minCPUCount)
		}

		framework.Logf("SMT level %d", smtLevel)

		cpuPolicyOptions := map[string]string{
			cpumanager.FullPCPUsOnlyOption:            "true",
			cpumanager.DistributeCPUsAcrossNUMAOption: "false",
		}
		newCfg := configureCPUManagerInKubelet(oldCfg,
			&cpuManagerKubeletArguments{
				policyName:              string(cpumanager.PolicyStatic),
				reservedSystemCPUs:      cpuset.New(0),
				enableCPUManagerOptions: true,
				options:                 cpuPolicyOptions,
			},
		)
		updateKubeletConfig(ctx, f, newCfg, true)

		ctnAttrs := []ctnAttribute{
			{
				ctnName:    "test-gu-container-distribute-cpus-across-numa-disabled",
				cpuRequest: "2000m",
				cpuLimit:   "2000m",
			},
		}
		pod := makeCPUManagerPod("test-pod-distribute-cpus-across-numa-disabled", ctnAttrs)
		pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

		for _, cnt := range pod.Spec.Containers {
			ginkgo.By(fmt.Sprintf("validating the container %s on Gu pod %s", cnt.Name, pod.Name))

			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, cnt.Name)
			framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]", cnt.Name, pod.Name)

			cpus := getContainerAllowedCPUsFromLogs(pod.Name, cnt.Name, logs)

			validateSMTAlignment(cpus, smtLevel, pod, &cnt)
			gomega.Expect(cpus).To(BePackedCPUs())
		}
		deletePodSyncByName(ctx, f, pod.Name)
		// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
		// this is in turn needed because we will have an unavoidable (in the current framework) race with th
		// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
		waitForAllContainerRemoval(ctx, pod.Name, pod.Namespace)
	})

	ginkgo.It("should assign CPUs distributed across NUMA with distribute-cpus-across-numa and pcpu-only policy options enabled", func(ctx context.Context) {
		var cpusNumPerNUMA, numaNodeNum int

		fullCPUsOnlyOpt := fmt.Sprintf("option=%s", cpumanager.FullPCPUsOnlyOption)
		_, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)
		smtLevel := getSMTLevel()
		framework.Logf("SMT level %d", smtLevel)

		// strict SMT alignment is trivially verified and granted on non-SMT systems
		if smtLevel < minSMTLevel {
			e2eskipper.Skipf("Skipping CPU Manager %s tests since SMT disabled", fullCPUsOnlyOpt)
		}

		// our tests want to allocate a full core, so we need at least 2*2=4 virtual cpus
		minCPUCount := int64(smtLevel * minCPUCapacity)
		if cpuAlloc < minCPUCount {
			e2eskipper.Skipf("Skipping CPU Manager %s tests since the CPU capacity < %d", fullCPUsOnlyOpt, minCPUCount)
		}

		// this test is intended to be run on a multi-node NUMA system and
		// a system with at least 4 cores per socket, hostcheck skips test
		// if above requirements are not satisfied
		numaNodeNum, _, _, cpusNumPerNUMA = hostCheck()

		cpuPolicyOptions := map[string]string{
			cpumanager.FullPCPUsOnlyOption:            "true",
			cpumanager.DistributeCPUsAcrossNUMAOption: "true",
		}
		newCfg := configureCPUManagerInKubelet(oldCfg,
			&cpuManagerKubeletArguments{
				policyName:              string(cpumanager.PolicyStatic),
				reservedSystemCPUs:      cpuset.New(0),
				enableCPUManagerOptions: true,
				options:                 cpuPolicyOptions,
			},
		)
		updateKubeletConfig(ctx, f, newCfg, true)
		// 'distribute-cpus-across-numa' policy option ensures that CPU allocations are evenly distributed
		//  across NUMA nodes in cases where more than one NUMA node is required to satisfy the allocation.
		// So, we want to ensure that the CPU Request exceeds the number of CPUs that can fit within a single
		// NUMA node. We have to pick cpuRequest such that:
		// 1. CPURequest > cpusNumPerNUMA
		// 2. Not occupy all the CPUs on the node ande leave room for reserved CPU
		// 3. CPURequest is a multiple if number of NUMA nodes to allow equal CPU distribution across NUMA nodes
		//
		// In summary: cpusNumPerNUMA < CPURequest < ((cpusNumPerNuma * numaNodeNum) - reservedCPUscount)
		// Considering all these constraints we select: CPURequest= (cpusNumPerNUMA-smtLevel)*numaNodeNum

		cpuReq := (cpusNumPerNUMA - smtLevel) * numaNodeNum
		ctnAttrs := []ctnAttribute{
			{
				ctnName:    "test-gu-container-distribute-cpus-across-numa",
				cpuRequest: fmt.Sprintf("%d", cpuReq),
				cpuLimit:   fmt.Sprintf("%d", cpuReq),
			},
		}
		pod := makeCPUManagerPod("test-pod-distribute-cpus-across-numa", ctnAttrs)
		pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

		for _, cnt := range pod.Spec.Containers {
			ginkgo.By(fmt.Sprintf("validating the container %s on Gu pod %s", cnt.Name, pod.Name))

			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, cnt.Name)
			framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]", cnt.Name, pod.Name)

			cpus := getContainerAllowedCPUsFromLogs(pod.Name, cnt.Name, logs)

			validateSMTAlignment(cpus, smtLevel, pod, &cnt)
			// We expect a perfectly even spilit i.e. equal distribution across NUMA Node as the CPU Request is 4*smtLevel*numaNodeNum.
			expectedSpread := cpus.Size() / numaNodeNum
			gomega.Expect(cpus).To(BeDistributedCPUs(expectedSpread))
		}
		deletePodSyncByName(ctx, f, pod.Name)
		// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
		// this is in turn needed because we will have an unavoidable (in the current framework) race with th
		// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
		waitForAllContainerRemoval(ctx, pod.Name, pod.Namespace)
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		updateKubeletConfig(ctx, f, oldCfg, true)
	})
}

func runSMTAlignmentNegativeTests(ctx context.Context, f *framework.Framework) {
	// negative test: try to run a container whose requests aren't a multiple of SMT level, expect a rejection
	ctnAttrs := []ctnAttribute{
		{
			ctnName:    "gu-container-neg",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
	}
	pod := makeCPUManagerPod("gu-pod", ctnAttrs)
	// CreateSync would wait for pod to become Ready - which will never happen if production code works as intended!
	pod = e2epod.NewPodClient(f).Create(ctx, pod)

	err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Failed", 30*time.Second, func(pod *v1.Pod) (bool, error) {
		if pod.Status.Phase != v1.PodPending {
			return true, nil
		}
		return false, nil
	})
	framework.ExpectNoError(err)
	pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	if pod.Status.Phase != v1.PodFailed {
		framework.Failf("pod %s not failed: %v", pod.Name, pod.Status)
	}
	if !isSMTAlignmentError(pod) {
		framework.Failf("pod %s failed for wrong reason: %q", pod.Name, pod.Status.Reason)
	}

	deletePodSyncByName(ctx, f, pod.Name)
	// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
	// this is in turn needed because we will have an unavoidable (in the current framework) race with th
	// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
	waitForAllContainerRemoval(ctx, pod.Name, pod.Namespace)
}

func runSMTAlignmentPositiveTests(ctx context.Context, f *framework.Framework, smtLevel int, strictReservedCPUs cpuset.CPUSet) {
	// positive test: try to run a container whose requests are a multiple of SMT level, check allocated cores
	// 1. are core siblings
	// 2. take a full core
	// WARNING: this assumes 2-way SMT systems - we don't know how to access other SMT levels.
	//          this means on more-than-2-way SMT systems this test will prove nothing
	ctnAttrs := []ctnAttribute{
		{
			ctnName:    "gu-container-pos",
			cpuRequest: "2000m",
			cpuLimit:   "2000m",
		},
	}
	pod := makeCPUManagerPod("gu-pod", ctnAttrs)
	pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

	for _, cnt := range pod.Spec.Containers {
		ginkgo.By(fmt.Sprintf("validating the container %s on Gu pod %s", cnt.Name, pod.Name))

		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, cnt.Name)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]", cnt.Name, pod.Name)

		cpus := getContainerAllowedCPUsFromLogs(pod.Name, cnt.Name, logs)

		gomega.Expect(cpus.Intersection(strictReservedCPUs).IsEmpty()).To(gomega.BeTrueBecause("cpuset %q should not contain strict reserved cpus %q", cpus.String(), strictReservedCPUs.String()))
		validateSMTAlignment(cpus, smtLevel, pod, &cnt)
	}

	deletePodSyncByName(ctx, f, pod.Name)
	// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
	// this is in turn needed because we will have an unavoidable (in the current framework) race with th
	// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
	waitForAllContainerRemoval(ctx, pod.Name, pod.Namespace)
}

func validateSMTAlignment(cpus cpuset.CPUSet, smtLevel int, pod *v1.Pod, cnt *v1.Container) {
	framework.Logf("validating cpus: %v", cpus)

	if cpus.Size()%smtLevel != 0 {
		framework.Failf("pod %q cnt %q received non-smt-multiple cpuset %v (SMT level %d)", pod.Name, cnt.Name, cpus, smtLevel)
	}

	// now check all the given cpus are thread siblings.
	// to do so the easiest way is to rebuild the expected set of siblings from all the cpus we got.
	// if the expected set matches the given set, the given set was good.
	siblingsCPUs := cpuset.New()
	for _, cpuID := range cpus.UnsortedList() {
		threadSiblings, err := cpuset.Parse(strings.TrimSpace(getCPUSiblingList(int64(cpuID))))
		framework.ExpectNoError(err, "parsing cpuset from logs for [%s] of pod [%s]", cnt.Name, pod.Name)
		siblingsCPUs = siblingsCPUs.Union(threadSiblings)
	}

	framework.Logf("siblings cpus: %v", siblingsCPUs)
	if !siblingsCPUs.Equals(cpus) {
		framework.Failf("pod %q cnt %q received non-smt-aligned cpuset %v (expected %v)", pod.Name, cnt.Name, cpus, siblingsCPUs)
	}
}

func isSMTAlignmentError(pod *v1.Pod) bool {
	re := regexp.MustCompile(`SMT.*Alignment.*Error`)
	return re.MatchString(pod.Status.Reason)
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("CPU Manager", framework.WithSerial(), feature.CPUManager, func() {
	f := framework.NewDefaultFramework("cpu-manager-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("With kubeconfig updated with static CPU Manager policy run the CPU Manager tests", func() {
		runCPUManagerTests(f)
	})
})
