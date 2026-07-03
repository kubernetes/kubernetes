//go:build windows

/*
Copyright The Kubernetes Authors.

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

package e2enodewindows

import (
	"context"
	"fmt"
	"reflect"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2enodekubelet "k8s.io/kubernetes/test/e2e_node_windows/kubeletconfig"
	admissionapi "k8s.io/pod-security-admission/api"
)

/*
 * Windows CPU Affinity Node E2E Tests
 *
 * These tests verify that the CPU manager correctly sets Windows CPU group
 * affinity on containers via the CRI, as implemented in:
 *   - pkg/kubelet/cm/cpumanager/cpu_manager_windows.go
 *   - pkg/kubelet/cm/internal_container_lifecycle_windows.go
 *
 * Prerequisites:
 *   - Windows node with kubelet running as a Windows service named "kubelet"
 *   - At least 2 allocatable integer CPUs on the node
 *   - containerd as the container runtime
 *
 * Linux-only features intentionally NOT covered here:
 *   - SMT/HT alignment (FullPCPUsOnlyOption) — no equivalent in Windows CRI
 *   - Uncore cache alignment (PreferAlignByUnCoreCacheOption) — no L3 topology via Windows CRI
 *   - CFS quota management — cgroup concept, not applicable on Windows
 *   - Strict CPU reservation option — implemented via cgroup on Linux
 */

var _ = SIGWindowsDescribe(feature.CPUManager, feature.Windows, ginkgo.Ordered, ginkgo.ContinueOnFailure, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("cpu-manager-windows")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var (
		oldCfg    *kubeletconfig.KubeletConfiguration
		criClient internalapi.RuntimeService
		podMap    map[string]*v1.Pod
	)

	// createPodSync creates a pod, waits for it to be running, registers it
	// in podMap for cleanup, and returns the updated pod object.
	var createPodSync func(ctx context.Context, pod *v1.Pod) *v1.Pod

	ginkgo.BeforeAll(func(ctx context.Context) {
		var err error
		oldCfg, err = getCurrentKubeletConfig(ctx)
		framework.ExpectNoError(err, "failed to get current kubelet config")

		criClient, _, err = getCRIClient(ctx)
		framework.ExpectNoError(err, "failed to get CRI client")
	})

	ginkgo.AfterAll(func(ctx context.Context) {
		updateWindowsKubeletConfig(ctx, f, oldCfg)
	})

	ginkgo.BeforeEach(func(ctx context.Context) {
		podMap = make(map[string]*v1.Pod)
		createPodSync = func(ctx context.Context, pod *v1.Pod) *v1.Pod {
			newPod := e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(newPod.UID)] = newPod
			return newPod
		}
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		for _, pod := range podMap {
			e2epod.NewPodClient(f).DeleteSync(ctx, pod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
		}
	})

	// -------------------------------------------------------------------------
	// Non-guaranteed (burstable / best-effort) pod tests
	// These mirror the Linux "running non-guaranteed pods tests" group.
	// -------------------------------------------------------------------------
	ginkgo.When("running non-guaranteed pods", ginkgo.Label("non-guaranteed"), func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			updateWindowsKubeletConfigIfNeeded(ctx, f, buildWindowsCPUManagerKubeletConfig(oldCfg, true))
		})

		ginkgo.It("should not set CPU affinity on a burstable container", func(ctx context.Context) {
			pod := makeWindowsCPUManagerPod("burstable-pod", []windowsCtnAttribute{
				{name: "burstable-ctr", cpuRequest: "100m", cpuLimit: "500m"},
			})
			ginkgo.By("creating the burstable pod")
			pod = createPodSync(ctx, pod)

			ginkgo.By("verifying no exclusive CPU affinity is set")
			affinities, err := getWindowsContainerCPUAffinity(ctx, criClient, pod, "burstable-ctr")
			framework.ExpectNoError(err)
			cpuCount := countCPUsInAffinities(affinities)
			hostCPUs := int(getLocalNode(ctx, f).Status.Capacity.Cpu().Value())
			gomega.Expect(cpuCount).To(gomega.Equal(hostCPUs),
				"burstable container must not receive exclusive CPU affinity: got %d CPUs (host has %d)", cpuCount, hostCPUs)
		})

		// Mirrors: "should let the container access all the online non-exclusively-allocated
		// CPUs when using a reserved CPUs set" (Linux).
		// On Windows we verify via CRI: the guaranteed container has affinity set and the
		// burstable container has none, meaning the burstable runs on the shared pool.
		ginkgo.It("should set affinity only on the guaranteed container when coexisting with a burstable pod", func(ctx context.Context) {
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), 2) // 1 for guaranteed + 1 reserved

			guPod := makeWindowsCPUManagerPod("gu-pod", []windowsCtnAttribute{
				{name: "gu-ctr", cpuRequest: "1000m", cpuLimit: "1000m"},
			})
			ginkgo.By("creating the guaranteed pod")
			guPod = createPodSync(ctx, guPod)

			buPod := makeWindowsCPUManagerPod("bu-pod", []windowsCtnAttribute{
				{name: "bu-ctr", cpuRequest: "100m", cpuLimit: "300m"},
			})
			ginkgo.By("creating the burstable pod")
			buPod = createPodSync(ctx, buPod)

			ginkgo.By("verifying guaranteed container has CPU affinity set")
			gomega.Eventually(ctx, func(ctx context.Context) (int, error) {
				aff, err := getWindowsContainerCPUAffinity(ctx, criClient, guPod, "gu-ctr")
				if err != nil {
					return 0, err
				}
				return countCPUsInAffinities(aff), nil
			}, 30*time.Second, 2*time.Second).Should(gomega.Equal(1),
				"guaranteed container should have exactly 1 CPU affinity")

			ginkgo.By("verifying burstable container runs on the shared pool, not exclusively pinned")
			buAff, err := getWindowsContainerCPUAffinity(ctx, criClient, buPod, "bu-ctr")
			framework.ExpectNoError(err)
			guAff, err := getWindowsContainerCPUAffinity(ctx, criClient, guPod, "gu-ctr")
			framework.ExpectNoError(err)
			// The shared pool used by burstable containers shrinks as guaranteed pods
			// take exclusive CPUs, so the expected size is hostCPUs - guaranteedExclusive.
			buCPUs := countCPUsInAffinities(buAff)
			guCPUs := countCPUsInAffinities(guAff)
			hostCPUs := int(getLocalNode(ctx, f).Status.Capacity.Cpu().Value())
			sharedPool := hostCPUs - guCPUs
			gomega.Expect(buCPUs).To(gomega.Equal(sharedPool),
				"burstable container must not receive exclusive CPU affinity: got %d CPUs (host=%d, guaranteed=%d, shared pool=%d)",
				buCPUs, hostCPUs, guCPUs, sharedPool)
		})
	})

	// -------------------------------------------------------------------------
	// Guaranteed pod tests (feature gate ON)
	// These mirror the Linux "running guaranteed pod tests" group.
	// -------------------------------------------------------------------------
	ginkgo.When("running guaranteed pods with exclusive CPU allocation", ginkgo.Label("guaranteed", "exclusive-cpus"), func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			updateWindowsKubeletConfigIfNeeded(ctx, f, buildWindowsCPUManagerKubeletConfig(oldCfg, true))
		})

		// Mirrors: "should allocate exclusively a CPU to a 1-container pod".
		ginkgo.It("should set CPU affinity with exactly 1 CPU for a single-container pod", func(ctx context.Context) {
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), 1)

			pod := makeWindowsCPUManagerPod("gu-pod-1cpu", []windowsCtnAttribute{
				{name: "gu-ctr", cpuRequest: "1000m", cpuLimit: "1000m"},
			})
			ginkgo.By("creating the guaranteed pod")
			pod = createPodSync(ctx, pod)

			ginkgo.By("verifying CPU affinity is set to exactly 1 CPU")
			gomega.Eventually(ctx, func(ctx context.Context) (int, error) {
				aff, err := getWindowsContainerCPUAffinity(ctx, criClient, pod, "gu-ctr")
				if err != nil {
					return 0, err
				}
				return countCPUsInAffinities(aff), nil
			}, 30*time.Second, 2*time.Second).Should(gomega.Equal(1),
				"expected exactly 1 CPU in affinity mask")

			ginkgo.By("verifying the host job-object affinity agrees with the CRI report")
			verifyHostMatchesCRI(ctx, criClient, pod, "gu-ctr")
		})

		// Mirrors: "should allocate exclusively a even number of CPUs to a 1-container pod".
		ginkgo.It("should set CPU affinity with exactly 2 CPUs for a single-container pod", func(ctx context.Context) {
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), 2)

			pod := makeWindowsCPUManagerPod("gu-pod-2cpu", []windowsCtnAttribute{
				{name: "gu-ctr", cpuRequest: "2000m", cpuLimit: "2000m"},
			})
			ginkgo.By("creating the guaranteed pod requesting 2 CPUs")
			pod = createPodSync(ctx, pod)

			ginkgo.By("verifying CPU affinity is set to exactly 2 CPUs")
			gomega.Eventually(ctx, func(ctx context.Context) (int, error) {
				aff, err := getWindowsContainerCPUAffinity(ctx, criClient, pod, "gu-ctr")
				if err != nil {
					return 0, err
				}
				return countCPUsInAffinities(aff), nil
			}, 30*time.Second, 2*time.Second).Should(gomega.Equal(2),
				"expected exactly 2 CPUs in affinity mask")

			ginkgo.By("verifying the host job-object affinity agrees with the CRI report")
			verifyHostMatchesCRI(ctx, criClient, pod, "gu-ctr")
		})

		// Mirrors: "should allocate exclusively a odd number of CPUs to a 1-container pod".
		ginkgo.It("should set CPU affinity with exactly 3 CPUs for a single-container pod", func(ctx context.Context) {
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), 3)

			pod := makeWindowsCPUManagerPod("gu-pod-3cpu", []windowsCtnAttribute{
				{name: "gu-ctr", cpuRequest: "3000m", cpuLimit: "3000m"},
			})
			ginkgo.By("creating the guaranteed pod requesting 3 CPUs")
			pod = createPodSync(ctx, pod)

			ginkgo.By("verifying CPU affinity is set to exactly 3 CPUs")
			gomega.Eventually(ctx, func(ctx context.Context) (int, error) {
				aff, err := getWindowsContainerCPUAffinity(ctx, criClient, pod, "gu-ctr")
				if err != nil {
					return 0, err
				}
				return countCPUsInAffinities(aff), nil
			}, 30*time.Second, 2*time.Second).Should(gomega.Equal(3),
				"expected exactly 3 CPUs in affinity mask")

			ginkgo.By("verifying the host job-object affinity agrees with the CRI report")
			verifyHostMatchesCRI(ctx, criClient, pod, "gu-ctr")
		})

		// Mirrors: "should allocate exclusively CPUs to a multi-container pod (1+2)".
		ginkgo.It("should assign non-overlapping CPU affinity to each container in a multi-container pod (1+2)", func(ctx context.Context) {
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), 3)

			pod := makeWindowsCPUManagerPod("gu-pod-1plus2", []windowsCtnAttribute{
				{name: "gu-ctr-1", cpuRequest: "1000m", cpuLimit: "1000m"},
				{name: "gu-ctr-2", cpuRequest: "2000m", cpuLimit: "2000m"},
			})
			ginkgo.By("creating the guaranteed pod with containers requesting 1 and 2 CPUs")
			pod = createPodSync(ctx, pod)

			var aff1, aff2 []*runtimeapi.WindowsCpuGroupAffinity
			ginkgo.By("verifying each container gets the correct CPU count")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				var err error
				aff1, err = getWindowsContainerCPUAffinity(ctx, criClient, pod, "gu-ctr-1")
				if err != nil {
					return err
				}
				aff2, err = getWindowsContainerCPUAffinity(ctx, criClient, pod, "gu-ctr-2")
				if err != nil {
					return err
				}
				c1, c2 := countCPUsInAffinities(aff1), countCPUsInAffinities(aff2)
				if c1 != 1 || c2 != 2 {
					return fmt.Errorf("want cpu counts (1,2), got (%d,%d)", c1, c2)
				}
				return nil
			}, 30*time.Second, 2*time.Second).Should(gomega.Succeed())

			ginkgo.By("verifying the CPU affinity masks do not overlap")
			gomega.Expect(windowsAffinitiesOverlap(aff1, aff2)).To(gomega.BeFalse(),
				"containers in the same pod must receive non-overlapping CPU affinity masks")

			ginkgo.By("verifying the host job-object affinity agrees with the CRI report and masks are disjoint")
			verifyHostMatchesCRI(ctx, criClient, pod, "gu-ctr-1")
			verifyHostMatchesCRI(ctx, criClient, pod, "gu-ctr-2")
			verifyHostMasksDisjoint(pod, "gu-ctr-1", pod, "gu-ctr-2")
		})

		// Mirrors: "should allocate exclusively CPUs to a multi-container pod (3+2)".
		ginkgo.It("should assign non-overlapping CPU affinity to each container in a multi-container pod (3+2)", func(ctx context.Context) {
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), 5)

			pod := makeWindowsCPUManagerPod("gu-pod-3plus2", []windowsCtnAttribute{
				{name: "gu-ctr-1", cpuRequest: "3000m", cpuLimit: "3000m"},
				{name: "gu-ctr-2", cpuRequest: "2000m", cpuLimit: "2000m"},
			})
			ginkgo.By("creating the guaranteed pod with containers requesting 3 and 2 CPUs")
			pod = createPodSync(ctx, pod)

			var aff1, aff2 []*runtimeapi.WindowsCpuGroupAffinity
			gomega.Eventually(ctx, func(ctx context.Context) error {
				var err error
				aff1, err = getWindowsContainerCPUAffinity(ctx, criClient, pod, "gu-ctr-1")
				if err != nil {
					return err
				}
				aff2, err = getWindowsContainerCPUAffinity(ctx, criClient, pod, "gu-ctr-2")
				if err != nil {
					return err
				}
				c1, c2 := countCPUsInAffinities(aff1), countCPUsInAffinities(aff2)
				if c1 != 3 || c2 != 2 {
					return fmt.Errorf("want cpu counts (3,2), got (%d,%d)", c1, c2)
				}
				return nil
			}, 30*time.Second, 2*time.Second).Should(gomega.Succeed())

			gomega.Expect(windowsAffinitiesOverlap(aff1, aff2)).To(gomega.BeFalse(),
				"containers in the same pod must receive non-overlapping CPU affinity masks")

			ginkgo.By("verifying the host job-object affinity agrees with the CRI report and masks are disjoint")
			verifyHostMatchesCRI(ctx, criClient, pod, "gu-ctr-1")
			verifyHostMatchesCRI(ctx, criClient, pod, "gu-ctr-2")
			verifyHostMasksDisjoint(pod, "gu-ctr-1", pod, "gu-ctr-2")
		})

		// Mirrors: "should allocate exclusively a CPU to multiple 1-container pods".
		ginkgo.It("should assign non-overlapping CPU affinity across separate guaranteed pods (2+2)", func(ctx context.Context) {
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), 4)

			pod1 := makeWindowsCPUManagerPod("gu-pod-a", []windowsCtnAttribute{
				{name: "gu-ctr-a", cpuRequest: "2000m", cpuLimit: "2000m"},
			})
			ginkgo.By("creating guaranteed pod 1")
			pod1 = createPodSync(ctx, pod1)

			pod2 := makeWindowsCPUManagerPod("gu-pod-b", []windowsCtnAttribute{
				{name: "gu-ctr-b", cpuRequest: "2000m", cpuLimit: "2000m"},
			})
			ginkgo.By("creating guaranteed pod 2")
			pod2 = createPodSync(ctx, pod2)

			var affA, affB []*runtimeapi.WindowsCpuGroupAffinity
			ginkgo.By("verifying both containers each get exactly 2 CPUs")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				var err error
				affA, err = getWindowsContainerCPUAffinity(ctx, criClient, pod1, "gu-ctr-a")
				if err != nil {
					return err
				}
				affB, err = getWindowsContainerCPUAffinity(ctx, criClient, pod2, "gu-ctr-b")
				if err != nil {
					return err
				}
				cA, cB := countCPUsInAffinities(affA), countCPUsInAffinities(affB)
				if cA != 2 || cB != 2 {
					return fmt.Errorf("want cpu counts (2,2), got (%d,%d)", cA, cB)
				}
				return nil
			}, 30*time.Second, 2*time.Second).Should(gomega.Succeed())

			ginkgo.By("verifying the two pods' CPU affinity masks do not overlap")
			gomega.Expect(windowsAffinitiesOverlap(affA, affB)).To(gomega.BeFalse(),
				"guaranteed pods must not share exclusively-allocated CPUs")

			ginkgo.By("verifying the host job-object affinity agrees with the CRI report and masks are disjoint")
			verifyHostMatchesCRI(ctx, criClient, pod1, "gu-ctr-a")
			verifyHostMatchesCRI(ctx, criClient, pod2, "gu-ctr-b")
			verifyHostMasksDisjoint(pod1, "gu-ctr-a", pod2, "gu-ctr-b")
		})
	})

	// -------------------------------------------------------------------------
	// Feature gate disabled tests
	// Mirrors the Linux "running guaranteed pod tests with feature gates disabled"
	// group, adapted for the Windows-specific WindowsCPUAndMemoryAffinity gate.
	// When the feature gate is off, cpu_manager_windows.go's updateContainerCPUSet
	// returns immediately without calling UpdateContainerResources, so the CRI
	// should report no AffinityCpus even for guaranteed containers.
	// -------------------------------------------------------------------------
	ginkgo.When("running guaranteed pods with WindowsCPUAndMemoryAffinity feature gate disabled", ginkgo.Label("guaranteed", "feature-gate-disabled"), func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			updateWindowsKubeletConfigIfNeeded(ctx, f, buildWindowsCPUManagerKubeletConfig(oldCfg, false))
		})

		ginkgo.It("should NOT set CPU affinity on a 1-CPU guaranteed container when feature gate is off", func(ctx context.Context) {
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), 1)

			pod := makeWindowsCPUManagerPod("gu-pod-no-affinity", []windowsCtnAttribute{
				{name: "gu-ctr", cpuRequest: "1000m", cpuLimit: "1000m"},
			})
			ginkgo.By("creating the guaranteed pod")
			pod = createPodSync(ctx, pod)

			// Give the CPU manager reconcile loop (1 s period) a few cycles to act,
			// then confirm it still has not set any exclusive affinity.  On Windows
			// "no affinity assigned" is reported either as an empty list or as a full
			// mask covering every host CPU; anything in between is a strict subset
			// (i.e. exclusive pinning) which the feature gate must prevent.
			ginkgo.By("verifying no exclusive CPU affinity is set even for a guaranteed container")
			hostCPUs := int(getLocalNode(ctx, f).Status.Capacity.Cpu().Value())
			gomega.Consistently(ctx, func(ctx context.Context) error {
				aff, err := getWindowsContainerCPUAffinity(ctx, criClient, pod, "gu-ctr")
				if err != nil {
					return err
				}
				cpuCount := countCPUsInAffinities(aff)
				if cpuCount != 0 && cpuCount < hostCPUs {
					return fmt.Errorf("feature gate off: guaranteed container received exclusive affinity: got %d CPUs (host has %d)", cpuCount, hostCPUs)
				}
				return nil
			}, 10*time.Second, 2*time.Second).Should(gomega.Succeed())
		})

		ginkgo.It("should NOT set CPU affinity on a multi-container guaranteed pod when feature gate is off", func(ctx context.Context) {
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), 3)

			pod := makeWindowsCPUManagerPod("gu-pod-mc-no-affinity", []windowsCtnAttribute{
				{name: "gu-ctr-1", cpuRequest: "1000m", cpuLimit: "1000m"},
				{name: "gu-ctr-2", cpuRequest: "2000m", cpuLimit: "2000m"},
			})
			ginkgo.By("creating the guaranteed multi-container pod")
			pod = createPodSync(ctx, pod)

			ginkgo.By("verifying no exclusive CPU affinity is set on either container")
			hostCPUs := int(getLocalNode(ctx, f).Status.Capacity.Cpu().Value())
			gomega.Consistently(ctx, func(ctx context.Context) error {
				aff1, err := getWindowsContainerCPUAffinity(ctx, criClient, pod, "gu-ctr-1")
				if err != nil {
					return err
				}
				aff2, err := getWindowsContainerCPUAffinity(ctx, criClient, pod, "gu-ctr-2")
				if err != nil {
					return err
				}
				c1 := countCPUsInAffinities(aff1)
				c2 := countCPUsInAffinities(aff2)
				if c1 != 0 && c1 < hostCPUs {
					return fmt.Errorf("feature gate off: gu-ctr-1 received exclusive affinity: got %d CPUs (host has %d)", c1, hostCPUs)
				}
				if c2 != 0 && c2 < hostCPUs {
					return fmt.Errorf("feature gate off: gu-ctr-2 received exclusive affinity: got %d CPUs (host has %d)", c2, hostCPUs)
				}
				return nil
			}, 10*time.Second, 2*time.Second).Should(gomega.Succeed())
		})
	})

	// -------------------------------------------------------------------------
	// Init / sidecar container tests
	// Windows supports init containers and restartable init (sidecar) containers.
	// These mirror the Linux "checking the sidecar containers" group.
	//
	// Verification approach differs from Linux: on Windows we cannot read the
	// cpuset filesystem.  Instead:
	//   - For terminated init containers we can only observe the outcome on the
	//     main container (CPUs released and re-used).
	//   - For running sidecar (restartable init) containers we query the CRI.
	// -------------------------------------------------------------------------
	ginkgo.When("running pods with init containers", ginkgo.Label("guaranteed", "init-containers"), func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			updateWindowsKubeletConfigIfNeeded(ctx, f, buildWindowsCPUManagerKubeletConfig(oldCfg, true))
		})

		// Mirrors: "should reuse init container exclusive CPUs, but not sidecar container
		// exclusive CPUs".
		// A terminated (non-restartable) init container releases its exclusive CPUs so
		// that the regular app container can use them.  A restartable sidecar init
		// container holds its CPUs for its entire lifetime.
		// We verify:
		//   1. The sidecar container (restartable init) holds exactly 1 exclusive CPU.
		//   2. The app container holds exactly 1 exclusive CPU.
		//   3. The sidecar and app CPUs do not overlap (both are exclusive).
		ginkgo.It("sidecar container should hold exclusive CPUs separately from the app container", func(ctx context.Context) {
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), 2) // 1 non-restartable init + 1 sidecar reused + 1 app = 2 total (init CPUs reused)

			var restartAlways = v1.ContainerRestartPolicyAlways

			pod := makeWindowsCPUManagerInitPod("gu-sidecar-pod",
				[]windowsCtnAttribute{
					// non-restartable init: terminates quickly, CPUs are reused
					{name: "init-ctr", cpuRequest: "1000m", cpuLimit: "1000m"},
					// restartable sidecar: stays alive, holds exclusive CPUs
					{name: "sidecar-ctr", cpuRequest: "1000m", cpuLimit: "1000m", restartPolicy: &restartAlways},
				},
				// app container
				windowsCtnAttribute{name: "app-ctr", cpuRequest: "1000m", cpuLimit: "1000m"},
			)
			ginkgo.By("creating the pod with a non-restartable init container, a sidecar, and an app container")
			pod = createPodSync(ctx, pod)

			ginkgo.By("verifying sidecar container holds exactly 1 CPU")
			var sidecarAff, appAff []*runtimeapi.WindowsCpuGroupAffinity
			gomega.Eventually(ctx, func(ctx context.Context) error {
				var err error
				sidecarAff, err = getWindowsContainerCPUAffinity(ctx, criClient, pod, "sidecar-ctr")
				if err != nil {
					return err
				}
				appAff, err = getWindowsContainerCPUAffinity(ctx, criClient, pod, "app-ctr")
				if err != nil {
					return err
				}
				cs, ca := countCPUsInAffinities(sidecarAff), countCPUsInAffinities(appAff)
				if cs != 1 || ca != 1 {
					return fmt.Errorf("want sidecar=1 app=1, got sidecar=%d app=%d", cs, ca)
				}
				return nil
			}, 60*time.Second, 2*time.Second).Should(gomega.Succeed(),
				"both sidecar and app containers must each hold exactly 1 exclusive CPU")

			ginkgo.By("verifying sidecar and app container CPU affinity masks do not overlap")
			gomega.Expect(windowsAffinitiesOverlap(sidecarAff, appAff)).To(gomega.BeFalse(),
				"sidecar and app containers must not share exclusively-allocated CPUs")
		})

		// A pod whose only init container is non-restartable: after the init
		// container completes, the app container gets exclusive CPUs (which may
		// overlap with the init container's former CPUs — that is expected and
		// correct; we only verify the app container count here).
		ginkgo.It("app container should get exclusive CPUs after a non-restartable init container completes", func(ctx context.Context) {
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), 1)

			pod := makeWindowsCPUManagerInitPod("gu-init-pod",
				[]windowsCtnAttribute{
					// init container: terminates after a brief sleep
					{name: "init-ctr", cpuRequest: "1000m", cpuLimit: "1000m"},
				},
				windowsCtnAttribute{name: "app-ctr", cpuRequest: "1000m", cpuLimit: "1000m"},
			)
			ginkgo.By("creating the pod with a non-restartable init container")
			pod = createPodSync(ctx, pod)

			ginkgo.By("verifying the app container holds exactly 1 exclusive CPU after init completes")
			gomega.Eventually(ctx, func(ctx context.Context) (int, error) {
				aff, err := getWindowsContainerCPUAffinity(ctx, criClient, pod, "app-ctr")
				if err != nil {
					return 0, err
				}
				return countCPUsInAffinities(aff), nil
			}, 60*time.Second, 2*time.Second).Should(gomega.Equal(1),
				"app container must hold exactly 1 exclusive CPU")
		})
	})
	// -------------------------------------------------------------------------
	// Strict vs non-strict CPU reservation
	// With the strict-cpu-reservation policy option the reserved CPUs are removed
	// from the shared pool, so shared-pool (burstable) containers are confined to
	// (online - reserved). Without it (the default) the reserved CPUs remain part
	// of the shared pool and stay usable by burstable containers.
	// -------------------------------------------------------------------------
	ginkgo.When("running with the strict-cpu-reservation policy option", ginkgo.Label("non-guaranteed", "strict-cpu-reservation"), func() {
		// Toggling strict-cpu-reservation changes whether the reserved CPU belongs
		// to the default pool, which invalidates the persisted CPU manager
		// checkpoint. Reset to a clean non-strict state on the way out so the
		// checkpoint left behind is compatible with the rest of the suite.
		ginkgo.AfterEach(func(ctx context.Context) {
			updateWindowsKubeletConfigClearState(ctx, f, buildWindowsCPUManagerKubeletConfig(oldCfg, true))
		})

		ginkgo.It("should exclude the reserved CPU from the burstable shared pool when strict-cpu-reservation is enabled", func(ctx context.Context) {
			node := getLocalNode(ctx, f)
			hostCPUs := int(node.Status.Capacity.Cpu().Value())
			if hostCPUs < 2 {
				ginkgo.Skip(fmt.Sprintf("strict-cpu-reservation test needs >= 2 CPUs (1 reserved + shared pool), node has %d", hostCPUs))
			}

			ginkgo.By("enabling the static CPU manager with strict-cpu-reservation and CPU 0 reserved")
			updateWindowsKubeletConfigClearState(ctx, f, buildWindowsStrictCPUReservationConfig(oldCfg))

			pod := createPodSync(ctx, makeWindowsCPUManagerPod("strict-burstable-pod", []windowsCtnAttribute{
				{name: "bu-ctr", cpuRequest: "100m", cpuLimit: "300m"},
			}))

			// The reconcile loop applies the shared pool shortly after the container
			// starts; with strict reservation the reserved CPU 0 is excluded from it.
			ginkgo.By("verifying the burstable container is confined to (host - reserved) CPUs")
			gomega.Eventually(ctx, func(ctx context.Context) (int, error) {
				aff, err := getWindowsContainerCPUAffinity(ctx, criClient, pod, "bu-ctr")
				if err != nil {
					return 0, err
				}
				return countCPUsInAffinities(aff), nil
			}, 60*time.Second, 2*time.Second).Should(gomega.Equal(hostCPUs-1),
				"with strict-cpu-reservation the burstable shared pool must exclude the 1 reserved CPU (host=%d)", hostCPUs)
		})

		ginkgo.It("should keep the reserved CPU in the burstable shared pool when strict-cpu-reservation is disabled (default)", func(ctx context.Context) {
			node := getLocalNode(ctx, f)
			hostCPUs := int(node.Status.Capacity.Cpu().Value())
			if hostCPUs < 2 {
				ginkgo.Skip(fmt.Sprintf("test needs >= 2 CPUs, node has %d", hostCPUs))
			}

			ginkgo.By("enabling the static CPU manager without strict-cpu-reservation (CPU 0 reserved)")
			updateWindowsKubeletConfigClearState(ctx, f, buildWindowsCPUManagerKubeletConfig(oldCfg, true))

			pod := createPodSync(ctx, makeWindowsCPUManagerPod("nonstrict-burstable-pod", []windowsCtnAttribute{
				{name: "bu-ctr", cpuRequest: "100m", cpuLimit: "300m"},
			}))

			// Without strict reservation and with no exclusive allocations, the
			// shared pool is the whole machine, including the reserved CPU.
			ginkgo.By("verifying the burstable container can use all host CPUs (reserved included)")
			gomega.Eventually(ctx, func(ctx context.Context) (int, error) {
				aff, err := getWindowsContainerCPUAffinity(ctx, criClient, pod, "bu-ctr")
				if err != nil {
					return 0, err
				}
				return countCPUsInAffinities(aff), nil
			}, 60*time.Second, 2*time.Second).Should(gomega.Equal(hostCPUs),
				"without strict-cpu-reservation the burstable shared pool must include the reserved CPU (host=%d)", hostCPUs)
		})
	})

	// -------------------------------------------------------------------------
	// Dynamic shared-pool resizing (reconcile loop)
	// The CPU manager reconcile loop updates a *running* shared-pool container's
	// affinity as guaranteed pods take and release exclusive CPUs. This verifies
	// the live transition, not just the steady state at pod start.
	// -------------------------------------------------------------------------
	ginkgo.When("dynamically resizing the shared pool", ginkgo.Label("guaranteed", "non-guaranteed", "shared-pool", "reconcile"), func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			updateWindowsKubeletConfigIfNeeded(ctx, f, buildWindowsCPUManagerKubeletConfig(oldCfg, true))
		})

		ginkgo.It("should shrink and grow a running burstable container's affinity as a guaranteed pod comes and goes", func(ctx context.Context) {
			node := getLocalNode(ctx, f)
			hostCPUs := int(node.Status.Capacity.Cpu().Value())
			// Need >= 2 CPUs: CPU 0 reserved (which stays in the shared pool in
			// non-strict mode) plus at least one non-reserved CPU for the guaranteed
			// pod's exclusive allocation. The shared pool is then never empty, so the
			// burstable mask shrinks from hostCPUs to hostCPUs-1 and back.
			if hostCPUs < 2 {
				ginkgo.Skip(fmt.Sprintf("dynamic shared-pool test needs >= 2 CPUs, node has %d", hostCPUs))
			}

			buPod := createPodSync(ctx, makeWindowsCPUManagerPod("dyn-bu-pod", []windowsCtnAttribute{
				{name: "bu-ctr", cpuRequest: "100m", cpuLimit: "300m"},
			}))
			burstableCPUCount := func(ctx context.Context) (int, error) {
				aff, err := getWindowsContainerCPUAffinity(ctx, criClient, buPod, "bu-ctr")
				if err != nil {
					return 0, err
				}
				return countCPUsInAffinities(aff), nil
			}

			ginkgo.By("waiting for the burstable container to occupy the full shared pool")
			gomega.Eventually(ctx, burstableCPUCount, 60*time.Second, 2*time.Second).Should(gomega.Equal(hostCPUs),
				"with no exclusive allocations the burstable shared pool should span all %d host CPUs", hostCPUs)

			ginkgo.By("creating a guaranteed pod that takes 1 exclusive CPU")
			guPod := createPodSync(ctx, makeWindowsCPUManagerPod("dyn-gu-pod", []windowsCtnAttribute{
				{name: "gu-ctr", cpuRequest: "1000m", cpuLimit: "1000m"},
			}))

			ginkgo.By("verifying the running burstable container's affinity shrinks by the 1 exclusive CPU")
			gomega.Eventually(ctx, burstableCPUCount, 60*time.Second, 2*time.Second).Should(gomega.Equal(hostCPUs-1),
				"the reconcile loop should remove the guaranteed pod's exclusive CPU from the running burstable container's mask")

			ginkgo.By("deleting the guaranteed pod to release its exclusive CPU")
			e2epod.NewPodClient(f).DeleteSync(ctx, guPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			delete(podMap, string(guPod.UID))

			ginkgo.By("verifying the running burstable container's affinity grows back to the full shared pool")
			gomega.Eventually(ctx, burstableCPUCount, 60*time.Second, 2*time.Second).Should(gomega.Equal(hostCPUs),
				"the reconcile loop should return the released CPU to the running burstable container's mask")
		})
	})
})

// -------------------------------------------------------------------------
// Pod / container creation helpers
// -------------------------------------------------------------------------

// windowsCtnAttribute describes a single container's CPU resource requirements
// for use with makeWindowsCPUManagerPod / makeWindowsCPUManagerInitPod.
type windowsCtnAttribute struct {
	name          string
	cpuRequest    string
	cpuLimit      string
	restartPolicy *v1.ContainerRestartPolicy
}

// makeWindowsCPUManagerPod builds a Pod spec for Windows CPU affinity tests.
// Containers run a PowerShell sleep so they stay alive without Linux-specific
// cgroup mounts or /proc filesystem access.
func makeWindowsCPUManagerPod(podName string, attrs []windowsCtnAttribute) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers:    buildWindowsContainers(attrs),
			NodeSelector:  map[string]string{"kubernetes.io/os": "windows"},
		},
	}
}

// verifyHostMatchesCRI asserts that the Windows kernel's job-object affinity
// for ctnName agrees bit-for-bit with the affinity CRI reports. This proves the
// runtime actually applied the mask, not just that containerd recorded it.
func verifyHostMatchesCRI(ctx context.Context, criClient internalapi.RuntimeService, pod *v1.Pod, ctnName string) {
	ginkgo.GinkgoHelper()
	criAff, err := getWindowsContainerCPUAffinity(ctx, criClient, pod, ctnName)
	framework.ExpectNoError(err, "failed to fetch CRI affinity for host comparison of %q", ctnName)
	framework.ExpectNoError(
		validateHostJobAffinityProcessIsolated(pod, ctnName, criAff),
		"host job-object affinity does not match CRI-reported affinity for %q", ctnName)
}

// verifyHostMasksDisjoint asserts that the kernel-applied affinity masks of two
// containers (in the same or different pods) share no CPU — the host-level
// counterpart to the CRI windowsAffinitiesOverlap check.
func verifyHostMasksDisjoint(pod1 *v1.Pod, ctn1 string, pod2 *v1.Pod, ctn2 string) {
	ginkgo.GinkgoHelper()
	h1, err := getHostJobAffinity(pod1, ctn1)
	framework.ExpectNoError(err, "failed to read host job affinity for %q", ctn1)
	h2, err := getHostJobAffinity(pod2, ctn2)
	framework.ExpectNoError(err, "failed to read host job affinity for %q", ctn2)
	gomega.Expect(hostJobAffinitiesOverlap(h1, h2)).To(gomega.BeFalse(),
		"containers %q and %q must not share exclusively-allocated CPUs at the kernel level", ctn1, ctn2)
}

// makeWindowsCPUManagerInitPod builds a Pod spec that has init containers
// (possibly restartable / sidecar) followed by a regular app container.
func makeWindowsCPUManagerInitPod(podName string, initAttrs []windowsCtnAttribute, appAttr windowsCtnAttribute) *v1.Pod {
	initContainers := buildWindowsContainers(initAttrs)
	for i := range initContainers {
		if initAttrs[i].restartPolicy != nil {
			initContainers[i].RestartPolicy = initAttrs[i].restartPolicy
		} else {
			// Non-restartable init containers exit quickly.
			initContainers[i].Command = []string{"powershell.exe", "-Command", "Start-Sleep -Seconds 2"}
		}
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName},
		Spec: v1.PodSpec{
			RestartPolicy:  v1.RestartPolicyNever,
			InitContainers: initContainers,
			Containers:     buildWindowsContainers([]windowsCtnAttribute{appAttr}),
			NodeSelector:   map[string]string{"kubernetes.io/os": "windows"},
		},
	}
}

// buildWindowsContainers converts a slice of windowsCtnAttribute into v1.Container
// objects suitable for Windows nodes (PowerShell command, no Linux volume mounts).
func buildWindowsContainers(attrs []windowsCtnAttribute) []v1.Container {
	var containers []v1.Container
	for _, attr := range attrs {
		requests := v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("128Mi"),
		}
		if attr.cpuRequest != "" {
			requests[v1.ResourceCPU] = resource.MustParse(attr.cpuRequest)
		}
		limits := v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("128Mi"),
		}
		if attr.cpuLimit != "" {
			limits[v1.ResourceCPU] = resource.MustParse(attr.cpuLimit)
		}
		containers = append(containers, v1.Container{
			Name:  attr.name,
			Image: busyboxImage,
			Resources: v1.ResourceRequirements{
				Requests: requests,
				Limits:   limits,
			},
			// Long-running sleep; powershell.exe is available in the Windows BusyBox image.
			Command: []string{"powershell.exe", "-Command", "Start-Sleep -Seconds 86400"},
		})
	}
	return containers
}

// -------------------------------------------------------------------------
// Kubelet configuration helpers
// -------------------------------------------------------------------------

// buildWindowsCPUManagerKubeletConfig returns a KubeletConfiguration with the
// static CPU manager policy.  When featureGateOn is true, the
// WindowsCPUAndMemoryAffinity feature gate is also enabled.
func buildWindowsCPUManagerKubeletConfig(oldCfg *kubeletconfig.KubeletConfiguration, featureGateOn bool) *kubeletconfig.KubeletConfiguration {
	newCfg := oldCfg.DeepCopy()
	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = make(map[string]bool)
	}
	newCfg.FeatureGates[string(features.WindowsCPUAndMemoryAffinity)] = featureGateOn
	newCfg.CPUManagerPolicy = string(cpumanager.PolicyStatic)
	newCfg.CPUManagerReconcilePeriod = metav1.Duration{Duration: 1 * time.Second}
	// Reserve CPU 0 so the remaining CPUs are available for guaranteed pods.
	newCfg.ReservedSystemCPUs = "0"
	return newCfg
}

// buildWindowsStrictCPUReservationConfig returns a static CPU manager config
// (feature gate on) with the strict-cpu-reservation policy option enabled, which
// removes the reserved CPUs from the shared pool. CPUManagerPolicyOptions is GA
// and locked on, so only the option map needs to be set.
func buildWindowsStrictCPUReservationConfig(oldCfg *kubeletconfig.KubeletConfiguration) *kubeletconfig.KubeletConfiguration {
	newCfg := buildWindowsCPUManagerKubeletConfig(oldCfg, true)
	newCfg.CPUManagerPolicyOptions = map[string]string{
		cpumanager.StrictCPUReservationOption: "true",
	}
	return newCfg
}

// updateWindowsKubeletConfig stops the kubelet Windows service, writes the new
// configuration file, and restarts the service.
func updateWindowsKubeletConfig(ctx context.Context, f *framework.Framework, cfg *kubeletconfig.KubeletConfiguration) {
	ginkgo.GinkgoHelper()
	kubeletStart := mustStopKubelet(ctx, f)
	framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(cfg), "failed to write kubelet config file")
	kubeletStart(ctx)
}

// updateWindowsKubeletConfigClearState is like updateWindowsKubeletConfig but
// also removes the CPU/memory manager state files while the kubelet is stopped.
// This is required when the new configuration invalidates the persisted
// checkpoint (e.g. toggling strict-cpu-reservation, which changes whether the
// reserved CPU belongs to the default pool) — otherwise the kubelet refuses to
// start with "invalid state, please drain node and remove policy state file".
// It mirrors the Linux updateKubeletConfig(..., deleteStateFiles=true).
func updateWindowsKubeletConfigClearState(ctx context.Context, f *framework.Framework, cfg *kubeletconfig.KubeletConfiguration) {
	ginkgo.GinkgoHelper()
	kubeletStart := mustStopKubelet(ctx, f)
	deleteStateFile(cpuManagerStateFile)
	deleteStateFile(memoryManagerStateFile)
	framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(cfg), "failed to write kubelet config file")
	kubeletStart(ctx)
}

// updateWindowsKubeletConfigIfNeeded calls updateWindowsKubeletConfig only when
// the desired configuration differs from what is currently running, avoiding
// unnecessary kubelet restarts between tests in the same When block.
func updateWindowsKubeletConfigIfNeeded(ctx context.Context, f *framework.Framework, desired *kubeletconfig.KubeletConfiguration) {
	ginkgo.GinkgoHelper()
	current, err := getCurrentKubeletConfig(ctx)
	framework.ExpectNoError(err, "failed to get current kubelet config")
	if equalKubeletConfiguration(current, desired) {
		framework.Logf("kubelet configuration already matches desired state, skipping restart")
		return
	}
	updateWindowsKubeletConfig(ctx, f, desired)
}

// equalKubeletConfiguration returns true when the two configurations are
// semantically equal (ignoring TypeMeta which is not meaningful for comparison).
func equalKubeletConfiguration(a, b *kubeletconfig.KubeletConfiguration) bool {
	a = a.DeepCopy()
	b = b.DeepCopy()
	a.TypeMeta = metav1.TypeMeta{}
	b.TypeMeta = metav1.TypeMeta{}
	return reflect.DeepEqual(a, b)
}

// -------------------------------------------------------------------------
// Skip / node helpers
// -------------------------------------------------------------------------

// skipIfAllocatableCPUsLessThan skips the current test when the node does not
// have enough allocatable integer CPUs to satisfy the test.
// One CPU (CPU 0) is always reserved by buildWindowsCPUManagerKubeletConfig,
// so the minimum allocatable count is (requested + 1).
func skipIfAllocatableCPUsLessThan(node *v1.Node, requested int) {
	ginkgo.GinkgoHelper()
	allocatable := node.Status.Allocatable[v1.ResourceCPU]
	need := int64(requested + 1) // +1 for the reserved CPU
	if allocatable.Value() < need {
		ginkgo.Skip(fmt.Sprintf(
			"skipping: node has %d allocatable CPUs but test needs %d (including 1 reserved)",
			allocatable.Value(), need))
	}
}
