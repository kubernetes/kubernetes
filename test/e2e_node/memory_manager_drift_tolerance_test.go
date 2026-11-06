//go:build linux

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

package e2enode

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2enodekubelet "k8s.io/kubernetes/test/e2e_node/kubeletconfig"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	memoryDriftToleranceBytes = 128 * 1024 * 1024
	memoryDriftShiftBytes     = 4 * 1024 * 1024
	memoryDriftGraceBytes     = 64 * 1024 * 1024
)

var _ = SIGDescribe("Memory Manager Drift Tolerance", framework.WithDisruptive(), framework.WithSerial(), feature.MemoryManager, framework.WithFeatureGate(features.MemoryManagerDriftTolerance), func() {
	f := framework.NewDefaultFramework("memory-manager-drift-tolerance")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var oldCfg *kubeletconfig.KubeletConfiguration

	ginkgo.BeforeEach(func(ctx context.Context) {
		var err error
		if oldCfg == nil {
			oldCfg, err = getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)
		}
		ginkgo.DeferCleanup(func(ctx context.Context) {
			if e2enode.HealthCheck(kubeletHealthCheckURL) {
				updateKubeletConfig(ctx, f, oldCfg, true)
				return
			}
			deleteStateFile(cpuManagerStateFile)
			deleteStateFile(memoryManagerStateFile)
			deleteStateFile(usernsStateFiles)
			framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(oldCfg))
			kubeletServiceName := findKubeletServiceName(false)
			stdout, err := exec.CommandContext(ctx, "sudo", "systemctl", "restart", kubeletServiceName).CombinedOutput()
			framework.ExpectNoError(err, "Failed to restart kubelet with systemctl: %v, %s", err, string(stdout))
			waitForKubeletToStart(ctx, f)
		})
	})

	ginkgo.It("should report the tolerance and a zero drift for every NUMA node after a fresh start", func(ctx context.Context) {
		ginkgo.By("Setting memory-drift-tolerance to 128Mi")
		setMemoryDriftTolerance(ctx, f, oldCfg, "128Mi")
		checkpoint := waitForMemoryManagerState(ctx)

		ginkgo.By("Checking the drift metrics")
		gomega.Eventually(ctx, getKubeletMetrics, time.Minute, 15*time.Second).Should(
			matchMemoryDriftMetrics(timelessSample(memoryDriftToleranceBytes), checkpoint.MachineState, nil))
	})

	ginkgo.It("should start again and keep the memory assignments after a kubelet restart with the tolerance set to auto", func(ctx context.Context) {
		ginkgo.By("Setting memory-drift-tolerance to auto")
		setMemoryDriftTolerance(ctx, f, oldCfg, "auto")

		ginkgo.By("Creating the test pod")
		pod := createMemoryDriftTestPod(ctx, f)
		recorded := waitForMemoryManagerState(ctx)

		ginkgo.By("Restarting the kubelet")
		restartKubelet(ctx, true)
		waitForKubeletToStart(ctx, f)

		ginkgo.By("Checking the drift metrics")
		gomega.Eventually(ctx, getKubeletMetrics, time.Minute, 15*time.Second).Should(
			matchMemoryDriftMetrics(timelessSampleAtLeast(memoryDriftGraceBytes+1), recorded.MachineState, nil))

		ginkgo.By("Checking that the memory assignment of the test pod is kept")
		checkpoint, err := getMemoryManagerState()
		framework.ExpectNoError(err)
		gomega.Expect(checkpoint.Entries).To(gomega.HaveKeyWithValue(string(pod.UID), gomega.HaveKeyWithValue(pod.Spec.Containers[0].Name, gomega.Not(gomega.BeEmpty()))))
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))
	})

	ginkgo.It("should tolerate a recorded NUMA node memory drift within the bound and re-baseline the state", func(ctx context.Context) {
		ginkgo.By("Setting memory-drift-tolerance to 128Mi")
		setMemoryDriftTolerance(ctx, f, oldCfg, "128Mi")

		ginkgo.By("Creating the test pod")
		pod := createMemoryDriftTestPod(ctx, f)
		recorded := waitForMemoryManagerState(ctx)
		gomega.Expect(recorded.MachineState).To(gomega.HaveKey(0))
		gomega.Expect(recorded.MachineState[0].MemoryMap).To(gomega.HaveKey(v1.ResourceMemory))
		machineTotal := recorded.MachineState[0].MemoryMap[v1.ResourceMemory].TotalMemSize

		ginkgo.By("Shifting the recorded memory of NUMA node 0 while the kubelet is stopped")
		withStoppedKubelet(ctx, f, false, func() {
			shiftRecordedMemory(ctx, 0, memoryDriftShiftBytes)
		})

		ginkgo.By("Checking the drift metrics")
		gomega.Eventually(ctx, getKubeletMetrics, time.Minute, 15*time.Second).Should(
			matchMemoryDriftMetrics(timelessSample(memoryDriftToleranceBytes), recorded.MachineState, map[int]uint64{0: memoryDriftShiftBytes}))

		ginkgo.By("Checking that the state is re-baselined and the memory assignment of the test pod is kept")
		checkpoint, err := getMemoryManagerState()
		framework.ExpectNoError(err)
		gomega.Expect(checkpoint.MachineState).To(gomega.HaveKey(0))
		gomega.Expect(checkpoint.MachineState[0].MemoryMap).To(gomega.HaveKey(v1.ResourceMemory))
		gomega.Expect(checkpoint.MachineState[0].MemoryMap[v1.ResourceMemory].TotalMemSize).To(gomega.Equal(machineTotal), "the recorded total memory of NUMA node 0 is re-baselined")
		gomega.Expect(checkpoint.Entries).To(gomega.HaveKeyWithValue(string(pod.UID), gomega.HaveKeyWithValue(pod.Spec.Containers[0].Name, gomega.Not(gomega.BeEmpty()))))
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace))
	})
})

func setMemoryDriftTolerance(ctx context.Context, f *framework.Framework, oldCfg *kubeletconfig.KubeletConfiguration, tolerance string) {
	ginkgo.GinkgoHelper()

	newCfg := oldCfg.DeepCopy()
	updateKubeletConfigWithMemoryManagerParams(newCfg,
		&memoryManagerKubeletParams{
			policy: staticPolicy,
			systemReservedMemory: []kubeletconfig.MemoryReservation{
				{
					NumaNode: 0,
					Limits: v1.ResourceList{
						resourceMemory: resource.MustParse("1100Mi"),
					},
				},
			},
			systemReserved: map[string]string{resourceMemory: "500Mi"},
			kubeReserved:   map[string]string{resourceMemory: "500Mi"},
			evictionHard:   map[string]string{evictionHardMemory: "100Mi"},
		},
	)
	newCfg.FeatureGates[string(features.MemoryManagerDriftTolerance)] = true
	newCfg.MemoryManagerPolicyOptions = map[string]string{memorymanager.MemoryDriftTolerance: tolerance}
	updateKubeletConfig(ctx, f, newCfg, true)
}

func createMemoryDriftTestPod(ctx context.Context, f *framework.Framework) *v1.Pod {
	ginkgo.GinkgoHelper()

	pod := e2epod.NewPodClient(f).CreateSync(ctx, makeMemoryManagerPod("memmngrpod", nil,
		[]memoryManagerCtnAttributes{
			{
				ctnName: "memmngrcnt",
				cpus:    "100m",
				memory:  "64Mi",
			},
		}),
	)
	ginkgo.DeferCleanup(func(ctx context.Context) {
		deletePodSyncByName(ctx, f, pod.Name)
	})
	return pod
}

func waitForMemoryManagerState(ctx context.Context) *state.MemoryManagerCheckpoint {
	ginkgo.GinkgoHelper()

	var checkpoint *state.MemoryManagerCheckpoint
	gomega.Eventually(ctx, func() error {
		var err error
		checkpoint, err = getMemoryManagerState()
		if err != nil {
			return err
		}
		if len(checkpoint.MachineState) == 0 {
			return fmt.Errorf("the memory manager state file %s has no machine state yet", memoryManagerStateFile)
		}
		return nil
	}).WithTimeout(time.Minute).WithPolling(5 * time.Second).Should(gomega.Succeed())
	return checkpoint
}

func shiftRecordedMemory(ctx context.Context, nodeID int, shift uint64) {
	ginkgo.GinkgoHelper()

	st, err := state.NewCheckpointState(klog.FromContext(ctx), filepath.Dir(memoryManagerStateFile), filepath.Base(memoryManagerStateFile), staticPolicy)
	framework.ExpectNoError(err)

	machineState := st.GetMachineState()
	gomega.Expect(machineState).To(gomega.HaveKey(nodeID))
	gomega.Expect(machineState[nodeID].MemoryMap).To(gomega.HaveKey(v1.ResourceMemory))

	memoryTable := machineState[nodeID].MemoryMap[v1.ResourceMemory]
	memoryTable.TotalMemSize += shift
	memoryTable.Allocatable += shift
	memoryTable.Free += shift
	st.SetMachineState(machineState)
}

func matchMemoryDriftMetrics(tolerance types.GomegaMatcher, machineState state.NUMANodeMap, driftBytes map[int]uint64) types.GomegaMatcher {
	drift := gstruct.Elements{}
	for nodeID := range machineState {
		drift[strconv.Itoa(nodeID)] = timelessSample(driftBytes[nodeID])
	}
	return gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
		"kubelet_memory_manager_drift_tolerance_bytes": gstruct.MatchAllElements(nodeID, gstruct.Elements{
			"": tolerance,
		}),
		"kubelet_memory_manager_drift_bytes": gstruct.MatchAllElements(makeCustomLabelID(metrics.AlignedNUMANode), drift),
	})
}
