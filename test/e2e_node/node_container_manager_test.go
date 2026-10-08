//go:build linux

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
	"os"
	"maps"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/stats/pidlimit"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2enodekubelet "k8s.io/kubernetes/test/e2e_node/kubeletconfig"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// hugepageReservations describes per-size hugepage reservations for use in
func setDesiredConfiguration(initialConfig *kubeletconfig.KubeletConfiguration, cgroupManager cm.CgroupManager, hpSystemReserved map[string]string, hpKubeReserved map[string]string) {
	initialConfig.EnforceNodeAllocatable = []string{"pods", kubeReservedCgroup, systemReservedCgroup}
	initialConfig.SystemReserved = map[string]string{
		string(v1.ResourceCPU):    "100m",
		string(v1.ResourceMemory): "100Mi",
		string(pidlimit.PIDs):     "1000",
	}
	initialConfig.KubeReserved = map[string]string{
		string(v1.ResourceCPU):    "100m",
		string(v1.ResourceMemory): "100Mi",
		string(pidlimit.PIDs):     "738",
	}
	maps.Copy(initialConfig.SystemReserved, hpSystemReserved)
	maps.Copy(initialConfig.KubeReserved, hpKubeReserved)
	initialConfig.EvictionHard = map[string]string{"memory.available": "100Mi"}
	// Necessary for allocatable cgroup creation.
	initialConfig.CgroupsPerQOS = true
	initialConfig.KubeReservedCgroup = kubeReservedCgroup
	initialConfig.SystemReservedCgroup = systemReservedCgroup

	if initialConfig.CgroupDriver == "systemd" {
		initialConfig.KubeReservedCgroup = cm.NewCgroupName(cm.RootCgroupName, kubeReservedCgroup).ToSystemd()
		initialConfig.SystemReservedCgroup = cm.NewCgroupName(cm.RootCgroupName, systemReservedCgroup).ToSystemd()
	}
}

var _ = SIGDescribe("Node Container Manager", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("node-container-manager")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	f.Describe("Validate Node Allocatable", feature.NodeAllocatable, func() {
		ginkgo.It("sets up the node and runs the test", func(ctx context.Context) {
			framework.ExpectNoError(runTest(ctx, f))
		})
		ginkgo.It("should handle introducing hugepage reservation on a running node", func(ctx context.Context) {
			framework.ExpectNoError(runHugepagesUpgradeTest(ctx, f))
		})
	})
	f.Describe("Validate CGroup management", func() {
		// Regression test for https://issues.k8s.io/125923
		// In this issue there's a race involved with systemd which seems to manifest most likely, or perhaps only
		// (data gathered so far seems inconclusive) on the very first boot of the machine, so restarting the kubelet
		// seems not sufficient. OTOH, the exact reproducer seems to require a dedicate lane with only this test, or
		// to reboot the machine before to run this test. Both are practically unrealistic in CI.
		// The closest approximation is this test in this current form, using a kubelet restart. This at least
		// acts as non regression testing, so it still brings value.
		ginkgo.It("should correctly start with cpumanager none policy in use with systemd", func(ctx context.Context) {
			ginkgo.Skip("currently broken")

			if !IsCgroup2UnifiedMode() {
				ginkgo.Skip("this test requires cgroups v2")
			}

			var err error
			var oldCfg *kubeletconfig.KubeletConfiguration
			// Get current kubelet configuration
			oldCfg, err = getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)

			ginkgo.DeferCleanup(func(ctx context.Context) {
				if oldCfg != nil {
					// Update the Kubelet configuration.
					framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(oldCfg))

					ginkgo.By("Restarting the kubelet")
					restartKubelet(ctx, true)

					waitForKubeletToStart(ctx, f)
					ginkgo.By("Started the kubelet")
				}
			})

			newCfg := oldCfg.DeepCopy()
			// Change existing kubelet configuration
			newCfg.CPUManagerPolicy = "none"
			newCfg.CgroupDriver = "systemd"
			newCfg.FailCgroupV1 = true // extra safety. We want to avoid false negatives though, so we added the skip check earlier

			// Update the Kubelet configuration.
			framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(newCfg))

			ginkgo.By("Restarting the kubelet")
			restartKubelet(ctx, true)

			waitForKubeletToStart(ctx, f)
			ginkgo.By("Started the kubelet")

			gomega.Consistently(ctx, func(ctx context.Context) bool {
				return getNodeReadyStatus(ctx, f) && e2enode.HealthCheck(kubeletHealthCheckURL)
			}).WithTimeout(2 * time.Minute).WithPolling(2 * time.Second).Should(gomega.BeTrueBecause("node keeps reporting ready status"))
		})
	})
})

func expectFileValToEqual(filePath string, expectedValue, delta int64) error {
	out, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read file %q", filePath)
	}
	actual, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return fmt.Errorf("failed to parse output %v", err)
	}

	// Ensure that values are within a delta range to work around rounding errors.
	if (actual < (expectedValue - delta)) || (actual > (expectedValue + delta)) {
		return fmt.Errorf("Expected value at %q to be between %d and %d. Got %d", filePath, (expectedValue - delta), (expectedValue + delta), actual)
	}
	return nil
}

// hugepageReserved maps hugepage resource names (e.g. "hugepages-2Mi") to
// the total reserved quantity string (system + kube). Empty map means no
// hugepage reservations.
type hugepageReserved map[v1.ResourceName]string

type allocatableLimits struct {
	cpu, memory, pids *resource.Quantity
	hugepages         map[v1.ResourceName]*resource.Quantity
}

func getAllocatableLimits(cpu, memory, pids string, hpReserved hugepageReserved, capacity v1.ResourceList) allocatableLimits {
	var result allocatableLimits
	result.hugepages = make(map[v1.ResourceName]*resource.Quantity)

	for k, v := range capacity {
		if k == v1.ResourceCPU {
			c := v.DeepCopy()
			result.cpu = &c
			result.cpu.Sub(resource.MustParse(cpu))
		}
		if k == v1.ResourceMemory {
			c := v.DeepCopy()
			result.memory = &c
			result.memory.Sub(resource.MustParse(memory))
		}
		if reserved, ok := hpReserved[k]; ok {
			c := v.DeepCopy()
			result.hugepages[k] = &c
			result.hugepages[k].Sub(resource.MustParse(reserved))
		}
	}
	// Process IDs are not a node allocatable, so we have to do this ad hoc
	pidlimits, err := pidlimit.Stats()
	if err == nil && pidlimits != nil && pidlimits.MaxPID != nil {
		result.pids = resource.NewQuantity(int64(*pidlimits.MaxPID), resource.DecimalSI)
		result.pids.Sub(resource.MustParse(pids))
	}
	return result
}

const (
	kubeReservedCgroup    = "kube-reserved"
	systemReservedCgroup  = "system-reserved"
	nodeAllocatableCgroup = "kubepods"
)

func createIfNotExists(cm cm.CgroupManager, cgroupConfig *cm.CgroupConfig) error {
	if !cm.Exists(cgroupConfig.Name) {
		if err := cm.Create(klog.Background(), cgroupConfig); err != nil {
			return err
		}
	}
	return nil
}

func createTemporaryCgroupsForReservation(cgroupManager cm.CgroupManager) error {
	// Create kube reserved cgroup
	cgroupConfig := &cm.CgroupConfig{
		Name: cm.NewCgroupName(cm.RootCgroupName, kubeReservedCgroup),
	}
	if err := createIfNotExists(cgroupManager, cgroupConfig); err != nil {
		return err
	}
	// Create system reserved cgroup
	cgroupConfig.Name = cm.NewCgroupName(cm.RootCgroupName, systemReservedCgroup)

	return createIfNotExists(cgroupManager, cgroupConfig)
}

func destroyTemporaryCgroupsForReservation(cgroupManager cm.CgroupManager) error {
	// Create kube reserved cgroup
	cgroupConfig := &cm.CgroupConfig{
		Name: cm.NewCgroupName(cm.RootCgroupName, kubeReservedCgroup),
	}
	if err := cgroupManager.Destroy(klog.Background(), cgroupConfig); err != nil {
		return err
	}
	cgroupConfig.Name = cm.NewCgroupName(cm.RootCgroupName, systemReservedCgroup)
	return cgroupManager.Destroy(klog.Background(), cgroupConfig)
}

// convertSharesToWeight converts from cgroup v1 cpu.shares to cgroup v2 cpu.weight
func convertSharesToWeight(shares int64) int64 {
	return 1 + ((shares-2)*9999)/262142
}

func runTest(ctx context.Context, f *framework.Framework) error {
	var oldCfg *kubeletconfig.KubeletConfiguration
	subsystems, err := cm.GetCgroupSubsystems()
	if err != nil {
		return err
	}
	// Get current kubelet configuration
	oldCfg, err = getCurrentKubeletConfig(ctx)
	if err != nil {
		return err
	}

	// Create a cgroup manager object for manipulating cgroups.
	cgroupManager := cm.NewCgroupManager(klog.Background(), subsystems, oldCfg.CgroupDriver)

	ginkgo.DeferCleanup(destroyTemporaryCgroupsForReservation, cgroupManager)
	ginkgo.DeferCleanup(func(ctx context.Context) {
		if oldCfg != nil {
			// Update the Kubelet configuration.
			ginkgo.By("Stopping the kubelet")
			restartKubelet := mustStopKubelet(ctx, f)

			// wait until the kubelet health check will fail
			gomega.Eventually(ctx, func() bool {
				return e2enode.HealthCheck(kubeletHealthCheckURL)
			}, time.Minute, time.Second).Should(gomega.BeFalseBecause("expected kubelet health check to be failed"))

			framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(oldCfg))

			ginkgo.By("Restarting the kubelet")
			restartKubelet(ctx)
		}
	})
	if err := createTemporaryCgroupsForReservation(cgroupManager); err != nil {
		return err
	}

	newCfg := oldCfg.DeepCopy()

	// Provision hugepages before configuring kubelet so setDesiredConfiguration
	// can include them in system-reserved and kube-reserved.
	// 2Mi is required - setHugepages skips the test if not supported.
	hugepages := map[string]int{hugepagesResourceName2Mi: 4} // 4 × 2Mi = 8Mi capacity
	setHugepages(ctx, hugepages)
	ginkgo.DeferCleanup(releaseHugepages, hugepages)

	// 1Gi is best-effort - runtime allocation may fail if the host lacks
	// contiguous memory. Try directly and only include if successful.
	has1GiHugepages := false
	if isHugePageAvailable(hugepagesSize1G) {
		if err := configureHugePages(hugepagesSize1G, 2, nil); err == nil {
			has1GiHugepages = true
			hugepages[hugepagesResourceName1Gi] = 2 // 2 × 1Gi = 2Gi capacity
		} else {
			framework.Logf("Failed to allocate 1Gi hugepage at runtime, 1Gi validation will not be exercised: %v", err)
		}
	}

	// setDesiredConfiguration reserves: 2Mi sys + 2Mi kube = 4Mi for 2Mi pages.
	setDesiredConfiguration(newCfg, cgroupManager, map[string]string{hugepagesResourceName2Mi: "4Mi"}, nil)
	if has1GiHugepages {
		newCfg.SystemReserved["hugepages-1Gi"] = "1Gi"
	}

	// Set the new kubelet configuration.
	// Update the Kubelet configuration.
	ginkgo.By("Stopping the kubelet")
	restartKubelet := mustStopKubelet(ctx, f)

	expectedNAPodCgroup := cm.NewCgroupName(cm.RootCgroupName, nodeAllocatableCgroup)

	// Cleanup from the previous kubelet, to verify the new one creates it correctly
	if err := cgroupManager.Destroy(klog.Background(), &cm.CgroupConfig{
		Name: cm.NewCgroupName(expectedNAPodCgroup),
	}); err != nil {
		return err
	}
	if cgroupManager.Exists(expectedNAPodCgroup) {
		return fmt.Errorf("Expected Node Allocatable Cgroup %q not to exist", expectedNAPodCgroup)
	}

	framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(newCfg))

	ginkgo.By("Starting the kubelet")
	restartKubelet(ctx)

	if err != nil {
		return err
	}
	// Set new config and current config.
	currentConfig := newCfg

	if !cgroupManager.Exists(expectedNAPodCgroup) {
		return fmt.Errorf("Expected Node Allocatable Cgroup %q to exist", expectedNAPodCgroup)
	}

	memoryLimitFile := "memory.limit_in_bytes"
	if IsCgroup2UnifiedMode() {
		memoryLimitFile = "memory.max"
	}

	// Build the per-size hugepage reservation map: total = system + kube.
	hpReserved := hugepageReserved{
		v1.ResourceName(hugepagesResourceName2Mi): "4Mi", // 2Mi sys + 2Mi kube
	}
	if has1GiHugepages {
		hpReserved[v1.ResourceName(hugepagesResourceName1Gi)] = "1Gi" // 1Gi sys only
	}

	// Map hugepage resource names to cgroup controller names.
	hpCgroupCtrl := map[v1.ResourceName]string{
		v1.ResourceName(hugepagesResourceName2Mi): "hugetlb.2MB",
		v1.ResourceName(hugepagesResourceName1Gi): "hugetlb.1GB",
	}

	// TODO: Update cgroupManager to expose a Status interface to get current Cgroup Settings.
	// The node may not have updated capacity and allocatable yet, so check that it happens eventually.
	var capacity v1.ResourceList
	gomega.Eventually(ctx, func(ctx context.Context) error {
		nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		if err != nil {
			return err
		}
		if len(nodeList.Items) != 1 {
			return fmt.Errorf("Unexpected number of node objects for node e2e. Expects only one node: %+v", nodeList)
		}
		cgroupName := nodeAllocatableCgroup
		if currentConfig.CgroupDriver == "systemd" {
			cgroupName = nodeAllocatableCgroup + ".slice"
		}

		node := nodeList.Items[0]
		capacity = node.Status.Capacity
		alloc := getAllocatableLimits("200m", "200Mi", "1738", hpReserved, capacity)
		// Total Memory reservation is 200Mi excluding eviction thresholds.
		// Expect CPU shares on node allocatable cgroup to equal allocatable.
		shares := int64(cm.MilliCPUToShares(alloc.cpu.MilliValue()))
		if IsCgroup2UnifiedMode() {
			// convert to the cgroup v2 cpu.weight value
			if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["cpu"], cgroupName, "cpu.weight"), convertSharesToWeight(shares), 10); err != nil {
				return err
			}
		} else {
			if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["cpu"], cgroupName, "cpu.shares"), shares, 10); err != nil {
				return err
			}
		}
		// Expect Memory limit on node allocatable cgroup to equal allocatable.
		if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["memory"], cgroupName, memoryLimitFile), alloc.memory.Value(), 0); err != nil {
			return err
		}
		// Expect PID limit on node allocatable cgroup to equal allocatable.
		if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["pids"], cgroupName, "pids.max"), alloc.pids.Value(), 0); err != nil {
			return err
		}
		// Expect hugepage limits on kubepods cgroup to equal allocatable per size.
		for hpName, hpAlloc := range alloc.hugepages {
			ctrl := hpCgroupCtrl[hpName]
			if err := expectFileValToEqual(hugetlbLimitFile(subsystems, cgroupName, ctrl), hpAlloc.Value(), 0); err != nil {
				return fmt.Errorf("kubepods %s check: %w", ctrl, err)
			}
		}

		// Check that Allocatable reported to scheduler includes eviction thresholds.
		schedulerAllocatable := node.Status.Allocatable
		// Memory allocatable should take into account eviction thresholds.
		// Process IDs are not a scheduler resource and as such cannot be tested here.
		allocSched := getAllocatableLimits("200m", "300Mi", "1738", hpReserved, capacity)
		// Hugepages are pre-allocated physical memory unavailable for regular use.
		// The kubelet subtracts full hugepage capacity from memory allocatable.
		for hpName := range hpReserved {
			if hpCap, ok := capacity[hpName]; ok {
				allocSched.memory.Sub(hpCap)
			}
		}
		// Expect allocatable to include all resources in capacity.
		if len(schedulerAllocatable) != len(capacity) {
			return fmt.Errorf("Expected all resources in capacity to be found in allocatable")
		}
		// CPU based evictions are not supported.
		if allocSched.cpu.Cmp(schedulerAllocatable[v1.ResourceCPU]) != 0 {
			return fmt.Errorf("Unexpected cpu allocatable value exposed by the node. Expected: %v, got: %v, capacity: %v", allocSched.cpu, schedulerAllocatable[v1.ResourceCPU], capacity[v1.ResourceCPU])
		}
		if allocSched.memory.Cmp(schedulerAllocatable[v1.ResourceMemory]) != 0 {
			return fmt.Errorf("Unexpected memory allocatable value exposed by the node. Expected: %v, got: %v, capacity: %v", allocSched.memory, schedulerAllocatable[v1.ResourceMemory], capacity[v1.ResourceMemory])
		}

		// Expect hugepage scheduler allocatable to match per size.
		for hpName, hpAlloc := range allocSched.hugepages {
			allocHP, ok := schedulerAllocatable[hpName]
			if !ok {
				return fmt.Errorf("%s not found in node allocatable", hpName)
			}
			if allocHP.Cmp(*hpAlloc) != 0 {
				return fmt.Errorf("expected allocatable %s %s, got %s", hpName, hpAlloc.String(), allocHP.String())
			}
		}

		return nil
	}, time.Minute, 5*time.Second).Should(gomega.Succeed())

	cgroupPath := ""
	if currentConfig.CgroupDriver == "systemd" {
		cgroupPath = cm.NewCgroupName(cm.RootCgroupName, kubeReservedCgroup).ToSystemd()
	} else {
		cgroupPath = cgroupManager.Name(cm.NewCgroupName(cm.RootCgroupName, kubeReservedCgroup))
	}
	// Expect CPU shares on kube reserved cgroup to equal it's reservation which is `100m`.
	kubeReservedCPU := resource.MustParse(currentConfig.KubeReserved[string(v1.ResourceCPU)])
	shares := int64(cm.MilliCPUToShares(kubeReservedCPU.MilliValue()))
	if IsCgroup2UnifiedMode() {
		if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["cpu"], cgroupPath, "cpu.weight"), convertSharesToWeight(shares), 10); err != nil {
			return err
		}
	} else {
		if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["cpu"], cgroupPath, "cpu.shares"), shares, 10); err != nil {
			return err
		}
	}
	// Expect Memory limit kube reserved cgroup to equal configured value `100Mi`.
	kubeReservedMemory := resource.MustParse(currentConfig.KubeReserved[string(v1.ResourceMemory)])
	if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["memory"], cgroupPath, memoryLimitFile), kubeReservedMemory.Value(), 0); err != nil {
		return err
	}
	// Expect process ID limit kube reserved cgroup to equal configured value `738`.
	kubeReservedPIDs := resource.MustParse(currentConfig.KubeReserved[string(pidlimit.PIDs)])
	if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["pids"], cgroupPath, "pids.max"), kubeReservedPIDs.Value(), 0); err != nil {
		return err
	}
	// Expect hugepage limits on kube-reserved cgroup to match configured values.
	for hpName, ctrl := range hpCgroupCtrl {
		if valStr, ok := currentConfig.KubeReserved[string(hpName)]; ok {
			kubeReservedHP := resource.MustParse(valStr)
			if err := expectFileValToEqual(hugetlbLimitFile(subsystems, cgroupPath, ctrl), kubeReservedHP.Value(), 0); err != nil {
				return fmt.Errorf("kube-reserved %s check: %w", ctrl, err)
			}
		}
	}

	if currentConfig.CgroupDriver == "systemd" {
		cgroupPath = cm.NewCgroupName(cm.RootCgroupName, systemReservedCgroup).ToSystemd()
	} else {
		cgroupPath = cgroupManager.Name(cm.NewCgroupName(cm.RootCgroupName, systemReservedCgroup))
	}

	// Expect CPU shares on system reserved cgroup to equal it's reservation which is `100m`.
	systemReservedCPU := resource.MustParse(currentConfig.SystemReserved[string(v1.ResourceCPU)])
	shares = int64(cm.MilliCPUToShares(systemReservedCPU.MilliValue()))
	if IsCgroup2UnifiedMode() {
		if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["cpu"], cgroupPath, "cpu.weight"), convertSharesToWeight(shares), 10); err != nil {
			return err
		}
	} else {
		if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["cpu"], cgroupPath, "cpu.shares"), shares, 10); err != nil {
			return err
		}
	}
	// Expect Memory limit on node allocatable cgroup to equal allocatable.
	systemReservedMemory := resource.MustParse(currentConfig.SystemReserved[string(v1.ResourceMemory)])
	if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["memory"], cgroupPath, memoryLimitFile), systemReservedMemory.Value(), 0); err != nil {
		return err
	}
	// Expect process ID limit system reserved cgroup to equal configured value `1000`.
	systemReservedPIDs := resource.MustParse(currentConfig.SystemReserved[string(pidlimit.PIDs)])
	if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["pids"], cgroupPath, "pids.max"), systemReservedPIDs.Value(), 0); err != nil {
		return err
	}
	// Expect hugepage limits on system-reserved cgroup to match configured values.
	for hpName, ctrl := range hpCgroupCtrl {
		if valStr, ok := currentConfig.SystemReserved[string(hpName)]; ok {
			systemReservedHP := resource.MustParse(valStr)
			if err := expectFileValToEqual(hugetlbLimitFile(subsystems, cgroupPath, ctrl), systemReservedHP.Value(), 0); err != nil {
				return fmt.Errorf("system-reserved %s check: %w", ctrl, err)
			}
		}
	}

	// Verify the QoS cgroup reconciliation cycle (runs every 1 minute) does
	// not overwrite the kubepods hugepage limits back to unbounded.
	cgroupName := nodeAllocatableCgroup
	if currentConfig.CgroupDriver == "systemd" {
		cgroupName = nodeAllocatableCgroup + ".slice"
	}
	alloc := getAllocatableLimits("200m", "200Mi", "1738", hpReserved, capacity)

	ginkgo.By("Waiting for QoS cgroup reconciliation cycle")
	gomega.Consistently(ctx, func() error {
		for hpName, hpAlloc := range alloc.hugepages {
			ctrl := hpCgroupCtrl[hpName]
			if err := expectFileValToEqual(hugetlbLimitFile(subsystems, cgroupName, ctrl), hpAlloc.Value(), 0); err != nil {
				return fmt.Errorf("QoS reconciliation %s check: %w", ctrl, err)
			}
		}
		return nil
	}).WithTimeout(90 * time.Second).WithPolling(10 * time.Second).Should(gomega.Succeed())

	return nil
}

// hugetlbLimitFile returns the path to the hugetlb limit file for the given
// cgroup and hugepage cgroup controller name (e.g. "hugetlb.2MB").
func hugetlbLimitFile(subsystems *cm.CgroupSubsystems, cgroupPath, hpCgroupCtrl string) string {
	if IsCgroup2UnifiedMode() {
		return filepath.Join(subsystems.MountPoints["hugetlb"], cgroupPath, hpCgroupCtrl+".max")
	}
	return filepath.Join(subsystems.MountPoints["hugetlb"], cgroupPath, hpCgroupCtrl+".limit_in_bytes")
}

// runHugepagesUpgradeTest verifies the upgrade behavior when hugepage
// reservation is introduced on a node that was previously running without it.
// Specifically: restart kubelet with --system-reserved including hugepages,
// verify the Memory Manager rejects the checkpoint mismatch, delete the
// checkpoint, restart again, and verify kubelet starts with reduced allocatable.
func runHugepagesUpgradeTest(ctx context.Context, f *framework.Framework) error {
	subsystems, err := cm.GetCgroupSubsystems()
	if err != nil {
		return err
	}

	oldCfg, err := getCurrentKubeletConfig(ctx)
	if err != nil {
		return err
	}

	ginkgo.DeferCleanup(func(ctx context.Context) {
		if oldCfg != nil {
			ginkgo.By("Restoring the kubelet configuration")
			// Delete the memory manager state file to avoid checkpoint mismatch on restore.
			deleteStateFile(memoryManagerStateFile)

			restartKubelet := mustStopKubelet(ctx, f)
			gomega.Eventually(ctx, func() bool {
				return e2enode.HealthCheck(kubeletHealthCheckURL)
			}, time.Minute, time.Second).Should(gomega.BeFalseBecause("expected kubelet health check to fail"))
			framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(oldCfg))
			restartKubelet(ctx)
		}
	})

	cgroupManager := cm.NewCgroupManager(klog.Background(), subsystems, oldCfg.CgroupDriver)
	if err := createTemporaryCgroupsForReservation(cgroupManager); err != nil {
		return err
	}
	ginkgo.DeferCleanup(destroyTemporaryCgroupsForReservation, cgroupManager)

	hugepages := map[string]int{hugepagesResourceName2Mi: 4}
	// test will skip if hugepages are not available on the machine
	setHugepages(ctx, hugepages)
	ginkgo.DeferCleanup(releaseHugepages, hugepages)

	// Step 1: Start kubelet with Memory Manager Static and --reserved-memory
	// but WITHOUT hugepages reservation. This creates a valid checkpoint.
	ginkgo.By("Starting kubelet with Memory Manager Static, no hugepage reservation")

	step1Cfg := oldCfg.DeepCopy()
	setDesiredConfiguration(step1Cfg, cgroupManager, nil, nil)
	step1Cfg.MemoryManagerPolicy = "Static"
	// reserved-memory must match system-reserved + kube-reserved for memory
	// (200Mi total) + eviction threshold (100Mi) = 300Mi. No hugepages.
	step1Cfg.ReservedMemory = []kubeletconfig.MemoryReservation{
		{
			NumaNode: 0,
			Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("300Mi"),
			},
		},
	}

	restartKubelet := mustStopKubelet(ctx, f)
	// Delete existing checkpoint to start clean.
	deleteStateFile(memoryManagerStateFile)
	framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(step1Cfg))
	ginkgo.By("Starting kubelet (step 1: no hugepage reservation)")
	restartKubelet(ctx)
	waitForKubeletToStart(ctx, f)

	// Step 2: Restart kubelet adding hugepages to system-reserved AND
	// reserved-memory. The Memory Manager checkpoint from step 1 has a
	// different allocatable, so the kubelet should fail to start.
	ginkgo.By("Restarting kubelet with hugepage reservation (step 2: expect failure)")

	step2Cfg := step1Cfg.DeepCopy()
	step2Cfg.SystemReserved["hugepages-2Mi"] = "2Mi"
	step2Cfg.ReservedMemory = []kubeletconfig.MemoryReservation{
		{
			NumaNode: 0,
			Limits: v1.ResourceList{
				v1.ResourceMemory:                resource.MustParse("300Mi"),
				v1.ResourceName("hugepages-2Mi"): resource.MustParse("2Mi"),
			},
		},
	}

	kubeletServiceName := findKubeletServiceName(true)
	ginkgo.By("Stopping the kubelet (step 2)")
	restartKubelet = mustStopKubelet(ctx, f)

	framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(step2Cfg))

	// Restart kubelet - it should fail because the Memory Manager checkpoint
	// has a different allocatable than the new config expects.
	// Use raw systemctl because mustStopKubelet's restart closure asserts
	// success, which would fail the test.
	exec.CommandContext(ctx, "sudo", "systemctl", "restart", kubeletServiceName).CombinedOutput() //nolint:errcheck // intentional: we expect this restart to fail

	// Kubelet should NOT become healthy because of the checkpoint mismatch.
	gomega.Consistently(ctx, func() bool {
		return e2enode.HealthCheck(kubeletHealthCheckURL)
	}).WithTimeout(30 * time.Second).WithPolling(5 * time.Second).Should(gomega.BeFalseBecause(
		"kubelet should fail to start due to memory manager checkpoint mismatch"))

	// Step 3: Delete the checkpoint file and restart. Kubelet should start
	// successfully with reduced allocatable.
	ginkgo.By("Deleting memory manager checkpoint and restarting (step 3)")
	deleteStateFile(memoryManagerStateFile)

	restartKubelet(ctx)
	waitForKubeletToStart(ctx, f)

	// Verify allocatable is reduced by the reserved amount.
	// 2Mi: 4 pages × 2Mi = 8Mi capacity, step2 reserves 2Mi → allocatable = 6Mi.
	hp2MiCapBytes := int64(hugepages[hugepagesResourceName2Mi]) * 2 * 1024 * 1024
	expected2Mi := resource.NewQuantity(hp2MiCapBytes-2*1024*1024, resource.BinarySI)

	gomega.Eventually(ctx, func(ctx context.Context) error {
		nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		if err != nil {
			return err
		}
		if len(nodeList.Items) != 1 {
			return fmt.Errorf("expected 1 node, got %d", len(nodeList.Items))
		}
		allocHP2Mi, ok := nodeList.Items[0].Status.Allocatable[v1.ResourceName("hugepages-2Mi")]
		if !ok {
			return fmt.Errorf("hugepages-2Mi not found in node allocatable")
		}
		if allocHP2Mi.Cmp(*expected2Mi) != 0 {
			return fmt.Errorf("expected allocatable hugepages-2Mi %s, got %s", expected2Mi.String(), allocHP2Mi.String())
		}
		return nil
	}, time.Minute, 5*time.Second).Should(gomega.Succeed())

	return nil
}
