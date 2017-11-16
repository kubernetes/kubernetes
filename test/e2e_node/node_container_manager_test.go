// +build linux

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
	"fmt"
	"io/ioutil"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

func setDesiredConfiguration(initialConfig *kubeletconfig.KubeletConfiguration) {
	initialConfig.EnforceNodeAllocatable = []string{"pods", "kube-reserved", "system-reserved"}
	initialConfig.SystemReserved = kubeletconfig.ConfigurationMap{
		string(v1.ResourceCPU):    "100m",
		string(v1.ResourceMemory): "100Mi",
	}
	initialConfig.KubeReserved = kubeletconfig.ConfigurationMap{
		string(v1.ResourceCPU):    "100m",
		string(v1.ResourceMemory): "100Mi",
	}
	initialConfig.EvictionHard = "memory.available<100Mi"
	// Necessary for allocatable cgroup creation.
	initialConfig.CgroupsPerQOS = true
	initialConfig.KubeReservedCgroup = kubeReservedCgroup
	initialConfig.SystemReservedCgroup = systemReservedCgroup
}

var _ = framework.KubeDescribe("Node Container Manager [Serial]", func() {
	f := framework.NewDefaultFramework("node-container-manager")
	Describe("Validate Node Allocatable", func() {
		It("set's up the node and runs the test", func() {
			framework.ExpectNoError(runTest(f))
		})
	})

})

func expectFileValToEqual(filePath string, expectedValue, delta int64) error {
	out, err := ioutil.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read file %q", filePath)
	}
	actual, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return fmt.Errorf("failed to parse output %v", err)
	}

	// Ensure that values are within a delta range to work arounding rounding errors.
	if (actual < (expectedValue - delta)) || (actual > (expectedValue + delta)) {
		return fmt.Errorf("Expected value at %q to be between %d and %d. Got %d", filePath, (expectedValue - delta), (expectedValue + delta), actual)
	}
	return nil
}

func getAllocatableLimits(cpu, memory string, capacity v1.ResourceList) (*resource.Quantity, *resource.Quantity) {
	var allocatableCPU, allocatableMemory *resource.Quantity
	// Total cpu reservation is 200m.
	for k, v := range capacity {
		if k == v1.ResourceCPU {
			allocatableCPU = v.Copy()
			allocatableCPU.Sub(resource.MustParse(cpu))
		}
		if k == v1.ResourceMemory {
			allocatableMemory = v.Copy()
			allocatableMemory.Sub(resource.MustParse(memory))
		}
	}
	return allocatableCPU, allocatableMemory
}

const (
	kubeReservedCgroup   = "/kube_reserved"
	systemReservedCgroup = "/system_reserved"
)

func createIfNotExists(cm cm.CgroupManager, cgroupConfig *cm.CgroupConfig) error {
	if !cm.Exists(cgroupConfig.Name) {
		if err := cm.Create(cgroupConfig); err != nil {
			return err
		}
	}
	return nil
}

func createTemporaryCgroupsForReservation(cgroupManager cm.CgroupManager) error {
	// Create kube reserved cgroup
	cgroupConfig := &cm.CgroupConfig{
		Name: cm.CgroupName(kubeReservedCgroup),
	}
	if err := createIfNotExists(cgroupManager, cgroupConfig); err != nil {
		return err
	}
	// Create system reserved cgroup
	cgroupConfig.Name = cm.CgroupName(systemReservedCgroup)

	return createIfNotExists(cgroupManager, cgroupConfig)
}

func destroyTemporaryCgroupsForReservation(cgroupManager cm.CgroupManager) error {
	// Create kube reserved cgroup
	cgroupConfig := &cm.CgroupConfig{
		Name: cm.CgroupName(kubeReservedCgroup),
	}
	if err := cgroupManager.Destroy(cgroupConfig); err != nil {
		return err
	}
	cgroupConfig.Name = cm.CgroupName(systemReservedCgroup)
	return cgroupManager.Destroy(cgroupConfig)
}

func runTest(f *framework.Framework) error {
	var oldCfg *kubeletconfig.KubeletConfiguration
	subsystems, err := cm.GetCgroupSubsystems()
	if err != nil {
		return err
	}
	// Get current kubelet configuration
	oldCfg, err = getCurrentKubeletConfig()
	if err != nil {
		return err
	}

	// Create a cgroup manager object for manipulating cgroups.
	cgroupManager := cm.NewCgroupManager(subsystems, oldCfg.CgroupDriver)

	defer destroyTemporaryCgroupsForReservation(cgroupManager)
	defer func() {
		if oldCfg != nil {
			framework.ExpectNoError(setKubeletConfiguration(f, oldCfg))
		}
	}()
	if err := createTemporaryCgroupsForReservation(cgroupManager); err != nil {
		return err
	}
	newCfg := oldCfg.DeepCopy()
	// Change existing kubelet configuration
	setDesiredConfiguration(newCfg)
	// Set the new kubelet configuration.
	err = setKubeletConfiguration(f, newCfg)
	if err != nil {
		return err
	}
	// Set new config and current config.
	currentConfig := newCfg

	expectedNAPodCgroup := path.Join(currentConfig.CgroupRoot, "kubepods")
	if !cgroupManager.Exists(cm.CgroupName(expectedNAPodCgroup)) {
		return fmt.Errorf("Expected Node Allocatable Cgroup Does not exist")
	}
	// TODO: Update cgroupManager to expose a Status interface to get current Cgroup Settings.
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{})
	if err != nil {
		return err
	}
	if len(nodeList.Items) != 1 {
		return fmt.Errorf("Unexpected number of node objects for node e2e. Expects only one node: %+v", nodeList)
	}
	node := nodeList.Items[0]
	capacity := node.Status.Capacity
	allocatableCPU, allocatableMemory := getAllocatableLimits("200m", "200Mi", capacity)
	// Total Memory reservation is 200Mi excluding eviction thresholds.
	// Expect CPU shares on node allocatable cgroup to equal allocatable.
	if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["cpu"], "kubepods", "cpu.shares"), int64(cm.MilliCPUToShares(allocatableCPU.MilliValue())), 10); err != nil {
		return err
	}
	// Expect Memory limit on node allocatable cgroup to equal allocatable.
	if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["memory"], "kubepods", "memory.limit_in_bytes"), allocatableMemory.Value(), 0); err != nil {
		return err
	}

	// Check that Allocatable reported to scheduler includes eviction thresholds.
	schedulerAllocatable := node.Status.Allocatable
	// Memory allocatable should take into account eviction thresholds.
	allocatableCPU, allocatableMemory = getAllocatableLimits("200m", "300Mi", capacity)
	// Expect allocatable to include all resources in capacity.
	if len(schedulerAllocatable) != len(capacity) {
		return fmt.Errorf("Expected all resources in capacity to be found in allocatable")
	}
	// CPU based evictions are not supported.
	if allocatableCPU.Cmp(schedulerAllocatable[v1.ResourceCPU]) != 0 {
		return fmt.Errorf("Unexpected cpu allocatable value exposed by the node. Expected: %v, got: %v, capacity: %v", allocatableCPU, schedulerAllocatable[v1.ResourceCPU], capacity[v1.ResourceCPU])
	}
	if allocatableMemory.Cmp(schedulerAllocatable[v1.ResourceMemory]) != 0 {
		return fmt.Errorf("Unexpected cpu allocatable value exposed by the node. Expected: %v, got: %v, capacity: %v", allocatableCPU, schedulerAllocatable[v1.ResourceCPU], capacity[v1.ResourceMemory])
	}

	if !cgroupManager.Exists(cm.CgroupName(kubeReservedCgroup)) {
		return fmt.Errorf("Expected kube reserved cgroup Does not exist")
	}
	// Expect CPU shares on kube reserved cgroup to equal it's reservation which is `100m`.
	kubeReservedCPU := resource.MustParse(currentConfig.KubeReserved[string(v1.ResourceCPU)])
	if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["cpu"], kubeReservedCgroup, "cpu.shares"), int64(cm.MilliCPUToShares(kubeReservedCPU.MilliValue())), 10); err != nil {
		return err
	}
	// Expect Memory limit kube reserved cgroup to equal configured value `100Mi`.
	kubeReservedMemory := resource.MustParse(currentConfig.KubeReserved[string(v1.ResourceMemory)])
	if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["memory"], kubeReservedCgroup, "memory.limit_in_bytes"), kubeReservedMemory.Value(), 0); err != nil {
		return err
	}
	if !cgroupManager.Exists(cm.CgroupName(systemReservedCgroup)) {
		return fmt.Errorf("Expected system reserved cgroup Does not exist")
	}
	// Expect CPU shares on system reserved cgroup to equal it's reservation which is `100m`.
	systemReservedCPU := resource.MustParse(currentConfig.SystemReserved[string(v1.ResourceCPU)])
	if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["cpu"], systemReservedCgroup, "cpu.shares"), int64(cm.MilliCPUToShares(systemReservedCPU.MilliValue())), 10); err != nil {
		return err
	}
	// Expect Memory limit on node allocatable cgroup to equal allocatable.
	systemReservedMemory := resource.MustParse(currentConfig.SystemReserved[string(v1.ResourceMemory)])
	if err := expectFileValToEqual(filepath.Join(subsystems.MountPoints["memory"], systemReservedCgroup, "memory.limit_in_bytes"), systemReservedMemory.Value(), 0); err != nil {
		return err
	}
	return nil
}
