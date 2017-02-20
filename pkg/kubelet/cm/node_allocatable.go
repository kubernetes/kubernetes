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

package cm

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api/v1"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
)

const (
	defaultNodeAllocatableCgroupName = "/kubepods"
	nodeAllocatableEnforcementKey    = "pods"
	systemReservedEnforcementKey     = "system-reserved"
	kubeReservedEnforcementKey       = "kube-reserved"
)

func createNodeAllocatableCgroups(nc NodeAllocatableConfig, nodeAllocatable v1.ResourceList, cgroupManager CgroupManager) error {
	cgroupConfig := &CgroupConfig{
		Name: CgroupName(defaultNodeAllocatableCgroupName),
	}
	if err := cgroupManager.Create(cgroupConfig); err != nil {
		glog.Errorf("Failed to create %q cgroup and apply limits")
		return err
	}
	return nil
}

// Enforce Node Allocatable Cgroup settings.
func enforceNodeAllocatableCgroups(nc NodeAllocatableConfig, nodeAllocatable v1.ResourceList, cgroupManager CgroupManager) error {
	glog.V(4).Infof("Attempting to enforce Node Allocatable with config: %+v", nc)
	glog.V(4).Infof("Node Allocatable resources: %+v", nodeAllocatable)
	// Create top level cgroups for all pods if necessary.
	if nc.EnforceNodeAllocatable.Has(nodeAllocatableEnforcementKey) {
		cgroupConfig := &CgroupConfig{
			Name:               CgroupName(defaultNodeAllocatableCgroupName),
			ResourceParameters: getCgroupConfig(nodeAllocatable),
		}
		glog.V(4).Infof("Updating Node Allocatable cgroup with %d cpu shares and %d bytes of memory", cgroupConfig.ResourceParameters.CpuShares, cgroupConfig.ResourceParameters.Memory)
		if err := cgroupManager.Update(cgroupConfig); err != nil {
			glog.Errorf("Failed to create %q cgroup and apply limits")
			return err
		}
	}
	// Now apply kube reserved and system reserved limits if required.
	if nc.EnforceNodeAllocatable.Has(systemReservedEnforcementKey) {
		glog.V(2).Infof("Enforcing system reserved on cgroup %q with limits: %+v", nc.SystemReservedCgroupName, nc.SystemReserved)
		if err := enforceExistingCgroup(cgroupManager, nc.SystemReservedCgroupName, nc.SystemReserved); err != nil {
			return fmt.Errorf("failed to enforce System Reserved Cgroup Limits: %v", err)
		}
	}
	if nc.EnforceNodeAllocatable.Has(kubeReservedEnforcementKey) {
		glog.V(2).Infof("Enforcing kube reserved on cgroup %q with limits: %+v", nc.KubeReservedCgroupName, nc.KubeReserved)
		if err := enforceExistingCgroup(cgroupManager, nc.KubeReservedCgroupName, nc.KubeReserved); err != nil {
			return fmt.Errorf("failed to enforce Kube Reserved Cgroup Limits: %v", err)
		}
	}
	return nil
}

func enforceExistingCgroup(cgroupManager CgroupManager, cName string, rl v1.ResourceList) error {
	cgroupConfig := &CgroupConfig{
		Name:               CgroupName(cName),
		ResourceParameters: getCgroupConfig(rl),
	}
	glog.V(4).Infof("Enforcing limits on cgroup %q with %d cpu shares and %d bytes of memory", cName, cgroupConfig.ResourceParameters.CpuShares, cgroupConfig.ResourceParameters.Memory)
	if !cgroupManager.Exists(cgroupConfig.Name) {
		return fmt.Errorf("%q cgroup does not exist", cgroupConfig.Name)
	}
	if err := cgroupManager.Update(cgroupConfig); err != nil {
		return err
	}
	return nil
}

func getCgroupConfig(rl v1.ResourceList) *ResourceConfig {
	// TODO(vishh): Set CPU Quota if necessary.
	if rl == nil {
		return nil
	}
	var rc ResourceConfig
	if q, exists := rl[v1.ResourceMemory]; exists {
		// Memory is defined in bytes.
		val := q.Value()
		rc.Memory = &val
	}
	if q, exists := rl[v1.ResourceCPU]; exists {
		// CPU is defined in milli-cores.
		val := MilliCPUToShares(q.MilliValue())
		rc.CpuShares = &val
	}
	return &rc
}

func (cm *containerManagerImpl) getNodeAllocatableInternal(includeHardEviction bool) v1.ResourceList {
	var evictionReservation v1.ResourceList
	if includeHardEviction {
		evictionReservation = hardEvictionReservation(cm.HardEvictionThresholds, cm.capacity)
	}
	result := make(v1.ResourceList)
	for k, v := range cm.capacity {
		value := *(v.Copy())
		if cm.NodeConfig.SystemReserved != nil {
			value.Sub(cm.NodeConfig.SystemReserved[k])
		}
		if cm.NodeConfig.KubeReserved != nil {
			value.Sub(cm.NodeConfig.KubeReserved[k])
		}
		if evictionReservation != nil {
			value.Sub(evictionReservation[k])
		}
		if value.Sign() < 0 {
			// Negative Allocatable resources don't make sense.
			value.Set(0)
		}
		result[k] = value
	}
	return result

}

// GetNodeAllocatable returns amount of compute resource available for pods.
func (cm *containerManagerImpl) GetNodeAllocatable() v1.ResourceList {
	return cm.getNodeAllocatableInternal(!cm.NodeConfig.IgnoreHardEvictionThreshold)
}

// hardEvictionReservation returns a resourcelist that includes reservation of resources based on hard eviction thresholds.
func hardEvictionReservation(thresholds []evictionapi.Threshold, capacity v1.ResourceList) v1.ResourceList {
	if len(thresholds) == 0 {
		return nil
	}
	ret := v1.ResourceList{}
	for _, threshold := range thresholds {
		if threshold.Operator != evictionapi.OpLessThan {
			continue
		}
		switch threshold.Signal {
		case evictionapi.SignalMemoryAvailable:
			memoryCapacity := capacity[v1.ResourceMemory]
			value := evictionapi.GetThresholdQuantity(threshold.Value, &memoryCapacity)
			ret[v1.ResourceMemory] = *value
		}
	}
	return ret
}
