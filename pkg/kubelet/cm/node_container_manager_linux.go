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

package cm

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	libcontainercgroups "github.com/opencontainers/cgroups"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/stats/pidlimit"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	defaultNodeAllocatableCgroupName = "kubepods"
)

// createNodeAllocatableCgroups creates Node Allocatable Cgroup when CgroupsPerQOS flag is specified as true
func (cm *containerManagerImpl) createNodeAllocatableCgroups(logger klog.Logger) error {
	nodeAllocatable := cm.internalCapacity
	// Use Node Allocatable limits instead of capacity if the user requested enforcing node allocatable.
	nc := cm.NodeConfig.NodeAllocatableConfig
	if cm.CgroupsPerQOS && nc.EnforceNodeAllocatable.Has(kubetypes.NodeAllocatableEnforcementKey) {
		nodeAllocatable = cm.getNodeAllocatableInternalAbsolute()
	}

	cgroupConfig := &CgroupConfig{
		Name: cm.cgroupRoot,
		// The default limits for cpu shares can be very low which can lead to CPU starvation for pods.
		ResourceParameters: cm.getCgroupConfig(nodeAllocatable, false),
	}
	if cm.cgroupManager.Exists(cgroupConfig.Name) {
		return nil
	}
	if err := cm.cgroupManager.Create(logger, cgroupConfig); err != nil {
		logger.Error(err, "Failed to create cgroup", "cgroupName", cm.cgroupRoot)
		return err
	}
	return nil
}

// enforceNodeAllocatableCgroups enforce Node Allocatable Cgroup settings.
func (cm *containerManagerImpl) enforceNodeAllocatableCgroups(logger klog.Logger) error {
	nc := cm.NodeConfig.NodeAllocatableConfig

	// We need to update limits on node allocatable cgroup no matter what because
	// default cpu shares on cgroups are low and can cause cpu starvation.
	nodeAllocatable := cm.internalCapacity
	// Use Node Allocatable limits instead of capacity if the user requested enforcing node allocatable.
	if cm.CgroupsPerQOS && nc.EnforceNodeAllocatable.Has(kubetypes.NodeAllocatableEnforcementKey) {
		nodeAllocatable = cm.getNodeAllocatableInternalAbsolute()
	}

	logger.V(4).Info("Attempting to enforce Node Allocatable", "config", nc)

	cgroupConfig := &CgroupConfig{
		Name:               cm.cgroupRoot,
		ResourceParameters: cm.getCgroupConfig(nodeAllocatable, false),
	}

	// Stale memory.min/memory.low from a previously enabled MemoryQoS state can
	// persist across kubelet restarts. Reset to 0 so rollback takes effect.
	if !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.MemoryQoS) && libcontainercgroups.IsCgroup2UnifiedMode() {
		rp := cgroupConfig.ResourceParameters
		if rp.Unified == nil {
			rp.Unified = make(map[string]string)
		}
		rp.Unified[Cgroup2MemoryMin] = "0"
		rp.Unified[Cgroup2MemoryLow] = "0"
	}

	// Using ObjectReference for events as the node maybe not cached; refer to #42701 for detail.
	nodeRef := nodeRefFromNode(cm.nodeInfo.Name)

	// If Node Allocatable is enforced on a node that has not been drained or is updated on an existing node to a lower value,
	// existing memory usage across pods might be higher than current Node Allocatable Memory Limits.
	// Pod Evictions are expected to bring down memory usage to below Node Allocatable limits.
	// Until evictions happen retry cgroup updates.
	// Update limits on non root cgroup-root to be safe since the default limits for CPU can be too low.
	// Check if cgroupRoot is set to a non-empty value (empty would be the root container)
	if len(cm.cgroupRoot) > 0 {
		go func() {
			for {
				err := cm.updateNodeAllocatableCgroup(logger, cgroupConfig)
				if err == nil {
					cm.recorder.Event(nodeRef, v1.EventTypeNormal, events.SuccessfulNodeAllocatableEnforcement, "Updated Node Allocatable limit across pods")
					return
				}
				message := fmt.Sprintf("Failed to update Node Allocatable Limits %q: %v", cm.cgroupRoot, err)
				cm.recorder.Event(nodeRef, v1.EventTypeWarning, events.FailedNodeAllocatableEnforcement, message)
				time.Sleep(time.Minute)
			}
		}()
	}
	// Now apply kube reserved and system reserved limits if required.
	if nc.EnforceNodeAllocatable.Has(kubetypes.SystemReservedEnforcementKey) {
		logger.V(2).Info("Enforcing system reserved on cgroup", "cgroupName", nc.SystemReservedCgroupName, "limits", nc.SystemReserved)
		if err := cm.enforceExistingCgroup(logger, nc.SystemReservedCgroupName, nc.SystemReserved, false); err != nil {
			message := fmt.Sprintf("Failed to enforce System Reserved Cgroup Limits on %q: %v", nc.SystemReservedCgroupName, err)
			cm.recorder.Event(nodeRef, v1.EventTypeWarning, events.FailedNodeAllocatableEnforcement, message)
			return errors.New(message)
		}
		cm.recorder.Eventf(nodeRef, v1.EventTypeNormal, events.SuccessfulNodeAllocatableEnforcement, "Updated limits on system reserved cgroup %v", nc.SystemReservedCgroupName)
	}
	if nc.EnforceNodeAllocatable.Has(kubetypes.KubeReservedEnforcementKey) {
		logger.V(2).Info("Enforcing kube reserved on cgroup", "cgroupName", nc.KubeReservedCgroupName, "limits", nc.KubeReserved)
		if err := cm.enforceExistingCgroup(logger, nc.KubeReservedCgroupName, nc.KubeReserved, false); err != nil {
			message := fmt.Sprintf("Failed to enforce Kube Reserved Cgroup Limits on %q: %v", nc.KubeReservedCgroupName, err)
			cm.recorder.Event(nodeRef, v1.EventTypeWarning, events.FailedNodeAllocatableEnforcement, message)
			return errors.New(message)
		}
		cm.recorder.Eventf(nodeRef, v1.EventTypeNormal, events.SuccessfulNodeAllocatableEnforcement, "Updated limits on kube reserved cgroup %v", nc.KubeReservedCgroupName)
	}

	if nc.EnforceNodeAllocatable.Has(kubetypes.SystemReservedCompressibleEnforcementKey) {
		logger.V(2).Info("Enforcing system reserved compressible on cgroup", "cgroupName", nc.SystemReservedCgroupName, "limits", nc.SystemReserved)
		if err := cm.enforceExistingCgroup(logger, nc.SystemReservedCgroupName, nc.SystemReserved, true); err != nil {
			message := fmt.Sprintf("Failed to enforce System Reserved Compressible Cgroup Limits on %q: %v", nc.SystemReservedCgroupName, err)
			cm.recorder.Event(nodeRef, v1.EventTypeWarning, events.FailedNodeAllocatableEnforcement, message)
			return errors.New(message)
		}
		cm.recorder.Eventf(nodeRef, v1.EventTypeNormal, events.SuccessfulNodeAllocatableEnforcement, "Updated limits on system reserved cgroup %v", nc.SystemReservedCgroupName)
	}

	if nc.EnforceNodeAllocatable.Has(kubetypes.KubeReservedCompressibleEnforcementKey) {
		logger.V(2).Info("Enforcing kube reserved compressible on cgroup", "cgroupName", nc.KubeReservedCgroupName, "limits", nc.KubeReserved)
		if err := cm.enforceExistingCgroup(logger, nc.KubeReservedCgroupName, nc.KubeReserved, true); err != nil {
			message := fmt.Sprintf("Failed to enforce Kube Reserved Compressible Cgroup Limits on %q: %v", nc.KubeReservedCgroupName, err)
			cm.recorder.Event(nodeRef, v1.EventTypeWarning, events.FailedNodeAllocatableEnforcement, message)
			return errors.New(message)
		}
		cm.recorder.Eventf(nodeRef, v1.EventTypeNormal, events.SuccessfulNodeAllocatableEnforcement, "Updated limits on kube reserved cgroup %v", nc.KubeReservedCgroupName)
	}
	return nil
}

// updateNodeAllocatableCgroup avoids lowering a cgroup v2 memory limit below
// current usage. cgroup v2 may reclaim memory and invoke the memcg OOM killer
// when memory.max is lowered below usage instead of rejecting the update.
//
// When the requested memory limit cannot be applied yet, the remaining resource
// settings (CPU shares, pids limit, cpuset, unified settings) are still applied
// on their own, and the memory limit is clamped to the current usage so that the
// pod cgroup starts exercising pressure on the workload. Clamping keeps the
// eviction signal moving: available memory is derived from the installed limit,
// so leaving an older, looser limit in place would let the retry loop run forever
// without ever creating the pressure it waits for.
//
// An error is still returned while the target limit has not been reached, so the
// caller keeps retrying until usage drops below the requested value.
//
// The check is best-effort because usage may change before the cgroup update.
func (cm *containerManagerImpl) updateNodeAllocatableCgroup(logger klog.Logger, config *CgroupConfig) error {
	if cm.cgroupManager.Version() != 2 || config.ResourceParameters == nil {
		return cm.cgroupManager.Update(logger, config)
	}

	memory := config.ResourceParameters.Memory
	// Both cgroup v2 backends treat zero as "unset": the cgroupfs conversion
	// writes an empty value and systemd only adds MemoryMax for a nonzero value.
	// Negative values are already rejected by validateNodeAllocatable. Neither
	// reaches memory.max, so there is nothing to guard against.
	if memory == nil || *memory <= 0 {
		return cm.cgroupManager.Update(logger, config)
	}

	usage, err := cm.cgroupManager.MemoryUsage(config.Name)
	if err != nil {
		// Apply everything except memory so an unrelated read failure does not
		// hold back independent settings such as the CPU share correction.
		if applyErr := cm.cgroupManager.Update(logger, configWithoutMemory(config)); applyErr != nil {
			logger.Error(applyErr, "Failed to update Node Allocatable cgroup without memory", "cgroupName", config.Name)
		}
		return fmt.Errorf("failed to read current Node Allocatable memory usage for cgroup %q: %w", config.Name, err)
	}

	if usage <= *memory {
		return cm.cgroupManager.Update(logger, config)
	}

	// Usage is already above the target. Lowering memory.max to the target now
	// would reclaim and potentially OOM the node, so clamp to the current usage
	// instead: the write is safe, it applies pressure to the pod cgroup, and
	// eviction can then bring usage down towards the target on a later retry.
	clamped := configWithoutMemory(config)
	clamped.ResourceParameters.Memory = &usage

	if err := cm.cgroupManager.Update(logger, clamped); err != nil {
		return fmt.Errorf("failed to clamp Node Allocatable memory limit for cgroup %q to current usage %d: %w", config.Name, usage, err)
	}
	return fmt.Errorf("current memory usage %d exceeds requested limit %d, clamped to current usage", usage, *memory)
}

// configWithoutMemory returns a copy of config whose resource parameters carry no
// memory limit, leaving memory.max untouched. The caller's configuration is never
// mutated, so the retry loop keeps retrying the original target.
func configWithoutMemory(config *CgroupConfig) *CgroupConfig {
	params := *config.ResourceParameters
	params.Memory = nil

	copied := *config
	copied.ResourceParameters = &params
	return &copied
}

// enforceExistingCgroup updates the limits `rl` on existing cgroup `cName` using `cgroupManager` interface.
func (cm *containerManagerImpl) enforceExistingCgroup(logger klog.Logger, cNameStr string, rl v1.ResourceList, compressibleResources bool) error {
	cName := cm.cgroupManager.CgroupName(cNameStr)
	rp := cm.getCgroupConfig(rl, compressibleResources)
	if rp == nil {
		return fmt.Errorf("%q cgroup is not configured properly", cName)
	}

	// Enforce MemoryQoS for cgroups of kube-reserved/system-reserved. For more information,
	// see https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/2570-memory-qos
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.MemoryQoS) {
		if rp.Memory != nil {
			if rp.Unified == nil {
				rp.Unified = make(map[string]string)
			}
			rp.Unified[Cgroup2MemoryMin] = strconv.FormatInt(*rp.Memory, 10)
		}
	}

	cgroupConfig := &CgroupConfig{
		Name:               cName,
		ResourceParameters: rp,
	}
	logger.V(4).Info("Enforcing limits on cgroup", "cgroupName", cName, "cpuShares", cgroupConfig.ResourceParameters.CPUShares, "memory", cgroupConfig.ResourceParameters.Memory, "pidsLimit", cgroupConfig.ResourceParameters.PidsLimit)
	if err := cm.cgroupManager.Validate(cgroupConfig.Name); err != nil {
		return err
	}
	if err := cm.cgroupManager.Update(logger, cgroupConfig); err != nil {
		return err
	}
	return nil
}

// getCgroupConfig returns a ResourceConfig object that can be used to create or update cgroups via CgroupManager interface.
func (cm *containerManagerImpl) getCgroupConfig(rl v1.ResourceList, compressibleResourcesOnly bool) *ResourceConfig {
	rc := getCgroupConfigInternal(rl, compressibleResourcesOnly)
	if rc == nil {
		return nil
	}

	// In the case of a None policy, cgroupv2 and systemd cgroup manager, we must make sure systemd is aware of the cpuset cgroup.
	// By default, systemd will not create it, as we've not chosen to delegate it, and we haven't included it in the Apply() request.
	// However, this causes a bug where kubelet restarts unnecessarily (cpuset cgroup is created in the cgroupfs, but systemd
	// doesn't know about it and deletes it, and then kubelet doesn't continue because the cgroup isn't configured as expected).
	// An alternative is to delegate the `cpuset` cgroup to the kubelet, but that would require some plumbing in libcontainer,
	// and this is sufficient.
	// Only do so when the configured CPU manager does not support exclusive assignment;
	// a policy that supports exclusive assignment manages cpusets itself.
	if !cm.cpuManager.CanAllocateExclusively(v1.ResourceCPU) {
		rc.CPUSet = cm.cpuManager.GetAllCPUs()
	}

	return rc
}

// getCgroupConfigInternal are the pieces of getCgroupConfig that don't require the cm object.
// This is added to unit test without needing to create a full containerManager
func getCgroupConfigInternal(rl v1.ResourceList, compressibleResourcesOnly bool) *ResourceConfig {
	// TODO(vishh): Set CPU Quota if necessary.
	if rl == nil {
		return nil
	}
	var rc ResourceConfig

	setCompressibleResources := func() {
		if q, exists := rl[v1.ResourceCPU]; exists {
			// CPU is defined in milli-cores.
			val := MilliCPUToShares(q.MilliValue())
			rc.CPUShares = &val
		}
	}

	// Only return compressible resources
	if compressibleResourcesOnly {
		setCompressibleResources()
	} else {
		if q, exists := rl[v1.ResourceMemory]; exists {
			// Memory is defined in bytes.
			val := q.Value()
			rc.Memory = &val
		}

		setCompressibleResources()

		if q, exists := rl[pidlimit.PIDs]; exists {
			val := q.Value()
			rc.PidsLimit = &val
		}
		rc.HugePageLimit = HugePageLimits(rl)
	}
	return &rc
}

// GetNodeAllocatableAbsolute returns the absolute value of Node Allocatable which is primarily useful for enforcement.
// Note that not all resources that are available on the node are included in the returned list of resources.
// Returns a ResourceList.
func (cm *containerManagerImpl) GetNodeAllocatableAbsolute() v1.ResourceList {
	cm.RLock()
	defer cm.RUnlock()
	return cm.getNodeAllocatableAbsoluteImpl(cm.capacity)
}

func (cm *containerManagerImpl) getNodeAllocatableAbsoluteImpl(capacity v1.ResourceList) v1.ResourceList {
	result := make(v1.ResourceList)
	for k, v := range capacity {
		value := v.DeepCopy()
		if cm.NodeConfig.SystemReserved != nil {
			value.Sub(cm.NodeConfig.SystemReserved[k])
		}
		if cm.NodeConfig.KubeReserved != nil {
			value.Sub(cm.NodeConfig.KubeReserved[k])
		}
		if value.Sign() < 0 {
			// Negative Allocatable resources don't make sense.
			value.Set(0)
		}
		result[k] = value
	}
	return result
}

// getNodeAllocatableInternalAbsolute is similar to getNodeAllocatableAbsolute except that
// it also includes internal resources (currently process IDs and swap).  It is intended for setting
// up top level cgroups only.
func (cm *containerManagerImpl) getNodeAllocatableInternalAbsolute() v1.ResourceList {
	return cm.getNodeAllocatableAbsoluteImpl(cm.internalCapacity)
}

// GetNodeAllocatableReservation returns amount of compute or storage resource that have to be reserved on this node from scheduling.
func (cm *containerManagerImpl) GetNodeAllocatableReservation() v1.ResourceList {
	cm.RLock()
	defer cm.RUnlock()
	evictionReservation := hardEvictionReservation(cm.HardEvictionThresholds, cm.capacity)
	result := make(v1.ResourceList)
	for k := range cm.capacity {
		value := resource.NewQuantity(0, resource.DecimalSI)
		if cm.NodeConfig.SystemReserved != nil {
			value.Add(cm.NodeConfig.SystemReserved[k])
		}
		if cm.NodeConfig.KubeReserved != nil {
			value.Add(cm.NodeConfig.KubeReserved[k])
		}
		if evictionReservation != nil {
			value.Add(evictionReservation[k])
		}
		if !value.IsZero() {
			result[k] = *value
		}
	}
	return result
}

// validateNodeAllocatable ensures that the user specified Node Allocatable Configuration doesn't reserve more than the node capacity.
// Returns error if the configuration is invalid, nil otherwise.
func (cm *containerManagerImpl) validateNodeAllocatable() error {
	cm.RLock()
	defer cm.RUnlock()
	var errors []string
	nar := cm.GetNodeAllocatableReservation()
	for k, v := range nar {
		value := cm.capacity[k].DeepCopy()
		value.Sub(v)

		if value.Sign() < 0 {
			errors = append(errors, fmt.Sprintf("Resource %q has a reservation of %v but capacity of %v. Expected capacity >= reservation.", k, v, cm.capacity[k]))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("invalid Node Allocatable configuration. %s", strings.Join(errors, " "))
	}
	return nil
}

// Using ObjectReference for events as the node maybe not cached; refer to #42701 for detail.
func nodeRefFromNode(nodeName string) *v1.ObjectReference {
	return &v1.ObjectReference{
		APIVersion: "v1",
		Kind:       "Node",
		Name:       nodeName,
		Namespace:  "",
	}
}
