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
	"os"
	"path"
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
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/stats/pidlimit"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	defaultNodeAllocatableCgroupName = "kubepods"
	// systemPartitionCgroupName is the name of the system partition root, created
	// as a direct child of the node allocatable cgroup.
	systemPartitionCgroupName = "system"
)

// systemPartitionEnabled returns true if a system partition is configured on
// this node.
func systemPartitionEnabled(nodeConfig NodeConfig) bool {
	if !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.NodeSystemPartition) {
		return false
	}
	// The partition is a sub-hierarchy of the QoS hierarchy, so it cannot exist
	// without it.
	if !nodeConfig.CgroupsPerQOS {
		return false
	}
	// The system partition is limited to cgroup v2 only.
	if !libcontainercgroups.IsCgroup2UnifiedMode() {
		return false
	}
	return nodeConfig.SystemPartition != nil && nodeConfig.SystemPartition.Namespaces.Len() > 0
}

// systemPartitionQOSContainersInfo returns the system partition's QoS container
// roots under the given node allocatable cgroup root. The names are computed
// even when no system partition is configured, so that cgroups left behind by an
// earlier one stay visible to the orphan pod cgroup cleanup.
func systemPartitionQOSContainersInfo(cgroupRoot CgroupName) QOSContainersInfo {
	partitionRoot := NewCgroupName(cgroupRoot, systemPartitionCgroupName)
	return QOSContainersInfo{
		Guaranteed: partitionRoot,
		Burstable:  NewCgroupName(partitionRoot, strings.ToLower(string(v1.PodQOSBurstable))),
		BestEffort: NewCgroupName(partitionRoot, strings.ToLower(string(v1.PodQOSBestEffort))),
	}
}

// cgroupPathExists reports whether the cgroup directory is present in any of the
// mounted subsystems.
func (cm *containerManagerImpl) cgroupPathExists(name CgroupName) bool {
	cgroupfsName := cm.cgroupManager.Name(name)
	for _, mountPoint := range cm.subsystems.MountPoints {
		if _, err := os.Stat(path.Join(mountPoint, cgroupfsName)); err == nil {
			return true
		}
	}
	return false
}

// cgroupHasChildren reports whether the cgroup has any child cgroup in any of the
// mounted subsystems.
func (cm *containerManagerImpl) cgroupHasChildren(name CgroupName) (bool, error) {
	cgroupfsName := cm.cgroupManager.Name(name)
	for _, mountPoint := range cm.subsystems.MountPoints {
		entries, err := os.ReadDir(path.Join(mountPoint, cgroupfsName))
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return false, err
		}
		for _, entry := range entries {
			if entry.IsDir() {
				return true, nil
			}
		}
	}
	return false, nil
}

// cleanupSystemPartitionCgroups removes the cgroups of a system partition that is
// no longer configured, so that disabling the feature does not leak the hierarchy.
// It only removes empty cgroups. Pod cgroups still inside the partition are left
// to the orphan pod cgroup cleanup, which kills their processes and waits for
// volume teardown first. That runs asynchronously, so this is a periodic task
// that retries until the partition is empty.
func (cm *containerManagerImpl) cleanupSystemPartitionCgroups(logger klog.Logger) {
	qosContainersInfo := systemPartitionQOSContainersInfo(cm.cgroupRoot)
	partitionRoot := qosContainersInfo.Guaranteed
	if !cm.cgroupPathExists(partitionRoot) {
		return
	}

	// The QoS roots have to go before the partition root that parents them.
	for _, name := range []CgroupName{qosContainersInfo.Burstable, qosContainersInfo.BestEffort, partitionRoot} {
		if !cm.cgroupPathExists(name) {
			continue
		}
		hasChildren, err := cm.cgroupHasChildren(name)
		if err != nil {
			logger.Error(err, "Failed to inspect leftover system partition cgroup", "cgroupName", name)
			return
		}
		if hasChildren {
			logger.V(4).Info("Leftover system partition cgroup still has pod cgroups, deferring removal", "cgroupName", name)
			return
		}
		if err := cm.cgroupManager.Destroy(logger, &CgroupConfig{Name: name}); err != nil {
			logger.Error(err, "Failed to remove leftover system partition cgroup", "cgroupName", name)
			return
		}
		logger.V(2).Info("Removed leftover system partition cgroup", "cgroupName", name)
	}
}

// systemPartitionConfig returns the system partition configuration in effect, or
// nil if no system partition is configured on this node.
func (cm *containerManagerImpl) systemPartitionConfig() *SystemPartitionConfig {
	if cm.systemPartitionQOSManager == nil {
		return nil
	}
	return cm.NodeConfig.SystemPartition
}

// systemPartitionCgroupConfig returns the cgroup configuration of the system
// partition root. The node allocatable cgroup derives its limits from node
// capacity, but these are the fixed budget the administrator set aside for
// system Pods.
func (cm *containerManagerImpl) systemPartitionCgroupConfig() *CgroupConfig {
	spc := cm.NodeConfig.SystemPartition
	resourceParameters := &ResourceConfig{}
	if spc.MemoryLimit != nil {
		memoryLimit := *spc.MemoryLimit
		resourceParameters.Memory = &memoryLimit
	}
	if !spc.CPUSet.IsEmpty() {
		resourceParameters.CPUSet = spc.CPUSet
	}
	return &CgroupConfig{
		Name:               cm.systemPartitionRoot,
		ResourceParameters: resourceParameters,
	}
}

// createSystemPartitionCgroups creates the system partition root cgroup so that
// the partition's QoS hierarchy can be created underneath it.
func (cm *containerManagerImpl) createSystemPartitionCgroups(logger klog.Logger) error {
	cgroupConfig := cm.systemPartitionCgroupConfig()
	if cm.cgroupManager.Exists(cgroupConfig.Name) {
		return nil
	}
	logger.V(2).Info("Creating system partition cgroup", "cgroupName", cgroupConfig.Name)
	if err := cm.cgroupManager.Create(logger, cgroupConfig); err != nil {
		logger.Error(err, "Failed to create cgroup", "cgroupName", cgroupConfig.Name)
		return err
	}
	return nil
}

// enforceSystemPartitionCgroups applies the configured partition limits to the
// system partition root. createSystemPartitionCgroups is a no-op once the cgroup
// exists, so on a kubelet restart this is what applies them.
func (cm *containerManagerImpl) enforceSystemPartitionCgroups(logger klog.Logger) error {
	cgroupConfig := cm.systemPartitionCgroupConfig()
	spc := cm.NodeConfig.SystemPartition
	logger.V(2).Info("Enforcing system partition limits", "cgroupName", cgroupConfig.Name,
		"memoryLimit", spc.MemoryLimit, "cpuset", spc.CPUSet.String())
	if err := cm.cgroupManager.Update(logger, cgroupConfig); err != nil {
		return fmt.Errorf("failed to enforce system partition limits on %q: %w", cgroupConfig.Name, err)
	}
	// Report the limit from where it is written, so the gauge and the cgroup can
	// never disagree.
	if spc.MemoryLimit != nil {
		metrics.PartitionMemoryLimitBytes.WithLabelValues(systemPartitionCgroupName).Set(float64(*spc.MemoryLimit))
	}
	return nil
}

// PartitionMemoryUsage returns the memory currently charged to each partition's
// cgroup, keyed by partition name. It is read at collection time, so the caller
// gets what the kernel holds against the limit right now.
func (cm *containerManagerImpl) PartitionMemoryUsage() map[string]int64 {
	if cm.systemPartitionQOSManager == nil {
		return nil
	}
	usage, err := cm.cgroupManager.MemoryUsage(cm.systemPartitionRoot)
	if err != nil {
		return nil
	}
	return map[string]int64{systemPartitionCgroupName: usage}
}

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
				err := cm.cgroupManager.Update(logger, cgroupConfig)
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
	// Only do so on None policy, as Static policy will do its own updating of the cpuset.
	// Please see the comment on policy none's GetAllocatableCPUs
	if cm.cpuManager.GetAllocatableCPUs().IsEmpty() {
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
// it also includes internal resources (currently process IDs).  It is intended for setting
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
