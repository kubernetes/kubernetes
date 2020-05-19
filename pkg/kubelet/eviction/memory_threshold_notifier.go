/*
Copyright 2018 The Kubernetes Authors.

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

package eviction

import (
	"fmt"
	"time"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/api/resource"
	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
)

const (
	memoryUsageAttribute = "memory.usage_in_bytes"
	// this prevents constantly updating the memcg notifier if synchronize
	// is run frequently.
	notifierRefreshInterval = 10 * time.Second
)

type memoryThresholdNotifier struct {
	threshold  evictionapi.Threshold
	cgroupPath string
	events     chan struct{}
	factory    NotifierFactory
	handler    func(string)
	notifier   CgroupNotifier
}

var _ ThresholdNotifier = &memoryThresholdNotifier{}

// NewMemoryThresholdNotifier creates a ThresholdNotifier which is designed to respond to the given threshold.
// UpdateThreshold must be called once before the threshold will be active.
func NewMemoryThresholdNotifier(threshold evictionapi.Threshold, cgroupRoot string, factory NotifierFactory, handler func(string)) (ThresholdNotifier, error) {
	cgroups, err := cm.GetCgroupSubsystems()
	if err != nil {
		return nil, err
	}
	cgpath, found := cgroups.MountPoints["memory"]
	if !found || len(cgpath) == 0 {
		return nil, fmt.Errorf("memory cgroup mount point not found")
	}
	if isAllocatableEvictionThreshold(threshold) {
		// for allocatable thresholds, point the cgroup notifier at the allocatable cgroup
		cgpath += cgroupRoot
	}
	return &memoryThresholdNotifier{
		threshold:  threshold,
		cgroupPath: cgpath,
		events:     make(chan struct{}),
		handler:    handler,
		factory:    factory,
	}, nil
}

func (m *memoryThresholdNotifier) Start() {
	klog.Infof("eviction manager: created %s", m.Description())
	for range m.events {
		m.handler(fmt.Sprintf("eviction manager: %s crossed", m.Description()))
	}
}

func (m *memoryThresholdNotifier) UpdateThreshold(summary *statsapi.Summary) error {
	memoryStats := summary.Node.Memory
	if isAllocatableEvictionThreshold(m.threshold) {
		allocatableContainer, err := getSysContainer(summary.Node.SystemContainers, statsapi.SystemContainerPods)
		if err != nil {
			return err
		}
		memoryStats = allocatableContainer.Memory
	}
	if memoryStats == nil || memoryStats.UsageBytes == nil || memoryStats.WorkingSetBytes == nil || memoryStats.AvailableBytes == nil {
		return fmt.Errorf("summary was incomplete.  Expected MemoryStats and all subfields to be non-nil, but got %+v", memoryStats)
	}
	// Set threshold on usage to capacity - eviction_hard + inactive_file,
	// since we want to be notified when working_set = capacity - eviction_hard
	inactiveFile := resource.NewQuantity(int64(*memoryStats.UsageBytes-*memoryStats.WorkingSetBytes), resource.BinarySI)
	capacity := resource.NewQuantity(int64(*memoryStats.AvailableBytes+*memoryStats.WorkingSetBytes), resource.BinarySI)
	evictionThresholdQuantity := evictionapi.GetThresholdQuantity(m.threshold.Value, capacity)
	memcgThreshold := capacity.DeepCopy()
	memcgThreshold.Sub(*evictionThresholdQuantity)
	memcgThreshold.Add(*inactiveFile)

	klog.V(3).Infof("eviction manager: setting %s to %s\n", m.Description(), memcgThreshold.String())
	if m.notifier != nil {
		m.notifier.Stop()
	}
	newNotifier, err := m.factory.NewCgroupNotifier(m.cgroupPath, memoryUsageAttribute, memcgThreshold.Value())
	if err != nil {
		return err
	}
	m.notifier = newNotifier
	go m.notifier.Start(m.events)
	return nil
}

func (m *memoryThresholdNotifier) Description() string {
	var hard, allocatable string
	if isHardEvictionThreshold(m.threshold) {
		hard = "hard "
	} else {
		hard = "soft "
	}
	if isAllocatableEvictionThreshold(m.threshold) {
		allocatable = "allocatable "
	}
	return fmt.Sprintf("%s%smemory eviction threshold", hard, allocatable)
}

var _ NotifierFactory = &CgroupNotifierFactory{}

// CgroupNotifierFactory knows how to make CgroupNotifiers which integrate with the kernel
type CgroupNotifierFactory struct{}

// NewCgroupNotifier implements the NotifierFactory interface
func (n *CgroupNotifierFactory) NewCgroupNotifier(path, attribute string, threshold int64) (CgroupNotifier, error) {
	return NewCgroupNotifier(path, attribute, threshold)
}
