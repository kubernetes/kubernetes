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

package cm

import (
	"sync"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
)

// CGroupPods records the relationship between Pod UID and its cgroup name
// CGroupPods wraps the mapping between pod UID and cgroup name with a mutex
// since the test in TestDeleteOrphanedMirrorPods involves goroutine.
// The mutex prevents data race.
type CGroupPods struct {
	Lock *sync.RWMutex
	Pods map[types.UID]CgroupName
}

// Remove deletes all the Pod UIDs which map to the given cgroup name
func (cgp CGroupPods) Remove(cgroupName CgroupName) {
	cgp.Lock.Lock()
	defer cgp.Lock.Unlock()

	for uid, v := range cgp.Pods {
		if len(v) != len(cgroupName) {
			continue
		}
		found := true
		for i := range v {
			if v[i] != cgroupName[i] {
				found = false
				break
			}
		}
		if found {
			delete(cgp.Pods, uid)
		}
	}
}

// GetAllPodsFromCgroups returns map of Pod UID to the cgroup name
func (cgp CGroupPods) GetAllPodsFromCgroups() map[types.UID]CgroupName {
	cgp.Lock.RLock()
	defer cgp.Lock.RUnlock()

	cgroupPods := make(map[types.UID]CgroupName)
	for k, v := range cgp.Pods {
		cgroupPods[k] = v
	}
	return cgroupPods
}

// GetCgroupName retrieves cgroup name for the given Pod UID
func (cgp CGroupPods) GetCgroupName(uid types.UID) CgroupName {
	cgp.Lock.RLock()
	defer cgp.Lock.RUnlock()
	return cgp.Pods[uid]
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
		case evictionapi.SignalNodeFsAvailable:
			storageCapacity := capacity[v1.ResourceEphemeralStorage]
			value := evictionapi.GetThresholdQuantity(threshold.Value, &storageCapacity)
			ret[v1.ResourceEphemeralStorage] = *value
		}
	}
	return ret
}
