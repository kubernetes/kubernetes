/*
Copyright 2025 The Kubernetes Authors.

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

package queue

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// unschedulableEntitiesQueuer is a wrapper for unschedulableEntities related operations.
type unschedulableEntitiesQueuer interface {
	// add adds an entity to the unschedulable queue.
	// The event should show which event triggered the addition and is used for the metric recording.
	add(logger klog.Logger, entity framework.QueuedEntityInfo, event string)
	// update updates an entity in the unschedulable queue.
	update(entity framework.QueuedEntityInfo, gatedBefore bool)
	// delete removes an entity from the unschedulable queue.
	// The `gated` parameter is used to figure out which metric should be decreased.
	delete(entity framework.QueuedEntityInfo, gated bool)
	// get returns the matching entity from the unschedulable queue.
	get(entity framework.QueuedEntityInfo) (framework.QueuedEntityInfo, bool)
	// has returns true if the matching entity exists in the unschedulable queue.
	has(entity framework.QueuedEntityInfo) bool
	// forEach calls fn for each entity in the unschedulable queue.
	forEach(fn func(framework.QueuedEntityInfo))
	// listPods returns all pods in the unschedulable queue.
	listPods() []*v1.Pod
	// len returns the number of entities in the unschedulable queue.
	len() int
}

// unschedulableEntities holds pods and pod groups that cannot be scheduled.
type unschedulableEntities struct {
	// entityInfoMap is a map keyed by an entity's key and the value is the QueuedEntityInfo.
	entityInfoMap map[string]framework.QueuedEntityInfo
	keyFunc       func(framework.QueuedEntityInfo) string
	// unschedulableRecorder and gatedRecorder track the number of entities in the unschedulable queue.
	// unschedulableRecorder tracks standard unschedulable entities, while gatedRecorder tracks entities
	// that are specifically blocked by scheduling gates.
	unschedulableRecorder, gatedRecorder metrics.MetricRecorder
}

// newUnschedulableEntities initializes a new object of unschedulableEntities.
func newUnschedulableEntities(unschedulableRecorder, gatedRecorder metrics.MetricRecorder) *unschedulableEntities {
	return &unschedulableEntities{
		entityInfoMap:         make(map[string]framework.QueuedEntityInfo),
		keyFunc:               queuedEntityKeyFunc,
		unschedulableRecorder: unschedulableRecorder,
		gatedRecorder:         gatedRecorder,
	}
}

// updateMetricsOnStateChange handles the metric accounting when an entity changes
// between Gated and Unschedulable states.
func (u *unschedulableEntities) updateMetricsOnStateChange(gatedBefore, isGated bool, size int) {
	if gatedBefore == isGated {
		return
	}

	if gatedBefore {
		// Transition: Gated -> Ungated
		u.gatedRecorder.Add(-size)
		u.unschedulableRecorder.Add(size)
	} else {
		// Transition: Ungated -> Gated
		u.gatedRecorder.Add(size)
		u.unschedulableRecorder.Add(-size)
	}
}

// add adds an entity to the unschedulable entityInfoMap.
// The event should show which event triggered the addition and is used for the metric recording.
func (u *unschedulableEntities) add(logger klog.Logger, entity framework.QueuedEntityInfo, event string) {
	entityKey := u.keyFunc(entity)
	if entity.Gated() {
		u.gatedRecorder.Add(entity.Size())
	} else {
		u.unschedulableRecorder.Add(entity.Size())
	}
	if metrics.SchedulerQueueIncomingPods != nil {
		metrics.SchedulerQueueIncomingPods.WithLabelValues("unschedulable", event).Add(float64(entity.Size()))
	}
	logger.V(5).Info("Entity moved to an internal scheduling queue", "type", entity.Type(), "entity", klog.KObj(entity), "event", event, "queue", unschedulableQ)
	u.entityInfoMap[entityKey] = entity
}

// update updates an entity in the unschedulable entityInfoMap.
func (u *unschedulableEntities) update(entity framework.QueuedEntityInfo, gatedBefore bool) {
	entityKey := u.keyFunc(entity)
	if _, exists := u.entityInfoMap[entityKey]; exists {
		u.updateMetricsOnStateChange(gatedBefore, entity.Gated(), entity.Size())
		u.entityInfoMap[entityKey] = entity
	}
}

// delete deletes an entity from the unschedulable entityInfoMap.
// The `gated` parameter is used to figure out which metric should be decreased.
func (u *unschedulableEntities) delete(entity framework.QueuedEntityInfo, gated bool) {
	entityKey := u.keyFunc(entity)
	if _, exists := u.entityInfoMap[entityKey]; exists {
		if gated {
			u.gatedRecorder.Add(-entity.Size())
		} else {
			u.unschedulableRecorder.Add(-entity.Size())
		}
	}
	delete(u.entityInfoMap, entityKey)
}

// get returns the QueuedEntityInfo if an entity with the same key is found in the map.
// It returns nil otherwise.
func (u *unschedulableEntities) get(entity framework.QueuedEntityInfo) (framework.QueuedEntityInfo, bool) {
	entityKey := u.keyFunc(entity)
	if entity, exists := u.entityInfoMap[entityKey]; exists {
		return entity, true
	}
	return nil, false
}

// has returns true if the matching entity exists in the unschedulable entityInfoMap.
func (u *unschedulableEntities) has(entity framework.QueuedEntityInfo) bool {
	_, exists := u.get(entity)
	return exists
}

// forEach calls fn for each entity in the unschedulable queue.
func (u *unschedulableEntities) forEach(fn func(framework.QueuedEntityInfo)) {
	for _, entity := range u.entityInfoMap {
		fn(entity)
	}
}

// listPods returns all pods in the unschedulable queue.
func (u *unschedulableEntities) listPods() []*v1.Pod {
	var result []*v1.Pod
	for _, entity := range u.entityInfoMap {
		entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
			result = append(result, pInfo.Pod)
			return true
		})
	}
	return result
}

// len returns the number of entities in the unschedulable queue.
func (u *unschedulableEntities) len() int {
	return len(u.entityInfoMap)
}
