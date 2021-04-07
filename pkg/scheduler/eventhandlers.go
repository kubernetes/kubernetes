/*
Copyright 2019 The Kubernetes Authors.

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

package scheduler

import (
	"fmt"
	"reflect"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/scheduler/internal/queue"
	"k8s.io/kubernetes/pkg/scheduler/profile"
)

func (sched *Scheduler) onPvAdd(obj interface{}) {
	// Pods created when there are no PVs available will be stuck in
	// unschedulable queue. But unbound PVs created for static provisioning and
	// delay binding storage class are skipped in PV controller dynamic
	// provisioning and binding process, will not trigger events to schedule pod
	// again. So we need to move pods to active queue on PV add for this
	// scenario.
	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(queue.PvAdd)
}

func (sched *Scheduler) onPvUpdate(old, new interface{}) {
	// Scheduler.bindVolumesWorker may fail to update assumed pod volume
	// bindings due to conflicts if PVs are updated by PV controller or other
	// parties, then scheduler will add pod back to unschedulable queue. We
	// need to move pods to active queue on PV update for this scenario.
	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(queue.PvUpdate)
}

func (sched *Scheduler) onPvcAdd(obj interface{}) {
	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(queue.PvcAdd)
}

func (sched *Scheduler) onPvcUpdate(old, new interface{}) {
	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(queue.PvcUpdate)
}

func (sched *Scheduler) onStorageClassAdd(obj interface{}) {
	sc, ok := obj.(*storagev1.StorageClass)
	if !ok {
		klog.ErrorS(nil, "Cannot convert to *storagev1.StorageClass", "obj", obj)
		return
	}

	// CheckVolumeBindingPred fails if pod has unbound immediate PVCs. If these
	// PVCs have specified StorageClass name, creating StorageClass objects
	// with late binding will cause predicates to pass, so we need to move pods
	// to active queue.
	// We don't need to invalidate cached results because results will not be
	// cached for pod that has unbound immediate PVCs.
	if sc.VolumeBindingMode != nil && *sc.VolumeBindingMode == storagev1.VolumeBindingWaitForFirstConsumer {
		sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(queue.StorageClassAdd)
	}
}

func (sched *Scheduler) onServiceAdd(obj interface{}) {
	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(queue.ServiceAdd)
}

func (sched *Scheduler) onServiceUpdate(oldObj interface{}, newObj interface{}) {
	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(queue.ServiceUpdate)
}

func (sched *Scheduler) onServiceDelete(obj interface{}) {
	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(queue.ServiceDelete)
}

func (sched *Scheduler) addNodeToCache(obj interface{}) {
	node, ok := obj.(*v1.Node)
	if !ok {
		klog.ErrorS(nil, "Cannot convert to *v1.Node", "obj", obj)
		return
	}

	if err := sched.SchedulerCache.AddNode(node); err != nil {
		klog.ErrorS(err, "Scheduler cache AddNode failed")
	}

	klog.V(3).InfoS("Add event for node", "node", klog.KObj(node))
	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(queue.NodeAdd)
}

func (sched *Scheduler) updateNodeInCache(oldObj, newObj interface{}) {
	oldNode, ok := oldObj.(*v1.Node)
	if !ok {
		klog.ErrorS(nil, "Cannot convert oldObj to *v1.Node", "oldObj", oldObj)
		return
	}
	newNode, ok := newObj.(*v1.Node)
	if !ok {
		klog.ErrorS(nil, "Cannot convert newObj to *v1.Node", "newObj", newObj)
		return
	}

	if err := sched.SchedulerCache.UpdateNode(oldNode, newNode); err != nil {
		klog.ErrorS(err, "Scheduler cache UpdateNode failed")
	}

	// Only requeue unschedulable pods if the node became more schedulable.
	if event := nodeSchedulingPropertiesChange(newNode, oldNode); event != "" {
		sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(event)
	}
}

func (sched *Scheduler) deleteNodeFromCache(obj interface{}) {
	var node *v1.Node
	switch t := obj.(type) {
	case *v1.Node:
		node = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		node, ok = t.Obj.(*v1.Node)
		if !ok {
			klog.ErrorS(nil, "Cannot convert to *v1.Node", "obj", t.Obj)
			return
		}
	default:
		klog.ErrorS(nil, "Cannot convert to *v1.Node", "obj", t)
		return
	}
	klog.V(3).InfoS("Delete event for node", "node", klog.KObj(node))
	// NOTE: Updates must be written to scheduler cache before invalidating
	// equivalence cache, because we could snapshot equivalence cache after the
	// invalidation and then snapshot the cache itself. If the cache is
	// snapshotted before updates are written, we would update equivalence
	// cache with stale information which is based on snapshot of old cache.
	if err := sched.SchedulerCache.RemoveNode(node); err != nil {
		klog.ErrorS(err, "Scheduler cache RemoveNode failed")
	}
}

func (sched *Scheduler) onCSINodeAdd(obj interface{}) {
	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(queue.CSINodeAdd)
}

func (sched *Scheduler) onCSINodeUpdate(oldObj, newObj interface{}) {
	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(queue.CSINodeUpdate)
}

func (sched *Scheduler) addPodToSchedulingQueue(obj interface{}) {
	pod := obj.(*v1.Pod)
	klog.V(3).InfoS("Add event for unscheduled pod", "pod", klog.KObj(pod))
	if err := sched.SchedulingQueue.Add(pod); err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to queue %T: %v", obj, err))
	}
}

func (sched *Scheduler) updatePodInSchedulingQueue(oldObj, newObj interface{}) {
	oldPod, newPod := oldObj.(*v1.Pod), newObj.(*v1.Pod)
	// Bypass update event that carries identical objects; otherwise, a duplicated
	// Pod may go through scheduling and cause unexpected behavior (see #96071).
	if oldPod.ResourceVersion == newPod.ResourceVersion {
		return
	}
	if sched.skipPodUpdate(newPod) {
		return
	}
	if err := sched.SchedulingQueue.Update(oldPod, newPod); err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to update %T: %v", newObj, err))
	}
}

func (sched *Scheduler) deletePodFromSchedulingQueue(obj interface{}) {
	var pod *v1.Pod
	switch t := obj.(type) {
	case *v1.Pod:
		pod = obj.(*v1.Pod)
	case cache.DeletedFinalStateUnknown:
		var ok bool
		pod, ok = t.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("unable to convert object %T to *v1.Pod in %T", obj, sched))
			return
		}
	default:
		utilruntime.HandleError(fmt.Errorf("unable to handle object in %T: %T", sched, obj))
		return
	}
	klog.V(3).InfoS("Delete event for unscheduled pod", "pod", klog.KObj(pod))
	if err := sched.SchedulingQueue.Delete(pod); err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to dequeue %T: %v", obj, err))
	}
	fwk, err := sched.frameworkForPod(pod)
	if err != nil {
		// This shouldn't happen, because we only accept for scheduling the pods
		// which specify a scheduler name that matches one of the profiles.
		klog.ErrorS(err, "Unable to get profile", "pod", klog.KObj(pod))
		return
	}
	fwk.RejectWaitingPod(pod.UID)
}

func (sched *Scheduler) addPodToCache(obj interface{}) {
	pod, ok := obj.(*v1.Pod)
	if !ok {
		klog.ErrorS(nil, "Cannot convert to *v1.Pod", "obj", obj)
		return
	}
	klog.V(3).InfoS("Add event for scheduled pod", "pod", klog.KObj(pod))

	if err := sched.SchedulerCache.AddPod(pod); err != nil {
		klog.ErrorS(err, "Scheduler cache AddPod failed", "pod", klog.KObj(pod))
	}

	sched.SchedulingQueue.AssignedPodAdded(pod)
}

func (sched *Scheduler) updatePodInCache(oldObj, newObj interface{}) {
	oldPod, ok := oldObj.(*v1.Pod)
	if !ok {
		klog.ErrorS(nil, "Cannot convert oldObj to *v1.Pod", "oldObj", oldObj)
		return
	}
	newPod, ok := newObj.(*v1.Pod)
	if !ok {
		klog.ErrorS(nil, "Cannot convert newObj to *v1.Pod", "newObj", newObj)
		return
	}

	// A Pod delete event followed by an immediate Pod add event may be merged
	// into a Pod update event. In this case, we should invalidate the old Pod, and
	// then add the new Pod.
	if oldPod.UID != newPod.UID {
		sched.deletePodFromCache(oldObj)
		sched.addPodToCache(newObj)
		return
	}

	// NOTE: Updates must be written to scheduler cache before invalidating
	// equivalence cache, because we could snapshot equivalence cache after the
	// invalidation and then snapshot the cache itself. If the cache is
	// snapshotted before updates are written, we would update equivalence
	// cache with stale information which is based on snapshot of old cache.
	if err := sched.SchedulerCache.UpdatePod(oldPod, newPod); err != nil {
		klog.ErrorS(err, "Scheduler cache UpdatePod failed", "oldPod", klog.KObj(oldPod), "newPod", klog.KObj(newPod))
	}

	sched.SchedulingQueue.AssignedPodUpdated(newPod)
}

func (sched *Scheduler) deletePodFromCache(obj interface{}) {
	var pod *v1.Pod
	switch t := obj.(type) {
	case *v1.Pod:
		pod = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		pod, ok = t.Obj.(*v1.Pod)
		if !ok {
			klog.ErrorS(nil, "Cannot convert to *v1.Pod", "obj", t.Obj)
			return
		}
	default:
		klog.ErrorS(nil, "Cannot convert to *v1.Pod", "obj", t)
		return
	}
	klog.V(3).InfoS("Delete event for scheduled pod", "pod", klog.KObj(pod))
	// NOTE: Updates must be written to scheduler cache before invalidating
	// equivalence cache, because we could snapshot equivalence cache after the
	// invalidation and then snapshot the cache itself. If the cache is
	// snapshotted before updates are written, we would update equivalence
	// cache with stale information which is based on snapshot of old cache.
	if err := sched.SchedulerCache.RemovePod(pod); err != nil {
		klog.ErrorS(err, "Scheduler cache RemovePod failed", "pod", klog.KObj(pod))
	}

	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(queue.AssignedPodDelete)
}

// assignedPod selects pods that are assigned (scheduled and running).
func assignedPod(pod *v1.Pod) bool {
	return len(pod.Spec.NodeName) != 0
}

// responsibleForPod returns true if the pod has asked to be scheduled by the given scheduler.
func responsibleForPod(pod *v1.Pod, profiles profile.Map) bool {
	return profiles.HandlesSchedulerName(pod.Spec.SchedulerName)
}

// skipPodUpdate checks whether the specified pod update should be ignored.
// This function will return true if
//   - The pod has already been assumed, AND
//   - The pod has only its ResourceVersion, Spec.NodeName, Annotations,
//     ManagedFields, Finalizers and/or Conditions updated.
func (sched *Scheduler) skipPodUpdate(pod *v1.Pod) bool {
	// Non-assumed pods should never be skipped.
	isAssumed, err := sched.SchedulerCache.IsAssumedPod(pod)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to check whether pod %s/%s is assumed: %v", pod.Namespace, pod.Name, err))
		return false
	}
	if !isAssumed {
		return false
	}

	// Gets the assumed pod from the cache.
	assumedPod, err := sched.SchedulerCache.GetPod(pod)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to get assumed pod %s/%s from cache: %v", pod.Namespace, pod.Name, err))
		return false
	}

	// Compares the assumed pod in the cache with the pod update. If they are
	// equal (with certain fields excluded), this pod update will be skipped.
	f := func(pod *v1.Pod) *v1.Pod {
		p := pod.DeepCopy()
		// ResourceVersion must be excluded because each object update will
		// have a new resource version.
		p.ResourceVersion = ""
		// Spec.NodeName must be excluded because the pod assumed in the cache
		// is expected to have a node assigned while the pod update may nor may
		// not have this field set.
		p.Spec.NodeName = ""
		// Annotations must be excluded for the reasons described in
		// https://github.com/kubernetes/kubernetes/issues/52914.
		p.Annotations = nil
		// Same as above, when annotations are modified with ServerSideApply,
		// ManagedFields may also change and must be excluded
		p.ManagedFields = nil
		// The following might be changed by external controllers, but they don't
		// affect scheduling decisions.
		p.Finalizers = nil
		p.Status.Conditions = nil
		return p
	}
	assumedPodCopy, podCopy := f(assumedPod), f(pod)
	if !reflect.DeepEqual(assumedPodCopy, podCopy) {
		return false
	}
	klog.V(3).InfoS("Pod update ignored because changes won't affect scheduling", "pod", klog.KObj(pod))
	return true
}

// addAllEventHandlers is a helper function used in tests and in Scheduler
// to add event handlers for various informers.
func addAllEventHandlers(
	sched *Scheduler,
	informerFactory informers.SharedInformerFactory,
) {
	// scheduled pod cache
	informerFactory.Core().V1().Pods().Informer().AddEventHandler(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.Pod:
					return assignedPod(t)
				case cache.DeletedFinalStateUnknown:
					if pod, ok := t.Obj.(*v1.Pod); ok {
						return assignedPod(pod)
					}
					utilruntime.HandleError(fmt.Errorf("unable to convert object %T to *v1.Pod in %T", obj, sched))
					return false
				default:
					utilruntime.HandleError(fmt.Errorf("unable to handle object in %T: %T", sched, obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    sched.addPodToCache,
				UpdateFunc: sched.updatePodInCache,
				DeleteFunc: sched.deletePodFromCache,
			},
		},
	)
	// unscheduled pod queue
	informerFactory.Core().V1().Pods().Informer().AddEventHandler(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.Pod:
					return !assignedPod(t) && responsibleForPod(t, sched.Profiles)
				case cache.DeletedFinalStateUnknown:
					if pod, ok := t.Obj.(*v1.Pod); ok {
						return !assignedPod(pod) && responsibleForPod(pod, sched.Profiles)
					}
					utilruntime.HandleError(fmt.Errorf("unable to convert object %T to *v1.Pod in %T", obj, sched))
					return false
				default:
					utilruntime.HandleError(fmt.Errorf("unable to handle object in %T: %T", sched, obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    sched.addPodToSchedulingQueue,
				UpdateFunc: sched.updatePodInSchedulingQueue,
				DeleteFunc: sched.deletePodFromSchedulingQueue,
			},
		},
	)

	informerFactory.Core().V1().Nodes().Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    sched.addNodeToCache,
			UpdateFunc: sched.updateNodeInCache,
			DeleteFunc: sched.deleteNodeFromCache,
		},
	)

	informerFactory.Storage().V1().CSINodes().Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    sched.onCSINodeAdd,
			UpdateFunc: sched.onCSINodeUpdate,
		},
	)

	// On add and update of PVs.
	informerFactory.Core().V1().PersistentVolumes().Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			// MaxPDVolumeCountPredicate: since it relies on the counts of PV.
			AddFunc:    sched.onPvAdd,
			UpdateFunc: sched.onPvUpdate,
		},
	)

	// This is for MaxPDVolumeCountPredicate: add/update PVC will affect counts of PV when it is bound.
	informerFactory.Core().V1().PersistentVolumeClaims().Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    sched.onPvcAdd,
			UpdateFunc: sched.onPvcUpdate,
		},
	)

	// This is for ServiceAffinity: affected by the selector of the service is updated.
	// Also, if new service is added, equivalence cache will also become invalid since
	// existing pods may be "captured" by this service and change this predicate result.
	informerFactory.Core().V1().Services().Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    sched.onServiceAdd,
			UpdateFunc: sched.onServiceUpdate,
			DeleteFunc: sched.onServiceDelete,
		},
	)

	informerFactory.Storage().V1().StorageClasses().Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc: sched.onStorageClassAdd,
		},
	)
}

func nodeSchedulingPropertiesChange(newNode *v1.Node, oldNode *v1.Node) string {
	if nodeSpecUnschedulableChanged(newNode, oldNode) {
		return queue.NodeSpecUnschedulableChange
	}
	if nodeAllocatableChanged(newNode, oldNode) {
		return queue.NodeAllocatableChange
	}
	if nodeLabelsChanged(newNode, oldNode) {
		return queue.NodeLabelChange
	}
	if nodeTaintsChanged(newNode, oldNode) {
		return queue.NodeTaintChange
	}
	if nodeConditionsChanged(newNode, oldNode) {
		return queue.NodeConditionChange
	}

	return ""
}

func nodeAllocatableChanged(newNode *v1.Node, oldNode *v1.Node) bool {
	return !reflect.DeepEqual(oldNode.Status.Allocatable, newNode.Status.Allocatable)
}

func nodeLabelsChanged(newNode *v1.Node, oldNode *v1.Node) bool {
	return !reflect.DeepEqual(oldNode.GetLabels(), newNode.GetLabels())
}

func nodeTaintsChanged(newNode *v1.Node, oldNode *v1.Node) bool {
	return !reflect.DeepEqual(newNode.Spec.Taints, oldNode.Spec.Taints)
}

func nodeConditionsChanged(newNode *v1.Node, oldNode *v1.Node) bool {
	strip := func(conditions []v1.NodeCondition) map[v1.NodeConditionType]v1.ConditionStatus {
		conditionStatuses := make(map[v1.NodeConditionType]v1.ConditionStatus, len(conditions))
		for i := range conditions {
			conditionStatuses[conditions[i].Type] = conditions[i].Status
		}
		return conditionStatuses
	}
	return !reflect.DeepEqual(strip(oldNode.Status.Conditions), strip(newNode.Status.Conditions))
}

func nodeSpecUnschedulableChanged(newNode *v1.Node, oldNode *v1.Node) bool {
	return newNode.Spec.Unschedulable != oldNode.Spec.Unschedulable && !newNode.Spec.Unschedulable
}
