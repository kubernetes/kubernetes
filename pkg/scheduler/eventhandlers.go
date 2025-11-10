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
	"context"
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	corev1nodeaffinity "k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/dynamic-resource-allocation/deviceclass/extendedresourcecache"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

func (sched *Scheduler) addNodeToCache(obj interface{}) {
	evt := fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add}
	start := time.Now()
	defer metrics.EventHandlingLatency.WithLabelValues(evt.Label()).Observe(metrics.SinceInSeconds(start))
	logger := sched.logger
	node, ok := obj.(*v1.Node)
	if !ok {
		utilruntime.HandleErrorWithLogger(logger, nil, "Cannot convert to *v1.Node", "obj", obj)
		return
	}

	logger.V(3).Info("Add event for node", "node", klog.KObj(node))
	nodeInfo := sched.Cache.AddNode(logger, node)
	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, evt, nil, node, preCheckForNode(logger, nodeInfo))
}

func (sched *Scheduler) updateNodeInCache(oldObj, newObj interface{}) {
	start := time.Now()
	logger := sched.logger
	oldNode, ok := oldObj.(*v1.Node)
	if !ok {
		utilruntime.HandleErrorWithLogger(logger, nil, "Cannot convert oldObj to *v1.Node", "oldObj", oldObj)
		return
	}
	newNode, ok := newObj.(*v1.Node)
	if !ok {
		utilruntime.HandleErrorWithLogger(logger, nil, "Cannot convert newObj to *v1.Node", "newObj", newObj)
		return
	}

	logger.V(4).Info("Update event for node", "node", klog.KObj(newNode))
	nodeInfo := sched.Cache.UpdateNode(logger, oldNode, newNode)
	events := framework.NodeSchedulingPropertiesChange(newNode, oldNode)

	// Save the time it takes to update the node in the cache.
	updatingDuration := metrics.SinceInSeconds(start)

	// Only requeue unschedulable pods if the node became more schedulable.
	for _, evt := range events {
		startMoving := time.Now()
		sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, evt, oldNode, newNode, preCheckForNode(logger, nodeInfo))
		movingDuration := metrics.SinceInSeconds(startMoving)

		metrics.EventHandlingLatency.WithLabelValues(evt.Label()).Observe(updatingDuration + movingDuration)
	}
}

func (sched *Scheduler) deleteNodeFromCache(obj interface{}) {
	evt := fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Delete}
	start := time.Now()
	defer metrics.EventHandlingLatency.WithLabelValues(evt.Label()).Observe(metrics.SinceInSeconds(start))

	logger := sched.logger
	var node *v1.Node
	switch t := obj.(type) {
	case *v1.Node:
		node = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		node, ok = t.Obj.(*v1.Node)
		if !ok {
			utilruntime.HandleErrorWithLogger(logger, nil, "Cannot convert to *v1.Node", "obj", t.Obj)
			return
		}
	default:
		utilruntime.HandleErrorWithLogger(logger, nil, "Cannot convert to *v1.Node", "obj", t)
		return
	}

	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, evt, node, nil, nil)

	logger.V(3).Info("Delete event for node", "node", klog.KObj(node))
	if err := sched.Cache.RemoveNode(logger, node); err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Scheduler cache RemoveNode failed")
	}
}

func (sched *Scheduler) addPod(obj interface{}) {
	logger := sched.logger
	pod, ok := obj.(*v1.Pod)
	if !ok {
		utilruntime.HandleErrorWithLogger(logger, nil, "Cannot convert to *v1.Pod", "obj", obj)
		return
	}

	if sched.WorkloadManager != nil {
		// Register pod into workload manager before adding to the cache or scheduling queue.
		sched.WorkloadManager.AddPod(pod)
	}
	if assignedPod(pod) {
		sched.addAssignedPodToCache(pod)
	} else if responsibleForPod(pod, sched.Profiles) {
		sched.addPodToSchedulingQueue(pod)
	}
}

func (sched *Scheduler) updatePod(oldObj, newObj interface{}) {
	logger := sched.logger
	oldPod, ok := oldObj.(*v1.Pod)
	if !ok {
		utilruntime.HandleErrorWithLogger(logger, nil, "Cannot convert oldObj to *v1.Pod", "oldObj", oldObj)
		return
	}
	newPod, ok := newObj.(*v1.Pod)
	if !ok {
		utilruntime.HandleErrorWithLogger(logger, nil, "Cannot convert newObj to *v1.Pod", "newObj", newObj)
		return
	}

	if sched.WorkloadManager != nil {
		// Update pod in workload manager before updating it in the cache or scheduling queue.
		sched.WorkloadManager.UpdatePod(oldPod, newPod)
	}
	if assignedPod(oldPod) {
		sched.updateAssignedPodInCache(oldPod, newPod)
	} else if assignedPod(newPod) {
		// This update means binding operation. We can treat it as adding the pod to a cache
		// (addition to the cache will handle this binding appropriately).
		sched.addAssignedPodToCache(newPod)
		if responsibleForPod(oldPod, sched.Profiles) {
			// Pod shouldn't be in the scheduling queue, but in unlikely event that the pod has been bound
			// by another component, it should be removed from scheduling queue for correctness.
			// Passing "true" means that removal from the scheduling queue is caused by a binding event,
			// not by removal of the pod from the cluster.
			sched.deletePodFromSchedulingQueue(oldPod, true)
		}
	} else if responsibleForPod(oldPod, sched.Profiles) {
		sched.updatePodInSchedulingQueue(oldPod, newPod)
	}
}

func (sched *Scheduler) deletePod(obj interface{}) {
	logger := sched.logger
	var pod *v1.Pod
	switch t := obj.(type) {
	case *v1.Pod:
		pod = t
		if sched.WorkloadManager != nil {
			// Delete pod from workload manager before deleting the pod from cache or scheduling queue.
			sched.WorkloadManager.DeletePod(pod)
		}
		if assignedPod(pod) {
			sched.deleteAssignedPodFromCache(pod)
		} else if responsibleForPod(pod, sched.Profiles) {
			// Passing "false" means that removal from the scheduling queue is caused by
			// removal of the pod from the cluster, not by a binding event.
			sched.deletePodFromSchedulingQueue(pod, false)
		}
		return
	case cache.DeletedFinalStateUnknown:
		var ok bool
		pod, ok = t.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleErrorWithLogger(logger, nil, "Cannot convert to *v1.Pod", "obj", t.Obj)
			return
		}
		if sched.WorkloadManager != nil {
			// Delete pod from workload manager before deleting the pod from cache or scheduling queue.
			sched.WorkloadManager.DeletePod(pod)
		}
		// The carried object may be stale, so we don't use it to check if
		// it's assigned or not. Attempting to cleanup anyways.
		sched.deleteAssignedPodFromCache(pod)
		if responsibleForPod(pod, sched.Profiles) {
			// Passing "false" means that removal from the scheduling queue is caused by
			// removal of the pod from the cluster, not by a binding event.
			sched.deletePodFromSchedulingQueue(pod, false)
		}
		return
	default:
		utilruntime.HandleErrorWithLogger(logger, nil, "Unable to handle object", "objType", fmt.Sprintf("%T", obj), "obj", obj)
		return
	}
}

func (sched *Scheduler) addPodToSchedulingQueue(pod *v1.Pod) {
	start := time.Now()
	defer metrics.EventHandlingLatency.WithLabelValues(framework.EventUnscheduledPodAdd.Label()).Observe(metrics.SinceInSeconds(start))

	logger := sched.logger
	logger.V(3).Info("Add event for unscheduled pod", "pod", klog.KObj(pod))
	sched.SchedulingQueue.Add(logger, pod)
	if utilfeature.DefaultFeatureGate.Enabled(features.GangScheduling) {
		sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, framework.EventUnscheduledPodAdd, nil, pod, nil)
	}
}

func (sched *Scheduler) syncPodWithDispatcher(pod *v1.Pod) *v1.Pod {
	enrichedObj, err := sched.APIDispatcher.SyncObject(pod)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to sync pod %s/%s with API dispatcher: %w", pod.Namespace, pod.Name, err))
		return pod
	}
	enrichedPod, ok := enrichedObj.(*v1.Pod)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("cannot convert enrichedObj of type %T to *v1.Pod", enrichedObj))
		return pod
	}
	return enrichedPod
}

// handleAssumedPodDeletion is an event handler that deals with the deletion of an assumed pod.
// We must remove it from the scheduler's cache immediately to prevent it from blocking resources for other pending pods,
// causing unnecessary preemption attempts. Note that PreBinding/Binding will continue, but is eventually expected to fail
// as the pod does not exist in the kube-apiserver anymore and so in the scheduler cache.
func (sched *Scheduler) handleAssumedPodDeletion(pod *v1.Pod) {
	logger := sched.logger
	// We must operate on the pod from the scheduler's cache, not the one from the event.
	// The cached version has the assigned NodeName and represents the resources being consumed.
	assumedPod, err := sched.Cache.GetPod(pod)
	if err != nil {
		// This is not an error. The pod may have already completed its binding cycle and been
		// removed from the cache. Nothing more to do.
		logger.V(5).Info("Assumed pod was already forgotten", "pod", klog.KObj(pod))
		return
	}
	pod = assumedPod

	fwk, err := sched.frameworkForPod(pod)
	if err != nil {
		// This shouldn't happen, because we only accept for scheduling the pods
		// which specify a scheduler name that matches one of the profiles.
		utilruntime.HandleErrorWithLogger(logger, err, "Unable to get profile for pod", "pod", klog.KObj(pod))
		return
	}

	// The pod might be in one of two states:
	// 1. If the pod is waiting on WaitOnPermit, we reject it. This causes the pod's scheduling
	//    cycle to quickly fail gracefully, and it will clean itself up via `handleBindingCycleError`.
	if !fwk.RejectWaitingPod(pod.UID) {
		// 2. If the pod is no longer waiting (e.g., it's in PreBind or Bind), we can't quickly reject it.
		//    We must explicitly remove it from the cache here to free up its assumed resources.
		if err := sched.Cache.ForgetPod(logger, pod); err != nil {
			utilruntime.HandleErrorWithLogger(logger, err, "Scheduler cache ForgetPod failed", "pod", klog.KObj(pod))
		}
	}

	// The removal of this assumed pod may have freed up resources. We trigger the AssignedPodDelete event
	// to move other unscheduled pods, giving them a chance to be scheduled.
	// If the forgotten pod reserved some resources in memory,
	// it will wake up the pods again after freeing up the resources in `handleBindingCycleError`.
	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, framework.EventAssignedPodDelete, pod, nil, nil)
}

func (sched *Scheduler) updatePodInSchedulingQueue(oldPod, newPod *v1.Pod) {
	start := time.Now()
	logger := sched.logger
	// Bypass update event that carries identical objects; otherwise, a duplicated
	// Pod may go through scheduling and cause unexpected behavior (see #96071).
	if oldPod.ResourceVersion == newPod.ResourceVersion {
		return
	}

	defer metrics.EventHandlingLatency.WithLabelValues(framework.EventUnscheduledPodUpdate.Label()).Observe(metrics.SinceInSeconds(start))
	for _, evt := range framework.PodSchedulingPropertiesChange(newPod, oldPod) {
		if evt.Label() != framework.EventUnscheduledPodUpdate.Label() {
			defer metrics.EventHandlingLatency.WithLabelValues(evt.Label()).Observe(metrics.SinceInSeconds(start))
		}
	}

	if sched.APIDispatcher != nil {
		// If the API dispatcher is available, sync the new pod with the details.
		// However, at the moment the updated newPod is discarded and this logic will be handled in the future releases.
		_ = sched.syncPodWithDispatcher(newPod)
	}

	isAssumed, err := sched.Cache.IsAssumedPod(newPod)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to check whether pod is assumed", "pod", klog.KObj(newPod))
	}
	if isAssumed {
		if newPod.DeletionTimestamp != nil && oldPod.DeletionTimestamp == nil {
			// Assumed pod deletion has started. We should handle that differently,
			// because we can't update such pod in any structure directly.
			sched.handleAssumedPodDeletion(newPod)
		}
		return
	}

	logger.V(4).Info("Update event for unscheduled pod", "pod", klog.KObj(newPod))
	sched.SchedulingQueue.Update(logger, oldPod, newPod)
	if hasNominatedNodeNameChanged(oldPod, newPod) {
		// Nominated node changed in pod, so we need to treat it as if the pod was deleted from the old nominated node,
		// because the scheduler treats such a pod as if it was already assigned when scheduling lower or equal priority pods.
		sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, framework.EventAssignedPodDelete, oldPod, nil, getLEPriorityPreCheck(corev1helpers.PodPriority(oldPod)))
	}
}

// hasNominatedNodeNameChanged returns true when nominated node name has existed but changed.
func hasNominatedNodeNameChanged(oldPod, newPod *v1.Pod) bool {
	return len(oldPod.Status.NominatedNodeName) > 0 && oldPod.Status.NominatedNodeName != newPod.Status.NominatedNodeName
}

func (sched *Scheduler) deletePodFromSchedulingQueue(pod *v1.Pod, inBinding bool) {
	start := time.Now()
	defer metrics.EventHandlingLatency.WithLabelValues(framework.EventUnscheduledPodDelete.Label()).Observe(metrics.SinceInSeconds(start))

	logger := sched.logger

	logger.V(3).Info("Delete event for unscheduled pod", "pod", klog.KObj(pod))
	sched.SchedulingQueue.Delete(pod)
	if inBinding {
		// In the case of a binding, the rest can be skipped because it is not really a pod removal operation, but a binding.
		// Any necessary notifications will be sent by the binding process, unless it was an unlikely external binding.
		// In that case, we need to notify about the release of resources that were held by different assume/nomination
		// once the https://github.com/kubernetes/kubernetes/issues/134859 is fixed.
		return
	}
	isAssumed, err := sched.Cache.IsAssumedPod(pod)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to check whether pod is assumed", "pod", klog.KObj(pod))
	}
	if isAssumed {
		// Assumed pod is deleted. We should handle that differently,
		// because we can't delete such pod from any structure directly.
		sched.handleAssumedPodDeletion(pod)
	} else if pod.Status.NominatedNodeName != "" {
		// When a pod that had nominated node is deleted, it can unblock scheduling of other pods,
		// because the lower or equal priority pods treat such a pod as if it was assigned.
		// Note that a nominated pod can fall into `handleAssumedPodDeletion` case as well,
		// but in that case the `MoveAllToActiveOrBackoffQueue` already covered lower priority pods.
		sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, framework.EventAssignedPodDelete, pod, nil, getLEPriorityPreCheck(corev1helpers.PodPriority(pod)))
	}
}

// getLEPriorityPreCheck is a PreEnqueueCheck function that selects only lower or equal priority pods.
func getLEPriorityPreCheck(priority int32) queue.PreEnqueueCheck {
	return func(pod *v1.Pod) bool {
		return corev1helpers.PodPriority(pod) <= priority
	}
}

func (sched *Scheduler) addAssignedPodToCache(pod *v1.Pod) {
	start := time.Now()
	defer metrics.EventHandlingLatency.WithLabelValues(framework.EventAssignedPodAdd.Label()).Observe(metrics.SinceInSeconds(start))

	logger := sched.logger

	logger.V(3).Info("Add event for scheduled pod", "pod", klog.KObj(pod))
	if err := sched.Cache.AddPod(logger, pod); err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Scheduler cache AddPod failed", "pod", klog.KObj(pod))
	}

	// SchedulingQueue.AssignedPodAdded has a problem:
	// It internally pre-filters Pods to move to activeQ,
	// while taking only in-tree plugins into consideration.
	// Consequently, if custom plugins that subscribes Pod/Add events reject Pods,
	// those Pods will never be requeued to activeQ by an assigned Pod related events,
	// and they may be stuck in unschedulableQ.
	//
	// Here we use MoveAllToActiveOrBackoffQueue only when QueueingHint is enabled.
	// (We cannot switch to MoveAllToActiveOrBackoffQueue right away because of throughput concern.)
	if utilfeature.DefaultFeatureGate.Enabled(features.SchedulerQueueingHints) {
		sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, framework.EventAssignedPodAdd, nil, pod, nil)
	} else {
		sched.SchedulingQueue.AssignedPodAdded(logger, pod)
	}
}

func (sched *Scheduler) updateAssignedPodInCache(oldPod, newPod *v1.Pod) {
	start := time.Now()
	defer metrics.EventHandlingLatency.WithLabelValues(framework.EventAssignedPodUpdate.Label()).Observe(metrics.SinceInSeconds(start))

	logger := sched.logger

	if sched.APIDispatcher != nil {
		// If the API dispatcher is available, sync the new pod with the details.
		// However, at the moment the updated newPod is discarded and this logic will be handled in the future releases.
		_ = sched.syncPodWithDispatcher(newPod)
	}

	logger.V(4).Info("Update event for scheduled pod", "pod", klog.KObj(oldPod))
	if err := sched.Cache.UpdatePod(logger, oldPod, newPod); err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Scheduler cache UpdatePod failed", "pod", klog.KObj(oldPod))
	}

	events := framework.PodSchedulingPropertiesChange(newPod, oldPod)

	// Save the time it takes to update the pod in the cache.
	updatingDuration := metrics.SinceInSeconds(start)

	for _, evt := range events {
		startMoving := time.Now()
		// SchedulingQueue.AssignedPodUpdated has a problem:
		// It internally pre-filters Pods to move to activeQ,
		// while taking only in-tree plugins into consideration.
		// Consequently, if custom plugins that subscribes Pod/Update events reject Pods,
		// those Pods will never be requeued to activeQ by an assigned Pod related events,
		// and they may be stuck in unschedulableQ.
		//
		// Here we use MoveAllToActiveOrBackoffQueue only when QueueingHint is enabled.
		// (We cannot switch to MoveAllToActiveOrBackoffQueue right away because of throughput concern.)
		if utilfeature.DefaultFeatureGate.Enabled(features.SchedulerQueueingHints) {
			sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, evt, oldPod, newPod, nil)
		} else {
			sched.SchedulingQueue.AssignedPodUpdated(logger, oldPod, newPod, evt)
		}
		movingDuration := metrics.SinceInSeconds(startMoving)
		metrics.EventHandlingLatency.WithLabelValues(evt.Label()).Observe(updatingDuration + movingDuration)
	}
}

func (sched *Scheduler) deleteAssignedPodFromCache(pod *v1.Pod) {
	start := time.Now()
	defer metrics.EventHandlingLatency.WithLabelValues(framework.EventAssignedPodDelete.Label()).Observe(metrics.SinceInSeconds(start))

	logger := sched.logger

	logger.V(3).Info("Delete event for scheduled pod", "pod", klog.KObj(pod))
	if err := sched.Cache.RemovePod(logger, pod); err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Scheduler cache RemovePod failed", "pod", klog.KObj(pod))
	}

	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, framework.EventAssignedPodDelete, pod, nil, nil)
}

// assignedPod selects pods that are assigned (scheduled and running).
func assignedPod(pod *v1.Pod) bool {
	return len(pod.Spec.NodeName) != 0
}

// responsibleForPod returns true if the pod has asked to be scheduled by the given scheduler.
func responsibleForPod(pod *v1.Pod, profiles profile.Map) bool {
	return profiles.HandlesSchedulerName(pod.Spec.SchedulerName)
}

const (
	// syncedPollPeriod controls how often you look at the status of your sync funcs
	syncedPollPeriod = 100 * time.Millisecond
)

// WaitForHandlersSync waits for EventHandlers to sync.
// It returns true if it was successful, false if the controller should shut down
func (sched *Scheduler) WaitForHandlersSync(ctx context.Context) error {
	return wait.PollUntilContextCancel(ctx, syncedPollPeriod, true, func(ctx context.Context) (done bool, err error) {
		for _, handler := range sched.registeredHandlers {
			if !handler.HasSynced() {
				return false, nil
			}
		}
		return true, nil
	})
}

// addAllEventHandlers is a helper function used in tests and in Scheduler
// to add event handlers for various informers.
func addAllEventHandlers(
	sched *Scheduler,
	informerFactory informers.SharedInformerFactory,
	dynInformerFactory dynamicinformer.DynamicSharedInformerFactory,
	resourceClaimCache *assumecache.AssumeCache,
	resourceSliceTracker *resourceslicetracker.Tracker,
	draManager fwk.SharedDRAManager,
	gvkMap map[fwk.EventResource]fwk.ActionType,
) error {
	var (
		handlerRegistration cache.ResourceEventHandlerRegistration
		err                 error
		handlers            []cache.ResourceEventHandlerRegistration
	)

	logger := sched.logger

	if handlerRegistration, err = informerFactory.Core().V1().Pods().Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    sched.addPod,
		UpdateFunc: sched.updatePod,
		DeleteFunc: sched.deletePod,
	}); err != nil {
		return err
	}
	handlers = append(handlers, handlerRegistration)

	if handlerRegistration, err = informerFactory.Core().V1().Nodes().Informer().AddEventHandler(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    sched.addNodeToCache,
			UpdateFunc: sched.updateNodeInCache,
			DeleteFunc: sched.deleteNodeFromCache,
		},
	); err != nil {
		return err
	}
	handlers = append(handlers, handlerRegistration)

	buildEvtResHandler := func(at fwk.ActionType, resource fwk.EventResource) cache.ResourceEventHandlerFuncs {
		funcs := cache.ResourceEventHandlerFuncs{}
		if at&fwk.Add != 0 {
			evt := fwk.ClusterEvent{Resource: resource, ActionType: fwk.Add}
			funcs.AddFunc = func(obj interface{}) {
				start := time.Now()
				defer metrics.EventHandlingLatency.WithLabelValues(evt.Label()).Observe(metrics.SinceInSeconds(start))
				if resource == fwk.StorageClass && !utilfeature.DefaultFeatureGate.Enabled(features.SchedulerQueueingHints) {
					sc, ok := obj.(*storagev1.StorageClass)
					if !ok {
						utilruntime.HandleErrorWithLogger(logger, nil, "Cannot convert to *storagev1.StorageClass", "obj", obj)
						return
					}

					// CheckVolumeBindingPred fails if pod has unbound immediate PVCs. If these
					// PVCs have specified StorageClass name, creating StorageClass objects
					// with late binding will cause predicates to pass, so we need to move pods
					// to active queue.
					// We don't need to invalidate cached results because results will not be
					// cached for pod that has unbound immediate PVCs.
					if sc.VolumeBindingMode == nil || *sc.VolumeBindingMode != storagev1.VolumeBindingWaitForFirstConsumer {
						return
					}
				}
				sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, evt, nil, obj, nil)
			}
		}
		if at&fwk.Update != 0 {
			evt := fwk.ClusterEvent{Resource: resource, ActionType: fwk.Update}
			funcs.UpdateFunc = func(old, obj interface{}) {
				start := time.Now()
				sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, evt, old, obj, nil)
				metrics.EventHandlingLatency.WithLabelValues(evt.Label()).Observe(metrics.SinceInSeconds(start))
			}
		}
		if at&fwk.Delete != 0 {
			evt := fwk.ClusterEvent{Resource: resource, ActionType: fwk.Delete}
			funcs.DeleteFunc = func(obj interface{}) {
				start := time.Now()
				sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, evt, obj, nil, nil)
				metrics.EventHandlingLatency.WithLabelValues(evt.Label()).Observe(metrics.SinceInSeconds(start))
			}
		}
		return funcs
	}

	for gvk, at := range gvkMap {
		switch gvk {
		case fwk.Node, fwk.Pod:
			// Do nothing.
		case fwk.CSINode:
			if handlerRegistration, err = informerFactory.Storage().V1().CSINodes().Informer().AddEventHandler(
				buildEvtResHandler(at, fwk.CSINode),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		case fwk.CSIDriver:
			if handlerRegistration, err = informerFactory.Storage().V1().CSIDrivers().Informer().AddEventHandler(
				buildEvtResHandler(at, fwk.CSIDriver),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		case fwk.CSIStorageCapacity:
			if handlerRegistration, err = informerFactory.Storage().V1().CSIStorageCapacities().Informer().AddEventHandler(
				buildEvtResHandler(at, fwk.CSIStorageCapacity),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		case fwk.PersistentVolume:
			// MaxPDVolumeCountPredicate: since it relies on the counts of PV.
			//
			// PvAdd: Pods created when there are no PVs available will be stuck in
			// unschedulable queue. But unbound PVs created for static provisioning and
			// delay binding storage class are skipped in PV controller dynamic
			// provisioning and binding process, will not trigger events to schedule pod
			// again. So we need to move pods to active queue on PV add for this
			// scenario.
			//
			// PvUpdate: Scheduler.bindVolumesWorker may fail to update assumed pod volume
			// bindings due to conflicts if PVs are updated by PV controller or other
			// parties, then scheduler will add pod back to unschedulable queue. We
			// need to move pods to active queue on PV update for this scenario.
			if handlerRegistration, err = informerFactory.Core().V1().PersistentVolumes().Informer().AddEventHandler(
				buildEvtResHandler(at, fwk.PersistentVolume),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		case fwk.PersistentVolumeClaim:
			// MaxPDVolumeCountPredicate: add/update PVC will affect counts of PV when it is bound.
			if handlerRegistration, err = informerFactory.Core().V1().PersistentVolumeClaims().Informer().AddEventHandler(
				buildEvtResHandler(at, fwk.PersistentVolumeClaim),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		case fwk.ResourceClaim:
			if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
				handlerRegistration = resourceClaimCache.AddEventHandler(
					buildEvtResHandler(at, fwk.ResourceClaim),
				)
				handlers = append(handlers, handlerRegistration)
			}
		case fwk.ResourceSlice:
			if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
				if handlerRegistration, err = resourceSliceTracker.AddEventHandler(
					buildEvtResHandler(at, fwk.ResourceSlice),
				); err != nil {
					return err
				}
				handlers = append(handlers, handlerRegistration)
			}
		case fwk.DeviceClass:
			if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
				handler := cache.ResourceEventHandler(buildEvtResHandler(at, fwk.DeviceClass))
				if utilfeature.DefaultFeatureGate.Enabled(features.DRAExtendedResource) {
					// Inject updating of the cache before the scheduler event handlers ("chaining")
					// to ensure that the cache gets updated before the scheduler kicks off
					// pod scheduling based on a DeviceClass event.
					//
					// We know that this is a DefaultDRAManager and we know that it
					// uses an ExtendedResourceCache, so no need for type checks.
					erCache := draManager.DeviceClassResolver().(*extendedresourcecache.ExtendedResourceCache)
					erCache.AddEventHandler(handler)
					handler = erCache
				}
				if handlerRegistration, err = informerFactory.Resource().V1().DeviceClasses().Informer().AddEventHandler(
					handler,
				); err != nil {
					return err
				}
				handlers = append(handlers, handlerRegistration)
			}
		case fwk.StorageClass:
			if handlerRegistration, err = informerFactory.Storage().V1().StorageClasses().Informer().AddEventHandler(
				buildEvtResHandler(at, fwk.StorageClass),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		case fwk.VolumeAttachment:
			if handlerRegistration, err = informerFactory.Storage().V1().VolumeAttachments().Informer().AddEventHandler(
				buildEvtResHandler(at, fwk.VolumeAttachment),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		case fwk.Workload:
			if utilfeature.DefaultFeatureGate.Enabled(features.GenericWorkload) {
				if handlerRegistration, err = informerFactory.Scheduling().V1alpha1().Workloads().Informer().AddEventHandler(
					buildEvtResHandler(at, fwk.Workload),
				); err != nil {
					return err
				}
				handlers = append(handlers, handlerRegistration)
			}
		default:
			// Tests may not instantiate dynInformerFactory.
			if dynInformerFactory == nil {
				continue
			}
			// GVK is expected to be at least 3-folded, separated by dots.
			// <kind in plural>.<version>.<group>
			// Valid examples:
			// - foos.v1.example.com
			// - bars.v1beta1.a.b.c
			// Invalid examples:
			// - foos.v1 (2 sections)
			// - foo.v1.example.com (the first section should be plural)
			if strings.Count(string(gvk), ".") < 2 {
				utilruntime.HandleErrorWithLogger(logger, nil, "Incorrect event registration", "gvk", gvk)
				continue
			}
			// Fall back to try dynamic informers.
			gvr, _ := schema.ParseResourceArg(string(gvk))
			dynInformer := dynInformerFactory.ForResource(*gvr).Informer()
			if handlerRegistration, err = dynInformer.AddEventHandler(
				buildEvtResHandler(at, gvk),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		}
	}
	sched.registeredHandlers = handlers
	return nil
}

func preCheckForNode(logger klog.Logger, nodeInfo *framework.NodeInfo) queue.PreEnqueueCheck {
	if utilfeature.DefaultFeatureGate.Enabled(features.SchedulerQueueingHints) {
		// QHint is initially created from the motivation of replacing this preCheck.
		// It assumes that the scheduler only has in-tree plugins, which is problematic for our extensibility.
		// Here, we skip preCheck if QHint is enabled, and we eventually remove it after QHint is graduated.
		return nil
	}

	// Note: the following checks doesn't take preemption into considerations, in very rare
	// cases (e.g., node resizing), "pod" may still fail a check but preemption helps. We deliberately
	// chose to ignore those cases as unschedulable pods will be re-queued eventually.
	return func(pod *v1.Pod) bool {
		admissionResults := AdmissionCheck(pod, nodeInfo, false)
		if len(admissionResults) != 0 {
			return false
		}
		_, isUntolerated := corev1helpers.FindMatchingUntoleratedTaint(logger, nodeInfo.Node().Spec.Taints, pod.Spec.Tolerations,
			helper.DoNotScheduleTaintsFilterFunc(),
			utilfeature.DefaultFeatureGate.Enabled(features.TaintTolerationComparisonOperators))
		return !isUntolerated
	}
}

// AdmissionCheck calls the filtering logic of noderesources/nodeport/nodeAffinity/nodename
// and returns the failure reasons. It's used in kubelet(pkg/kubelet/lifecycle/predicate.go) and scheduler.
// It returns the first failure if `includeAllFailures` is set to false; otherwise
// returns all failures.
func AdmissionCheck(pod *v1.Pod, nodeInfo *framework.NodeInfo, includeAllFailures bool) []AdmissionResult {
	var admissionResults []AdmissionResult
	insufficientResources := noderesources.Fits(pod, nodeInfo, nil, noderesources.ResourceRequestsOptions{
		EnablePodLevelResources:   utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources),
		EnableDRAExtendedResource: utilfeature.DefaultFeatureGate.Enabled(features.DRAExtendedResource),
	})
	if len(insufficientResources) != 0 {
		for i := range insufficientResources {
			admissionResults = append(admissionResults, AdmissionResult{InsufficientResource: &insufficientResources[i]})
		}
		if !includeAllFailures {
			return admissionResults
		}
	}

	if matches, _ := corev1nodeaffinity.GetRequiredNodeAffinity(pod).Match(nodeInfo.Node()); !matches {
		admissionResults = append(admissionResults, AdmissionResult{Name: nodeaffinity.Name, Reason: nodeaffinity.ErrReasonPod})
		if !includeAllFailures {
			return admissionResults
		}
	}
	if !nodename.Fits(pod, nodeInfo) {
		admissionResults = append(admissionResults, AdmissionResult{Name: nodename.Name, Reason: nodename.ErrReason})
		if !includeAllFailures {
			return admissionResults
		}
	}
	if !nodeports.Fits(pod, nodeInfo) {
		admissionResults = append(admissionResults, AdmissionResult{Name: nodeports.Name, Reason: nodeports.ErrReason})
		if !includeAllFailures {
			return admissionResults
		}
	}
	return admissionResults
}

// AdmissionResult describes the reason why Scheduler can't admit the pod.
// If the reason is a resource fit one, then AdmissionResult.InsufficientResource includes the details.
type AdmissionResult struct {
	Name                 string
	Reason               string
	InsufficientResource *noderesources.InsufficientResource
}
