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
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	corev1nodeaffinity "k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/internal/queue"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

func (sched *Scheduler) onStorageClassAdd(obj interface{}) {
	logger := sched.logger
	sc, ok := obj.(*storagev1.StorageClass)
	if !ok {
		logger.Error(nil, "Cannot convert to *storagev1.StorageClass", "obj", obj)
		return
	}

	// CheckVolumeBindingPred fails if pod has unbound immediate PVCs. If these
	// PVCs have specified StorageClass name, creating StorageClass objects
	// with late binding will cause predicates to pass, so we need to move pods
	// to active queue.
	// We don't need to invalidate cached results because results will not be
	// cached for pod that has unbound immediate PVCs.
	if sc.VolumeBindingMode != nil && *sc.VolumeBindingMode == storagev1.VolumeBindingWaitForFirstConsumer {
		sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, queue.StorageClassAdd, nil, sc, nil)
	}
}

func (sched *Scheduler) addNodeToCache(obj interface{}) {
	logger := sched.logger
	node, ok := obj.(*v1.Node)
	if !ok {
		logger.Error(nil, "Cannot convert to *v1.Node", "obj", obj)
		return
	}

	logger.V(3).Info("Add event for node", "node", klog.KObj(node))
	nodeInfo := sched.Cache.AddNode(logger, node)
	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, queue.NodeAdd, nil, node, preCheckForNode(nodeInfo))
}

func (sched *Scheduler) updateNodeInCache(oldObj, newObj interface{}) {
	logger := sched.logger
	oldNode, ok := oldObj.(*v1.Node)
	if !ok {
		logger.Error(nil, "Cannot convert oldObj to *v1.Node", "oldObj", oldObj)
		return
	}
	newNode, ok := newObj.(*v1.Node)
	if !ok {
		logger.Error(nil, "Cannot convert newObj to *v1.Node", "newObj", newObj)
		return
	}

	logger.V(4).Info("Update event for node", "node", klog.KObj(newNode))
	nodeInfo := sched.Cache.UpdateNode(logger, oldNode, newNode)
	// Only requeue unschedulable pods if the node became more schedulable.
	for _, evt := range nodeSchedulingPropertiesChange(newNode, oldNode) {
		sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, evt, oldNode, newNode, preCheckForNode(nodeInfo))
	}
}

func (sched *Scheduler) deleteNodeFromCache(obj interface{}) {
	logger := sched.logger
	var node *v1.Node
	switch t := obj.(type) {
	case *v1.Node:
		node = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		node, ok = t.Obj.(*v1.Node)
		if !ok {
			logger.Error(nil, "Cannot convert to *v1.Node", "obj", t.Obj)
			return
		}
	default:
		logger.Error(nil, "Cannot convert to *v1.Node", "obj", t)
		return
	}

	logger.V(3).Info("Delete event for node", "node", klog.KObj(node))
	if err := sched.Cache.RemoveNode(logger, node); err != nil {
		logger.Error(err, "Scheduler cache RemoveNode failed")
	}
}

func (sched *Scheduler) addPodToSchedulingQueue(obj interface{}) {
	logger := sched.logger
	pod := obj.(*v1.Pod)
	logger.V(3).Info("Add event for unscheduled pod", "pod", klog.KObj(pod))
	if err := sched.SchedulingQueue.Add(logger, pod); err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to queue %T: %v", obj, err))
	}
}

func (sched *Scheduler) updatePodInSchedulingQueue(oldObj, newObj interface{}) {
	logger := sched.logger
	oldPod, newPod := oldObj.(*v1.Pod), newObj.(*v1.Pod)
	// Bypass update event that carries identical objects; otherwise, a duplicated
	// Pod may go through scheduling and cause unexpected behavior (see #96071).
	if oldPod.ResourceVersion == newPod.ResourceVersion {
		return
	}

	isAssumed, err := sched.Cache.IsAssumedPod(newPod)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to check whether pod %s/%s is assumed: %v", newPod.Namespace, newPod.Name, err))
	}
	if isAssumed {
		return
	}

	logger.V(4).Info("Update event for unscheduled pod", "pod", klog.KObj(newPod))
	if err := sched.SchedulingQueue.Update(logger, oldPod, newPod); err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to update %T: %v", newObj, err))
	}
}

func (sched *Scheduler) deletePodFromSchedulingQueue(obj interface{}) {
	logger := sched.logger
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

	logger.V(3).Info("Delete event for unscheduled pod", "pod", klog.KObj(pod))
	if err := sched.SchedulingQueue.Delete(pod); err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to dequeue %T: %v", obj, err))
	}
	fwk, err := sched.frameworkForPod(pod)
	if err != nil {
		// This shouldn't happen, because we only accept for scheduling the pods
		// which specify a scheduler name that matches one of the profiles.
		logger.Error(err, "Unable to get profile", "pod", klog.KObj(pod))
		return
	}
	// If a waiting pod is rejected, it indicates it's previously assumed and we're
	// removing it from the scheduler cache. In this case, signal a AssignedPodDelete
	// event to immediately retry some unscheduled Pods.
	if fwk.RejectWaitingPod(pod.UID) {
		sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, queue.AssignedPodDelete, pod, nil, nil)
	}
}

func (sched *Scheduler) addPodToCache(obj interface{}) {
	logger := sched.logger
	pod, ok := obj.(*v1.Pod)
	if !ok {
		logger.Error(nil, "Cannot convert to *v1.Pod", "obj", obj)
		return
	}

	logger.V(3).Info("Add event for scheduled pod", "pod", klog.KObj(pod))
	if err := sched.Cache.AddPod(logger, pod); err != nil {
		logger.Error(err, "Scheduler cache AddPod failed", "pod", klog.KObj(pod))
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
		sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, queue.AssignedPodAdd, nil, pod, nil)
	} else {
		sched.SchedulingQueue.AssignedPodAdded(logger, pod)
	}
}

func (sched *Scheduler) updatePodInCache(oldObj, newObj interface{}) {
	logger := sched.logger
	oldPod, ok := oldObj.(*v1.Pod)
	if !ok {
		logger.Error(nil, "Cannot convert oldObj to *v1.Pod", "oldObj", oldObj)
		return
	}
	newPod, ok := newObj.(*v1.Pod)
	if !ok {
		logger.Error(nil, "Cannot convert newObj to *v1.Pod", "newObj", newObj)
		return
	}

	logger.V(4).Info("Update event for scheduled pod", "pod", klog.KObj(oldPod))
	if err := sched.Cache.UpdatePod(logger, oldPod, newPod); err != nil {
		logger.Error(err, "Scheduler cache UpdatePod failed", "pod", klog.KObj(oldPod))
	}

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
		sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, queue.AssignedPodUpdate, oldPod, newPod, nil)
	} else {
		sched.SchedulingQueue.AssignedPodUpdated(logger, oldPod, newPod)
	}
}

func (sched *Scheduler) deletePodFromCache(obj interface{}) {
	logger := sched.logger
	var pod *v1.Pod
	switch t := obj.(type) {
	case *v1.Pod:
		pod = t
	case cache.DeletedFinalStateUnknown:
		var ok bool
		pod, ok = t.Obj.(*v1.Pod)
		if !ok {
			logger.Error(nil, "Cannot convert to *v1.Pod", "obj", t.Obj)
			return
		}
	default:
		logger.Error(nil, "Cannot convert to *v1.Pod", "obj", t)
		return
	}

	logger.V(3).Info("Delete event for scheduled pod", "pod", klog.KObj(pod))
	if err := sched.Cache.RemovePod(logger, pod); err != nil {
		logger.Error(err, "Scheduler cache RemovePod failed", "pod", klog.KObj(pod))
	}

	sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, queue.AssignedPodDelete, pod, nil, nil)
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
	gvkMap map[framework.GVK]framework.ActionType,
) error {
	var (
		handlerRegistration cache.ResourceEventHandlerRegistration
		err                 error
		handlers            []cache.ResourceEventHandlerRegistration
	)
	// scheduled pod cache
	if handlerRegistration, err = informerFactory.Core().V1().Pods().Informer().AddEventHandler(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.Pod:
					return assignedPod(t)
				case cache.DeletedFinalStateUnknown:
					if _, ok := t.Obj.(*v1.Pod); ok {
						// The carried object may be stale, so we don't use it to check if
						// it's assigned or not. Attempting to cleanup anyways.
						return true
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
	); err != nil {
		return err
	}
	handlers = append(handlers, handlerRegistration)

	// unscheduled pod queue
	if handlerRegistration, err = informerFactory.Core().V1().Pods().Informer().AddEventHandler(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.Pod:
					return !assignedPod(t) && responsibleForPod(t, sched.Profiles)
				case cache.DeletedFinalStateUnknown:
					if pod, ok := t.Obj.(*v1.Pod); ok {
						// The carried object may be stale, so we don't use it to check if
						// it's assigned or not.
						return responsibleForPod(pod, sched.Profiles)
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
	); err != nil {
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

	logger := sched.logger
	buildEvtResHandler := func(at framework.ActionType, gvk framework.GVK, shortGVK string) cache.ResourceEventHandlerFuncs {
		funcs := cache.ResourceEventHandlerFuncs{}
		if at&framework.Add != 0 {
			evt := framework.ClusterEvent{Resource: gvk, ActionType: framework.Add, Label: fmt.Sprintf("%vAdd", shortGVK)}
			funcs.AddFunc = func(obj interface{}) {
				sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, evt, nil, obj, nil)
			}
		}
		if at&framework.Update != 0 {
			evt := framework.ClusterEvent{Resource: gvk, ActionType: framework.Update, Label: fmt.Sprintf("%vUpdate", shortGVK)}
			funcs.UpdateFunc = func(old, obj interface{}) {
				sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, evt, old, obj, nil)
			}
		}
		if at&framework.Delete != 0 {
			evt := framework.ClusterEvent{Resource: gvk, ActionType: framework.Delete, Label: fmt.Sprintf("%vDelete", shortGVK)}
			funcs.DeleteFunc = func(obj interface{}) {
				sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, evt, obj, nil, nil)
			}
		}
		return funcs
	}

	for gvk, at := range gvkMap {
		switch gvk {
		case framework.Node, framework.Pod:
			// Do nothing.
		case framework.CSINode:
			if handlerRegistration, err = informerFactory.Storage().V1().CSINodes().Informer().AddEventHandler(
				buildEvtResHandler(at, framework.CSINode, "CSINode"),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		case framework.CSIDriver:
			if handlerRegistration, err = informerFactory.Storage().V1().CSIDrivers().Informer().AddEventHandler(
				buildEvtResHandler(at, framework.CSIDriver, "CSIDriver"),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		case framework.CSIStorageCapacity:
			if handlerRegistration, err = informerFactory.Storage().V1().CSIStorageCapacities().Informer().AddEventHandler(
				buildEvtResHandler(at, framework.CSIStorageCapacity, "CSIStorageCapacity"),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		case framework.PersistentVolume:
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
				buildEvtResHandler(at, framework.PersistentVolume, "Pv"),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		case framework.PersistentVolumeClaim:
			// MaxPDVolumeCountPredicate: add/update PVC will affect counts of PV when it is bound.
			if handlerRegistration, err = informerFactory.Core().V1().PersistentVolumeClaims().Informer().AddEventHandler(
				buildEvtResHandler(at, framework.PersistentVolumeClaim, "Pvc"),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		case framework.PodSchedulingContext:
			if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
				if handlerRegistration, err = informerFactory.Resource().V1alpha2().PodSchedulingContexts().Informer().AddEventHandler(
					buildEvtResHandler(at, framework.PodSchedulingContext, "PodSchedulingContext"),
				); err != nil {
					return err
				}
				handlers = append(handlers, handlerRegistration)
			}
		case framework.ResourceClaim:
			if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
				handlerRegistration = resourceClaimCache.AddEventHandler(
					buildEvtResHandler(at, framework.ResourceClaim, "ResourceClaim"),
				)
				handlers = append(handlers, handlerRegistration)
			}
		case framework.ResourceClass:
			if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
				if handlerRegistration, err = informerFactory.Resource().V1alpha2().ResourceClasses().Informer().AddEventHandler(
					buildEvtResHandler(at, framework.ResourceClass, "ResourceClass"),
				); err != nil {
					return err
				}
				handlers = append(handlers, handlerRegistration)
			}
		case framework.ResourceClaimParameters:
			if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
				if handlerRegistration, err = informerFactory.Resource().V1alpha2().ResourceClaimParameters().Informer().AddEventHandler(
					buildEvtResHandler(at, framework.ResourceClaimParameters, "ResourceClaimParameters"),
				); err != nil {
					return err
				}
				handlers = append(handlers, handlerRegistration)
			}
		case framework.ResourceClassParameters:
			if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
				if handlerRegistration, err = informerFactory.Resource().V1alpha2().ResourceClassParameters().Informer().AddEventHandler(
					buildEvtResHandler(at, framework.ResourceClassParameters, "ResourceClassParameters"),
				); err != nil {
					return err
				}
				handlers = append(handlers, handlerRegistration)
			}
		case framework.StorageClass:
			if at&framework.Add != 0 {
				if handlerRegistration, err = informerFactory.Storage().V1().StorageClasses().Informer().AddEventHandler(
					cache.ResourceEventHandlerFuncs{
						AddFunc: sched.onStorageClassAdd,
					},
				); err != nil {
					return err
				}
				handlers = append(handlers, handlerRegistration)
			}
			if at&framework.Update != 0 {
				if handlerRegistration, err = informerFactory.Storage().V1().StorageClasses().Informer().AddEventHandler(
					cache.ResourceEventHandlerFuncs{
						UpdateFunc: func(old, obj interface{}) {
							sched.SchedulingQueue.MoveAllToActiveOrBackoffQueue(logger, queue.StorageClassUpdate, old, obj, nil)
						},
					},
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
				logger.Error(nil, "incorrect event registration", "gvk", gvk)
				continue
			}
			// Fall back to try dynamic informers.
			gvr, _ := schema.ParseResourceArg(string(gvk))
			dynInformer := dynInformerFactory.ForResource(*gvr).Informer()
			if handlerRegistration, err = dynInformer.AddEventHandler(
				buildEvtResHandler(at, gvk, strings.Title(gvr.Resource)),
			); err != nil {
				return err
			}
			handlers = append(handlers, handlerRegistration)
		}
	}
	sched.registeredHandlers = handlers
	return nil
}

func nodeSchedulingPropertiesChange(newNode *v1.Node, oldNode *v1.Node) []framework.ClusterEvent {
	var events []framework.ClusterEvent

	if nodeSpecUnschedulableChanged(newNode, oldNode) {
		events = append(events, queue.NodeSpecUnschedulableChange)
	}
	if nodeAllocatableChanged(newNode, oldNode) {
		events = append(events, queue.NodeAllocatableChange)
	}
	if nodeLabelsChanged(newNode, oldNode) {
		events = append(events, queue.NodeLabelChange)
	}
	if nodeTaintsChanged(newNode, oldNode) {
		events = append(events, queue.NodeTaintChange)
	}
	if nodeConditionsChanged(newNode, oldNode) {
		events = append(events, queue.NodeConditionChange)
	}
	if nodeAnnotationsChanged(newNode, oldNode) {
		events = append(events, queue.NodeAnnotationChange)
	}

	return events
}

func nodeAllocatableChanged(newNode *v1.Node, oldNode *v1.Node) bool {
	return !equality.Semantic.DeepEqual(oldNode.Status.Allocatable, newNode.Status.Allocatable)
}

func nodeLabelsChanged(newNode *v1.Node, oldNode *v1.Node) bool {
	return !equality.Semantic.DeepEqual(oldNode.GetLabels(), newNode.GetLabels())
}

func nodeTaintsChanged(newNode *v1.Node, oldNode *v1.Node) bool {
	return !equality.Semantic.DeepEqual(newNode.Spec.Taints, oldNode.Spec.Taints)
}

func nodeConditionsChanged(newNode *v1.Node, oldNode *v1.Node) bool {
	strip := func(conditions []v1.NodeCondition) map[v1.NodeConditionType]v1.ConditionStatus {
		conditionStatuses := make(map[v1.NodeConditionType]v1.ConditionStatus, len(conditions))
		for i := range conditions {
			conditionStatuses[conditions[i].Type] = conditions[i].Status
		}
		return conditionStatuses
	}
	return !equality.Semantic.DeepEqual(strip(oldNode.Status.Conditions), strip(newNode.Status.Conditions))
}

func nodeSpecUnschedulableChanged(newNode *v1.Node, oldNode *v1.Node) bool {
	return newNode.Spec.Unschedulable != oldNode.Spec.Unschedulable && !newNode.Spec.Unschedulable
}

func nodeAnnotationsChanged(newNode *v1.Node, oldNode *v1.Node) bool {
	return !equality.Semantic.DeepEqual(oldNode.GetAnnotations(), newNode.GetAnnotations())
}

func preCheckForNode(nodeInfo *framework.NodeInfo) queue.PreEnqueueCheck {
	// Note: the following checks doesn't take preemption into considerations, in very rare
	// cases (e.g., node resizing), "pod" may still fail a check but preemption helps. We deliberately
	// chose to ignore those cases as unschedulable pods will be re-queued eventually.
	return func(pod *v1.Pod) bool {
		admissionResults := AdmissionCheck(pod, nodeInfo, false)
		if len(admissionResults) != 0 {
			return false
		}
		_, isUntolerated := corev1helpers.FindMatchingUntoleratedTaint(nodeInfo.Node().Spec.Taints, pod.Spec.Tolerations, func(t *v1.Taint) bool {
			return t.Effect == v1.TaintEffectNoSchedule
		})
		return !isUntolerated
	}
}

// AdmissionCheck calls the filtering logic of noderesources/nodeport/nodeAffinity/nodename
// and returns the failure reasons. It's used in kubelet(pkg/kubelet/lifecycle/predicate.go) and scheduler.
// It returns the first failure if `includeAllFailures` is set to false; otherwise
// returns all failures.
func AdmissionCheck(pod *v1.Pod, nodeInfo *framework.NodeInfo, includeAllFailures bool) []AdmissionResult {
	var admissionResults []AdmissionResult
	insufficientResources := noderesources.Fits(pod, nodeInfo)
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
