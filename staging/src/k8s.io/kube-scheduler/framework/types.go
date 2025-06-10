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

package framework

import (
	"fmt"

	v1 "k8s.io/api/core/v1"

	"k8s.io/klog/v2"
)

// ActionType is an integer to represent one type of resource change.
// Different ActionTypes can be bit-wised to compose new semantics.
type ActionType int64

// Constants for ActionTypes.
// CAUTION for contributors: When you add a new ActionType, you must update the following:
// - The list of basicActionTypes, podActionTypes, and nodeActionTypes at k/k/pkg/scheduler/framework/types.go
// - String() method.
const (
	Add ActionType = 1 << iota
	Delete

	// UpdateNodeXYZ is only applicable for Node events.
	// If you use UpdateNodeXYZ,
	// your plugin's QueueingHint is only executed for the specific sub-Update event.
	// It's better to narrow down the scope of the event by using them instead of just using Update event
	// for better performance in requeueing.
	UpdateNodeAllocatable
	UpdateNodeLabel
	// UpdateNodeTaint is an update for node's taints or node.Spec.Unschedulable.
	UpdateNodeTaint
	UpdateNodeCondition
	UpdateNodeAnnotation

	// UpdatePodXYZ is only applicable for Pod events.
	// If you use UpdatePodXYZ,
	// your plugin's QueueingHint is only executed for the specific sub-Update event.
	// It's better to narrow down the scope of the event by using them instead of Update event
	// for better performance in requeueing.
	UpdatePodLabel
	// UpdatePodScaleDown is an update for pod's scale down (i.e., any resource request is reduced).
	UpdatePodScaleDown
	// UpdatePodToleration is an addition for pod's tolerations.
	// (Due to API validation, we can add, but cannot modify or remove tolerations.)
	UpdatePodToleration
	// UpdatePodSchedulingGatesEliminated is an update for pod's scheduling gates, which eliminates all scheduling gates in the Pod.
	UpdatePodSchedulingGatesEliminated
	// UpdatePodGeneratedResourceClaim is an update of the list of ResourceClaims generated for the pod.
	// Depends on the DynamicResourceAllocation feature gate.
	UpdatePodGeneratedResourceClaim

	All ActionType = 1<<iota - 1

	// Use the general Update type if you don't either know or care the specific sub-Update type to use.
	Update = UpdateNodeAllocatable | UpdateNodeLabel | UpdateNodeTaint | UpdateNodeCondition | UpdateNodeAnnotation | UpdatePodLabel | UpdatePodScaleDown | UpdatePodToleration | UpdatePodSchedulingGatesEliminated | UpdatePodGeneratedResourceClaim

	// None is a special ActionType that is only used internally.
	None ActionType = 0
)

func (a ActionType) String() string {
	switch a {
	case Add:
		return "Add"
	case Delete:
		return "Delete"
	case UpdateNodeAllocatable:
		return "UpdateNodeAllocatable"
	case UpdateNodeLabel:
		return "UpdateNodeLabel"
	case UpdateNodeTaint:
		return "UpdateNodeTaint"
	case UpdateNodeCondition:
		return "UpdateNodeCondition"
	case UpdateNodeAnnotation:
		return "UpdateNodeAnnotation"
	case UpdatePodLabel:
		return "UpdatePodLabel"
	case UpdatePodScaleDown:
		return "UpdatePodScaleDown"
	case UpdatePodToleration:
		return "UpdatePodToleration"
	case UpdatePodSchedulingGatesEliminated:
		return "UpdatePodSchedulingGatesEliminated"
	case UpdatePodGeneratedResourceClaim:
		return "UpdatePodGeneratedResourceClaim"
	case All:
		return "All"
	case Update:
		return "Update"
	}

	// Shouldn't reach here.
	return ""
}

// EventResource is basically short for group/version/kind, which can uniquely represent a particular API resource.
type EventResource string

// Constants for GVKs.
//
// CAUTION for contributors: When you add a new EventResource, you must register a new one to allResources at k/k/pkg/scheduler/framework/types.go
//
// Note:
// - UpdatePodXYZ or UpdateNodeXYZ: triggered by updating particular parts of a Pod or a Node, e.g. updatePodLabel.
// Use specific events rather than general ones (updatePodLabel vs update) can make the requeueing process more efficient
// and consume less memory as less events will be cached at scheduler.
const (
	// There are a couple of notes about how the scheduler notifies the events of Pods:
	// - Add: add events could be triggered by either a newly created Pod or an existing Pod that is scheduled to a Node.
	// - Delete: delete events could be triggered by:
	//           - a Pod that is deleted
	//           - a Pod that was assumed, but gets un-assumed due to some errors in the binding cycle.
	//           - an existing Pod that was unscheduled but gets scheduled to a Node.
	//
	// Note that the Pod event type includes the events for the unscheduled Pod itself.
	// i.e., when unscheduled Pods are updated, the scheduling queue checks with Pod/Update QueueingHint(s) whether the update may make the pods schedulable,
	// and requeues them to activeQ/backoffQ when at least one QueueingHint(s) return Queue.
	// Plugins **have to** implement a QueueingHint for Pod/Update event
	// if the rejection from them could be resolved by updating unscheduled Pods themselves.
	// Example: Pods that require excessive resources may be rejected by the noderesources plugin,
	// if this unscheduled pod is updated to require fewer resources,
	// the previous rejection from noderesources plugin can be resolved.
	// this plugin would implement QueueingHint for Pod/Update event
	// that returns Queue when such label changes are made in unscheduled Pods.
	Pod EventResource = "Pod"

	// A note about NodeAdd event and UpdateNodeTaint event:
	// When QHint is disabled, NodeAdd often isn't worked expectedly because of the internal feature called preCheck.
	// It's definitely not something expected for plugin developers,
	// and registering UpdateNodeTaint event is the only mitigation for now.
	// So, kube-scheduler registers UpdateNodeTaint event for plugins that has NodeAdded event, but don't have UpdateNodeTaint event.
	// It has a bad impact for the requeuing efficiency though, a lot better than some Pods being stuck in the
	// unschedulable pod pool.
	// This problematic preCheck feature is disabled when QHint is enabled,
	// and eventually will be removed along with QHint graduation.
	// See: https://github.com/kubernetes/kubernetes/issues/110175
	Node                  EventResource = "Node"
	PersistentVolume      EventResource = "PersistentVolume"
	PersistentVolumeClaim EventResource = "PersistentVolumeClaim"
	CSINode               EventResource = "storage.k8s.io/CSINode"
	CSIDriver             EventResource = "storage.k8s.io/CSIDriver"
	VolumeAttachment      EventResource = "storage.k8s.io/VolumeAttachment"
	CSIStorageCapacity    EventResource = "storage.k8s.io/CSIStorageCapacity"
	StorageClass          EventResource = "storage.k8s.io/StorageClass"
	ResourceClaim         EventResource = "resource.k8s.io/ResourceClaim"
	ResourceSlice         EventResource = "resource.k8s.io/ResourceSlice"
	DeviceClass           EventResource = "resource.k8s.io/DeviceClass"

	// WildCard is a special EventResource to match all resources.
	// e.g., If you register `{Resource: "*", ActionType: All}` in EventsToRegister,
	// all coming clusterEvents will be admitted. Be careful to register it, it will
	// increase the computing pressure in requeueing unless you really need it.
	//
	// Meanwhile, if the coming clusterEvent is a wildcard one, all pods
	// will be moved from unschedulablePod pool to activeQ/backoffQ forcibly.
	WildCard EventResource = "*"
)

type ClusterEventWithHint struct {
	Event ClusterEvent
	// QueueingHintFn is executed for the Pod rejected by this plugin when the above Event happens,
	// and filters out events to reduce useless retry of Pod's scheduling.
	// It's an optional field. If not set,
	// the scheduling of Pods will be always retried with backoff when this Event happens.
	// (the same as Queue)
	QueueingHintFn QueueingHintFn
}

// QueueingHintFn returns a hint that signals whether the event can make a Pod,
// which was rejected by this plugin in the past scheduling cycle, schedulable or not.
// It's called before a Pod gets moved from unschedulableQ to backoffQ or activeQ.
// If it returns an error, we'll take the returned QueueingHint as `Queue` at the caller whatever we returned here so that
// we can prevent the Pod from being stuck in the unschedulable pod pool.
//
// - `pod`: the Pod to be enqueued, which is rejected by this plugin in the past.
// - `oldObj` `newObj`: the object involved in that event.
//   - For example, the given event is "Node deleted", the `oldObj` will be that deleted Node.
//   - `oldObj` is nil if the event is add event.
//   - `newObj` is nil if the event is delete event.
type QueueingHintFn func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (QueueingHint, error)

type QueueingHint int

const (
	// QueueSkip implies that the cluster event has no impact on
	// scheduling of the pod.
	QueueSkip QueueingHint = iota

	// Queue implies that the Pod may be schedulable by the event.
	Queue
)

func (s QueueingHint) String() string {
	switch s {
	case QueueSkip:
		return "QueueSkip"
	case Queue:
		return "Queue"
	}
	return ""
}

// ClusterEvent abstracts how a system resource's state gets changed.
// Resource represents the standard API resources such as Pod, Node, etc.
// ActionType denotes the specific change such as Add, Update or Delete.
type ClusterEvent struct {
	Resource   EventResource
	ActionType ActionType

	// CustomLabel describes this cluster event.
	// It's an optional field to control Label(), which is used in logging and metrics.
	// Normally, it's not necessary to set this field; only used for special events like UnschedulableTimeout.
	CustomLabel string
}

// Label is used for logging and metrics.
func (ce ClusterEvent) Label() string {
	if ce.CustomLabel != "" {
		return ce.CustomLabel
	}

	return fmt.Sprintf("%v%v", ce.Resource, ce.ActionType)
}
