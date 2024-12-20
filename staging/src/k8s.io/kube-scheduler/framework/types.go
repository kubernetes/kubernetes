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
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"

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

// NodeInfo is node level aggregated information.
type NodeInfo interface {
	// Node returns overall information about this node.
	Node() *v1.Node
	// GetPods returns Pods running on the node.
	GetPods() []PodInfo
	// GetPodsWithAffinity returns the subset of pods with affinity.
	GetPodsWithAffinity() []PodInfo
	// GetPodsWithRequiredAntiAffinity returns the subset of pods with required anti-affinity.
	GetPodsWithRequiredAntiAffinity() []PodInfo
	// GetUsedPorts returns the ports allocated on the node.
	GetUsedPorts() HostPortInfo
	// GetRequested returns total requested resources of all pods on this node. This includes assumed
	// pods, which scheduler has sent for binding, but may not be scheduled yet.
	GetRequested() Resource
	// GetNonZeroRequested return total requested resources of all pods on this node with a minimum value
	// applied to each container's CPU and memory requests. This does not reflect
	// the actual resource requests for this node, but is used to avoid scheduling
	// many zero-request pods onto one node.
	GetNonZeroRequested() Resource
	// We store allocatedResources (which is Node.Status.Allocatable.*) explicitly
	// as int64, to avoid conversions and accessing map.
	GetAllocatable() Resource
	// GetImageStates returns the entry of an image if and only if this image is on the node. The entry can be used for
	// checking an image's existence and advanced usage (e.g., image locality scheduling policy) based on the image
	// state information.
	GetImageStates() map[string]*ImageStateSummary
	// GetPVCRefCounts returns a mapping of PVC names to the number of pods on the node using it.
	// Keys are in the format "namespace/name".
	GetPVCRefCounts() map[string]int
	// Whenever NodeInfo changes, generation is bumped.
	// This is used to avoid cloning it if the object didn't change.
	GetGeneration() int64
	// Snapshot returns a copy of this node, Except that ImageStates is copied without the Nodes field.
	Snapshot() NodeInfo
	// String returns representation of human readable format of this NodeInfo.
	String() string

	// AddPodInfo adds pod information to this NodeInfo.
	// Consider using this instead of AddPod if a PodInfo is already computed.
	AddPodInfo(podInfo PodInfo)
	// RemovePod subtracts pod information from this NodeInfo.
	RemovePod(logger klog.Logger, pod *v1.Pod) error
	// SetNode sets the overall node information.
	SetNode(node *v1.Node)
}

// QueuedPodInfo is a Pod wrapper with additional information related to
// the pod's status in the scheduling queue, such as the timestamp when
// it's added to the queue.
type QueuedPodInfo interface {
	// GetPodInfo returns the PodInfo object wrapped by this QueuedPodInfo instance.
	GetPodInfo() PodInfo
	// GetTimestamp returns the time pod added to the scheduling queue.
	GetTimestamp() time.Time
	// GetAttempts returns the number of all schedule attempts before successfully scheduled.
	// It's used to record the # attempts metric.
	GetAttempts() int
	// GetBackoffExpiration returns the time when the Pod will complete its backoff.
	// If the SchedulerPopFromBackoffQ feature is enabled, the value is aligned to the backoff ordering window.
	// Then, two Pods with the same BackoffExpiration (time bucket) are ordered by priority and eventually the timestamp,
	// to make sure popping from the backoffQ considers priority of pods that are close to the expiration time.
	GetBackoffExpiration() time.Time
	// GetUnschedulableCount returns the total number of the scheduling attempts that this Pod gets unschedulable.
	// Basically it equals Attempts, but when the Pod fails with the Error status (e.g., the network error),
	// this count won't be incremented.
	// It's used to calculate the backoff time this Pod is obliged to get before retrying.
	GetUnschedulableCount() int
	// GetConsecutiveErrorsCount returns the number of the error status that this Pod gets sequentially.
	// This count is reset when the Pod gets another status than Error.
	//
	// If the error status is returned (e.g., kube-apiserver is unstable), we don't want to immediately retry the Pod and hence need a backoff retry mechanism
	// because that might push more burden to the kube-apiserver.
	// But, we don't want to calculate the backoff time in the same way as the normal unschedulable reason
	// since the purpose is different; the backoff for a unschedulable status etc is for the punishment of wasting the scheduling cycles,
	// whereas the backoff for the error status is for the protection of the kube-apiserver.
	// That's why we need to distinguish ConsecutiveErrorsCount for the error status and UnschedulableCount for the unschedulable status.
	// See https://github.com/kubernetes/kubernetes/issues/128744 for the discussion.
	GetConsecutiveErrorsCount() int
	// GetInitialAttemptTimestamp returns the time when the pod is added to the queue for the first time. The pod may be added
	// back to the queue multiple times before it's successfully scheduled.
	// It shouldn't be updated once initialized. It's used to record the e2e scheduling
	// latency for a pod.
	GetInitialAttemptTimestamp() *time.Time
	// GetUnschedulablePlugins records the plugin names that the Pod failed with Unschedulable or UnschedulableAndUnresolvable status
	// at specific extension points: PreFilter, Filter, Reserve, or Permit (WaitOnPermit).
	// If Pods are rejected at other extension points,
	// they're assumed to be unexpected errors (e.g., temporal network issue, plugin implementation issue, etc)
	// and retried soon after a backoff period.
	// That is because such failures could be solved regardless of incoming cluster events (registered in EventsToRegister).
	GetUnschedulablePlugins() sets.Set[string]
	// GetPendingPlugins records the plugin names that the Pod failed with Pending status.
	GetPendingPlugins() sets.Set[string]
	// GetGatingPlugin records the plugin name that gated the Pod at PreEnqueue.
	GetGatingPlugin() string
	// GetGatingPluginEvents records the events registered by the plugin that gated the Pod at PreEnqueue.
	// We have it as a cache purpose to avoid re-computing which event(s) might ungate the Pod.
	GetGatingPluginEvents() []ClusterEvent
}

// PodInfo is a wrapper to a Pod with additional pre-computed information to
// accelerate processing. This information is typically immutable (e.g., pre-processed
// inter-pod affinity selectors).
type PodInfo interface {
	// GetPod returns the wrapped Pod
	GetPod() *v1.Pod
	// GetRequiredAffinityTerms returns the precomputed affinity terms.
	GetRequiredAffinityTerms() []AffinityTerm
	// GetRequiredAffinitRequiredAntiAffinityTermsyTerms returns the precomputed anti-affinity terms.
	GetRequiredAntiAffinityTerms() []AffinityTerm
	// GetPreferredAffinityTerms returns the precomputed affinity terms with weights.
	GetPreferredAffinityTerms() []WeightedAffinityTerm
	// GetPreferredAntiAffinityTerms returns the precomputed anti-affinity terms with weights.
	GetPreferredAntiAffinityTerms() []WeightedAffinityTerm
	// CalculateResource is only intended to be used by NodeInfo.
	CalculateResource() PodResource
}

// PodResource contains the result of CalculateResource and is intended to be used only internally.
type PodResource struct {
	Resource Resource
	Non0CPU  int64
	Non0Mem  int64
}

// AffinityTerm is a processed version of v1.PodAffinityTerm.
type AffinityTerm struct {
	Namespaces        sets.Set[string]
	Selector          labels.Selector
	TopologyKey       string
	NamespaceSelector labels.Selector
}

// Matches returns true if the pod matches the label selector and namespaces or namespace selector.
func (at *AffinityTerm) Matches(pod *v1.Pod, nsLabels labels.Set) bool {
	if at.Namespaces.Has(pod.Namespace) || at.NamespaceSelector.Matches(nsLabels) {
		return at.Selector.Matches(labels.Set(pod.Labels))
	}
	return false
}

// WeightedAffinityTerm is a "processed" representation of v1.WeightedAffinityTerm.
type WeightedAffinityTerm struct {
	AffinityTerm
	Weight int32
}

// Resource is a collection of compute resources.
type Resource interface {
	GetMilliCPU() int64
	GetMemory() int64
	GetEphemeralStorage() int64
	// We return AllowedPodNumber (which is Node.Status.Allocatable.Pods().Value())
	// explicitly as int, to avoid conversions and improve performance.
	GetAllowedPodNumber() int
	// ScalarResources returns a map for resource names to their scalar values
	GetScalarResources() map[v1.ResourceName]int64
	// SetMaxResource compares with ResourceList and takes max value for each Resource.
	SetMaxResource(rl v1.ResourceList)
}

// ImageStateSummary provides summarized information about the state of an image.
type ImageStateSummary struct {
	// Size of the image
	Size int64
	// Used to track how many nodes have this image, it is computed from the Nodes field below
	// during the execution of Snapshot.
	NumNodes int
	// A set of node names for nodes having this image present. This field is used for
	// keeping track of the nodes during update/add/remove events.
	Nodes sets.Set[string]
}

// Snapshot returns a copy without Nodes field of ImageStateSummary
func (iss *ImageStateSummary) Snapshot() *ImageStateSummary {
	return &ImageStateSummary{
		Size:     iss.Size,
		NumNodes: iss.Nodes.Len(),
	}
}

// DefaultBindAllHostIP defines the default ip address used to bind to all host.
const DefaultBindAllHostIP = "0.0.0.0"

// ProtocolPort represents a protocol port pair, e.g. tcp:80.
type ProtocolPort struct {
	Protocol string
	Port     int32
}

// NewProtocolPort creates a ProtocolPort instance.
func NewProtocolPort(protocol string, port int32) *ProtocolPort {
	pp := &ProtocolPort{
		Protocol: protocol,
		Port:     port,
	}

	if len(pp.Protocol) == 0 {
		pp.Protocol = string(v1.ProtocolTCP)
	}

	return pp
}

// HostPortInfo stores mapping from ip to a set of ProtocolPort
type HostPortInfo map[string]map[ProtocolPort]struct{}

// Add adds (ip, protocol, port) to HostPortInfo
func (h HostPortInfo) Add(ip, protocol string, port int32) {
	if port <= 0 {
		return
	}

	h.sanitize(&ip, &protocol)

	pp := NewProtocolPort(protocol, port)
	if _, ok := h[ip]; !ok {
		h[ip] = map[ProtocolPort]struct{}{
			*pp: {},
		}
		return
	}

	h[ip][*pp] = struct{}{}
}

// Remove removes (ip, protocol, port) from HostPortInfo
func (h HostPortInfo) Remove(ip, protocol string, port int32) {
	if port <= 0 {
		return
	}

	h.sanitize(&ip, &protocol)

	pp := NewProtocolPort(protocol, port)
	if m, ok := h[ip]; ok {
		delete(m, *pp)
		if len(h[ip]) == 0 {
			delete(h, ip)
		}
	}
}

// Len returns the total number of (ip, protocol, port) tuple in HostPortInfo
func (h HostPortInfo) Len() int {
	length := 0
	for _, m := range h {
		length += len(m)
	}
	return length
}

// CheckConflict checks if the input (ip, protocol, port) conflicts with the existing
// ones in HostPortInfo.
func (h HostPortInfo) CheckConflict(ip, protocol string, port int32) bool {
	if port <= 0 {
		return false
	}

	h.sanitize(&ip, &protocol)

	pp := NewProtocolPort(protocol, port)

	// If ip is 0.0.0.0 check all IP's (protocol, port) pair
	if ip == DefaultBindAllHostIP {
		for _, m := range h {
			if _, ok := m[*pp]; ok {
				return true
			}
		}
		return false
	}

	// If ip isn't 0.0.0.0, only check IP and 0.0.0.0's (protocol, port) pair
	for _, key := range []string{DefaultBindAllHostIP, ip} {
		if m, ok := h[key]; ok {
			if _, ok2 := m[*pp]; ok2 {
				return true
			}
		}
	}

	return false
}

// sanitize the parameters
func (h HostPortInfo) sanitize(ip, protocol *string) {
	if len(*ip) == 0 {
		*ip = DefaultBindAllHostIP
	}
	if len(*protocol) == 0 {
		*protocol = string(v1.ProtocolTCP)
	}
}
