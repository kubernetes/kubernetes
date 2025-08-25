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

// This file defines the scheduling framework plugin interfaces.

package framework

import (
	"context"
	"math"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
)

// NodeScoreList declares a list of nodes and their scores.
type NodeScoreList []NodeScore

// NodeScore is a struct with node name and score.
type NodeScore struct {
	Name  string
	Score int64
}

// NodeToStatusReader is a read-only interface of NodeToStatus passed to each PostFilter plugin.
type NodeToStatusReader interface {
	// Get returns the status for given nodeName.
	// If the node is not in the map, the AbsentNodesStatus is returned.
	Get(nodeName string) *fwk.Status
	// NodesForStatusCode returns a list of NodeInfos for the nodes that have a given status code.
	// It returns the NodeInfos for all matching nodes denoted by AbsentNodesStatus as well.
	NodesForStatusCode(nodeLister NodeInfoLister, code fwk.Code) ([]fwk.NodeInfo, error)
}

// NodeToStatusMap is an alias for NodeToStatusReader to keep partial backwards compatibility.
// NodeToStatusReader should be used if possible.
type NodeToStatusMap = NodeToStatusReader

// NodeToStatus contains the statuses of the Nodes where the incoming Pod was not schedulable.
type NodeToStatus struct {
	// nodeToStatus contains specific statuses of the nodes.
	nodeToStatus map[string]*fwk.Status
	// absentNodesStatus defines a status for all nodes that are absent in nodeToStatus map.
	// By default, all absent nodes are UnschedulableAndUnresolvable.
	absentNodesStatus *fwk.Status
}

// NewDefaultNodeToStatus creates NodeToStatus without any node in the map.
// The absentNodesStatus is set by default to UnschedulableAndUnresolvable.
func NewDefaultNodeToStatus() *NodeToStatus {
	return NewNodeToStatus(make(map[string]*fwk.Status), fwk.NewStatus(fwk.UnschedulableAndUnresolvable))
}

// NewNodeToStatus creates NodeToStatus initialized with given nodeToStatus and absentNodesStatus.
func NewNodeToStatus(nodeToStatus map[string]*fwk.Status, absentNodesStatus *fwk.Status) *NodeToStatus {
	return &NodeToStatus{
		nodeToStatus:      nodeToStatus,
		absentNodesStatus: absentNodesStatus,
	}
}

// Get returns the status for given nodeName. If the node is not in the map, the absentNodesStatus is returned.
func (m *NodeToStatus) Get(nodeName string) *fwk.Status {
	if status, ok := m.nodeToStatus[nodeName]; ok {
		return status
	}
	return m.absentNodesStatus
}

// Set sets status for given nodeName.
func (m *NodeToStatus) Set(nodeName string, status *fwk.Status) {
	m.nodeToStatus[nodeName] = status
}

// Len returns length of nodeToStatus map. It is not aware of number of absent nodes.
func (m *NodeToStatus) Len() int {
	return len(m.nodeToStatus)
}

// AbsentNodesStatus returns absentNodesStatus value.
func (m *NodeToStatus) AbsentNodesStatus() *fwk.Status {
	return m.absentNodesStatus
}

// SetAbsentNodesStatus sets absentNodesStatus value.
func (m *NodeToStatus) SetAbsentNodesStatus(status *fwk.Status) {
	m.absentNodesStatus = status
}

// ForEachExplicitNode runs fn for each node which status is explicitly set.
// Imporatant note, it runs the fn only for nodes with a status explicitly registered,
// and hence may not run the fn for all existing nodes.
// For example, if PreFilter rejects all Nodes, the scheduler would NOT set a failure status to every Node,
// but set a failure status as AbsentNodesStatus.
// You're supposed to get a status from AbsentNodesStatus(), and consider all other nodes that are rejected by them.
func (m *NodeToStatus) ForEachExplicitNode(fn func(nodeName string, status *fwk.Status)) {
	for nodeName, status := range m.nodeToStatus {
		fn(nodeName, status)
	}
}

// NodesForStatusCode returns a list of NodeInfos for the nodes that matches a given status code.
// If the absentNodesStatus matches the code, all existing nodes are fetched using nodeLister
// and filtered using NodeToStatus.Get.
// If the absentNodesStatus doesn't match the code, nodeToStatus map is used to create a list of nodes
// and nodeLister.Get is used to obtain NodeInfo for each.
func (m *NodeToStatus) NodesForStatusCode(nodeLister NodeInfoLister, code fwk.Code) ([]fwk.NodeInfo, error) {
	var resultNodes []fwk.NodeInfo

	if m.AbsentNodesStatus().Code() == code {
		allNodes, err := nodeLister.List()
		if err != nil {
			return nil, err
		}
		if m.Len() == 0 {
			// All nodes are absent and status code is matching, so can return all nodes.
			return allNodes, nil
		}
		// Need to find all the nodes that are absent or have a matching code using the allNodes.
		for _, node := range allNodes {
			nodeName := node.Node().Name
			if status := m.Get(nodeName); status.Code() == code {
				resultNodes = append(resultNodes, node)
			}
		}
		return resultNodes, nil
	}

	m.ForEachExplicitNode(func(nodeName string, status *fwk.Status) {
		if status.Code() == code {
			if nodeInfo, err := nodeLister.Get(nodeName); err == nil {
				resultNodes = append(resultNodes, nodeInfo)
			}
		}
	})

	return resultNodes, nil
}

// NodePluginScores is a struct with node name and scores for that node.
type NodePluginScores struct {
	// Name is node name.
	Name string
	// Scores is scores from plugins and extenders.
	Scores []PluginScore
	// TotalScore is the total score in Scores.
	TotalScore int64
}

// PluginScore is a struct with plugin/extender name and score.
type PluginScore struct {
	// Name is the name of plugin or extender.
	Name  string
	Score int64
}

const (
	// MaxNodeScore is the maximum score a Score plugin is expected to return.
	MaxNodeScore int64 = 100

	// MinNodeScore is the minimum score a Score plugin is expected to return.
	MinNodeScore int64 = 0

	// MaxTotalScore is the maximum total score.
	MaxTotalScore int64 = math.MaxInt64
)

// PodsToActivateKey is a reserved state key for stashing pods.
// If the stashed pods are present in unschedulablePods or backoffQ，they will be
// activated (i.e., moved to activeQ) in two phases:
// - end of a scheduling cycle if it succeeds (will be cleared from `PodsToActivate` if activated)
// - end of a binding cycle if it succeeds
var PodsToActivateKey fwk.StateKey = "kubernetes.io/pods-to-activate"

// PodsToActivate stores pods to be activated.
type PodsToActivate struct {
	sync.Mutex
	// Map is keyed with namespaced pod name, and valued with the pod.
	Map map[string]*v1.Pod
}

// Clone just returns the same state.
func (s *PodsToActivate) Clone() fwk.StateData {
	return s
}

// NewPodsToActivate instantiates a PodsToActivate object.
func NewPodsToActivate() *PodsToActivate {
	return &PodsToActivate{Map: make(map[string]*v1.Pod)}
}

// WaitingPod represents a pod currently waiting in the permit phase.
type WaitingPod interface {
	// GetPod returns a reference to the waiting pod.
	GetPod() *v1.Pod
	// GetPendingPlugins returns a list of pending Permit plugin's name.
	GetPendingPlugins() []string
	// Allow declares the waiting pod is allowed to be scheduled by the plugin named as "pluginName".
	// If this is the last remaining plugin to allow, then a success signal is delivered
	// to unblock the pod.
	Allow(pluginName string)
	// Reject declares the waiting pod unschedulable.
	Reject(pluginName, msg string)
}

// Plugin is the parent type for all the scheduling framework plugins.
type Plugin interface {
	Name() string
}

// PreEnqueuePlugin is an interface that must be implemented by "PreEnqueue" plugins.
// These plugins are called prior to adding Pods to activeQ.
// Note: an preEnqueue plugin is expected to be lightweight and efficient, so it's not expected to
// involve expensive calls like accessing external endpoints; otherwise it'd block other
// Pods' enqueuing in event handlers.
type PreEnqueuePlugin interface {
	Plugin
	// PreEnqueue is called prior to adding Pods to activeQ.
	PreEnqueue(ctx context.Context, p *v1.Pod) *fwk.Status
}

// LessFunc is the function to sort pod info
type LessFunc func(podInfo1, podInfo2 fwk.QueuedPodInfo) bool

// QueueSortPlugin is an interface that must be implemented by "QueueSort" plugins.
// These plugins are used to sort pods in the scheduling queue. Only one queue sort
// plugin may be enabled at a time.
type QueueSortPlugin interface {
	Plugin
	// Less are used to sort pods in the scheduling queue.
	Less(fwk.QueuedPodInfo, fwk.QueuedPodInfo) bool
}

// EnqueueExtensions is an optional interface that plugins can implement to efficiently
// move unschedulable Pods in internal scheduling queues.
// In the scheduler, Pods can be unschedulable by PreEnqueue, PreFilter, Filter, Reserve, and Permit plugins,
// and Pods rejected by these plugins are requeued based on this extension point.
// Failures from other extension points are regarded as temporal errors (e.g., network failure),
// and the scheduler requeue Pods without this extension point - always requeue Pods to activeQ after backoff.
// This is because such temporal errors cannot be resolved by specific cluster events,
// and we have no choose but keep retrying scheduling until the failure is resolved.
//
// Plugins that make pod unschedulable (PreEnqueue, PreFilter, Filter, Reserve, and Permit plugins) must implement this interface,
// otherwise the default implementation will be used, which is less efficient in requeueing Pods rejected by the plugin.
//
// Also, if EventsToRegister returns an empty list, that means the Pods failed by the plugin are not requeued by any events,
// which doesn't make sense in most cases (very likely misuse)
// since the pods rejected by the plugin could be stuck in the unschedulable pod pool forever.
//
// If plugins other than above extension points support this interface, they are just ignored.
type EnqueueExtensions interface {
	Plugin
	// EventsToRegister returns a series of possible events that may cause a Pod
	// failed by this plugin schedulable. Each event has a callback function that
	// filters out events to reduce useless retry of Pod's scheduling.
	// The events will be registered when instantiating the internal scheduling queue,
	// and leveraged to build event handlers dynamically.
	// When it returns an error, the scheduler fails to start.
	// Note: the returned list needs to be determined at a startup,
	// and the scheduler only evaluates it once during start up.
	// Do not change the result during runtime, for example, based on the cluster's state etc.
	//
	// Appropriate implementation of this function will make Pod's re-scheduling accurate and performant.
	EventsToRegister(context.Context) ([]fwk.ClusterEventWithHint, error)
}

// PreFilterExtensions is an interface that is included in plugins that allow specifying
// callbacks to make incremental updates to its supposedly pre-calculated
// state.
type PreFilterExtensions interface {
	// AddPod is called by the framework while trying to evaluate the impact
	// of adding podToAdd to the node while scheduling podToSchedule.
	AddPod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status
	// RemovePod is called by the framework while trying to evaluate the impact
	// of removing podToRemove from the node while scheduling podToSchedule.
	RemovePod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status
}

// PreFilterPlugin is an interface that must be implemented by "PreFilter" plugins.
// These plugins are called at the beginning of the scheduling cycle.
type PreFilterPlugin interface {
	Plugin
	// PreFilter is called at the beginning of the scheduling cycle. All PreFilter
	// plugins must return success or the pod will be rejected. PreFilter could optionally
	// return a PreFilterResult to influence which nodes to evaluate downstream. This is useful
	// for cases where it is possible to determine the subset of nodes to process in O(1) time.
	// When PreFilterResult filters out some Nodes, the framework considers Nodes that are filtered out as getting "UnschedulableAndUnresolvable".
	// i.e., those Nodes will be out of the candidates of the preemption.
	//
	// When it returns Skip status, returned PreFilterResult and other fields in status are just ignored,
	// and coupled Filter plugin/PreFilterExtensions() will be skipped in this scheduling cycle.
	PreFilter(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodes []fwk.NodeInfo) (*PreFilterResult, *fwk.Status)
	// PreFilterExtensions returns a PreFilterExtensions interface if the plugin implements one,
	// or nil if it does not. A Pre-filter plugin can provide extensions to incrementally
	// modify its pre-processed info. The framework guarantees that the extensions
	// AddPod/RemovePod will only be called after PreFilter, possibly on a cloned
	// CycleState, and may call those functions more than once before calling
	// Filter again on a specific node.
	PreFilterExtensions() PreFilterExtensions
}

// FilterPlugin is an interface for Filter plugins. These plugins are called at the
// filter extension point for filtering out hosts that cannot run a pod.
// This concept used to be called 'predicate' in the original scheduler.
// These plugins should return "Success", "Unschedulable" or "Error" in Status.code.
// However, the scheduler accepts other valid codes as well.
// Anything other than "Success" will lead to exclusion of the given host from
// running the pod.
type FilterPlugin interface {
	Plugin
	// Filter is called by the scheduling framework.
	// All FilterPlugins should return "Success" to declare that
	// the given node fits the pod. If Filter doesn't return "Success",
	// it will return "Unschedulable", "UnschedulableAndUnresolvable" or "Error".
	//
	// "Error" aborts pod scheduling and puts the pod into the backoff queue.
	//
	// For the node being evaluated, Filter plugins should look at the passed
	// nodeInfo reference for this particular node's information (e.g., pods
	// considered to be running on the node) instead of looking it up in the
	// NodeInfoSnapshot because we don't guarantee that they will be the same.
	// For example, during preemption, we may pass a copy of the original
	// nodeInfo object that has some pods removed from it to evaluate the
	// possibility of preempting them to schedule the target pod.
	//
	// Plugins are encouraged to check the context for cancellation.
	// Once canceled, they should return as soon as possible with
	// an UnschedulableAndUnresolvable status that includes the
	// `context.Cause(ctx)` error explanation. For example, the
	// context gets canceled when a sufficient number of suitable
	// nodes have been found and searching for more isn't necessary
	// anymore.
	Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status
}

// PostFilterPlugin is an interface for "PostFilter" plugins. These plugins are called
// after a pod cannot be scheduled.
type PostFilterPlugin interface {
	Plugin
	// PostFilter is called by the scheduling framework
	// when the scheduling cycle failed at PreFilter or Filter by Unschedulable or UnschedulableAndUnresolvable.
	// NodeToStatusReader has statuses that each Node got in PreFilter or Filter phase.
	//
	// If you're implementing a custom preemption with PostFilter, ignoring Nodes with UnschedulableAndUnresolvable is the responsibility of your plugin,
	// meaning NodeToStatusReader could have Nodes with UnschedulableAndUnresolvable
	// and the scheduling framework does call PostFilter plugins even when all Nodes in NodeToStatusReader are UnschedulableAndUnresolvable.
	//
	// A PostFilter plugin should return one of the following statuses:
	// - Unschedulable: the plugin gets executed successfully but the pod cannot be made schedulable.
	// - Success: the plugin gets executed successfully and the pod can be made schedulable.
	// - Error: the plugin aborts due to some internal error.
	//
	// Informational plugins should be configured ahead of other ones, and always return Unschedulable status.
	// Optionally, a non-nil PostFilterResult may be returned along with a Success status. For example,
	// a preemption plugin may choose to return nominatedNodeName, so that framework can reuse that to update the
	// preemptor pod's .spec.status.nominatedNodeName field.
	PostFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, filteredNodeStatusMap NodeToStatusReader) (*PostFilterResult, *fwk.Status)
}

// PreScorePlugin is an interface for "PreScore" plugin. PreScore is an
// informational extension point. Plugins will be called with a list of nodes
// that passed the filtering phase. A plugin may use this data to update internal
// state or to generate logs/metrics.
type PreScorePlugin interface {
	Plugin
	// PreScore is called by the scheduling framework after a list of nodes
	// passed the filtering phase. All prescore plugins must return success or
	// the pod will be rejected
	// When it returns Skip status, other fields in status are just ignored,
	// and coupled Score plugin will be skipped in this scheduling cycle.
	PreScore(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) *fwk.Status
}

// ScoreExtensions is an interface for Score extended functionality.
type ScoreExtensions interface {
	// NormalizeScore is called for all node scores produced by the same plugin's "Score"
	// method. A successful run of NormalizeScore will update the scores list and return
	// a success status.
	NormalizeScore(ctx context.Context, state fwk.CycleState, p *v1.Pod, scores NodeScoreList) *fwk.Status
}

// ScorePlugin is an interface that must be implemented by "Score" plugins to rank
// nodes that passed the filtering phase.
type ScorePlugin interface {
	Plugin
	// Score is called on each filtered node. It must return success and an integer
	// indicating the rank of the node. All scoring plugins must return success or
	// the pod will be rejected.
	Score(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeInfo fwk.NodeInfo) (int64, *fwk.Status)

	// ScoreExtensions returns a ScoreExtensions interface if it implements one, or nil if does not.
	ScoreExtensions() ScoreExtensions
}

// ReservePlugin is an interface for plugins with Reserve and Unreserve
// methods. These are meant to update the state of the plugin. This concept
// used to be called 'assume' in the original scheduler. These plugins should
// return only Success or Error in Status.code. However, the scheduler accepts
// other valid codes as well. Anything other than Success will lead to
// rejection of the pod.
type ReservePlugin interface {
	Plugin
	// Reserve is called by the scheduling framework when the scheduler cache is
	// updated. If this method returns a failed Status, the scheduler will call
	// the Unreserve method for all enabled ReservePlugins.
	Reserve(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status
	// Unreserve is called by the scheduling framework when a reserved pod was
	// rejected, an error occurred during reservation of subsequent plugins, or
	// in a later phase. The Unreserve method implementation must be idempotent
	// and may be called by the scheduler even if the corresponding Reserve
	// method for the same plugin was not called.
	Unreserve(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string)
}

// PreBindPlugin is an interface that must be implemented by "PreBind" plugins.
// These plugins are called before a pod being scheduled.
type PreBindPlugin interface {
	Plugin
	// PreBindPreFlight is called before PreBind, and the plugin is supposed to return Success, Skip, or Error status
	// to tell the scheduler whether the plugin will do something in PreBind or not.
	// If it returns Success, it means this PreBind plugin will handle this pod.
	// If it returns Skip, it means this PreBind plugin has nothing to do with the pod, and PreBind will be skipped.
	// This function should be lightweight, and shouldn't do any actual operation, e.g., creating a volume etc.
	PreBindPreFlight(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status
	// PreBind is called before binding a pod. All prebind plugins must return
	// success or the pod will be rejected and won't be sent for binding.
	PreBind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status
}

// PostBindPlugin is an interface that must be implemented by "PostBind" plugins.
// These plugins are called after a pod is successfully bound to a node.
type PostBindPlugin interface {
	Plugin
	// PostBind is called after a pod is successfully bound. These plugins are
	// informational. A common application of this extension point is for cleaning
	// up. If a plugin needs to clean-up its state after a pod is scheduled and
	// bound, PostBind is the extension point that it should register.
	PostBind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string)
}

// PermitPlugin is an interface that must be implemented by "Permit" plugins.
// These plugins are called before a pod is bound to a node.
type PermitPlugin interface {
	Plugin
	// Permit is called before binding a pod (and before prebind plugins). Permit
	// plugins are used to prevent or delay the binding of a Pod. A permit plugin
	// must return success or wait with timeout duration, or the pod will be rejected.
	// The pod will also be rejected if the wait timeout or the pod is rejected while
	// waiting. Note that if the plugin returns "wait", the framework will wait only
	// after running the remaining plugins given that no other plugin rejects the pod.
	Permit(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) (*fwk.Status, time.Duration)
}

// BindPlugin is an interface that must be implemented by "Bind" plugins. Bind
// plugins are used to bind a pod to a Node.
type BindPlugin interface {
	Plugin
	// Bind plugins will not be called until all pre-bind plugins have completed. Each
	// bind plugin is called in the configured order. A bind plugin may choose whether
	// or not to handle the given Pod. If a bind plugin chooses to handle a Pod, the
	// remaining bind plugins are skipped. When a bind plugin does not handle a pod,
	// it must return Skip in its Status code. If a bind plugin returns an Error, the
	// pod is rejected and will not be bound.
	Bind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status
}

// Framework manages the set of plugins in use by the scheduling framework.
// Configured plugins are called at specified points in a scheduling context.
type Framework interface {
	Handle

	// PreEnqueuePlugins returns the registered preEnqueue plugins.
	PreEnqueuePlugins() []PreEnqueuePlugin

	// EnqueueExtensions returns the registered Enqueue extensions.
	EnqueueExtensions() []EnqueueExtensions

	// QueueSortFunc returns the function to sort pods in scheduling queue
	QueueSortFunc() LessFunc

	// RunPreFilterPlugins runs the set of configured PreFilter plugins. It returns
	// *fwk.Status and its code is set to non-success if any of the plugins returns
	// anything but Success. If a non-success status is returned, then the scheduling
	// cycle is aborted.
	// It also returns a PreFilterResult, which may influence what or how many nodes to
	// evaluate downstream.
	// The third returns value contains PreFilter plugin that rejected some or all Nodes with PreFilterResult.
	// But, note that it doesn't contain any plugin when a plugin rejects this Pod with non-success status,
	// not with PreFilterResult.
	RunPreFilterPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod) (*PreFilterResult, *fwk.Status, sets.Set[string])

	// RunPostFilterPlugins runs the set of configured PostFilter plugins.
	// PostFilter plugins can either be informational, in which case should be configured
	// to execute first and return Unschedulable status, or ones that try to change the
	// cluster state to make the pod potentially schedulable in a future scheduling cycle.
	RunPostFilterPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, filteredNodeStatusMap NodeToStatusReader) (*PostFilterResult, *fwk.Status)

	// RunPreBindPlugins runs the set of configured PreBind plugins. It returns
	// *fwk.Status and its code is set to non-success if any of the plugins returns
	// anything but Success. If the Status code is "Unschedulable", it is
	// considered as a scheduling check failure, otherwise, it is considered as an
	// internal error. In either case the pod is not going to be bound.
	RunPreBindPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status

	// RunPreBindPreFlights runs the set of configured PreBindPreFlight functions from PreBind plugins.
	// It returns immediately if any of the plugins returns a non-skip status.
	RunPreBindPreFlights(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status

	// RunPostBindPlugins runs the set of configured PostBind plugins.
	RunPostBindPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string)

	// RunReservePluginsReserve runs the Reserve method of the set of
	// configured Reserve plugins. If any of these calls returns an error, it
	// does not continue running the remaining ones and returns the error. In
	// such case, pod will not be scheduled.
	RunReservePluginsReserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status

	// RunReservePluginsUnreserve runs the Unreserve method of the set of
	// configured Reserve plugins.
	RunReservePluginsUnreserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string)

	// RunPermitPlugins runs the set of configured Permit plugins. If any of these
	// plugins returns a status other than "Success" or "Wait", it does not continue
	// running the remaining plugins and returns an error. Otherwise, if any of the
	// plugins returns "Wait", then this function will create and add waiting pod
	// to a map of currently waiting pods and return status with "Wait" code.
	// Pod will remain waiting pod for the minimum duration returned by the Permit plugins.
	RunPermitPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status

	// WillWaitOnPermit returns whether this pod will wait on permit by checking if the pod is a waiting pod.
	WillWaitOnPermit(ctx context.Context, pod *v1.Pod) bool

	// WaitOnPermit will block, if the pod is a waiting pod, until the waiting pod is rejected or allowed.
	WaitOnPermit(ctx context.Context, pod *v1.Pod) *fwk.Status

	// RunBindPlugins runs the set of configured Bind plugins. A Bind plugin may choose
	// whether or not to handle the given Pod. If a Bind plugin chooses to skip the
	// binding, it should return code=5("skip") status. Otherwise, it should return "Error"
	// or "Success". If none of the plugins handled binding, RunBindPlugins returns
	// code=5("skip") status.
	RunBindPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status

	// HasFilterPlugins returns true if at least one Filter plugin is defined.
	HasFilterPlugins() bool

	// HasPostFilterPlugins returns true if at least one PostFilter plugin is defined.
	HasPostFilterPlugins() bool

	// HasScorePlugins returns true if at least one Score plugin is defined.
	HasScorePlugins() bool

	// ListPlugins returns a map of extension point name to list of configured Plugins.
	ListPlugins() *config.Plugins

	// ProfileName returns the profile name associated to a profile.
	ProfileName() string

	// PercentageOfNodesToScore returns percentageOfNodesToScore associated to a profile.
	PercentageOfNodesToScore() *int32

	// SetPodNominator sets the PodNominator
	SetPodNominator(nominator PodNominator)
	// SetPodActivator sets the PodActivator
	SetPodActivator(activator PodActivator)
	// SetAPICacher sets the APICacher
	SetAPICacher(apiCacher APICacher)

	// Close calls Close method of each plugin.
	Close() error
}

// Handle provides data and some tools that plugins can use. It is
// passed to the plugin factories at the time of plugin initialization. Plugins
// must store and use this handle to call framework functions.
type Handle interface {
	// PodNominator abstracts operations to maintain nominated Pods.
	PodNominator
	// PluginsRunner abstracts operations to run some plugins.
	PluginsRunner
	// PodActivator abstracts operations in the scheduling queue.
	PodActivator
	// SnapshotSharedLister returns listers from the latest NodeInfo Snapshot. The snapshot
	// is taken at the beginning of a scheduling cycle and remains unchanged until
	// a pod finishes "Permit" point.
	//
	// It should be used only during scheduling cycle:
	// - There is no guarantee that the information remains unchanged in the binding phase of scheduling.
	//   So, plugins shouldn't use it in the binding cycle (pre-bind/bind/post-bind/un-reserve plugin)
	//   otherwise, a concurrent read/write error might occur.
	// - There is no guarantee that the information is always up-to-date.
	//   So, plugins shouldn't use it in QueueingHint and PreEnqueue
	//   otherwise, they might make a decision based on stale information.
	//
	// Instead, they should use the resources getting from Informer created from SharedInformerFactory().
	SnapshotSharedLister() SharedLister

	// IterateOverWaitingPods acquires a read lock and iterates over the WaitingPods map.
	IterateOverWaitingPods(callback func(WaitingPod))

	// GetWaitingPod returns a waiting pod given its UID.
	GetWaitingPod(uid types.UID) WaitingPod

	// RejectWaitingPod rejects a waiting pod given its UID.
	// The return value indicates if the pod is waiting or not.
	RejectWaitingPod(uid types.UID) bool

	// ClientSet returns a kubernetes clientSet.
	ClientSet() clientset.Interface

	// KubeConfig returns the raw kube config.
	KubeConfig() *restclient.Config

	// EventRecorder returns an event recorder.
	EventRecorder() events.EventRecorder

	SharedInformerFactory() informers.SharedInformerFactory

	// SharedDRAManager can be used to obtain DRA objects, and track modifications to them in-memory - mainly by the DRA plugin.
	// A non-default implementation can be plugged into the framework to simulate the state of DRA objects.
	SharedDRAManager() SharedDRAManager

	// RunFilterPluginsWithNominatedPods runs the set of configured filter plugins for nominated pod on the given node.
	RunFilterPluginsWithNominatedPods(ctx context.Context, state fwk.CycleState, pod *v1.Pod, info fwk.NodeInfo) *fwk.Status

	// Extenders returns registered scheduler extenders.
	Extenders() []Extender

	// Parallelizer returns a parallelizer holding parallelism for scheduler.
	Parallelizer() parallelize.Parallelizer

	// APIDispatcher returns a fwk.APIDispatcher that can be used to dispatch API calls directly.
	// This is non-nil if the SchedulerAsyncAPICalls feature gate is enabled.
	APIDispatcher() fwk.APIDispatcher

	// APICacher returns an APICacher that coordinates API calls with the scheduler's internal cache.
	// Use this to ensure the scheduler's view of the cluster remains consistent.
	// This is non-nil if the SchedulerAsyncAPICalls feature gate is enabled.
	APICacher() APICacher
}

// APICacher defines methods that send API calls through the scheduler's cache
// before they are executed asynchronously by the APIDispatcher.
// This ensures the scheduler's internal state is updated optimistically,
// reflecting the intended outcome of the call.
// This methods should be used only if the SchedulerAsyncAPICalls feature gate is enabled.
type APICacher interface {
	// PatchPodStatus sends a patch request for a Pod's status.
	// The patch could be first applied to the cached Pod object and then the API call is executed asynchronously.
	// It returns a channel that can be used to wait for the call's completion.
	PatchPodStatus(pod *v1.Pod, condition *v1.PodCondition, nominatingInfo *NominatingInfo) (<-chan error, error)

	// BindPod sends a binding request. The binding could be first applied to the cached Pod object
	// and then the API call is executed asynchronously.
	// It returns a channel that can be used to wait for the call's completion.
	BindPod(binding *v1.Binding) (<-chan error, error)

	// WaitOnFinish blocks until the result of an API call is sent to the given onFinish channel
	// (returned by methods BindPod or PreemptPod).
	//
	// It returns the error received from the channel.
	// It also returns nil if the call was skipped or overwritten,
	// as these are considered successful lifecycle outcomes.
	// Direct onFinish channel read can be used to access these results.
	WaitOnFinish(ctx context.Context, onFinish <-chan error) error
}

// APICallImplementations define constructors for each fwk.APICall that is used by the scheduler internally.
type APICallImplementations[T, K fwk.APICall] struct {
	// PodStatusPatch is a constructor used to create fwk.APICall object for pod status patch.
	PodStatusPatch func(pod *v1.Pod, condition *v1.PodCondition, nominatingInfo *NominatingInfo) T
	// PodBinding is a constructor used to create fwk.APICall object for pod binding.
	PodBinding func(binding *v1.Binding) K
}

// PreFilterResult wraps needed info for scheduler framework to act upon PreFilter phase.
type PreFilterResult struct {
	// The set of nodes that should be considered downstream; if nil then
	// all nodes are eligible.
	NodeNames sets.Set[string]
}

func (p *PreFilterResult) AllNodes() bool {
	return p == nil || p.NodeNames == nil
}

func (p *PreFilterResult) Merge(in *PreFilterResult) *PreFilterResult {
	if p.AllNodes() && in.AllNodes() {
		return nil
	}

	r := PreFilterResult{}
	if p.AllNodes() {
		r.NodeNames = in.NodeNames.Clone()
		return &r
	}
	if in.AllNodes() {
		r.NodeNames = p.NodeNames.Clone()
		return &r
	}

	r.NodeNames = p.NodeNames.Intersection(in.NodeNames)
	return &r
}

type NominatingMode int

const (
	ModeNoop NominatingMode = iota
	ModeOverride
)

type NominatingInfo struct {
	NominatedNodeName string
	NominatingMode    NominatingMode
}

// PostFilterResult wraps needed info for scheduler framework to act upon PostFilter phase.
type PostFilterResult struct {
	*NominatingInfo
}

func NewPostFilterResultWithNominatedNode(name string) *PostFilterResult {
	return &PostFilterResult{
		NominatingInfo: &NominatingInfo{
			NominatedNodeName: name,
			NominatingMode:    ModeOverride,
		},
	}
}

func (ni *NominatingInfo) Mode() NominatingMode {
	if ni == nil {
		return ModeNoop
	}
	return ni.NominatingMode
}

// PodActivator abstracts operations in the scheduling queue.
type PodActivator interface {
	// Activate moves the given pods to activeQ.
	// If a pod isn't found in unschedulablePods or backoffQ and it's in-flight,
	// the wildcard event is registered so that the pod will be requeued when it comes back.
	// But, if a pod isn't found in unschedulablePods or backoffQ and it's not in-flight (i.e., completely unknown pod),
	// Activate would ignore the pod.
	Activate(logger klog.Logger, pods map[string]*v1.Pod)
}

// PodNominator abstracts operations to maintain nominated Pods.
type PodNominator interface {
	// AddNominatedPod adds the given pod to the nominator or
	// updates it if it already exists.
	AddNominatedPod(logger klog.Logger, pod fwk.PodInfo, nominatingInfo *NominatingInfo)
	// DeleteNominatedPodIfExists deletes nominatedPod from internal cache. It's a no-op if it doesn't exist.
	DeleteNominatedPodIfExists(pod *v1.Pod)
	// UpdateNominatedPod updates the <oldPod> with <newPod>.
	UpdateNominatedPod(logger klog.Logger, oldPod *v1.Pod, newPodInfo fwk.PodInfo)
	// NominatedPodsForNode returns nominatedPods on the given node.
	NominatedPodsForNode(nodeName string) []fwk.PodInfo
}

// PluginsRunner abstracts operations to run some plugins.
// This is used by preemption PostFilter plugins when evaluating the feasibility of
// scheduling the pod on nodes when certain running pods get evicted.
type PluginsRunner interface {
	// RunPreScorePlugins runs the set of configured PreScore plugins. If any
	// of these plugins returns any status other than "Success", the given pod is rejected.
	RunPreScorePlugins(context.Context, fwk.CycleState, *v1.Pod, []fwk.NodeInfo) *fwk.Status
	// RunScorePlugins runs the set of configured scoring plugins.
	// It returns a list that stores scores from each plugin and total score for each Node.
	// It also returns *fwk.Status, which is set to non-success if any of the plugins returns
	// a non-success status.
	RunScorePlugins(context.Context, fwk.CycleState, *v1.Pod, []fwk.NodeInfo) ([]NodePluginScores, *fwk.Status)
	// RunFilterPlugins runs the set of configured Filter plugins for pod on
	// the given node. Note that for the node being evaluated, the passed nodeInfo
	// reference could be different from the one in NodeInfoSnapshot map (e.g., pods
	// considered to be running on the node could be different). For example, during
	// preemption, we may pass a copy of the original nodeInfo object that has some pods
	// removed from it to evaluate the possibility of preempting them to
	// schedule the target pod.
	RunFilterPlugins(context.Context, fwk.CycleState, *v1.Pod, fwk.NodeInfo) *fwk.Status
	// RunPreFilterExtensionAddPod calls the AddPod interface for the set of configured
	// PreFilter plugins. It returns directly if any of the plugins return any
	// status other than Success.
	RunPreFilterExtensionAddPod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status
	// RunPreFilterExtensionRemovePod calls the RemovePod interface for the set of configured
	// PreFilter plugins. It returns directly if any of the plugins return any
	// status other than Success.
	RunPreFilterExtensionRemovePod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status
}
