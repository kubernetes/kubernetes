/*
Copyright 2014 The Kubernetes Authors.

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

package api

import (
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	restclient "k8s.io/client-go/rest"
)

const (
	// MaxUint defines the max unsigned int value.
	MaxUint = ^uint(0)
	// MaxInt defines the max signed int value.
	MaxInt = int(MaxUint >> 1)
	// MaxTotalPriority defines the max total priority value.
	MaxTotalPriority = MaxInt
	// MaxPriority defines the max priority value.
	MaxPriority = 10
	// MaxWeight defines the max weight value.
	MaxWeight = MaxInt / MaxPriority
	// DefaultPercentageOfNodesToScore defines the percentage of nodes of all nodes
	// that once found feasible, the scheduler stops looking for more nodes.
	DefaultPercentageOfNodesToScore = 50
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Policy describes a struct of a policy resource in api.
type Policy struct {
	metav1.TypeMeta
	// Holds the information to configure the fit predicate functions.
	// If unspecified, the default predicate functions will be applied.
	// If empty list, all predicates (except the mandatory ones) will be
	// bypassed.
	Predicates []PredicatePolicy
	// Holds the information to configure the priority functions.
	// If unspecified, the default priority functions will be applied.
	// If empty list, all priority functions will be bypassed.
	Priorities []PriorityPolicy
	// Holds the information to communicate with the extender(s)
	ExtenderConfigs []ExtenderConfig
	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 1-100.
	HardPodAffinitySymmetricWeight int32

	// When AlwaysCheckAllPredicates is set to true, scheduler checks all
	// the configured predicates even after one or more of them fails.
	// When the flag is set to false, scheduler skips checking the rest
	// of the predicates after it finds one predicate that failed.
	AlwaysCheckAllPredicates bool
}

// PredicatePolicy describes a struct of a predicate policy.
type PredicatePolicy struct {
	// Identifier of the predicate policy
	// For a custom predicate, the name can be user-defined
	// For the Kubernetes provided predicates, the name is the identifier of the pre-defined predicate
	Name string
	// Holds the parameters to configure the given predicate
	Argument *PredicateArgument
}

// PriorityPolicy describes a struct of a priority policy.
type PriorityPolicy struct {
	// Identifier of the priority policy
	// For a custom priority, the name can be user-defined
	// For the Kubernetes provided priority functions, the name is the identifier of the pre-defined priority function
	Name string
	// The numeric multiplier for the node scores that the priority function generates
	// The weight should be a positive integer
	Weight int
	// Holds the parameters to configure the given priority function
	Argument *PriorityArgument
}

// PredicateArgument represents the arguments to configure predicate functions in scheduler policy configuration.
// Only one of its members may be specified
type PredicateArgument struct {
	// The predicate that provides affinity for pods belonging to a service
	// It uses a label to identify nodes that belong to the same "group"
	ServiceAffinity *ServiceAffinity
	// The predicate that checks whether a particular node has a certain label
	// defined or not, regardless of value
	LabelsPresence *LabelsPresence
}

// PriorityArgument represents the arguments to configure priority functions in scheduler policy configuration.
// Only one of its members may be specified
type PriorityArgument struct {
	// The priority function that ensures a good spread (anti-affinity) for pods belonging to a service
	// It uses a label to identify nodes that belong to the same "group"
	ServiceAntiAffinity *ServiceAntiAffinity
	// The priority function that checks whether a particular node has a certain label
	// defined or not, regardless of value
	LabelPreference *LabelPreference
	// The RequestedToCapacityRatio priority function is parametrized with function shape.
	RequestedToCapacityRatioArguments *RequestedToCapacityRatioArguments
}

// ServiceAffinity holds the parameters that are used to configure the corresponding predicate in scheduler policy configuration.
type ServiceAffinity struct {
	// The list of labels that identify node "groups"
	// All of the labels should match for the node to be considered a fit for hosting the pod
	Labels []string
}

// LabelsPresence holds the parameters that are used to configure the corresponding predicate in scheduler policy configuration.
type LabelsPresence struct {
	// The list of labels that identify node "groups"
	// All of the labels should be either present (or absent) for the node to be considered a fit for hosting the pod
	Labels []string
	// The boolean flag that indicates whether the labels should be present or absent from the node
	Presence bool
}

// ServiceAntiAffinity holds the parameters that are used to configure the corresponding priority function
type ServiceAntiAffinity struct {
	// Used to identify node "groups"
	Label string
}

// LabelPreference holds the parameters that are used to configure the corresponding priority function
type LabelPreference struct {
	// Used to identify node "groups"
	Label string
	// This is a boolean flag
	// If true, higher priority is given to nodes that have the label
	// If false, higher priority is given to nodes that do not have the label
	Presence bool
}

// RequestedToCapacityRatioArguments holds arguments specific to RequestedToCapacityRatio priority function
type RequestedToCapacityRatioArguments struct {
	// Array of point defining priority function shape
	UtilizationShape []UtilizationShapePoint
}

// UtilizationShapePoint represents single point of priority function shape
type UtilizationShapePoint struct {
	// Utilization (x axis). Valid values are 0 to 100. Fully utilized node maps to 100.
	Utilization int
	// Score assigned to given utilization (y axis). Valid values are 0 to 10.
	Score int
}

// ExtenderManagedResource describes the arguments of extended resources
// managed by an extender.
type ExtenderManagedResource struct {
	// Name is the extended resource name.
	Name v1.ResourceName
	// IgnoredByScheduler indicates whether kube-scheduler should ignore this
	// resource when applying predicates.
	IgnoredByScheduler bool
}

// ExtenderConfig holds the parameters used to communicate with the extender. If a verb is unspecified/empty,
// it is assumed that the extender chose not to provide that extension.
type ExtenderConfig struct {
	// URLPrefix at which the extender is available
	URLPrefix string
	// Verb for the filter call, empty if not supported. This verb is appended to the URLPrefix when issuing the filter call to extender.
	FilterVerb string
	// Verb for the preempt call, empty if not supported. This verb is appended to the URLPrefix when issuing the preempt call to extender.
	PreemptVerb string
	// Verb for the prioritize call, empty if not supported. This verb is appended to the URLPrefix when issuing the prioritize call to extender.
	PrioritizeVerb string
	// The numeric multiplier for the node scores that the prioritize call generates.
	// The weight should be a positive integer
	Weight int
	// Verb for the bind call, empty if not supported. This verb is appended to the URLPrefix when issuing the bind call to extender.
	// If this method is implemented by the extender, it is the extender's responsibility to bind the pod to apiserver. Only one extender
	// can implement this function.
	BindVerb string
	// EnableHTTPS specifies whether https should be used to communicate with the extender
	EnableHTTPS bool
	// TLSConfig specifies the transport layer security config
	TLSConfig *restclient.TLSClientConfig
	// HTTPTimeout specifies the timeout duration for a call to the extender. Filter timeout fails the scheduling of the pod. Prioritize
	// timeout is ignored, k8s/other extenders priorities are used to select the node.
	HTTPTimeout time.Duration
	// NodeCacheCapable specifies that the extender is capable of caching node information,
	// so the scheduler should only send minimal information about the eligible nodes
	// assuming that the extender already cached full details of all nodes in the cluster
	NodeCacheCapable bool
	// ManagedResources is a list of extended resources that are managed by
	// this extender.
	// - A pod will be sent to the extender on the Filter, Prioritize and Bind
	//   (if the extender is the binder) phases iff the pod requests at least
	//   one of the extended resources in this list. If empty or unspecified,
	//   all pods will be sent to this extender.
	// - If IgnoredByScheduler is set to true for a resource, kube-scheduler
	//   will skip checking the resource in predicates.
	// +optional
	ManagedResources []ExtenderManagedResource
	// Ignorable specifies if the extender is ignorable, i.e. scheduling should not
	// fail when the extender returns an error or is not reachable.
	Ignorable bool
}

// ExtenderPreemptionResult represents the result returned by preemption phase of extender.
type ExtenderPreemptionResult struct {
	NodeNameToMetaVictims map[string]*MetaVictims
}

// ExtenderPreemptionArgs represents the arguments needed by the extender to preempt pods on nodes.
type ExtenderPreemptionArgs struct {
	// Pod being scheduled
	Pod *v1.Pod
	// Victims map generated by scheduler preemption phase
	// Only set NodeNameToMetaVictims if ExtenderConfig.NodeCacheCapable == true. Otherwise, only set NodeNameToVictims.
	NodeNameToVictims     map[string]*Victims
	NodeNameToMetaVictims map[string]*MetaVictims
}

// Victims represents:
//   pods:  a group of pods expected to be preempted.
//   numPDBViolations: the count of violations of PodDisruptionBudget
type Victims struct {
	Pods             []*v1.Pod
	NumPDBViolations int
}

// MetaPod represent identifier for a v1.Pod
type MetaPod struct {
	UID string
}

// MetaVictims represents:
//   pods:  a group of pods expected to be preempted.
//     Only Pod identifiers will be sent and user are expect to get v1.Pod in their own way.
//   numPDBViolations: the count of violations of PodDisruptionBudget
type MetaVictims struct {
	Pods             []*MetaPod
	NumPDBViolations int
}

// ExtenderArgs represents the arguments needed by the extender to filter/prioritize
// nodes for a pod.
type ExtenderArgs struct {
	// Pod being scheduled
	Pod *v1.Pod
	// List of candidate nodes where the pod can be scheduled; to be populated
	// only if ExtenderConfig.NodeCacheCapable == false
	Nodes *v1.NodeList
	// List of candidate node names where the pod can be scheduled; to be
	// populated only if ExtenderConfig.NodeCacheCapable == true
	NodeNames *[]string
}

// FailedNodesMap represents the filtered out nodes, with node names and failure messages
type FailedNodesMap map[string]string

// ExtenderFilterResult represents the results of a filter call to an extender
type ExtenderFilterResult struct {
	// Filtered set of nodes where the pod can be scheduled; to be populated
	// only if ExtenderConfig.NodeCacheCapable == false
	Nodes *v1.NodeList
	// Filtered set of nodes where the pod can be scheduled; to be populated
	// only if ExtenderConfig.NodeCacheCapable == true
	NodeNames *[]string
	// Filtered out nodes where the pod can't be scheduled and the failure messages
	FailedNodes FailedNodesMap
	// Error message indicating failure
	Error string
}

// ExtenderBindingArgs represents the arguments to an extender for binding a pod to a node.
type ExtenderBindingArgs struct {
	// PodName is the name of the pod being bound
	PodName string
	// PodNamespace is the namespace of the pod being bound
	PodNamespace string
	// PodUID is the UID of the pod being bound
	PodUID types.UID
	// Node selected by the scheduler
	Node string
}

// ExtenderBindingResult represents the result of binding of a pod to a node from an extender.
type ExtenderBindingResult struct {
	// Error message indicating failure
	Error string
}

// HostPriority represents the priority of scheduling to a particular host, higher priority is better.
type HostPriority struct {
	// Name of the host
	Host string
	// Score associated with the host
	Score int
}

// HostPriorityList declares a []HostPriority type.
type HostPriorityList []HostPriority

func (h HostPriorityList) Len() int {
	return len(h)
}

func (h HostPriorityList) Less(i, j int) bool {
	if h[i].Score == h[j].Score {
		return h[i].Host < h[j].Host
	}
	return h[i].Score < h[j].Score
}

func (h HostPriorityList) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}
