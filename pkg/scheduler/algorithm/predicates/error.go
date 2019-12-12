/*
Copyright 2016 The Kubernetes Authors.

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

package predicates

import (
	"fmt"

	"k8s.io/api/core/v1"
)

var (
	// The predicateName tries to be consistent as the predicate name used in DefaultAlgorithmProvider defined in
	// defaults.go (which tend to be stable for backward compatibility)

	// NOTE: If you add a new predicate failure error for a predicate that can never
	// be made to pass by removing pods, or you change an existing predicate so that
	// it can never be made to pass by removing pods, you need to add the predicate
	// failure error in nodesWherePreemptionMightHelp() in scheduler/core/generic_scheduler.go

	// ErrDiskConflict is used for NoDiskConflict predicate error.
	ErrDiskConflict = NewPredicateFailureError("NoDiskConflict", "node(s) had no available disk")
	// ErrVolumeZoneConflict is used for NoVolumeZoneConflict predicate error.
	ErrVolumeZoneConflict = NewPredicateFailureError("NoVolumeZoneConflict", "node(s) had no available volume zone")
	// ErrNodeSelectorNotMatch is used for MatchNodeSelector predicate error.
	ErrNodeSelectorNotMatch = NewPredicateFailureError("MatchNodeSelector", "node(s) didn't match node selector")
	// ErrPodAffinityNotMatch is used for MatchInterPodAffinity predicate error.
	ErrPodAffinityNotMatch = NewPredicateFailureError("MatchInterPodAffinity", "node(s) didn't match pod affinity/anti-affinity")
	// ErrPodAffinityRulesNotMatch is used for PodAffinityRulesNotMatch predicate error.
	ErrPodAffinityRulesNotMatch = NewPredicateFailureError("PodAffinityRulesNotMatch", "node(s) didn't match pod affinity rules")
	// ErrPodAntiAffinityRulesNotMatch is used for PodAntiAffinityRulesNotMatch predicate error.
	ErrPodAntiAffinityRulesNotMatch = NewPredicateFailureError("PodAntiAffinityRulesNotMatch", "node(s) didn't match pod anti-affinity rules")
	// ErrExistingPodsAntiAffinityRulesNotMatch is used for ExistingPodsAntiAffinityRulesNotMatch predicate error.
	ErrExistingPodsAntiAffinityRulesNotMatch = NewPredicateFailureError("ExistingPodsAntiAffinityRulesNotMatch", "node(s) didn't satisfy existing pods anti-affinity rules")
	// ErrTaintsTolerationsNotMatch is used for PodToleratesNodeTaints predicate error.
	ErrTaintsTolerationsNotMatch = NewPredicateFailureError("PodToleratesNodeTaints", "node(s) had taints that the pod didn't tolerate")
	// ErrPodNotMatchHostName is used for HostName predicate error.
	ErrPodNotMatchHostName = NewPredicateFailureError("HostName", "node(s) didn't match the requested hostname")
	// ErrPodNotFitsHostPorts is used for PodFitsHostPorts predicate error.
	ErrPodNotFitsHostPorts = NewPredicateFailureError("PodFitsHostPorts", "node(s) didn't have free ports for the requested pod ports")
	// ErrNodeLabelPresenceViolated is used for CheckNodeLabelPresence predicate error.
	ErrNodeLabelPresenceViolated = NewPredicateFailureError("CheckNodeLabelPresence", "node(s) didn't have the requested labels")
	// ErrServiceAffinityViolated is used for CheckServiceAffinity predicate error.
	ErrServiceAffinityViolated = NewPredicateFailureError("CheckServiceAffinity", "node(s) didn't match service affinity")
	// ErrMaxVolumeCountExceeded is used for MaxVolumeCount predicate error.
	ErrMaxVolumeCountExceeded = NewPredicateFailureError("MaxVolumeCount", "node(s) exceed max volume count")
	// ErrNodeUnderMemoryPressure is used for NodeUnderMemoryPressure predicate error.
	ErrNodeUnderMemoryPressure = NewPredicateFailureError("NodeUnderMemoryPressure", "node(s) had memory pressure")
	// ErrNodeUnderDiskPressure is used for NodeUnderDiskPressure predicate error.
	ErrNodeUnderDiskPressure = NewPredicateFailureError("NodeUnderDiskPressure", "node(s) had disk pressure")
	// ErrNodeUnderPIDPressure is used for NodeUnderPIDPressure predicate error.
	ErrNodeUnderPIDPressure = NewPredicateFailureError("NodeUnderPIDPressure", "node(s) had pid pressure")
	// ErrNodeNotReady is used for NodeNotReady predicate error.
	ErrNodeNotReady = NewPredicateFailureError("NodeNotReady", "node(s) were not ready")
	// ErrNodeNetworkUnavailable is used for NodeNetworkUnavailable predicate error.
	ErrNodeNetworkUnavailable = NewPredicateFailureError("NodeNetworkUnavailable", "node(s) had unavailable network")
	// ErrNodeUnschedulable is used for NodeUnschedulable predicate error.
	ErrNodeUnschedulable = NewPredicateFailureError("NodeUnschedulable", "node(s) were unschedulable")
	// ErrNodeUnknownCondition is used for NodeUnknownCondition predicate error.
	ErrNodeUnknownCondition = NewPredicateFailureError("NodeUnknownCondition", "node(s) had unknown conditions")
	// ErrVolumeNodeConflict is used for VolumeNodeAffinityConflict predicate error.
	ErrVolumeNodeConflict = NewPredicateFailureError("VolumeNodeAffinityConflict", "node(s) had volume node affinity conflict")
	// ErrVolumeBindConflict is used for VolumeBindingNoMatch predicate error.
	ErrVolumeBindConflict = NewPredicateFailureError("VolumeBindingNoMatch", "node(s) didn't find available persistent volumes to bind")
	// ErrTopologySpreadConstraintsNotMatch is used for EvenPodsSpread predicate error.
	ErrTopologySpreadConstraintsNotMatch = NewPredicateFailureError("EvenPodsSpreadNotMatch", "node(s) didn't match pod topology spread constraints")
	// ErrFakePredicate is used for test only. The fake predicates returning false also returns error
	// as ErrFakePredicate.
	ErrFakePredicate = NewPredicateFailureError("FakePredicateError", "Nodes failed the fake predicate")
)

var unresolvablePredicateFailureErrors = map[PredicateFailureReason]struct{}{
	ErrNodeSelectorNotMatch:      {},
	ErrPodAffinityRulesNotMatch:  {},
	ErrPodNotMatchHostName:       {},
	ErrTaintsTolerationsNotMatch: {},
	ErrNodeLabelPresenceViolated: {},
	// Node conditions won't change when scheduler simulates removal of preemption victims.
	// So, it is pointless to try nodes that have not been able to host the pod due to node
	// conditions. These include ErrNodeNotReady, ErrNodeUnderPIDPressure, ErrNodeUnderMemoryPressure, ....
	ErrNodeNotReady:            {},
	ErrNodeNetworkUnavailable:  {},
	ErrNodeUnderDiskPressure:   {},
	ErrNodeUnderPIDPressure:    {},
	ErrNodeUnderMemoryPressure: {},
	ErrNodeUnschedulable:       {},
	ErrNodeUnknownCondition:    {},
	ErrVolumeZoneConflict:      {},
	ErrVolumeNodeConflict:      {},
	ErrVolumeBindConflict:      {},
}

// UnresolvablePredicateExists checks if there is at least one unresolvable predicate failure reason.
func UnresolvablePredicateExists(reasons []PredicateFailureReason) bool {
	for _, r := range reasons {
		if _, ok := unresolvablePredicateFailureErrors[r]; ok {
			return true
		}
	}
	return false
}

// InsufficientResourceError is an error type that indicates what kind of resource limit is
// hit and caused the unfitting failure.
type InsufficientResourceError struct {
	// resourceName is the name of the resource that is insufficient
	ResourceName v1.ResourceName
	requested    int64
	used         int64
	capacity     int64
}

// NewInsufficientResourceError returns an InsufficientResourceError.
func NewInsufficientResourceError(resourceName v1.ResourceName, requested, used, capacity int64) *InsufficientResourceError {
	return &InsufficientResourceError{
		ResourceName: resourceName,
		requested:    requested,
		used:         used,
		capacity:     capacity,
	}
}

func (e *InsufficientResourceError) Error() string {
	return fmt.Sprintf("Node didn't have enough resource: %s, requested: %d, used: %d, capacity: %d",
		e.ResourceName, e.requested, e.used, e.capacity)
}

// GetReason returns the reason of the InsufficientResourceError.
func (e *InsufficientResourceError) GetReason() string {
	return fmt.Sprintf("Insufficient %v", e.ResourceName)
}

// GetInsufficientAmount returns the amount of the insufficient resource of the error.
func (e *InsufficientResourceError) GetInsufficientAmount() int64 {
	return e.requested - (e.capacity - e.used)
}

// PredicateFailureError describes a failure error of predicate.
type PredicateFailureError struct {
	PredicateName string
	PredicateDesc string
}

// NewPredicateFailureError creates a PredicateFailureError with message.
func NewPredicateFailureError(predicateName, predicateDesc string) *PredicateFailureError {
	return &PredicateFailureError{PredicateName: predicateName, PredicateDesc: predicateDesc}
}

func (e *PredicateFailureError) Error() string {
	return fmt.Sprintf("Predicate %s failed", e.PredicateName)
}

// GetReason returns the reason of the PredicateFailureError.
func (e *PredicateFailureError) GetReason() string {
	return e.PredicateDesc
}

// PredicateFailureReason interface represents the failure reason of a predicate.
type PredicateFailureReason interface {
	GetReason() string
}
