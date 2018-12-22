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
	ErrDiskConflict = newPredicateFailureError("NoDiskConflict", "node(s) had no available disk")
	// ErrVolumeZoneConflict is used for NoVolumeZoneConflict predicate error.
	ErrVolumeZoneConflict = newPredicateFailureError("NoVolumeZoneConflict", "node(s) had no available volume zone")
	// ErrNodeSelectorNotMatch is used for MatchNodeSelector predicate error.
	ErrNodeSelectorNotMatch = newPredicateFailureError("MatchNodeSelector", "node(s) didn't match node selector")
	// ErrPodAffinityNotMatch is used for MatchInterPodAffinity predicate error.
	ErrPodAffinityNotMatch = newPredicateFailureError("MatchInterPodAffinity", "node(s) didn't match pod affinity/anti-affinity")
	// ErrPodAffinityRulesNotMatch is used for PodAffinityRulesNotMatch predicate error.
	ErrPodAffinityRulesNotMatch = newPredicateFailureError("PodAffinityRulesNotMatch", "node(s) didn't match pod affinity rules")
	// ErrPodAntiAffinityRulesNotMatch is used for PodAntiAffinityRulesNotMatch predicate error.
	ErrPodAntiAffinityRulesNotMatch = newPredicateFailureError("PodAntiAffinityRulesNotMatch", "node(s) didn't match pod anti-affinity rules")
	// ErrExistingPodsAntiAffinityRulesNotMatch is used for ExistingPodsAntiAffinityRulesNotMatch predicate error.
	ErrExistingPodsAntiAffinityRulesNotMatch = newPredicateFailureError("ExistingPodsAntiAffinityRulesNotMatch", "node(s) didn't satisfy existing pods anti-affinity rules")
	// ErrTaintsTolerationsNotMatch is used for PodToleratesNodeTaints predicate error.
	ErrTaintsTolerationsNotMatch = newPredicateFailureError("PodToleratesNodeTaints", "node(s) had taints that the pod didn't tolerate")
	// ErrPodNotMatchHostName is used for HostName predicate error.
	ErrPodNotMatchHostName = newPredicateFailureError("HostName", "node(s) didn't match the requested hostname")
	// ErrPodNotFitsHostPorts is used for PodFitsHostPorts predicate error.
	ErrPodNotFitsHostPorts = newPredicateFailureError("PodFitsHostPorts", "node(s) didn't have free ports for the requested pod ports")
	// ErrNodeLabelPresenceViolated is used for CheckNodeLabelPresence predicate error.
	ErrNodeLabelPresenceViolated = newPredicateFailureError("CheckNodeLabelPresence", "node(s) didn't have the requested labels")
	// ErrServiceAffinityViolated is used for CheckServiceAffinity predicate error.
	ErrServiceAffinityViolated = newPredicateFailureError("CheckServiceAffinity", "node(s) didn't match service affinity")
	// ErrMaxVolumeCountExceeded is used for MaxVolumeCount predicate error.
	ErrMaxVolumeCountExceeded = newPredicateFailureError("MaxVolumeCount", "node(s) exceed max volume count")
	// ErrNodeUnderMemoryPressure is used for NodeUnderMemoryPressure predicate error.
	ErrNodeUnderMemoryPressure = newPredicateFailureError("NodeUnderMemoryPressure", "node(s) had memory pressure")
	// ErrNodeUnderDiskPressure is used for NodeUnderDiskPressure predicate error.
	ErrNodeUnderDiskPressure = newPredicateFailureError("NodeUnderDiskPressure", "node(s) had disk pressure")
	// ErrNodeUnderPIDPressure is used for NodeUnderPIDPressure predicate error.
	ErrNodeUnderPIDPressure = newPredicateFailureError("NodeUnderPIDPressure", "node(s) had pid pressure")
	// ErrNodeNotReady is used for NodeNotReady predicate error.
	ErrNodeNotReady = newPredicateFailureError("NodeNotReady", "node(s) were not ready")
	// ErrNodeNetworkUnavailable is used for NodeNetworkUnavailable predicate error.
	ErrNodeNetworkUnavailable = newPredicateFailureError("NodeNetworkUnavailable", "node(s) had unavailable network")
	// ErrNodeUnschedulable is used for NodeUnschedulable predicate error.
	ErrNodeUnschedulable = newPredicateFailureError("NodeUnschedulable", "node(s) were unschedulable")
	// ErrNodeUnknownCondition is used for NodeUnknownCondition predicate error.
	ErrNodeUnknownCondition = newPredicateFailureError("NodeUnknownCondition", "node(s) had unknown conditions")
	// ErrVolumeNodeConflict is used for VolumeNodeAffinityConflict predicate error.
	ErrVolumeNodeConflict = newPredicateFailureError("VolumeNodeAffinityConflict", "node(s) had volume node affinity conflict")
	// ErrVolumeBindConflict is used for VolumeBindingNoMatch predicate error.
	ErrVolumeBindConflict = newPredicateFailureError("VolumeBindingNoMatch", "node(s) didn't find available persistent volumes to bind")
	// ErrFakePredicate is used for test only. The fake predicates returning false also returns error
	// as ErrFakePredicate.
	ErrFakePredicate = newPredicateFailureError("FakePredicateError", "Nodes failed the fake predicate")
)

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

func newPredicateFailureError(predicateName, predicateDesc string) *PredicateFailureError {
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

// FailureReason describes a failure reason.
type FailureReason struct {
	reason string
}

// NewFailureReason creates a FailureReason with message.
func NewFailureReason(msg string) *FailureReason {
	return &FailureReason{reason: msg}
}

// GetReason returns the reason of the FailureReason.
func (e *FailureReason) GetReason() string {
	return e.reason
}
