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

	v1 "k8s.io/api/core/v1"
)

var (
	// The predicateName tries to be consistent as the predicate name used in DefaultAlgorithmProvider defined in
	// defaults.go (which tend to be stable for backward compatibility)

	// NOTE: If you add a new predicate failure error for a predicate that can never
	// be made to pass by removing pods, or you change an existing predicate so that
	// it can never be made to pass by removing pods, you need to add the predicate
	// failure error in nodesWherePreemptionMightHelp() in scheduler/core/generic_scheduler.go

	// ErrNodeSelectorNotMatch is used for MatchNodeSelector predicate error.
	ErrNodeSelectorNotMatch = NewPredicateFailureError("MatchNodeSelector", "node(s) didn't match node selector")
	// ErrPodNotMatchHostName is used for HostName predicate error.
	ErrPodNotMatchHostName = NewPredicateFailureError("HostName", "node(s) didn't match the requested hostname")
	// ErrPodNotFitsHostPorts is used for PodFitsHostPorts predicate error.
	ErrPodNotFitsHostPorts = NewPredicateFailureError("PodFitsHostPorts", "node(s) didn't have free ports for the requested pod ports")
	// ErrNodeUnknownCondition is used for NodeUnknownCondition predicate error.
	ErrNodeUnknownCondition = NewPredicateFailureError("NodeUnknownCondition", "node(s) had unknown conditions")
)

var unresolvablePredicateFailureErrors = map[PredicateFailureReason]struct{}{
	ErrNodeSelectorNotMatch: {},
	ErrPodNotMatchHostName:  {},
	// Node conditions won't change when scheduler simulates removal of preemption victims.
	// So, it is pointless to try nodes that have not been able to host the pod due to node
	// conditions.
	ErrNodeUnknownCondition: {},
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
