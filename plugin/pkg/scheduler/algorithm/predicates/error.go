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
	ErrDiskConflict              = newPredicateFailureError("NoDiskConflict")
	ErrVolumeZoneConflict        = newPredicateFailureError("NoVolumeZoneConflict")
	ErrNodeSelectorNotMatch      = newPredicateFailureError("MatchNodeSelector")
	ErrPodAffinityNotMatch       = newPredicateFailureError("MatchInterPodAffinity")
	ErrTaintsTolerationsNotMatch = newPredicateFailureError("PodToleratesNodeTaints")
	ErrPodNotMatchHostName       = newPredicateFailureError("HostName")
	ErrPodNotFitsHostPorts       = newPredicateFailureError("PodFitsHostPorts")
	ErrNodeLabelPresenceViolated = newPredicateFailureError("CheckNodeLabelPresence")
	ErrServiceAffinityViolated   = newPredicateFailureError("CheckServiceAffinity")
	ErrMaxVolumeCountExceeded    = newPredicateFailureError("MaxVolumeCount")
	ErrNodeUnderMemoryPressure   = newPredicateFailureError("NodeUnderMemoryPressure")
	ErrNodeUnderDiskPressure     = newPredicateFailureError("NodeUnderDiskPressure")
	ErrNodeOutOfDisk             = newPredicateFailureError("NodeOutOfDisk")
	ErrNodeNotReady              = newPredicateFailureError("NodeNotReady")
	ErrNodeNetworkUnavailable    = newPredicateFailureError("NodeNetworkUnavailable")
	ErrNodeUnschedulable         = newPredicateFailureError("NodeUnschedulable")
	ErrNodeUnknownCondition      = newPredicateFailureError("NodeUnknownCondition")
	ErrVolumeNodeConflict        = newPredicateFailureError("NoVolumeNodeConflict")
	// ErrFakePredicate is used for test only. The fake predicates returning false also returns error
	// as ErrFakePredicate.
	ErrFakePredicate = newPredicateFailureError("FakePredicateError")
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

func (e *InsufficientResourceError) GetReason() string {
	return fmt.Sprintf("Insufficient %v", e.ResourceName)
}

func (e *InsufficientResourceError) GetInsufficientAmount() int64 {
	return e.requested - (e.capacity - e.used)
}

type PredicateFailureError struct {
	PredicateName string
}

func newPredicateFailureError(predicateName string) *PredicateFailureError {
	return &PredicateFailureError{PredicateName: predicateName}
}

func (e *PredicateFailureError) Error() string {
	return fmt.Sprintf("Predicate %s failed", e.PredicateName)
}

func (e *PredicateFailureError) GetReason() string {
	return e.PredicateName
}

type FailureReason struct {
	reason string
}

func NewFailureReason(msg string) *FailureReason {
	return &FailureReason{reason: msg}
}

func (e *FailureReason) GetReason() string {
	return e.reason
}
