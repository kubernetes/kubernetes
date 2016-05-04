/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

import "fmt"

const (
	podCountResourceName string = "PodCount"
	cpuResourceName      string = "CPU"
	memoryResoureceName  string = "Memory"
)

var (
	// The predicateName tries to be consistent as the predicate name used in DefaultAlgorithmProvider defined in
	// defaults.go (which tend to be stable for backward compatibility)
	ErrDiskConflict              = newPredicateFailureError("NoDiskConflict")
	ErrVolumeZoneConflict        = newPredicateFailureError("NoVolumeZoneConflict")
	ErrNodeSelectorNotMatch      = newPredicateFailureError("MatchNodeSelector")
	ErrPodAffinityNotMatch       = newPredicateFailureError("MatchInterPodAffinity")
	ErrPodNotMatchHostName       = newPredicateFailureError("HostName")
	ErrPodNotFitsHostPorts       = newPredicateFailureError("PodFitsHostPorts")
	ErrNodeLabelPresenceViolated = newPredicateFailureError("CheckNodeLabelPresence")
	ErrServiceAffinityViolated   = newPredicateFailureError("CheckServiceAffinity")
	ErrMaxVolumeCountExceeded    = newPredicateFailureError("MaxVolumeCount")
	// ErrFakePredicate is used for test only. The fake predicates returning false also returns error
	// as ErrFakePredicate.
	ErrFakePredicate = newPredicateFailureError("FakePredicateError")
)

// InsufficientResourceError is an error type that indicates what kind of resource limit is
// hit and caused the unfitting failure.
type InsufficientResourceError struct {
	// resourceName is the name of the resource that is insufficient
	ResourceName string
	requested    int64
	used         int64
	capacity     int64
}

func newInsufficientResourceError(resourceName string, requested, used, capacity int64) *InsufficientResourceError {
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

type PredicateFailureError struct {
	PredicateName string
}

func newPredicateFailureError(predicateName string) *PredicateFailureError {
	return &PredicateFailureError{predicateName}
}

func (e *PredicateFailureError) Error() string {
	return fmt.Sprintf("Predicate %s failed", e.PredicateName)
}
