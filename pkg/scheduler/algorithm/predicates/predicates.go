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

package predicates

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	pluginhelper "k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// DEPRECATED: all the logic in this package exist only because kubelet uses it.

// InsufficientResourceError is an error type that indicates what kind of resource limit is
// hit and caused the unfitting failure.
type InsufficientResourceError struct {
	noderesources.InsufficientResource
}

func (e *InsufficientResourceError) Error() string {
	return fmt.Sprintf("Node didn't have enough resource: %s, requested: %d, used: %d, capacity: %d",
		e.ResourceName, e.Requested, e.Used, e.Capacity)
}

// PredicateFailureReason interface represents the failure reason of a predicate.
type PredicateFailureReason interface {
	GetReason() string
}

// GetReason returns the reason of the InsufficientResourceError.
func (e *InsufficientResourceError) GetReason() string {
	return fmt.Sprintf("Insufficient %v", e.ResourceName)
}

// GetInsufficientAmount returns the amount of the insufficient resource of the error.
func (e *InsufficientResourceError) GetInsufficientAmount() int64 {
	return e.Requested - (e.Capacity - e.Used)
}

// PredicateFailureError describes a failure error of predicate.
type PredicateFailureError struct {
	PredicateName string
	PredicateDesc string
}

func (e *PredicateFailureError) Error() string {
	return fmt.Sprintf("Predicate %s failed", e.PredicateName)
}

// GetReason returns the reason of the PredicateFailureError.
func (e *PredicateFailureError) GetReason() string {
	return e.PredicateDesc
}

// GeneralPredicates checks a group of predicates that the kubelet cares about.
func GeneralPredicates(pod *v1.Pod, _ interface{}, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	if nodeInfo.Node() == nil {
		return false, nil, fmt.Errorf("node not found")
	}

	var reasons []PredicateFailureReason
	for _, r := range noderesources.Fits(pod, nodeInfo, nil) {
		reasons = append(reasons, &InsufficientResourceError{InsufficientResource: r})
	}

	if !pluginhelper.PodMatchesNodeSelectorAndAffinityTerms(pod, nodeInfo.Node()) {
		reasons = append(reasons, &PredicateFailureError{nodeaffinity.Name, nodeaffinity.ErrReason})
	}
	if !nodename.Fits(pod, nodeInfo) {
		reasons = append(reasons, &PredicateFailureError{nodename.Name, nodename.ErrReason})
	}
	if !nodeports.Fits(pod, nodeInfo) {
		reasons = append(reasons, &PredicateFailureError{nodeports.Name, nodeports.ErrReason})
	}

	return len(reasons) == 0, reasons, nil
}
