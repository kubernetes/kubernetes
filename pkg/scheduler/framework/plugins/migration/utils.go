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

package migration

import (
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

const (
	// PrioritiesStateKey is the key in CycleState to PrioritiesStateData
	PrioritiesStateKey = "priorities"
)

// PredicateResultToFrameworkStatus converts a predicate result (PredicateFailureReason + error)
// to a framework status.
func PredicateResultToFrameworkStatus(reasons []predicates.PredicateFailureReason, err error) *framework.Status {
	if s := ErrorToFrameworkStatus(err); s != nil {
		return s
	}

	if len(reasons) == 0 {
		return nil
	}

	code := framework.Unschedulable
	if predicates.UnresolvablePredicateExists(reasons) {
		code = framework.UnschedulableAndUnresolvable
	}

	// We will keep all failure reasons.
	var failureReasons []string
	for _, reason := range reasons {
		failureReasons = append(failureReasons, reason.GetReason())
	}
	return framework.NewStatus(code, failureReasons...)
}

// ErrorToFrameworkStatus converts an error to a framework status.
func ErrorToFrameworkStatus(err error) *framework.Status {
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	return nil
}

// PrioritiesStateData is a pointer to PrioritiesMetadata.
type PrioritiesStateData struct {
	Reference interface{}
}

// Clone is supposed to make a copy of the data, but since this is just a pointer, we are practically
// just copying the pointer.
func (p *PrioritiesStateData) Clone() framework.StateData {
	return &PrioritiesStateData{
		Reference: p.Reference,
	}
}

// PriorityMetadata returns priority metadata stored in CycleState.
func PriorityMetadata(state *framework.CycleState) interface{} {
	if state == nil {
		return nil
	}

	var meta interface{}
	if s, err := state.Read(PrioritiesStateKey); err == nil {
		meta = s.(*PrioritiesStateData).Reference
	} else {
		klog.Errorf("reading key %q from CycleState, continuing without metadata: %v", PrioritiesStateKey, err)
	}
	return meta
}
