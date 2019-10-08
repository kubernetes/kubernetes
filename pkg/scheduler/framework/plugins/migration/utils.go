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
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

const (
	// PredicatesStateKey is the key in CycleState to PredicateStateData
	PredicatesStateKey = "predicates"

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

	if r := predicates.UnresolvablePredicateExists(reasons); r != nil {
		return framework.NewStatus(framework.UnschedulableAndUnresolvable, r.GetReason())
	}

	// We will just use the first reason.
	return framework.NewStatus(framework.Unschedulable, reasons[0].GetReason())
}

// ErrorToFrameworkStatus converts an error to a framework status.
func ErrorToFrameworkStatus(err error) *framework.Status {
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	return nil
}

// PredicatesStateData is a pointer to PredicateMetadata. In the normal case, StateData is supposed to
// be generated and stored in CycleState by a framework plugin (like a PreFilter pre-computing data for
// its corresponding Filter). However, during migration, the scheduler will inject a pointer to
// PredicateMetadata into CycleState. This "hack" is necessary because during migration Filters that implement
// predicates functionality will be calling into the existing predicate functions, and need
// to pass PredicateMetadata.
type PredicatesStateData struct {
	Reference interface{}
}

// Clone is supposed to make a copy of the data, but since this is just a pointer, we are practically
// just copying the pointer. This is ok because the actual reference to the PredicateMetadata
// copy that is made by generic_scheduler during preemption cycle will be injected again outside
// the framework.
func (p *PredicatesStateData) Clone() framework.StateData {
	return &PredicatesStateData{
		Reference: p.Reference,
	}
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
