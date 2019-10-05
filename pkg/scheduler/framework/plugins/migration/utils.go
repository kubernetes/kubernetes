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
