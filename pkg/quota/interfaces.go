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

package quota

import (
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

// UsageStatsOptions is an options structs that describes how stats should be calculated
type UsageStatsOptions struct {
	// Namespace where stats should be calculate
	Namespace string
	// Scopes that must match counted objects
	Scopes []api.ResourceQuotaScope
}

// UsageStats is result of measuring observed resource use in the system
type UsageStats struct {
	// Used maps resource to quantity used
	Used api.ResourceList
}

// Evaluator knows how to evaluate quota usage for a particular group kind
type Evaluator interface {
	// Constraints ensures that each required resource is present on item
	Constraints(required []api.ResourceName, item runtime.Object) error
	// Get returns the object with specified namespace and name
	Get(namespace, name string) (runtime.Object, error)
	// GroupKind returns the groupKind that this object knows how to evaluate
	GroupKind() unversioned.GroupKind
	// MatchesResources is the list of resources that this evaluator matches
	MatchesResources() []api.ResourceName
	// Matches returns true if the specified quota matches the input item
	Matches(resourceQuota *api.ResourceQuota, item runtime.Object) bool
	// OperationResources returns the set of resources that could be updated for the
	// specified operation for this kind.  If empty, admission control will ignore
	// quota processing for the operation.
	OperationResources(operation admission.Operation) []api.ResourceName
	// Usage returns the resource usage for the specified object
	Usage(object runtime.Object) api.ResourceList
	// UsageStats calculates latest observed usage stats for all objects
	UsageStats(options UsageStatsOptions) (UsageStats, error)
}

// Registry holds the list of evaluators associated to a particular group kind
type Registry interface {
	// Evaluators returns the set Evaluator objects registered to a groupKind
	Evaluators() map[unversioned.GroupKind]Evaluator
}
