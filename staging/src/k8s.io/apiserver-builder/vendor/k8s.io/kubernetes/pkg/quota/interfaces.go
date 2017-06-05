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

package quota

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
)

// UsageStatsOptions is an options structs that describes how stats should be calculated
type UsageStatsOptions struct {
	// Namespace where stats should be calculate
	Namespace string
	// Scopes that must match counted objects
	Scopes []api.ResourceQuotaScope
	// Resources are the set of resources to include in the measurement
	Resources []api.ResourceName
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
	// GroupKind returns the groupKind that this object knows how to evaluate
	GroupKind() schema.GroupKind
	// Handles determines if quota could be impacted by the specified operation.
	// If true, admission control must perform quota processing for the operation, otherwise it is safe to ignore quota.
	Handles(operation admission.Operation) bool
	// Matches returns true if the specified quota matches the input item
	Matches(resourceQuota *api.ResourceQuota, item runtime.Object) (bool, error)
	// MatchingResources takes the input specified list of resources and returns the set of resources evaluator matches.
	MatchingResources(input []api.ResourceName) []api.ResourceName
	// Usage returns the resource usage for the specified object
	Usage(item runtime.Object) (api.ResourceList, error)
	// UsageStats calculates latest observed usage stats for all objects
	UsageStats(options UsageStatsOptions) (UsageStats, error)
}

// Registry holds the list of evaluators associated to a particular group kind
type Registry interface {
	// Evaluators returns the set Evaluator objects registered to a groupKind
	Evaluators() map[schema.GroupKind]Evaluator
}

// UnionRegistry combines multiple registries.  Order matters because first registry to claim a GroupKind
// is the "winner"
type UnionRegistry []Registry

// Evaluators returns a mapping of evaluators by group kind.
func (r UnionRegistry) Evaluators() map[schema.GroupKind]Evaluator {
	ret := map[schema.GroupKind]Evaluator{}

	for i := len(r) - 1; i >= 0; i-- {
		for k, v := range r[i].Evaluators() {
			ret[k] = v
		}
	}

	return ret
}
