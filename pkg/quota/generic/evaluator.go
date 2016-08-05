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

package generic

import (
	"fmt"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/runtime"
)

// ConstraintsFunc takes a list of required resources that must match on the input item
type ConstraintsFunc func(required []api.ResourceName, item runtime.Object) error

// GetFuncByNamespace knows how to get a resource with specified namespace and name
type GetFuncByNamespace func(namespace, name string) (runtime.Object, error)

// ListFuncByNamespace knows how to list resources in a namespace
type ListFuncByNamespace func(namespace string, options api.ListOptions) (runtime.Object, error)

// MatchesScopeFunc knows how to evaluate if an object matches a scope
type MatchesScopeFunc func(scope api.ResourceQuotaScope, object runtime.Object) bool

// UsageFunc knows how to measure usage associated with an object
type UsageFunc func(object runtime.Object) api.ResourceList

// MatchesNoScopeFunc returns false on all match checks
func MatchesNoScopeFunc(scope api.ResourceQuotaScope, object runtime.Object) bool {
	return false
}

// ObjectCountConstraintsFunc returns ConstraintsFunc that returns nil if the
// specified resource name is in the required set of resource names
func ObjectCountConstraintsFunc(resourceName api.ResourceName) ConstraintsFunc {
	return func(required []api.ResourceName, item runtime.Object) error {
		if !quota.Contains(required, resourceName) {
			return fmt.Errorf("missing %s", resourceName)
		}
		return nil
	}
}

// ObjectCountUsageFunc is useful if you are only counting your object
// It always returns 1 as the usage for the named resource
func ObjectCountUsageFunc(resourceName api.ResourceName) UsageFunc {
	return func(object runtime.Object) api.ResourceList {
		return api.ResourceList{
			resourceName: resource.MustParse("1"),
		}
	}
}

// GenericEvaluator provides an implementation for quota.Evaluator
type GenericEvaluator struct {
	// Name used for logging
	Name string
	// The GroupKind that this evaluator tracks
	InternalGroupKind unversioned.GroupKind
	// The set of resources that are pertinent to the mapped operation
	InternalOperationResources map[admission.Operation][]api.ResourceName
	// The set of resource names this evaluator matches
	MatchedResourceNames []api.ResourceName
	// A function that knows how to evaluate a matches scope request
	MatchesScopeFunc MatchesScopeFunc
	// A function that knows how to return usage for an object
	UsageFunc UsageFunc
	// A function that knows how to list resources by namespace
	ListFuncByNamespace ListFuncByNamespace
	// A function that knows how to get resource in a namespace
	// This function must be specified if the evaluator needs to handle UPDATE
	GetFuncByNamespace GetFuncByNamespace
	// A function that checks required constraints are satisfied
	ConstraintsFunc ConstraintsFunc
}

// Ensure that GenericEvaluator implements quota.Evaluator
var _ quota.Evaluator = &GenericEvaluator{}

// Constraints checks required constraints are satisfied on the input object
func (g *GenericEvaluator) Constraints(required []api.ResourceName, item runtime.Object) error {
	return g.ConstraintsFunc(required, item)
}

// Get returns the object by namespace and name
func (g *GenericEvaluator) Get(namespace, name string) (runtime.Object, error) {
	return g.GetFuncByNamespace(namespace, name)
}

// OperationResources returns the set of resources that could be updated for the
// specified operation for this kind.  If empty, admission control will ignore
// quota processing for the operation.
func (g *GenericEvaluator) OperationResources(operation admission.Operation) []api.ResourceName {
	return g.InternalOperationResources[operation]
}

// GroupKind that this evaluator tracks
func (g *GenericEvaluator) GroupKind() unversioned.GroupKind {
	return g.InternalGroupKind
}

// MatchesResources is the list of resources that this evaluator matches
func (g *GenericEvaluator) MatchesResources() []api.ResourceName {
	return g.MatchedResourceNames
}

// Matches returns true if the evaluator matches the specified quota with the provided input item
func (g *GenericEvaluator) Matches(resourceQuota *api.ResourceQuota, item runtime.Object) bool {
	if resourceQuota == nil {
		return false
	}

	// verify the quota matches on resource, by default its false
	matchResource := false
	for resourceName := range resourceQuota.Status.Hard {
		if g.MatchesResource(resourceName) {
			matchResource = true
			break
		}
	}
	// by default, no scopes matches all
	matchScope := true
	for _, scope := range resourceQuota.Spec.Scopes {
		matchScope = matchScope && g.MatchesScope(scope, item)
	}
	return matchResource && matchScope
}

// MatchesResource returns true if this evaluator can match on the specified resource
func (g *GenericEvaluator) MatchesResource(resourceName api.ResourceName) bool {
	for _, matchedResourceName := range g.MatchedResourceNames {
		if resourceName == matchedResourceName {
			return true
		}
	}
	return false
}

// MatchesScope returns true if the input object matches the specified scope
func (g *GenericEvaluator) MatchesScope(scope api.ResourceQuotaScope, object runtime.Object) bool {
	return g.MatchesScopeFunc(scope, object)
}

// Usage returns the resource usage for the specified object
func (g *GenericEvaluator) Usage(object runtime.Object) api.ResourceList {
	return g.UsageFunc(object)
}

// UsageStats calculates latest observed usage stats for all objects
func (g *GenericEvaluator) UsageStats(options quota.UsageStatsOptions) (quota.UsageStats, error) {
	// default each tracked resource to zero
	result := quota.UsageStats{Used: api.ResourceList{}}
	for _, resourceName := range g.MatchedResourceNames {
		result.Used[resourceName] = resource.MustParse("0")
	}
	list, err := g.ListFuncByNamespace(options.Namespace, api.ListOptions{})
	if err != nil {
		return result, fmt.Errorf("%s: Failed to list %v: %v", g.Name, g.GroupKind(), err)
	}
	_, err = meta.ListAccessor(list)
	if err != nil {
		return result, fmt.Errorf("%s: Unable to understand list result, does not appear to be a list %#v", g.Name, list)
	}
	items, err := meta.ExtractList(list)
	if err != nil {
		return result, fmt.Errorf("%s: Unable to understand list result %#v (%v)", g.Name, list, err)
	}
	for _, item := range items {
		// need to verify that the item matches the set of scopes
		matchesScopes := true
		for _, scope := range options.Scopes {
			if !g.MatchesScope(scope, item) {
				matchesScopes = false
			}
		}
		// only count usage if there was a match
		if matchesScopes {
			result.Used = quota.Add(result.Used, g.Usage(item))
		}
	}
	return result, nil
}
