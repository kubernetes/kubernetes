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

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/informers"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/quota"
)

// ListResourceUsingInformerFunc returns a listing function based on the shared informer factory for the specified resource.
func ListResourceUsingInformerFunc(f informers.SharedInformerFactory, resource schema.GroupVersionResource) ListFuncByNamespace {
	return func(namespace string, options metav1.ListOptions) ([]runtime.Object, error) {
		labelSelector, err := labels.Parse(options.LabelSelector)
		if err != nil {
			return nil, err
		}
		informer, err := f.ForResource(resource)
		if err != nil {
			return nil, err
		}
		return informer.Lister().ByNamespace(namespace).List(labelSelector)
	}
}

// ListFuncByNamespace knows how to list resources in a namespace
type ListFuncByNamespace func(namespace string, options metav1.ListOptions) ([]runtime.Object, error)

// MatchesScopeFunc knows how to evaluate if an object matches a scope
type MatchesScopeFunc func(scope api.ResourceQuotaScope, object runtime.Object) (bool, error)

// UsageFunc knows how to measure usage associated with an object
type UsageFunc func(object runtime.Object) (api.ResourceList, error)

// MatchingResourceNamesFunc is a function that returns the list of resources matched
type MatchingResourceNamesFunc func(input []api.ResourceName) []api.ResourceName

// MatchesNoScopeFunc returns false on all match checks
func MatchesNoScopeFunc(scope api.ResourceQuotaScope, object runtime.Object) (bool, error) {
	return false, nil
}

// Matches returns true if the quota matches the specified item.
func Matches(resourceQuota *api.ResourceQuota, item runtime.Object, matchFunc MatchingResourceNamesFunc, scopeFunc MatchesScopeFunc) (bool, error) {
	if resourceQuota == nil {
		return false, fmt.Errorf("expected non-nil quota")
	}
	// verify the quota matches on at least one resource
	matchResource := len(matchFunc(quota.ResourceNames(resourceQuota.Status.Hard))) > 0
	// by default, no scopes matches all
	matchScope := true
	for _, scope := range resourceQuota.Spec.Scopes {
		innerMatch, err := scopeFunc(scope, item)
		if err != nil {
			return false, err
		}
		matchScope = matchScope && innerMatch
	}
	return matchResource && matchScope, nil
}

// CalculateUsageStats is a utility function that knows how to calculate aggregate usage.
func CalculateUsageStats(options quota.UsageStatsOptions,
	listFunc ListFuncByNamespace,
	scopeFunc MatchesScopeFunc,
	usageFunc UsageFunc) (quota.UsageStats, error) {
	// default each tracked resource to zero
	result := quota.UsageStats{Used: api.ResourceList{}}
	for _, resourceName := range options.Resources {
		result.Used[resourceName] = resource.Quantity{Format: resource.DecimalSI}
	}
	items, err := listFunc(options.Namespace, metav1.ListOptions{
		LabelSelector: labels.Everything().String(),
	})
	if err != nil {
		return result, fmt.Errorf("failed to list content: %v", err)
	}
	for _, item := range items {
		// need to verify that the item matches the set of scopes
		matchesScopes := true
		for _, scope := range options.Scopes {
			innerMatch, err := scopeFunc(scope, item)
			if err != nil {
				return result, nil
			}
			if !innerMatch {
				matchesScopes = false
			}
		}
		// only count usage if there was a match
		if matchesScopes {
			usage, err := usageFunc(item)
			if err != nil {
				return result, err
			}
			result.Used = quota.Add(result.Used, usage)
		}
	}
	return result, nil
}

// ObjectCountEvaluator provides an implementation for quota.Evaluator
// that associates usage of the specified resource based on the number of items
// returned by the specified listing function.
type ObjectCountEvaluator struct {
	// AllowCreateOnUpdate if true will ensure the evaluator tracks create
	// and update operations.
	AllowCreateOnUpdate bool
	// GroupKind that this evaluator tracks.
	InternalGroupKind schema.GroupKind
	// A function that knows how to list resources by namespace.
	// TODO move to dynamic client in future
	ListFuncByNamespace ListFuncByNamespace
	// Name associated with this resource in the quota.
	ResourceName api.ResourceName
}

// Constraints returns an error if the configured resource name is not in the required set.
func (o *ObjectCountEvaluator) Constraints(required []api.ResourceName, item runtime.Object) error {
	if !quota.Contains(required, o.ResourceName) {
		return fmt.Errorf("missing %s", o.ResourceName)
	}
	return nil
}

// GroupKind that this evaluator tracks
func (o *ObjectCountEvaluator) GroupKind() schema.GroupKind {
	return o.InternalGroupKind
}

// Handles returns true if the object count evaluator needs to track this attributes.
func (o *ObjectCountEvaluator) Handles(a admission.Attributes) bool {
	operation := a.GetOperation()
	return operation == admission.Create || (o.AllowCreateOnUpdate && operation == admission.Update)
}

// Matches returns true if the evaluator matches the specified quota with the provided input item
func (o *ObjectCountEvaluator) Matches(resourceQuota *api.ResourceQuota, item runtime.Object) (bool, error) {
	return Matches(resourceQuota, item, o.MatchingResources, MatchesNoScopeFunc)
}

// MatchingResources takes the input specified list of resources and returns the set of resources it matches.
func (o *ObjectCountEvaluator) MatchingResources(input []api.ResourceName) []api.ResourceName {
	return quota.Intersection(input, []api.ResourceName{o.ResourceName})
}

// Usage returns the resource usage for the specified object
func (o *ObjectCountEvaluator) Usage(object runtime.Object) (api.ResourceList, error) {
	quantity := resource.NewQuantity(1, resource.DecimalSI)
	return api.ResourceList{
		o.ResourceName: *quantity,
	}, nil
}

// UsageStats calculates aggregate usage for the object.
func (o *ObjectCountEvaluator) UsageStats(options quota.UsageStatsOptions) (quota.UsageStats, error) {
	return CalculateUsageStats(options, o.ListFuncByNamespace, MatchesNoScopeFunc, o.Usage)
}

// Verify implementation of interface at compile time.
var _ quota.Evaluator = &ObjectCountEvaluator{}
