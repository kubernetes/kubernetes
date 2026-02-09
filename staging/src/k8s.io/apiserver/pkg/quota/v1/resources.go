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

package v1

import (
	"sort"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
)

// Equals returns true if the two lists are equivalent
func Equals(a corev1.ResourceList, b corev1.ResourceList) bool {
	if len(a) != len(b) {
		return false
	}

	for key, value1 := range a {
		value2, found := b[key]
		if !found {
			return false
		}
		if value1.Cmp(value2) != 0 {
			return false
		}
	}

	return true
}

// LessThanOrEqual returns true if a < b for each key in b
// If false, it returns the keys in a that exceeded b
func LessThanOrEqual(a corev1.ResourceList, b corev1.ResourceList) (bool, []corev1.ResourceName) {
	result := true
	resourceNames := []corev1.ResourceName{}
	for key, value := range b {
		if other, found := a[key]; found {
			if other.Cmp(value) > 0 {
				result = false
				resourceNames = append(resourceNames, key)
			}
		}
	}
	return result, resourceNames
}

// Max returns the result of Max(a, b) for each named resource
func Max(a corev1.ResourceList, b corev1.ResourceList) corev1.ResourceList {
	result := corev1.ResourceList{}
	for key, value := range a {
		if other, found := b[key]; found {
			if value.Cmp(other) <= 0 {
				result[key] = other.DeepCopy()
				continue
			}
		}
		result[key] = value.DeepCopy()
	}
	for key, value := range b {
		if _, found := result[key]; !found {
			result[key] = value.DeepCopy()
		}
	}
	return result
}

// Add returns the result of a + b for each named resource
func Add(a corev1.ResourceList, b corev1.ResourceList) corev1.ResourceList {
	result := corev1.ResourceList{}
	for key, value := range a {
		quantity := value.DeepCopy()
		if other, found := b[key]; found {
			quantity.Add(other)
		}
		result[key] = quantity
	}
	for key, value := range b {
		if _, found := result[key]; !found {
			result[key] = value.DeepCopy()
		}
	}
	return result
}

// SubtractWithNonNegativeResult - subtracts and returns result of a - b but
// makes sure we don't return negative values to prevent negative resource usage.
func SubtractWithNonNegativeResult(a corev1.ResourceList, b corev1.ResourceList) corev1.ResourceList {
	zero := resource.MustParse("0")

	result := corev1.ResourceList{}
	for key, value := range a {
		quantity := value.DeepCopy()
		if other, found := b[key]; found {
			quantity.Sub(other)
		}
		if quantity.Cmp(zero) > 0 {
			result[key] = quantity
		} else {
			result[key] = zero
		}
	}

	for key := range b {
		if _, found := result[key]; !found {
			result[key] = zero
		}
	}
	return result
}

// Subtract returns the result of a - b for each named resource
func Subtract(a corev1.ResourceList, b corev1.ResourceList) corev1.ResourceList {
	result := corev1.ResourceList{}
	for key, value := range a {
		quantity := value.DeepCopy()
		if other, found := b[key]; found {
			quantity.Sub(other)
		}
		result[key] = quantity
	}
	for key, value := range b {
		if _, found := result[key]; !found {
			quantity := value.DeepCopy()
			quantity.Neg()
			result[key] = quantity
		}
	}
	return result
}

// Mask returns a new resource list that only has the values with the specified names
func Mask(resources corev1.ResourceList, names []corev1.ResourceName) corev1.ResourceList {
	nameSet := ToSet(names)
	result := corev1.ResourceList{}
	for key, value := range resources {
		if nameSet.Has(string(key)) {
			result[key] = value.DeepCopy()
		}
	}
	return result
}

// ResourceNames returns a list of all resource names in the ResourceList
func ResourceNames(resources corev1.ResourceList) []corev1.ResourceName {
	result := []corev1.ResourceName{}
	for resourceName := range resources {
		result = append(result, resourceName)
	}
	return result
}

// Contains returns true if the specified item is in the list of items
func Contains(items []corev1.ResourceName, item corev1.ResourceName) bool {
	for _, i := range items {
		if i == item {
			return true
		}
	}
	return false
}

// ContainsPrefix returns true if the specified item has a prefix that contained in given prefix Set
func ContainsPrefix(prefixSet []string, item corev1.ResourceName) bool {
	for _, prefix := range prefixSet {
		if strings.HasPrefix(string(item), prefix) {
			return true
		}
	}
	return false
}

// Intersection returns the intersection of both list of resources, deduped and sorted
func Intersection(a []corev1.ResourceName, b []corev1.ResourceName) []corev1.ResourceName {
	result := make([]corev1.ResourceName, 0, len(a))
	for _, item := range a {
		if Contains(result, item) {
			continue
		}
		if !Contains(b, item) {
			continue
		}
		result = append(result, item)
	}
	sort.Slice(result, func(i, j int) bool { return result[i] < result[j] })
	return result
}

// Difference returns the list of resources resulting from a-b, deduped and sorted
func Difference(a []corev1.ResourceName, b []corev1.ResourceName) []corev1.ResourceName {
	result := make([]corev1.ResourceName, 0, len(a))
	for _, item := range a {
		if Contains(b, item) || Contains(result, item) {
			continue
		}
		result = append(result, item)
	}
	sort.Slice(result, func(i, j int) bool { return result[i] < result[j] })
	return result
}

// IsZero returns true if each key maps to the quantity value 0
func IsZero(a corev1.ResourceList) bool {
	zero := resource.MustParse("0")
	for _, v := range a {
		if v.Cmp(zero) != 0 {
			return false
		}
	}
	return true
}

// RemoveZeros returns a new resource list that only has no zero values
func RemoveZeros(a corev1.ResourceList) corev1.ResourceList {
	result := corev1.ResourceList{}
	for key, value := range a {
		if !value.IsZero() {
			result[key] = value
		}
	}
	return result
}

// IsNegative returns the set of resource names that have a negative value.
func IsNegative(a corev1.ResourceList) []corev1.ResourceName {
	results := []corev1.ResourceName{}
	zero := resource.MustParse("0")
	for k, v := range a {
		if v.Cmp(zero) < 0 {
			results = append(results, k)
		}
	}
	return results
}

// ToSet takes a list of resource names and converts to a string set
func ToSet(resourceNames []corev1.ResourceName) sets.String {
	result := sets.NewString()
	for _, resourceName := range resourceNames {
		result.Insert(string(resourceName))
	}
	return result
}

// CalculateUsage calculates and returns the requested ResourceList usage.
// If an error is returned, usage only contains the resources which encountered no calculation errors.
func CalculateUsage(namespaceName string, scopes []corev1.ResourceQuotaScope, hardLimits corev1.ResourceList, registry Registry, scopeSelector *corev1.ScopeSelector) (corev1.ResourceList, error) {
	// find the intersection between the hard resources on the quota
	// and the resources this controller can track to know what we can
	// look to measure updated usage stats for
	hardResources := ResourceNames(hardLimits)
	potentialResources := []corev1.ResourceName{}
	evaluators := registry.List()
	for _, evaluator := range evaluators {
		potentialResources = append(potentialResources, evaluator.MatchingResources(hardResources)...)
	}
	// NOTE: the intersection just removes duplicates since the evaluator match intersects with hard
	matchedResources := Intersection(hardResources, potentialResources)

	errors := []error{}

	// sum the observed usage from each evaluator
	newUsage := corev1.ResourceList{}
	for _, evaluator := range evaluators {
		// only trigger the evaluator if it matches a resource in the quota, otherwise, skip calculating anything
		intersection := evaluator.MatchingResources(matchedResources)
		if len(intersection) == 0 {
			continue
		}

		usageStatsOptions := UsageStatsOptions{Namespace: namespaceName, Scopes: scopes, Resources: intersection, ScopeSelector: scopeSelector}
		stats, err := evaluator.UsageStats(usageStatsOptions)
		if err != nil {
			// remember the error
			errors = append(errors, err)
			// exclude resources which encountered calculation errors
			matchedResources = Difference(matchedResources, intersection)
			continue
		}
		newUsage = Add(newUsage, stats.Used)
	}

	// mask the observed usage to only the set of resources tracked by this quota
	// merge our observed usage with the quota usage status
	// if the new usage is different than the last usage, we will need to do an update
	newUsage = Mask(newUsage, matchedResources)
	return newUsage, utilerrors.NewAggregate(errors)
}
