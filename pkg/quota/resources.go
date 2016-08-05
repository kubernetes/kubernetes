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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/util/resources"
)

// CalculateUsage calculates and returns the requested ResourceList usage
func CalculateUsage(namespaceName string, scopes []api.ResourceQuotaScope, hardLimits api.ResourceList, registry Registry) (api.ResourceList, error) {
	// find the intersection between the hard resources on the quota
	// and the resources this controller can track to know what we can
	// look to measure updated usage stats for
	hardResources := resources.ResourceNames(hardLimits)
	potentialResources := []api.ResourceName{}
	evaluators := registry.Evaluators()
	for _, evaluator := range evaluators {
		potentialResources = append(potentialResources, evaluator.MatchesResources()...)
	}
	matchedResources := resources.Intersection(hardResources, potentialResources)

	// sum the observed usage from each evaluator
	newUsage := api.ResourceList{}
	usageStatsOptions := UsageStatsOptions{Namespace: namespaceName, Scopes: scopes}
	for _, evaluator := range evaluators {
		// only trigger the evaluator if it matches a resource in the quota, otherwise, skip calculating anything
		if intersection := resources.Intersection(evaluator.MatchesResources(), matchedResources); len(intersection) == 0 {
			continue
		}
		stats, err := evaluator.UsageStats(usageStatsOptions)
		if err != nil {
			return nil, err
		}
		newUsage = resources.Add(newUsage, stats.Used)
	}

	// mask the observed usage to only the set of resources tracked by this quota
	// merge our observed usage with the quota usage status
	// if the new usage is different than the last usage, we will need to do an update
	newUsage = resources.Mask(newUsage, matchedResources)
	return newUsage, nil
}
