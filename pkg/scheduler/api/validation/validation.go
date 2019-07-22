/*
Copyright 2015 The Kubernetes Authors.

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

package validation

import (
	"errors"
	"fmt"
	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/pkg/features"

	"k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"

	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
)

// ValidatePolicy checks for errors in the Config
// It does not return early so that it can find as many errors as possible
func ValidatePolicy(policy schedulerapi.Policy) error {
	var validationErrors []error
	if err := validateFeatureDependencies(policy); err != nil {
		validationErrors = append(validationErrors, err)
	}

	for _, priority := range policy.Priorities {
		if priority.Weight <= 0 || priority.Weight >= schedulerapi.MaxWeight {
			validationErrors = append(validationErrors, fmt.Errorf("Priority %s should have a positive weight applied to it or it has overflown", priority.Name))
		}
	}

	binders := 0
	extenderManagedResources := sets.NewString()
	for _, extender := range policy.ExtenderConfigs {
		if len(extender.PrioritizeVerb) > 0 && extender.Weight <= 0 {
			validationErrors = append(validationErrors, fmt.Errorf("Priority for extender %s should have a positive weight applied to it", extender.URLPrefix))
		}
		if extender.BindVerb != "" {
			binders++
		}
		for _, resource := range extender.ManagedResources {
			errs := validateExtendedResourceName(resource.Name)
			if len(errs) != 0 {
				validationErrors = append(validationErrors, errs...)
			}
			if extenderManagedResources.Has(string(resource.Name)) {
				validationErrors = append(validationErrors, fmt.Errorf("Duplicate extender managed resource name %s", string(resource.Name)))
			}
			extenderManagedResources.Insert(string(resource.Name))
		}
	}
	if binders > 1 {
		validationErrors = append(validationErrors, fmt.Errorf("Only one extender can implement bind, found %v", binders))
	}
	return utilerrors.NewAggregate(validationErrors)
}

var FeatureGateDependencies = map[featuregate.Feature]schedulerapi.FeatureDependency{
	features.TaintNodesByCondition: {
		NeededPredicateList: sets.NewString(predicates.PodToleratesNodeTaintsPred),
		NeededPriorityList:  nil,
		ExcludedPredicateList: sets.NewString(predicates.CheckNodeConditionPred,
			predicates.CheckNodeMemoryPressurePred,
			predicates.CheckNodeDiskPressurePred,
			predicates.CheckNodePIDPressurePred,
		),
		ExcludedPriorityList: nil,
	},
}

// validateFeatureDependencies checks dependent and excluded predicates/priorities for given feature gates
func validateFeatureDependencies(policy schedulerapi.Policy) error {
	var validationErrors []error
	for feature, featureDependencies := range FeatureGateDependencies {
		if !utilfeature.DefaultFeatureGate.Enabled(feature) {
			continue
		}
		for _, predicate := range policy.Predicates {
			if featureDependencies.ExcludedPredicateList.Has(predicate.Name) {
				validationErrors = append(validationErrors, fmt.Errorf("Predicate %s shouldn't be present when %s is enabled", predicate.Name, string(feature)))
			}
		}
		for _, priority := range policy.Priorities {
			if featureDependencies.ExcludedPriorityList.Has(priority.Name) {
				validationErrors = append(validationErrors, fmt.Errorf("Priority %s shouldn't be present when %s is enabled", priority.Name, string(feature)))
			}
		}

		for _, neededPredicate := range featureDependencies.NeededPredicateList.List() {
			found := false
			for _, predicate := range policy.Predicates {
				if predicate.Name == neededPredicate {
					found = true
				}
			}
			if !found {
				validationErrors = append(validationErrors, fmt.Errorf("Predicate %s should be present when %s is enabled", neededPredicate, string(feature)))
			}
		}

		for _, neededPriority := range featureDependencies.NeededPriorityList.List() {
			found := false
			for _, priority := range policy.Priorities {
				if priority.Name == neededPriority {
					found = true
				}
			}
			if !found {
				validationErrors = append(validationErrors, fmt.Errorf("Priority %s should be present when %s is enabled", neededPriority, string(feature)))
			}
		}
	}
	return utilerrors.NewAggregate(validationErrors)
}

// validateExtendedResourceName checks whether the specified name is a valid
// extended resource name.
func validateExtendedResourceName(name v1.ResourceName) []error {
	var validationErrors []error
	for _, msg := range validation.IsQualifiedName(string(name)) {
		validationErrors = append(validationErrors, errors.New(msg))
	}
	if len(validationErrors) != 0 {
		return validationErrors
	}
	if !v1helper.IsExtendedResourceName(name) {
		validationErrors = append(validationErrors, fmt.Errorf("%s is an invalid extended resource name", name))
	}
	return validationErrors
}
