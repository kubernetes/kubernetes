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

	v1 "k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// ValidatePolicy checks for errors in the Config
// It does not return early so that it can find as many errors as possible
func ValidatePolicy(policy schedulerapi.Policy) error {
	var validationErrors []error

	priorities := make(map[string]schedulerapi.PriorityPolicy, len(policy.Priorities))
	for _, priority := range policy.Priorities {
		if priority.Weight <= 0 || priority.Weight >= framework.MaxWeight {
			validationErrors = append(validationErrors, fmt.Errorf("Priority %s should have a positive weight applied to it or it has overflown", priority.Name))
		}

		validationErrors = append(validationErrors, validatePriorityRedeclared(priorities, priority))
	}

	predicates := make(map[string]schedulerapi.PredicatePolicy, len(policy.Predicates))
	for _, predicate := range policy.Predicates {
		validationErrors = append(validationErrors, validatePredicateRedeclared(predicates, predicate))
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

// validatePriorityRedeclared checks if any custom priorities have been declared multiple times in the policy config
// by examining the specified priority arguments
func validatePriorityRedeclared(priorities map[string]schedulerapi.PriorityPolicy, priority schedulerapi.PriorityPolicy) error {
	var priorityType string
	if priority.Argument != nil {
		if priority.Argument.LabelPreference != nil {
			priorityType = "LabelPreference"
		} else if priority.Argument.RequestedToCapacityRatioArguments != nil {
			priorityType = "RequestedToCapacityRatioArguments"
		} else if priority.Argument.ServiceAntiAffinity != nil {
			priorityType = "ServiceAntiAffinity"
		} else {
			return fmt.Errorf("No priority arguments set for priority %s", priority.Name)
		}
		if existing, alreadyDeclared := priorities[priorityType]; alreadyDeclared {
			return fmt.Errorf("Priority '%s' redeclares custom priority '%s', from:'%s'", priority.Name, priorityType, existing.Name)
		}
		priorities[priorityType] = priority
	}
	return nil
}

// validatePredicateRedeclared checks if any custom predicates have been declared multiple times in the policy config
// by examining the specified predicate arguments
func validatePredicateRedeclared(predicates map[string]schedulerapi.PredicatePolicy, predicate schedulerapi.PredicatePolicy) error {
	var predicateType string
	if predicate.Argument != nil {
		if predicate.Argument.LabelsPresence != nil {
			predicateType = "LabelsPresence"
		} else if predicate.Argument.ServiceAffinity != nil {
			predicateType = "ServiceAffinity"
		} else {
			return fmt.Errorf("No priority arguments set for priority %s", predicate.Name)
		}
		if existing, alreadyDeclared := predicates[predicateType]; alreadyDeclared {
			return fmt.Errorf("Predicate '%s' redeclares custom predicate '%s', from:'%s'", predicate.Name, predicateType, existing.Name)
		}
		predicates[predicateType] = predicate
	}
	return nil
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
