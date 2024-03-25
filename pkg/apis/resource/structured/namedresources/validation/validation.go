/*
Copyright 2022 The Kubernetes Authors.

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
	"fmt"
	"regexp"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	namedresourcescel "k8s.io/dynamic-resource-allocation/structured/namedresources/cel"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/resource"
)

var (
	validateInstanceName  = corevalidation.ValidateDNS1123Subdomain
	validateAttributeName = corevalidation.ValidateDNS1123Subdomain
)

type Options struct {
	// StoredExpressions must be true if and only if validating CEL
	// expressions that were already stored persistently. This makes
	// validation more permissive by enabling CEL definitions that are not
	// valid yet for new expressions.
	StoredExpressions bool
}

func ValidateResources(resources *resource.NamedResourcesResources, fldPath *field.Path) field.ErrorList {
	allErrs := validateInstances(resources.Instances, fldPath.Child("instances"))
	return allErrs
}

func validateInstances(instances []resource.NamedResourcesInstance, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	instanceNames := sets.New[string]()
	for i, instance := range instances {
		idxPath := fldPath.Index(i)
		instanceName := instance.Name
		allErrs = append(allErrs, validateInstanceName(instanceName, idxPath.Child("name"))...)
		if instanceNames.Has(instanceName) {
			allErrs = append(allErrs, field.Duplicate(idxPath.Child("name"), instanceName))
		} else {
			instanceNames.Insert(instanceName)
		}
		allErrs = append(allErrs, validateAttributes(instance.Attributes, idxPath.Child("attributes"))...)
	}
	return allErrs
}

var (
	numericIdentifier = `(0|[1-9]\d*)`

	preReleaseIdentifier = `(0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)`

	buildIdentifier = `[0-9a-zA-Z-]+`

	semverRe = regexp.MustCompile(`^` +

		// dot-separated version segments (e.g. 1.2.3)
		numericIdentifier + `\.` + numericIdentifier + `\.` + numericIdentifier +

		// optional dot-separated prerelease segments (e.g. -alpha.PRERELEASE.1)
		`(-` + preReleaseIdentifier + `(\.` + preReleaseIdentifier + `)*)?` +

		// optional dot-separated build identifier segments (e.g. +build.id.20240305)
		`(\+` + buildIdentifier + `(\.` + buildIdentifier + `)*)?` +

		`$`)
)

func validateAttributes(attributes []resource.NamedResourcesAttribute, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	attributeNames := sets.New[string]()
	for i, attribute := range attributes {
		idxPath := fldPath.Index(i)
		attributeName := attribute.Name
		allErrs = append(allErrs, validateAttributeName(attributeName, idxPath.Child("name"))...)
		if attributeNames.Has(attributeName) {
			allErrs = append(allErrs, field.Duplicate(idxPath.Child("name"), attributeName))
		} else {
			attributeNames.Insert(attributeName)
		}

		entries := sets.New[string]()
		if attribute.QuantityValue != nil {
			entries.Insert("quantity")
		}
		if attribute.BoolValue != nil {
			entries.Insert("bool")
		}
		if attribute.IntValue != nil {
			entries.Insert("int")
		}
		if attribute.IntSliceValue != nil {
			entries.Insert("intSlice")
		}
		if attribute.StringValue != nil {
			entries.Insert("string")
		}
		if attribute.StringSliceValue != nil {
			entries.Insert("stringSlice")
		}
		if attribute.VersionValue != nil {
			entries.Insert("version")
			if !semverRe.MatchString(*attribute.VersionValue) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("version"), *attribute.VersionValue, "must be a string compatible with semver.org spec 2.0.0"))
			}
		}

		switch len(entries) {
		case 0:
			allErrs = append(allErrs, field.Required(idxPath, "exactly one value must be set"))
		case 1:
			// Okay.
		default:
			allErrs = append(allErrs, field.Invalid(idxPath, sets.List(entries), "exactly one field must be set, not several"))
		}
	}
	return allErrs
}

func ValidateRequest(opts Options, request *resource.NamedResourcesRequest, fldPath *field.Path) field.ErrorList {
	return validateSelector(opts, request.Selector, fldPath.Child("selector"))
}

func ValidateFilter(opts Options, filter *resource.NamedResourcesFilter, fldPath *field.Path) field.ErrorList {
	return validateSelector(opts, filter.Selector, fldPath.Child("selector"))
}

func validateSelector(opts Options, selector string, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if selector == "" {
		allErrs = append(allErrs, field.Required(fldPath, ""))
	} else {
		envType := environment.NewExpressions
		if opts.StoredExpressions {
			envType = environment.StoredExpressions
		}
		result := namedresourcescel.Compiler.CompileCELExpression(selector, envType)
		if result.Error != nil {
			allErrs = append(allErrs, convertCELErrorToValidationError(fldPath, selector, result.Error))
		}
	}
	return allErrs
}

func convertCELErrorToValidationError(fldPath *field.Path, expression string, err *cel.Error) *field.Error {
	switch err.Type {
	case cel.ErrorTypeRequired:
		return field.Required(fldPath, err.Detail)
	case cel.ErrorTypeInvalid:
		return field.Invalid(fldPath, expression, err.Detail)
	case cel.ErrorTypeInternal:
		return field.InternalError(fldPath, err)
	}
	return field.InternalError(fldPath, fmt.Errorf("unsupported error type: %w", err))
}

func ValidateAllocationResult(result *resource.NamedResourcesAllocationResult, fldPath *field.Path) field.ErrorList {
	return validateInstanceName(result.Name, fldPath.Child("name"))
}
