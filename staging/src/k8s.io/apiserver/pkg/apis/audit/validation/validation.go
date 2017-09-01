/*
Copyright 2017 The Kubernetes Authors.

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
	"strings"

	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/audit"
)

func ValidatePolicy(policy *audit.Policy) field.ErrorList {
	var allErrs field.ErrorList
	rulePath := field.NewPath("rules")
	for i, rule := range policy.Rules {
		allErrs = append(allErrs, validatePolicyRule(rule, rulePath.Index(i))...)
	}
	return allErrs
}

func validatePolicyRule(rule audit.PolicyRule, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, validateLevel(rule.Level, fldPath.Child("level"))...)
	allErrs = append(allErrs, validateNonResourceURLs(rule.NonResourceURLs, fldPath.Child("nonResourceURLs"))...)
	allErrs = append(allErrs, validateResources(rule.Resources, fldPath.Child("resources"))...)

	if len(rule.NonResourceURLs) > 0 {
		if len(rule.Resources) > 0 || len(rule.Namespaces) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("nonResourceURLs"), rule.NonResourceURLs, "rules cannot apply to both regular resources and non-resource URLs"))
		}
	}

	return allErrs
}

var validLevels = []string{
	string(audit.LevelNone),
	string(audit.LevelMetadata),
	string(audit.LevelRequest),
	string(audit.LevelRequestResponse),
}

func validateLevel(level audit.Level, fldPath *field.Path) field.ErrorList {
	switch level {
	case audit.LevelNone, audit.LevelMetadata, audit.LevelRequest, audit.LevelRequestResponse:
		return nil
	case "":
		return field.ErrorList{field.Required(fldPath, "")}
	default:
		return field.ErrorList{field.NotSupported(fldPath, level, validLevels)}
	}
}

func validateNonResourceURLs(urls []string, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	for i, url := range urls {
		if url == "*" {
			continue
		}

		if !strings.HasPrefix(url, "/") {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(i), url, "non-resource URL rules must begin with a '/' character"))
		}

		if url != "" && strings.ContainsRune(url[:len(url)-1], '*') {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(i), url, "non-resource URL wildcards '*' must be the final character of the rule"))
		}
	}
	return allErrs
}

func validateResources(groupResources []audit.GroupResources, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	for _, groupResource := range groupResources {
		// The empty string represents the core API group.
		if len(groupResource.Group) == 0 {
			continue
		}

		// Group names must be lower case and be valid DNS subdomains.
		// reference: https://github.com/kubernetes/community/blob/master/contributors/devel/api-conventions.md
		// an error is returned for group name like rbac.authorization.k8s.io/v1beta1
		// rbac.authorization.k8s.io is the valid one
		if msgs := validation.NameIsDNSSubdomain(groupResource.Group, false); len(msgs) != 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("group"), groupResource.Group, strings.Join(msgs, ",")))
		}
	}
	return allErrs
}
