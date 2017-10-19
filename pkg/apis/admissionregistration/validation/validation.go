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
	"fmt"
	"strings"

	genericvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

func ValidateInitializerConfiguration(ic *admissionregistration.InitializerConfiguration) field.ErrorList {
	allErrors := genericvalidation.ValidateObjectMeta(&ic.ObjectMeta, false, genericvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))
	for i, initializer := range ic.Initializers {
		allErrors = append(allErrors, validateInitializer(&initializer, field.NewPath("initializers").Index(i))...)
	}
	return allErrors
}

func validateInitializer(initializer *admissionregistration.Initializer, fldPath *field.Path) field.ErrorList {
	var allErrors field.ErrorList
	// initlializer.Name must be fully qualified
	allErrors = append(allErrors, validation.IsFullyQualifiedName(fldPath.Child("name"), initializer.Name)...)

	for i, rule := range initializer.Rules {
		notAllowSubresources := false
		allErrors = append(allErrors, validateRule(&rule, fldPath.Child("rules").Index(i), notAllowSubresources)...)
	}
	return allErrors
}

func hasWildcard(slice []string) bool {
	for _, s := range slice {
		if s == "*" {
			return true
		}
	}
	return false
}

func validateResources(resources []string, fldPath *field.Path) field.ErrorList {
	var allErrors field.ErrorList
	if len(resources) == 0 {
		allErrors = append(allErrors, field.Required(fldPath, ""))
	}

	// */x
	resourcesWithWildcardSubresoures := sets.String{}
	// x/*
	subResoucesWithWildcardResource := sets.String{}
	// */*
	hasDoubleWildcard := false
	// *
	hasSingleWildcard := false
	// x
	hasResourceWithoutSubresource := false

	for i, resSub := range resources {
		if resSub == "" {
			allErrors = append(allErrors, field.Required(fldPath.Index(i), ""))
			continue
		}
		if resSub == "*/*" {
			hasDoubleWildcard = true
		}
		if resSub == "*" {
			hasSingleWildcard = true
		}
		parts := strings.SplitN(resSub, "/", 2)
		if len(parts) == 1 {
			hasResourceWithoutSubresource = resSub != "*"
			continue
		}
		res, sub := parts[0], parts[1]
		if _, ok := resourcesWithWildcardSubresoures[res]; ok {
			allErrors = append(allErrors, field.Invalid(fldPath.Index(i), resSub, fmt.Sprintf("if '%s/*' is present, must not specify %s", res, resSub)))
		}
		if _, ok := subResoucesWithWildcardResource[sub]; ok {
			allErrors = append(allErrors, field.Invalid(fldPath.Index(i), resSub, fmt.Sprintf("if '*/%s' is present, must not specify %s", sub, resSub)))
		}
		if sub == "*" {
			resourcesWithWildcardSubresoures[res] = struct{}{}
		}
		if res == "*" {
			subResoucesWithWildcardResource[sub] = struct{}{}
		}
	}
	if len(resources) > 1 && hasDoubleWildcard {
		allErrors = append(allErrors, field.Invalid(fldPath, resources, "if '*/*' is present, must not specify other resources"))
	}
	if hasSingleWildcard && hasResourceWithoutSubresource {
		allErrors = append(allErrors, field.Invalid(fldPath, resources, "if '*' is present, must not specify other resources without subresources"))
	}
	return allErrors
}

func validateResourcesNoSubResources(resources []string, fldPath *field.Path) field.ErrorList {
	var allErrors field.ErrorList
	if len(resources) == 0 {
		allErrors = append(allErrors, field.Required(fldPath, ""))
	}
	for i, resource := range resources {
		if resource == "" {
			allErrors = append(allErrors, field.Required(fldPath.Index(i), ""))
		}
		if strings.Contains(resource, "/") {
			allErrors = append(allErrors, field.Invalid(fldPath.Index(i), resource, "must not specify subresources"))
		}
	}
	if len(resources) > 1 && hasWildcard(resources) {
		allErrors = append(allErrors, field.Invalid(fldPath, resources, "if '*' is present, must not specify other resources"))
	}
	return allErrors
}

func validateRule(rule *admissionregistration.Rule, fldPath *field.Path, allowSubResource bool) field.ErrorList {
	var allErrors field.ErrorList
	if len(rule.APIGroups) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("apiGroups"), ""))
	}
	if len(rule.APIGroups) > 1 && hasWildcard(rule.APIGroups) {
		allErrors = append(allErrors, field.Invalid(fldPath.Child("apiGroups"), rule.APIGroups, "if '*' is present, must not specify other API groups"))
	}
	// Note: group could be empty, e.g., the legacy "v1" API
	if len(rule.APIVersions) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("apiVersions"), ""))
	}
	if len(rule.APIVersions) > 1 && hasWildcard(rule.APIVersions) {
		allErrors = append(allErrors, field.Invalid(fldPath.Child("apiVersions"), rule.APIVersions, "if '*' is present, must not specify other API versions"))
	}
	for i, version := range rule.APIVersions {
		if version == "" {
			allErrors = append(allErrors, field.Required(fldPath.Child("apiVersions").Index(i), ""))
		}
	}
	if allowSubResource {
		allErrors = append(allErrors, validateResources(rule.Resources, fldPath.Child("resources"))...)
	} else {
		allErrors = append(allErrors, validateResourcesNoSubResources(rule.Resources, fldPath.Child("resources"))...)
	}
	return allErrors
}

func ValidateInitializerConfigurationUpdate(newIC, oldIC *admissionregistration.InitializerConfiguration) field.ErrorList {
	return ValidateInitializerConfiguration(newIC)
}

func ValidateExternalAdmissionHookConfiguration(e *admissionregistration.ExternalAdmissionHookConfiguration) field.ErrorList {
	allErrors := genericvalidation.ValidateObjectMeta(&e.ObjectMeta, false, genericvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))
	for i, hook := range e.ExternalAdmissionHooks {
		allErrors = append(allErrors, validateExternalAdmissionHook(&hook, field.NewPath("externalAdmissionHooks").Index(i))...)
	}
	return allErrors
}

func validateExternalAdmissionHook(hook *admissionregistration.ExternalAdmissionHook, fldPath *field.Path) field.ErrorList {
	var allErrors field.ErrorList
	// hook.Name must be fully qualified
	allErrors = append(allErrors, validation.IsFullyQualifiedName(fldPath.Child("name"), hook.Name)...)

	for i, rule := range hook.Rules {
		allErrors = append(allErrors, validateRuleWithOperations(&rule, fldPath.Child("rules").Index(i))...)
	}
	if hook.FailurePolicy != nil && !supportedFailurePolicies.Has(string(*hook.FailurePolicy)) {
		allErrors = append(allErrors, field.NotSupported(fldPath.Child("failurePolicy"), *hook.FailurePolicy, supportedFailurePolicies.List()))
	}

	if len(hook.ClientConfig.URLPath) != 0 {
		allErrors = append(allErrors, validateURLPath(fldPath.Child("clientConfig", "urlPath"), hook.ClientConfig.URLPath)...)
	}

	return allErrors
}

func validateURLPath(fldPath *field.Path, urlPath string) field.ErrorList {
	var allErrors field.ErrorList
	if urlPath == "/" || len(urlPath) == 0 {
		return allErrors
	}
	if urlPath == "//" {
		allErrors = append(allErrors, field.Invalid(fldPath, urlPath, "segment[0] may not be empty"))
		return allErrors
	}

	if !strings.HasPrefix(urlPath, "/") {
		allErrors = append(allErrors, field.Invalid(fldPath, urlPath, "must start with a '/'"))
	}

	urlPathToCheck := urlPath[1:]
	if strings.HasSuffix(urlPathToCheck, "/") {
		urlPathToCheck = urlPathToCheck[:len(urlPathToCheck)-1]
	}
	steps := strings.Split(urlPathToCheck, "/")
	for i, step := range steps {
		if len(step) == 0 {
			allErrors = append(allErrors, field.Invalid(fldPath, urlPath, fmt.Sprintf("segment[%d] may not be empty", i)))
			continue
		}
		failures := validation.IsDNS1123Subdomain(step)
		for _, failure := range failures {
			allErrors = append(allErrors, field.Invalid(fldPath, urlPath, fmt.Sprintf("segment[%d]: %v", i, failure)))
		}
	}

	return allErrors
}

var supportedFailurePolicies = sets.NewString(
	string(admissionregistration.Ignore),
	string(admissionregistration.Fail),
)

var supportedOperations = sets.NewString(
	string(admissionregistration.OperationAll),
	string(admissionregistration.Create),
	string(admissionregistration.Update),
	string(admissionregistration.Delete),
	string(admissionregistration.Connect),
)

func hasWildcardOperation(operations []admissionregistration.OperationType) bool {
	for _, o := range operations {
		if o == admissionregistration.OperationAll {
			return true
		}
	}
	return false
}

func validateRuleWithOperations(ruleWithOperations *admissionregistration.RuleWithOperations, fldPath *field.Path) field.ErrorList {
	var allErrors field.ErrorList
	if len(ruleWithOperations.Operations) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("operations"), ""))
	}
	if len(ruleWithOperations.Operations) > 1 && hasWildcardOperation(ruleWithOperations.Operations) {
		allErrors = append(allErrors, field.Invalid(fldPath.Child("operations"), ruleWithOperations.Operations, "if '*' is present, must not specify other operations"))
	}
	for i, operation := range ruleWithOperations.Operations {
		if !supportedOperations.Has(string(operation)) {
			allErrors = append(allErrors, field.NotSupported(fldPath.Child("operations").Index(i), operation, supportedOperations.List()))
		}
	}
	allowSubResource := true
	allErrors = append(allErrors, validateRule(&ruleWithOperations.Rule, fldPath, allowSubResource)...)
	return allErrors
}

func ValidateExternalAdmissionHookConfigurationUpdate(newC, oldC *admissionregistration.ExternalAdmissionHookConfiguration) field.ErrorList {
	return ValidateExternalAdmissionHookConfiguration(newC)
}
