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

	genericvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	validationutil "k8s.io/apimachinery/pkg/util/validation"
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
	if len(initializer.Name) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("name"), ""))
	}
	if errs := validationutil.IsDNS1123Subdomain(initializer.Name); len(errs) > 0 {
		allErrors = append(allErrors, field.Invalid(fldPath.Child("name"), initializer.Name, strings.Join(errs, ",")))
	}
	if len(strings.Split(initializer.Name, ".")) < 3 {
		allErrors = append(allErrors, field.Invalid(fldPath.Child("name"), initializer.Name, "should be a domain with at least two dots"))
	}

	for i, rule := range initializer.Rules {
		allErrors = append(allErrors, validateRule(&rule, fldPath.Child("rules").Index(i))...)
	}
	// TODO: relax the validation rule when admissionregistration is beta.
	if initializer.FailurePolicy != nil && *initializer.FailurePolicy != admissionregistration.Ignore {
		allErrors = append(allErrors, field.NotSupported(fldPath.Child("failurePolicy"), *initializer.FailurePolicy, []string{string(admissionregistration.Ignore)}))
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

func validateRule(rule *admissionregistration.Rule, fldPath *field.Path) field.ErrorList {
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
	if len(rule.Resources) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("resources"), ""))
	}
	if len(rule.Resources) > 1 && hasWildcard(rule.Resources) {
		allErrors = append(allErrors, field.Invalid(fldPath.Child("Resources"), rule.Resources, "if '*' is present, must not specify other resources"))
	}
	for i, resource := range rule.Resources {
		if resource == "" {
			allErrors = append(allErrors, field.Required(fldPath.Child("resources").Index(i), ""))
		}
	}
	return allErrors
}

func ValidateInitializerConfigurationUpdate(newIC, oldIC *admissionregistration.InitializerConfiguration) field.ErrorList {
	return ValidateInitializerConfiguration(newIC)
}

func ValidateExternalAdmissionHookConfiguration(e *admissionregistration.ExternalAdmissionHookConfiguration) field.ErrorList {
	allErrors := genericvalidation.ValidateObjectMeta(&e.ObjectMeta, false, genericvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))
	for i, hook := range e.ExternalAdmissionHooks {
		allErrors = append(allErrors, validateExternalAdmissionHook(&hook, field.NewPath("externalAdmissionHook").Index(i))...)
	}
	return allErrors
}

func validateExternalAdmissionHook(hook *admissionregistration.ExternalAdmissionHook, fldPath *field.Path) field.ErrorList {
	var allErrors field.ErrorList
	// hook.Name must be fully qualified
	if len(hook.Name) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("name"), ""))
	}
	if errs := validationutil.IsDNS1123Subdomain(hook.Name); len(errs) > 0 {
		allErrors = append(allErrors, field.Invalid(fldPath.Child("name"), hook.Name, strings.Join(errs, ",")))
	}
	if len(strings.Split(hook.Name, ".")) < 3 {
		allErrors = append(allErrors, field.Invalid(fldPath.Child("name"), hook.Name, "should be a domain with at least two dots"))
	}

	for i, rule := range hook.Rules {
		allErrors = append(allErrors, validateRuleWithOperations(&rule, fldPath.Child("rules").Index(i))...)
	}
	// TODO: relax the validation rule when admissionregistration is beta.
	if hook.FailurePolicy != nil && *hook.FailurePolicy != admissionregistration.Ignore {
		allErrors = append(allErrors, field.NotSupported(fldPath.Child("failurePolicy"), *hook.FailurePolicy, []string{string(admissionregistration.Ignore)}))
	}
	return allErrors
}

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
	allErrors = append(allErrors, validateRule(&ruleWithOperations.Rule, fldPath)...)
	return allErrors
}

func ValidateExternalAdmissionHookConfigurationUpdate(newC, oldC *admissionregistration.ExternalAdmissionHookConfiguration) field.ErrorList {
	return ValidateExternalAdmissionHookConfiguration(newC)
}
