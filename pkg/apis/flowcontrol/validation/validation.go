/*
Copyright 2019 The Kubernetes Authors.

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
	"math/big"

	"k8s.io/apimachinery/pkg/api/validation"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
)

var ValidateFlowSchemaName = apimachineryvalidation.NameIsDNSSubdomain

var ValidatePriorityLevelConfigurationName = apimachineryvalidation.NameIsDNSSubdomain

var supportedDistinguisherMethods = sets.NewString(
	string(flowcontrol.FlowDistinguisherMethodByNamespaceType),
	string(flowcontrol.FlowDistinguisherMethodByUserType),
)

var supportedVerbs = sets.NewString(
	flowcontrol.VerbAll,
	"get",
	"list",
	"watch",
	"create",
	"update",
	"delete",
	"deletecollection",
	"patch",
)

var supportedSubjectKinds = sets.NewString(
	flowcontrol.ServiceAccountKind,
	flowcontrol.UserKind,
	flowcontrol.GroupKind,
)

const (
	maxHashBits = 60
)

func ValidateFlowSchema(fs *flowcontrol.FlowSchema) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&fs.ObjectMeta, false, ValidateFlowSchemaName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateFlowSchemaSpec(&fs.Spec, field.NewPath("spec"))...)
	allErrs = append(allErrs, ValidateFlowSchemaStatus(&fs.Status, field.NewPath("status"))...)
	return allErrs
}

func ValidateFlowSchemaSpec(spec *flowcontrol.FlowSchemaSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if spec.MatchingPrecedence < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("matchingPrecedence"), spec.MatchingPrecedence, "must be positive value"))
	}
	if spec.DistinguisherMethod != nil {
		if !supportedDistinguisherMethods.Has(string(spec.DistinguisherMethod.Type)) {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("distinguisherMethod").Child("type"), spec.DistinguisherMethod, supportedDistinguisherMethods.List()))
		}
	}
	if len(spec.PriorityLevelConfiguration.Name) > 0 {
		for _, msg := range ValidatePriorityLevelConfigurationName(spec.PriorityLevelConfiguration.Name, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("priorityLevelConfiguration").Child("name"), spec.PriorityLevelConfiguration.Name, msg))
		}
	} else {
		allErrs = append(allErrs, field.Required(fldPath.Child("priorityLevelConfiguration").Child("name"), "must reference to a priority level"))
	}
	if len(spec.Rules) > 0 {
		for i, rule := range spec.Rules {
			allErrs = append(allErrs, ValidateFlowSchemaPolicyRuleWithSubjects(&rule, fldPath.Child("rules").Index(i))...)
		}
	} else {
		allErrs = append(allErrs, field.Required(fldPath.Child("rules"), "rules must contain at least one value"))
	}
	return allErrs
}

func ValidateFlowSchemaPolicyRuleWithSubjects(rule *flowcontrol.PolicyRuleWithSubjects, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if len(rule.Subjects) > 0 {
		for i, subject := range rule.Subjects {
			allErrs = append(allErrs, ValidateFlowSchemaSubject(&subject, fldPath.Child("subjects").Index(i))...)
		}
	} else {
		allErrs = append(allErrs, field.Required(fldPath.Child("subjects"), "subjects must contain at least one value"))
	}
	allErrs = append(allErrs, ValidateFlowSchemaPolicyRule(&rule.Rule, fldPath.Child("rule"))...)
	return allErrs
}

func ValidateFlowSchemaSubject(subject *flowcontrol.Subject, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if len(subject.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	}
	switch subject.Kind {
	case flowcontrol.ServiceAccountKind:
		if len(subject.Name) > 0 {
			for _, msg := range validation.ValidateServiceAccountName(subject.Name, false) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), subject.Name, msg))
			}
		}
		if len(subject.APIGroup) > 0 {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("apiGroup"), subject.APIGroup, []string{""}))
		}
		if len(subject.Namespace) > 0 {
			for _, msg := range apimachineryvalidation.ValidateNamespaceName(subject.Namespace, false) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("namespace"), subject.Namespace, msg))
			}
		} else {
			allErrs = append(allErrs, field.Required(fldPath.Child("namespace"), "must specify namespace for service account"))
		}
	case flowcontrol.UserKind:
		// TODO(ericchiang): What other restrictions on user name are there?
		if subject.APIGroup != flowcontrol.GroupName {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("apiGroup"), subject.APIGroup, []string{flowcontrol.GroupName}))
		}
		if len(subject.Namespace) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("namespace"), subject.Namespace, "must not set namespace for user-kind subject"))
		}
	case flowcontrol.GroupKind:
		// TODO(ericchiang): What other restrictions on group name are there?
		if subject.APIGroup != flowcontrol.GroupName {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("apiGroup"), subject.APIGroup, []string{flowcontrol.GroupName}))
		}
		if len(subject.Namespace) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("namespace"), subject.Namespace, "must not set namespace for group-kind subject"))
		}
	default:
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("kind"), subject.Kind, supportedSubjectKinds.List()))
	}
	return allErrs
}

func ValidateFlowSchemaPolicyRule(rule *flowcontrol.PolicyRule, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	if len(rule.Verbs) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("verbs"), "verbs must contain at least one value"))
	}

	if len(rule.NonResourceURLs) > 0 {
		if len(rule.APIGroups) > 0 || len(rule.Resources) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("nonResourceURLs"), rule.NonResourceURLs, "rules cannot apply to both regular resources and non-resource URLs"))
		}
		if hasWildcard(rule.NonResourceURLs) && len(rule.NonResourceURLs) > 1 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("nonResourceURLs"), rule.NonResourceURLs, "if '*' is present, must not specify other non-resource URLs"))
		}
		return allErrs
	}
	if len(rule.APIGroups) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("apiGroups"), "resource rules must supply at least one api group"))
	}
	if len(rule.Resources) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("resources"), "resource rules must supply at least one resource"))
	}

	if hasWildcard(rule.Verbs) && len(rule.Verbs) > 1 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("verbs"), rule.Verbs, "if '*' is present, must not specify other verbs"))
	}
	if hasWildcard(rule.APIGroups) && len(rule.APIGroups) > 1 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("apiGroups"), rule.APIGroups, "if '*' is present, must not specify other api groups"))
	}
	if hasWildcard(rule.Resources) && len(rule.Resources) > 1 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("resources"), rule.Resources, "if '*' is present, must not specify other resources"))
	}
	if !supportedVerbs.IsSuperset(sets.NewString(rule.Verbs...)) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("verbs"), rule.Verbs, supportedVerbs.List()))
	}

	return allErrs
}

func ValidateFlowSchemaStatus(status *flowcontrol.FlowSchemaStatus, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	// conditions will not be validated
	return allErrs
}

func ValidatePriorityLevelConfiguration(pl *flowcontrol.PriorityLevelConfiguration) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&pl.ObjectMeta, false, ValidatePriorityLevelConfigurationName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidatePriorityLevelConfigurationSpec(&pl.Spec, pl.Name, field.NewPath("spec"))...)
	allErrs = append(allErrs, ValidatePriorityLevelConfigurationStatus(&pl.Status, field.NewPath("status"))...)
	return allErrs
}

func ValidatePriorityLevelConfigurationSpec(spec *flowcontrol.PriorityLevelConfigurationSpec, name string, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if name != flowcontrol.PriorityLevelConfigurationNameSystemTop && spec.Exempt {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("exempt"), "must not be exempt"))
	}
	if name != flowcontrol.PriorityLevelConfigurationNameWorkloadLow && spec.GlobalDefault {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("globalDefault"), "must not be global default"))
	}
	if !spec.Exempt {
		if spec.AssuredConcurrencyShares <= 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("assuredConcurrencyShares"), spec.AssuredConcurrencyShares, "must be positive"))
		}
		if spec.QueueLengthLimit <= 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("queueLengthLimit"), spec.QueueLengthLimit, "must be positive"))
		}
		if spec.Queues <= 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("queues"), spec.Queues, "must be positive"))
		}
		if spec.HandSize < 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("handSize"), spec.HandSize, "must be positive"))
		}
	} else {
		if spec.AssuredConcurrencyShares != 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("assuredConcurrencyShares"), spec.AssuredConcurrencyShares, "must be positive for exempt priority"))
		}
		if spec.QueueLengthLimit != 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("queueLengthLimit"), spec.QueueLengthLimit, "must be positive for exempt priority"))
		}
		if spec.Queues != 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("queues"), spec.Queues, "must be positive for exempt priority"))
		}
		if spec.HandSize != 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("handSize"), spec.HandSize, "must be positive for exempt priority"))
		}
	}
	if spec.HandSize > spec.Queues {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("handSize"), spec.HandSize,
			fmt.Sprintf("should not be greater than queues (%d)", spec.Queues)))
	}
	if !validateShuffleShardingParameters(spec.HandSize, spec.Queues) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("handSize"), spec.HandSize,
			fmt.Sprintf("falling factorial of handSize (%d) and queues (%d) exceeds %d bits", spec.HandSize, spec.Queues, maxHashBits)))
	}
	return allErrs
}

func ValidatePriorityLevelConfigurationStatus(status *flowcontrol.PriorityLevelConfigurationStatus, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	// conditions will not be validated
	return allErrs
}

func validateShuffleShardingParameters(handSize, queues int32) bool {
	// TODO: performance impact from bit-int multiplication?
	return int32(big.NewInt(int64(queues)).BitLen())*handSize <= maxHashBits
}

func hasWildcard(operations []string) bool {
	for _, o := range operations {
		if o == "*" {
			return true
		}
	}
	return false
}
