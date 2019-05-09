/*
Copyright 2018 The Kubernetes Authors.

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
	genericvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/flowregistration"
)

// TODO(aaron-prindle) make a more final version of the validation
// ValidateFlowSchema validates the FlowSchemas
func ValidateFlowSchema(as *flowregistration.FlowSchema) field.ErrorList {
	allErrs := genericvalidation.ValidateObjectMeta(&as.ObjectMeta, false, genericvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateFlowSchemaSpec(as.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateFlowSchemaSpec validates the spec for flow
func ValidateFlowSchemaSpec(s flowregistration.FlowSchemaSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, ValidateRequestPriority(s.RequestPriority, field.NewPath("requestPriority"))...)
	allErrs = append(allErrs, ValidateFlowDistinguisher(s.FlowDistinguisher, field.NewPath("flowDistinguisher"))...)
	allErrs = append(allErrs, ValidateMatches(s.Match, field.NewPath("match"))...)
	return allErrs
}

// ValidateRequestPriority validates the flow requestPriority
func ValidateRequestPriority(requestPriority flowregistration.RequestPriority, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if requestPriority.Name == "" {
		return field.ErrorList{field.Required(fldPath.Child("name"), "requestPriority.Name is required")}
	}
	return allErrs
}

// ValidateRequestPriority validates the RequestPriority
func ValidateFlowDistinguisher(fd flowregistration.FlowDistinguisher, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if fd.Source == "" {
		return field.ErrorList{field.Required(fldPath.Child("source"), "flowDistinguisher.source is required")}
	}
	allErrs = append(allErrs, validateFlowSource(fd.Source, fldPath.Child("level"))...)
	return allErrs
}

// ValidateMatch validates the match
func ValidateMatches(matches []*flowregistration.Match, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if len(matches) <= 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("match"), matches, "at least one match directive is required"))
	}
	// TODO(aaron-prindle) do this properly
	return allErrs
}

var validFlowSources = sets.NewString(
	string(flowregistration.FlowSourceUser),
	string(flowregistration.FlowSourceNamespace),
)

var validAndFields = sets.NewString(
	string(flowregistration.AndFieldUser),
	string(flowregistration.AndFieldGroups),
	string(flowregistration.AndFieldNamespace),
	string(flowregistration.AndFieldResource),
)

func validateFlowSource(fs flowregistration.FlowSource, fldPath *field.Path) field.ErrorList {
	if string(fs) == "" {
		return field.ErrorList{field.Required(fldPath, "")}
	}
	if !validFlowSources.Has(string(fs)) {
		return field.ErrorList{field.NotSupported(fldPath, fs, validFlowSources.List())}
	}
	return nil
}

// TODO(aaron-prindle) make this accept "AndField"?
func validateAndField(af string, fldPath *field.Path) field.ErrorList {
	if string(af) == "" {
		return field.ErrorList{field.Required(fldPath, "")}
	}
	if !validAndFields.Has(string(af)) {
		return field.ErrorList{field.NotSupported(fldPath, af, validFlowSources.List())}
	}
	return nil
}

// ValidateFlowSchemaUpdate validates an update to the object
func ValidateFlowSchemaUpdate(newC, oldC *flowregistration.FlowSchema) field.ErrorList {
	return ValidateFlowSchema(newC)
}
