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
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/bulk"
)

// ValidateBulkRequest performs validation for the given bulk request.
func ValidateBulkRequest(r *bulk.BulkRequest) field.ErrorList {
	var allErrs field.ErrorList

	if r.RequestID == "" {
		allErrs = append(allErrs, field.Required(field.NewPath("requestId"), "must specify request id"))
	}

	numOps := 0
	if r.Watch != nil {
		numOps++
		allErrs = append(allErrs, validateWatchOp(r.Watch, field.NewPath("watch"))...)
	}
	if r.StopWatch != nil {
		if numOps > 0 {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("stopWatch"), "may not specify more than 1 operation"))
		} else {
			numOps++
			allErrs = append(allErrs, validateStopWatchOp(r.StopWatch, field.NewPath("stopWatch"))...)
		}

	}
	return allErrs
}

func validateWatchOp(op *bulk.WatchOperation, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, validateWatchID(op.WatchID, fldPath.Child("watchID"))...)
	allErrs = append(allErrs, validateSelector(op.Selector, fldPath)...)
	return allErrs
}

func validateWatchID(watchID string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if watchID == "" {
		allErrs = append(allErrs, field.Required(fldPath, "must specify watch id"))
	}
	return allErrs
}

func validateSelector(s bulk.ResourceSelector, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if s.Resource == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("resource"), "must specify api resource"))
	}

	opts := s.Options
	if s.Name != "" && opts.FieldSelector != nil && !opts.FieldSelector.Empty() {
		// It doesn't make sense to ask for both a name and a field selector, since just the name is
		// sufficient to narrow down the request to a single object.
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), "both a name and a field selector provided; please provide one or the other."))
	}

	return allErrs
}

func validateStopWatchOp(op *bulk.StopWatchOperation, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, validateWatchID(op.WatchID, fldPath.Child("watchID"))...)
	return allErrs
}
