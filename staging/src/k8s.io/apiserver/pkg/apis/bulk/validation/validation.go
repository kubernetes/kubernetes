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
	tooManyOpsErr := "may not specify more than 1 operation"

	if r.Watch != nil {
		path := field.NewPath("watch")
		if numOps > 0 {
			allErrs = append(allErrs, field.Forbidden(path, tooManyOpsErr))
		}
		numOps++
		allErrs = append(allErrs, validateWatchOp(r.Watch, path)...)
	}
	if r.WatchList != nil {
		path := field.NewPath("watchList")
		if numOps > 0 {
			allErrs = append(allErrs, field.Forbidden(path, tooManyOpsErr))
		}
		numOps++
		allErrs = append(allErrs, validateWatchListOp(r.WatchList, path)...)
	}
	if r.StopWatch != nil {
		path := field.NewPath("stopWatch")
		if numOps > 0 {
			allErrs = append(allErrs, field.Forbidden(path, tooManyOpsErr))
		}
		numOps++
		allErrs = append(allErrs, validateStopWatchOp(r.StopWatch, path)...)
	}
	if r.List != nil {
		path := field.NewPath("list")
		if numOps > 0 {
			allErrs = append(allErrs, field.Forbidden(path, tooManyOpsErr))
		}
		numOps++
		allErrs = append(allErrs, validateListOp(r.List, path)...)
	}
	if r.Get != nil {
		path := field.NewPath("get")
		if numOps > 0 {
			allErrs = append(allErrs, field.Forbidden(path, tooManyOpsErr))
		}
		numOps++
		allErrs = append(allErrs, validateGetOp(r.Get, path)...)
	}
	return allErrs
}

func validateWatchOp(op *bulk.WatchOperation, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, validateWatchID(op.WatchID, fldPath.Child("watchID"))...)
	allErrs = append(allErrs, validateItemSelector(op.ItemSelector, fldPath)...)
	return allErrs
}

func validateWatchListOp(op *bulk.WatchListOperation, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, validateWatchID(op.WatchID, fldPath.Child("watchID"))...)
	allErrs = append(allErrs, validateListSelector(op.ListSelector, fldPath)...)
	return allErrs
}

func validateGetOp(op *bulk.GetOperation, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, validateItemSelector(op.ItemSelector, fldPath)...)
	return allErrs
}

func validateListOp(op *bulk.ListOperation, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, validateListSelector(op.ListSelector, fldPath)...)
	return allErrs
}

func validateStopWatchOp(op *bulk.StopWatchOperation, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, validateWatchID(op.WatchID, fldPath.Child("watchID"))...)
	return allErrs
}

func validateWatchID(watchID string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if watchID == "" {
		allErrs = append(allErrs, field.Required(fldPath, "must specify watch id"))
	}
	return allErrs
}

func validateItemSelector(s bulk.ItemSelector, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if s.Name == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), "must specify resource name"))
	}
	allErrs = append(allErrs, validateGroupVersionResource(s.GroupVersionResource, fldPath)...)
	// TODO: validate options
	return allErrs
}

func validateListSelector(s bulk.ListSelector, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, validateGroupVersionResource(s.GroupVersionResource, fldPath)...)
	// TODO: validate options
	return allErrs
}

func validateGroupVersionResource(s bulk.GroupVersionResource, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if s.Resource == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("resource"), "must specify api resource"))
	}
	return allErrs
}
