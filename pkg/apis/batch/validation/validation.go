/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	unversionedvalidation "k8s.io/kubernetes/pkg/api/unversioned/validation"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// The internal representation for Job is still in extensions group
// even though this validation code is for the endpoint used by the
// batch group.

func ValidateJob(job *extensions.Job) field.ErrorList {
	// Jobs and rcs have the same name validation
	allErrs := apivalidation.ValidateObjectMeta(&job.ObjectMeta, true, apivalidation.ValidateReplicationControllerName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateJobSpec(&job.Spec, field.NewPath("spec"))...)
	return allErrs
}

func ValidateJobSpec(spec *extensions.JobSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if spec.Parallelism != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*spec.Parallelism), fldPath.Child("parallelism"))...)
	}
	if spec.Completions != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*spec.Completions), fldPath.Child("completions"))...)
	}
	if spec.ActiveDeadlineSeconds != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*spec.ActiveDeadlineSeconds), fldPath.Child("activeDeadlineSeconds"))...)
	}
	if spec.Selector == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("selector"), ""))
	} else {
		allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(spec.Selector, fldPath.Child("selector"))...)
	}

	if selector, err := unversioned.LabelSelectorAsSelector(spec.Selector); err == nil {
		labels := labels.Set(spec.Template.Labels)
		if !selector.Matches(labels) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("template", "metadata", "labels"), spec.Template.Labels, "`selector` does not match template `labels`"))
		}
	}

	allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpec(&spec.Template, fldPath.Child("template"))...)
	if spec.Template.Spec.RestartPolicy != api.RestartPolicyOnFailure &&
		spec.Template.Spec.RestartPolicy != api.RestartPolicyNever {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("template", "spec", "restartPolicy"),
			spec.Template.Spec.RestartPolicy, []string{string(api.RestartPolicyOnFailure), string(api.RestartPolicyNever)}))
	}
	return allErrs
}

func ValidateJobStatus(status *extensions.JobStatus, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.Active), fldPath.Child("active"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.Succeeded), fldPath.Child("succeeded"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.Failed), fldPath.Child("failed"))...)
	return allErrs
}

func ValidateJobUpdate(job, oldJob *extensions.Job) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&oldJob.ObjectMeta, &job.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateJobSpecUpdate(job.Spec, oldJob.Spec, field.NewPath("spec"))...)
	return allErrs
}

func ValidateJobUpdateStatus(job, oldJob *extensions.Job) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&oldJob.ObjectMeta, &job.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateJobStatusUpdate(job.Status, oldJob.Status)...)
	return allErrs
}

func ValidateJobSpecUpdate(spec, oldSpec extensions.JobSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateJobSpec(&spec, fldPath)...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Completions, oldSpec.Completions, fldPath.Child("completions"))...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Selector, oldSpec.Selector, fldPath.Child("selector"))...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Template, oldSpec.Template, fldPath.Child("template"))...)
	return allErrs
}

func ValidateJobStatusUpdate(status, oldStatus extensions.JobStatus) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateJobStatus(&status, field.NewPath("status"))...)
	return allErrs
}
