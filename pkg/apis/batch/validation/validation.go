/*
Copyright 2016 The Kubernetes Authors.

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
	"github.com/robfig/cron"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/batch"
)

// TODO: generalize for other controller objects that will follow the same pattern, such as ReplicaSet and DaemonSet, and
// move to new location.  Replace batch.Job with an interface.
//
// ValidateGeneratedSelector validates that the generated selector on a controller object match the controller object
// metadata, and the labels on the pod template are as generated.
func ValidateGeneratedSelector(obj *batch.Job) field.ErrorList {
	allErrs := field.ErrorList{}
	if obj.Spec.ManualSelector != nil && *obj.Spec.ManualSelector {
		return allErrs
	}

	if obj.Spec.Selector == nil {
		return allErrs // This case should already have been checked in caller.  No need for more errors.
	}

	// If somehow uid was unset then we would get "controller-uid=" as the selector
	// which is bad.
	if obj.ObjectMeta.UID == "" {
		allErrs = append(allErrs, field.Required(field.NewPath("metadata").Child("uid"), ""))
	}

	// If selector generation was requested, then expected labels must be
	// present on pod template, and must match job's uid and name.  The
	// generated (not-manual) selectors/labels ensure no overlap with other
	// controllers.  The manual mode allows orphaning, adoption,
	// backward-compatibility, and experimentation with new
	// labeling/selection schemes.  Automatic selector generation should
	// have placed certain labels on the pod, but this could have failed if
	// the user added coflicting labels.  Validate that the expected
	// generated ones are there.

	allErrs = append(allErrs, apivalidation.ValidateHasLabel(obj.Spec.Template.ObjectMeta, field.NewPath("spec").Child("template").Child("metadata"), "controller-uid", string(obj.UID))...)
	allErrs = append(allErrs, apivalidation.ValidateHasLabel(obj.Spec.Template.ObjectMeta, field.NewPath("spec").Child("template").Child("metadata"), "job-name", string(obj.Name))...)
	expectedLabels := make(map[string]string)
	expectedLabels["controller-uid"] = string(obj.UID)
	expectedLabels["job-name"] = string(obj.Name)
	// Whether manually or automatically generated, the selector of the job must match the pods it will produce.
	if selector, err := metav1.LabelSelectorAsSelector(obj.Spec.Selector); err == nil {
		if !selector.Matches(labels.Set(expectedLabels)) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("spec").Child("selector"), obj.Spec.Selector, "`selector` not auto-generated"))
		}
	}

	return allErrs
}

func ValidateJob(job *batch.Job) field.ErrorList {
	// Jobs and rcs have the same name validation
	allErrs := apivalidation.ValidateObjectMeta(&job.ObjectMeta, true, apivalidation.ValidateReplicationControllerName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateGeneratedSelector(job)...)
	allErrs = append(allErrs, ValidateJobSpec(&job.Spec, field.NewPath("spec"))...)
	return allErrs
}

func ValidateJobSpec(spec *batch.JobSpec, fldPath *field.Path) field.ErrorList {
	allErrs := validateJobSpec(spec, fldPath)

	if spec.Selector == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("selector"), ""))
	} else {
		allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(spec.Selector, fldPath.Child("selector"))...)
	}

	// Whether manually or automatically generated, the selector of the job must match the pods it will produce.
	if selector, err := metav1.LabelSelectorAsSelector(spec.Selector); err == nil {
		labels := labels.Set(spec.Template.Labels)
		if !selector.Matches(labels) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("template", "metadata", "labels"), spec.Template.Labels, "`selector` does not match template `labels`"))
		}
	}
	return allErrs
}

func validateJobSpec(spec *batch.JobSpec, fldPath *field.Path) field.ErrorList {
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
	if spec.BackoffLimit != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*spec.BackoffLimit), fldPath.Child("backoffLimit"))...)
	}

	allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpec(&spec.Template, fldPath.Child("template"))...)
	if spec.Template.Spec.RestartPolicy != api.RestartPolicyOnFailure &&
		spec.Template.Spec.RestartPolicy != api.RestartPolicyNever {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("template", "spec", "restartPolicy"),
			spec.Template.Spec.RestartPolicy, []string{string(api.RestartPolicyOnFailure), string(api.RestartPolicyNever)}))
	}
	return allErrs
}

func ValidateJobStatus(status *batch.JobStatus, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.Active), fldPath.Child("active"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.Succeeded), fldPath.Child("succeeded"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.Failed), fldPath.Child("failed"))...)
	return allErrs
}

func ValidateJobUpdate(job, oldJob *batch.Job) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&job.ObjectMeta, &oldJob.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateJobSpecUpdate(job.Spec, oldJob.Spec, field.NewPath("spec"))...)
	return allErrs
}

func ValidateJobUpdateStatus(job, oldJob *batch.Job) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&job.ObjectMeta, &oldJob.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateJobStatusUpdate(job.Status, oldJob.Status)...)
	return allErrs
}

func ValidateJobSpecUpdate(spec, oldSpec batch.JobSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateJobSpec(&spec, fldPath)...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Completions, oldSpec.Completions, fldPath.Child("completions"))...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Selector, oldSpec.Selector, fldPath.Child("selector"))...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Template, oldSpec.Template, fldPath.Child("template"))...)
	return allErrs
}

func ValidateJobStatusUpdate(status, oldStatus batch.JobStatus) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateJobStatus(&status, field.NewPath("status"))...)
	return allErrs
}

func ValidateCronJob(scheduledJob *batch.CronJob) field.ErrorList {
	// CronJobs and rcs have the same name validation
	allErrs := apivalidation.ValidateObjectMeta(&scheduledJob.ObjectMeta, true, apivalidation.ValidateReplicationControllerName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateCronJobSpec(&scheduledJob.Spec, field.NewPath("spec"))...)
	return allErrs
}

func ValidateCronJobSpec(spec *batch.CronJobSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(spec.Schedule) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("schedule"), ""))
	} else {
		allErrs = append(allErrs, validateScheduleFormat(spec.Schedule, fldPath.Child("schedule"))...)
	}
	if spec.StartingDeadlineSeconds != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*spec.StartingDeadlineSeconds), fldPath.Child("startingDeadlineSeconds"))...)
	}
	allErrs = append(allErrs, validateConcurrencyPolicy(&spec.ConcurrencyPolicy, fldPath.Child("concurrencyPolicy"))...)
	allErrs = append(allErrs, ValidateJobTemplateSpec(&spec.JobTemplate, fldPath.Child("jobTemplate"))...)

	if spec.SuccessfulJobsHistoryLimit != nil {
		// zero is a valid SuccessfulJobsHistoryLimit
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*spec.SuccessfulJobsHistoryLimit), fldPath.Child("successfulJobsHistoryLimit"))...)
	}
	if spec.FailedJobsHistoryLimit != nil {
		// zero is a valid SuccessfulJobsHistoryLimit
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*spec.FailedJobsHistoryLimit), fldPath.Child("failedJobsHistoryLimit"))...)
	}

	return allErrs
}

func validateConcurrencyPolicy(concurrencyPolicy *batch.ConcurrencyPolicy, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	switch *concurrencyPolicy {
	case batch.AllowConcurrent, batch.ForbidConcurrent, batch.ReplaceConcurrent:
		break
	case "":
		allErrs = append(allErrs, field.Required(fldPath, ""))
	default:
		validValues := []string{string(batch.AllowConcurrent), string(batch.ForbidConcurrent), string(batch.ReplaceConcurrent)}
		allErrs = append(allErrs, field.NotSupported(fldPath, *concurrencyPolicy, validValues))
	}

	return allErrs
}

func validateScheduleFormat(schedule string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if _, err := cron.ParseStandard(schedule); err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, schedule, err.Error()))
	}

	return allErrs
}

func ValidateJobTemplate(job *batch.JobTemplate) field.ErrorList {
	// this method should be identical to ValidateJob
	allErrs := apivalidation.ValidateObjectMeta(&job.ObjectMeta, true, apivalidation.ValidateReplicationControllerName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateJobTemplateSpec(&job.Template, field.NewPath("template"))...)
	return allErrs
}

func ValidateJobTemplateSpec(spec *batch.JobTemplateSpec, fldPath *field.Path) field.ErrorList {
	allErrs := validateJobSpec(&spec.Spec, fldPath.Child("spec"))

	// jobtemplate will always have the selector automatically generated
	if spec.Spec.Selector != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("spec", "selector"), spec.Spec.Selector, "`selector` will be auto-generated"))
	}
	if spec.Spec.ManualSelector != nil && *spec.Spec.ManualSelector {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("spec", "manualSelector"), spec.Spec.ManualSelector, []string{"nil", "false"}))
	}
	return allErrs
}
