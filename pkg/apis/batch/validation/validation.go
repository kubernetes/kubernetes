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
	"fmt"
	"strings"
	"time"

	"github.com/robfig/cron/v3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	apimachineryvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/utils/pointer"
)

// maxParallelismForIndexJob is the maximum parallelism that an Indexed Job
// is allowed to have. This threshold allows to cap the length of
// .status.completedIndexes.
const maxParallelismForIndexedJob = 100000

// ValidateGeneratedSelector validates that the generated selector on a controller object match the controller object
// metadata, and the labels on the pod template are as generated.
//
// TODO: generalize for other controller objects that will follow the same pattern, such as ReplicaSet and DaemonSet, and
// move to new location.  Replace batch.Job with an interface.
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

// ValidateJob validates a Job and returns an ErrorList with any errors.
func ValidateJob(job *batch.Job, opts JobValidationOptions) field.ErrorList {
	// Jobs and rcs have the same name validation
	allErrs := apivalidation.ValidateObjectMeta(&job.ObjectMeta, true, apivalidation.ValidateReplicationControllerName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateGeneratedSelector(job)...)
	allErrs = append(allErrs, ValidateJobSpec(&job.Spec, field.NewPath("spec"), opts.PodValidationOptions)...)
	if !opts.AllowTrackingAnnotation && hasJobTrackingAnnotation(job) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("metadata").Child("annotations").Key(batch.JobTrackingFinalizer), "cannot add this annotation"))
	}
	if job.Spec.CompletionMode != nil && *job.Spec.CompletionMode == batch.IndexedCompletion && job.Spec.Completions != nil && *job.Spec.Completions > 0 {
		// For indexed job, the job controller appends a suffix (`-$INDEX`)
		// to the pod hostname when indexed job create pods.
		// The index could be maximum `.spec.completions-1`
		// If we don't validate this here, the indexed job will fail to create pods later.
		maximumPodHostname := fmt.Sprintf("%s-%d", job.ObjectMeta.Name, *job.Spec.Completions-1)
		if errs := apimachineryvalidation.IsDNS1123Label(maximumPodHostname); len(errs) > 0 {
			allErrs = append(allErrs, field.Invalid(field.NewPath("metadata").Child("name"), job.ObjectMeta.Name, fmt.Sprintf("will not able to create pod with invalid DNS label: %s", maximumPodHostname)))
		}
	}
	return allErrs
}

func hasJobTrackingAnnotation(job *batch.Job) bool {
	if job.Annotations == nil {
		return false
	}
	_, ok := job.Annotations[batch.JobTrackingFinalizer]
	return ok
}

// ValidateJobSpec validates a JobSpec and returns an ErrorList with any errors.
func ValidateJobSpec(spec *batch.JobSpec, fldPath *field.Path, opts apivalidation.PodValidationOptions) field.ErrorList {
	allErrs := validateJobSpec(spec, fldPath, opts)

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

func validateJobSpec(spec *batch.JobSpec, fldPath *field.Path, opts apivalidation.PodValidationOptions) field.ErrorList {
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
	if spec.TTLSecondsAfterFinished != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*spec.TTLSecondsAfterFinished), fldPath.Child("ttlSecondsAfterFinished"))...)
	}
	// CompletionMode might be nil when IndexedJob feature gate is disabled.
	if spec.CompletionMode != nil {
		if *spec.CompletionMode != batch.NonIndexedCompletion && *spec.CompletionMode != batch.IndexedCompletion {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("completionMode"), spec.CompletionMode, []string{string(batch.NonIndexedCompletion), string(batch.IndexedCompletion)}))
		}
		if *spec.CompletionMode == batch.IndexedCompletion {
			if spec.Completions == nil {
				allErrs = append(allErrs, field.Required(fldPath.Child("completions"), fmt.Sprintf("when completion mode is %s", batch.IndexedCompletion)))
			}
			if spec.Parallelism != nil && *spec.Parallelism > maxParallelismForIndexedJob {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("parallelism"), *spec.Parallelism, fmt.Sprintf("must be less than or equal to %d when completion mode is %s", maxParallelismForIndexedJob, batch.IndexedCompletion)))
			}
		}
	}

	allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpec(&spec.Template, fldPath.Child("template"), opts)...)

	// spec.Template.Spec.RestartPolicy can be defaulted as RestartPolicyAlways
	// by SetDefaults_PodSpec function when the user does not explicitly specify a value for it,
	// so we check both empty and RestartPolicyAlways cases here
	if spec.Template.Spec.RestartPolicy == api.RestartPolicyAlways || spec.Template.Spec.RestartPolicy == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("template", "spec", "restartPolicy"),
			fmt.Sprintf("valid values: %q, %q", api.RestartPolicyOnFailure, api.RestartPolicyNever)))
	} else if spec.Template.Spec.RestartPolicy != api.RestartPolicyOnFailure && spec.Template.Spec.RestartPolicy != api.RestartPolicyNever {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("template", "spec", "restartPolicy"),
			spec.Template.Spec.RestartPolicy, []string{string(api.RestartPolicyOnFailure), string(api.RestartPolicyNever)}))
	}
	return allErrs
}

// validateJobStatus validates a JobStatus and returns an ErrorList with any errors.
func validateJobStatus(status *batch.JobStatus, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.Active), fldPath.Child("active"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.Succeeded), fldPath.Child("succeeded"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.Failed), fldPath.Child("failed"))...)
	if status.Ready != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*status.Ready), fldPath.Child("ready"))...)
	}
	if status.UncountedTerminatedPods != nil {
		path := fldPath.Child("uncountedTerminatedPods")
		seen := sets.NewString()
		for i, k := range status.UncountedTerminatedPods.Succeeded {
			p := path.Child("succeeded").Index(i)
			if k == "" {
				allErrs = append(allErrs, field.Invalid(p, k, "must not be empty"))
			} else if seen.Has(string(k)) {
				allErrs = append(allErrs, field.Duplicate(p, k))
			} else {
				seen.Insert(string(k))
			}
		}
		for i, k := range status.UncountedTerminatedPods.Failed {
			p := path.Child("failed").Index(i)
			if k == "" {
				allErrs = append(allErrs, field.Invalid(p, k, "must not be empty"))
			} else if seen.Has(string(k)) {
				allErrs = append(allErrs, field.Duplicate(p, k))
			} else {
				seen.Insert(string(k))
			}
		}
	}
	return allErrs
}

// ValidateJobUpdate validates an update to a Job and returns an ErrorList with any errors.
func ValidateJobUpdate(job, oldJob *batch.Job, opts JobValidationOptions) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&job.ObjectMeta, &oldJob.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateJobSpecUpdate(job.Spec, oldJob.Spec, field.NewPath("spec"), opts)...)
	return allErrs
}

// ValidateJobUpdateStatus validates an update to the status of a Job and returns an ErrorList with any errors.
func ValidateJobUpdateStatus(job, oldJob *batch.Job) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&job.ObjectMeta, &oldJob.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateJobStatusUpdate(job.Status, oldJob.Status)...)
	return allErrs
}

// ValidateJobSpecUpdate validates an update to a JobSpec and returns an ErrorList with any errors.
func ValidateJobSpecUpdate(spec, oldSpec batch.JobSpec, fldPath *field.Path, opts JobValidationOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateJobSpec(&spec, fldPath, opts.PodValidationOptions)...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Completions, oldSpec.Completions, fldPath.Child("completions"))...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Selector, oldSpec.Selector, fldPath.Child("selector"))...)
	allErrs = append(allErrs, validatePodTemplateUpdate(spec, oldSpec, fldPath, opts)...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.CompletionMode, oldSpec.CompletionMode, fldPath.Child("completionMode"))...)
	return allErrs
}

func validatePodTemplateUpdate(spec, oldSpec batch.JobSpec, fldPath *field.Path, opts JobValidationOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	template := &spec.Template
	oldTemplate := &oldSpec.Template
	if opts.AllowMutableSchedulingDirectives {
		oldTemplate = oldSpec.Template.DeepCopy() // +k8s:verify-mutation:reason=clone
		switch {
		case template.Spec.Affinity == nil && oldTemplate.Spec.Affinity != nil:
			// allow the Affinity field to be cleared if the old template had no affinity directives other than NodeAffinity
			oldTemplate.Spec.Affinity.NodeAffinity = nil // +k8s:verify-mutation:reason=clone
			if (*oldTemplate.Spec.Affinity) == (api.Affinity{}) {
				oldTemplate.Spec.Affinity = nil // +k8s:verify-mutation:reason=clone
			}
		case template.Spec.Affinity != nil && oldTemplate.Spec.Affinity == nil:
			// allow the NodeAffinity field to skip immutability checking
			oldTemplate.Spec.Affinity = &api.Affinity{NodeAffinity: template.Spec.Affinity.NodeAffinity} // +k8s:verify-mutation:reason=clone
		case template.Spec.Affinity != nil && oldTemplate.Spec.Affinity != nil:
			// allow the NodeAffinity field to skip immutability checking
			oldTemplate.Spec.Affinity.NodeAffinity = template.Spec.Affinity.NodeAffinity // +k8s:verify-mutation:reason=clone
		}
		oldTemplate.Spec.NodeSelector = template.Spec.NodeSelector // +k8s:verify-mutation:reason=clone
		oldTemplate.Spec.Tolerations = template.Spec.Tolerations   // +k8s:verify-mutation:reason=clone
		oldTemplate.Annotations = template.Annotations             // +k8s:verify-mutation:reason=clone
		oldTemplate.Labels = template.Labels                       // +k8s:verify-mutation:reason=clone
	}
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(template, oldTemplate, fldPath.Child("template"))...)
	return allErrs
}

// ValidateJobStatusUpdate validates an update to a JobStatus and returns an ErrorList with any errors.
func ValidateJobStatusUpdate(status, oldStatus batch.JobStatus) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validateJobStatus(&status, field.NewPath("status"))...)
	return allErrs
}

// ValidateCronJobCreate validates a CronJob on creation and returns an ErrorList with any errors.
func ValidateCronJobCreate(cronJob *batch.CronJob, opts apivalidation.PodValidationOptions) field.ErrorList {
	// CronJobs and rcs have the same name validation
	allErrs := apivalidation.ValidateObjectMeta(&cronJob.ObjectMeta, true, apivalidation.ValidateReplicationControllerName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateCronJobSpec(&cronJob.Spec, nil, field.NewPath("spec"), opts)...)
	if len(cronJob.ObjectMeta.Name) > apimachineryvalidation.DNS1035LabelMaxLength-11 {
		// The cronjob controller appends a 11-character suffix to the cronjob (`-$TIMESTAMP`) when
		// creating a job. The job name length limit is 63 characters.
		// Therefore cronjob names must have length <= 63-11=52. If we don't validate this here,
		// then job creation will fail later.
		allErrs = append(allErrs, field.Invalid(field.NewPath("metadata").Child("name"), cronJob.ObjectMeta.Name, "must be no more than 52 characters"))
	}
	return allErrs
}

// ValidateCronJobUpdate validates an update to a CronJob and returns an ErrorList with any errors.
func ValidateCronJobUpdate(job, oldJob *batch.CronJob, opts apivalidation.PodValidationOptions) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&job.ObjectMeta, &oldJob.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateCronJobSpec(&job.Spec, &oldJob.Spec, field.NewPath("spec"), opts)...)

	// skip the 52-character name validation limit on update validation
	// to allow old cronjobs with names > 52 chars to be updated/deleted
	return allErrs
}

// validateCronJobSpec validates a CronJobSpec and returns an ErrorList with any errors.
func validateCronJobSpec(spec, oldSpec *batch.CronJobSpec, fldPath *field.Path, opts apivalidation.PodValidationOptions) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(spec.Schedule) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("schedule"), ""))
	} else {
		allErrs = append(allErrs, validateScheduleFormat(spec.Schedule, spec.TimeZone, fldPath.Child("schedule"))...)
	}

	if spec.StartingDeadlineSeconds != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*spec.StartingDeadlineSeconds), fldPath.Child("startingDeadlineSeconds"))...)
	}

	if oldSpec == nil || !pointer.StringEqual(oldSpec.TimeZone, spec.TimeZone) {
		allErrs = append(allErrs, validateTimeZone(spec.TimeZone, fldPath.Child("timeZone"))...)
	}

	allErrs = append(allErrs, validateConcurrencyPolicy(&spec.ConcurrencyPolicy, fldPath.Child("concurrencyPolicy"))...)
	allErrs = append(allErrs, ValidateJobTemplateSpec(&spec.JobTemplate, fldPath.Child("jobTemplate"), opts)...)

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

func validateScheduleFormat(schedule string, timeZone *string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if _, err := cron.ParseStandard(schedule); err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, schedule, err.Error()))
	}
	if strings.Contains(schedule, "TZ") && timeZone != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, schedule, "cannot use both timeZone field and TZ or CRON_TZ in schedule"))
	}

	return allErrs
}

func validateTimeZone(timeZone *string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if timeZone == nil {
		return allErrs
	}

	if len(*timeZone) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, timeZone, "timeZone must be nil or non-empty string"))
		return allErrs
	}

	if strings.EqualFold(*timeZone, "Local") {
		allErrs = append(allErrs, field.Invalid(fldPath, timeZone, "timeZone must be an explicit time zone as defined in https://www.iana.org/time-zones"))
	}

	if _, err := time.LoadLocation(*timeZone); err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, timeZone, err.Error()))
	}

	return allErrs
}

// ValidateJobTemplate validates a JobTemplate and returns an ErrorList with any errors.
func ValidateJobTemplate(job *batch.JobTemplate, opts apivalidation.PodValidationOptions) field.ErrorList {
	// this method should be identical to ValidateJob
	allErrs := apivalidation.ValidateObjectMeta(&job.ObjectMeta, true, apivalidation.ValidateReplicationControllerName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateJobTemplateSpec(&job.Template, field.NewPath("template"), opts)...)
	return allErrs
}

// ValidateJobTemplateSpec validates a JobTemplateSpec and returns an ErrorList with any errors.
func ValidateJobTemplateSpec(spec *batch.JobTemplateSpec, fldPath *field.Path, opts apivalidation.PodValidationOptions) field.ErrorList {
	allErrs := validateJobSpec(&spec.Spec, fldPath.Child("spec"), opts)

	// jobtemplate will always have the selector automatically generated
	if spec.Spec.Selector != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("spec", "selector"), spec.Spec.Selector, "`selector` will be auto-generated"))
	}
	if spec.Spec.ManualSelector != nil && *spec.Spec.ManualSelector {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("spec", "manualSelector"), spec.Spec.ManualSelector, []string{"nil", "false"}))
	}
	return allErrs
}

type JobValidationOptions struct {
	apivalidation.PodValidationOptions
	// Allow Job to have the annotation batch.kubernetes.io/job-tracking
	AllowTrackingAnnotation bool
	// Allow mutable node affinity, selector and tolerations of the template
	AllowMutableSchedulingDirectives bool
}
