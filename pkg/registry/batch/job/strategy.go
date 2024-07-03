/*
Copyright 2015 The Kubernetes Authors.

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

package job

import (
	"context"
	"fmt"
	"strconv"

	batchv1 "k8s.io/api/batch/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/job"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/apis/batch"
	batchvalidation "k8s.io/kubernetes/pkg/apis/batch/validation"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// jobStrategy implements verification logic for Replication Controllers.
type jobStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication Controller objects.
var Strategy = jobStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// DefaultGarbageCollectionPolicy returns OrphanDependents for batch/v1 for backwards compatibility,
// and DeleteDependents for all other versions.
func (jobStrategy) DefaultGarbageCollectionPolicy(ctx context.Context) rest.GarbageCollectionPolicy {
	var groupVersion schema.GroupVersion
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion = schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
	}
	switch groupVersion {
	case batchv1.SchemeGroupVersion:
		// for back compatibility
		return rest.OrphanDependents
	default:
		return rest.DeleteDependents
	}
}

// NamespaceScoped returns true because all jobs need to be within a namespace.
func (jobStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (jobStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"batch/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// PrepareForCreate clears the status of a job before creation.
func (jobStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	job := obj.(*batch.Job)
	generateSelectorIfNeeded(job)
	job.Status = batch.JobStatus{}

	job.Generation = 1

	if !utilfeature.DefaultFeatureGate.Enabled(features.JobPodFailurePolicy) {
		job.Spec.PodFailurePolicy = nil
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.JobManagedBy) {
		job.Spec.ManagedBy = nil
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.JobSuccessPolicy) {
		job.Spec.SuccessPolicy = nil
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.JobBackoffLimitPerIndex) {
		job.Spec.BackoffLimitPerIndex = nil
		job.Spec.MaxFailedIndexes = nil
		if job.Spec.PodFailurePolicy != nil {
			// We drop the FailIndex pod failure policy rules because
			// JobBackoffLimitPerIndex is disabled.
			index := 0
			for _, rule := range job.Spec.PodFailurePolicy.Rules {
				if rule.Action != batch.PodFailurePolicyActionFailIndex {
					job.Spec.PodFailurePolicy.Rules[index] = rule
					index++
				}
			}
			job.Spec.PodFailurePolicy.Rules = job.Spec.PodFailurePolicy.Rules[:index]
		}
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.JobPodReplacementPolicy) {
		job.Spec.PodReplacementPolicy = nil
	}

	pod.DropDisabledTemplateFields(&job.Spec.Template, nil)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (jobStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newJob := obj.(*batch.Job)
	oldJob := old.(*batch.Job)
	newJob.Status = oldJob.Status

	if !utilfeature.DefaultFeatureGate.Enabled(features.JobPodFailurePolicy) && oldJob.Spec.PodFailurePolicy == nil {
		newJob.Spec.PodFailurePolicy = nil
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.JobSuccessPolicy) && oldJob.Spec.SuccessPolicy == nil {
		newJob.Spec.SuccessPolicy = nil
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.JobBackoffLimitPerIndex) {
		if oldJob.Spec.BackoffLimitPerIndex == nil {
			newJob.Spec.BackoffLimitPerIndex = nil
		}
		if oldJob.Spec.MaxFailedIndexes == nil {
			newJob.Spec.MaxFailedIndexes = nil
		}
		// We keep pod failure policy rules with FailIndex actions (is any),
		// since the pod failure policy is immutable. Note that, if the old job
		// had BackoffLimitPerIndex set, the new Job will also have it, so the
		// validation of the pod failure policy with FailIndex rules will
		// continue to pass.
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.JobPodReplacementPolicy) && oldJob.Spec.PodReplacementPolicy == nil {
		newJob.Spec.PodReplacementPolicy = nil
	}

	pod.DropDisabledTemplateFields(&newJob.Spec.Template, &oldJob.Spec.Template)

	// Any changes to the spec increment the generation number.
	// See metav1.ObjectMeta description for more information on Generation.
	if !apiequality.Semantic.DeepEqual(newJob.Spec, oldJob.Spec) {
		newJob.Generation = oldJob.Generation + 1
	}

}

// Validate validates a new job.
func (jobStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	job := obj.(*batch.Job)
	opts := validationOptionsForJob(job, nil)
	return batchvalidation.ValidateJob(job, opts)
}

func validationOptionsForJob(newJob, oldJob *batch.Job) batchvalidation.JobValidationOptions {
	var newPodTemplate, oldPodTemplate *core.PodTemplateSpec
	if newJob != nil {
		newPodTemplate = &newJob.Spec.Template
	}
	if oldJob != nil {
		oldPodTemplate = &oldJob.Spec.Template
	}
	opts := batchvalidation.JobValidationOptions{
		PodValidationOptions:    pod.GetValidationOptionsFromPodTemplate(newPodTemplate, oldPodTemplate),
		AllowElasticIndexedJobs: utilfeature.DefaultFeatureGate.Enabled(features.ElasticIndexedJob),
		RequirePrefixedLabels:   true,
	}
	if oldJob != nil {
		opts.AllowInvalidLabelValueInSelector = opts.AllowInvalidLabelValueInSelector || metav1validation.LabelSelectorHasInvalidLabelValue(oldJob.Spec.Selector)

		// Updating node affinity, node selector and tolerations is allowed
		// only for suspended jobs that never started before.
		suspended := oldJob.Spec.Suspend != nil && *oldJob.Spec.Suspend
		notStarted := oldJob.Status.StartTime == nil
		opts.AllowMutableSchedulingDirectives = suspended && notStarted

		// Validation should not fail jobs if they don't have the new labels.
		// This can be removed once we have high confidence that both labels exist (1.30 at least)
		_, hadJobName := oldJob.Spec.Template.Labels[batch.JobNameLabel]
		_, hadControllerUid := oldJob.Spec.Template.Labels[batch.ControllerUidLabel]
		opts.RequirePrefixedLabels = hadJobName && hadControllerUid
	}
	return opts
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (jobStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	newJob := obj.(*batch.Job)
	var warnings []string
	if msgs := utilvalidation.IsDNS1123Label(newJob.Name); len(msgs) != 0 {
		warnings = append(warnings, fmt.Sprintf("metadata.name: this is used in Pod names and hostnames, which can result in surprising behavior; a DNS label is recommended: %v", msgs))
	}
	warnings = append(warnings, job.WarningsForJobSpec(ctx, field.NewPath("spec"), &newJob.Spec, nil)...)
	return warnings
}

// generateSelectorIfNeeded checks the job's manual selector flag and generates selector labels if the flag is true.
func generateSelectorIfNeeded(obj *batch.Job) {
	if !*obj.Spec.ManualSelector {
		generateSelector(obj)
	}
}

// generateSelector adds a selector to a job and labels to its template
// which can be used to uniquely identify the pods created by that job,
// if the user has requested this behavior.
func generateSelector(obj *batch.Job) {
	if obj.Spec.Template.Labels == nil {
		obj.Spec.Template.Labels = make(map[string]string)
	}
	// The job-name label is unique except in cases that are expected to be
	// quite uncommon, and is more user friendly than uid.  So, we add it as
	// a label.
	jobNameLabels := []string{batch.LegacyJobNameLabel, batch.JobNameLabel}
	for _, value := range jobNameLabels {
		_, found := obj.Spec.Template.Labels[value]
		if found {
			// User asked us to automatically generate a selector, but set manual labels.
			// If there is a conflict, we will reject in validation.
		} else {
			obj.Spec.Template.Labels[value] = string(obj.ObjectMeta.Name)
		}
	}

	// The controller-uid label makes the pods that belong to this job
	// only match this job.
	controllerUidLabels := []string{batch.LegacyControllerUidLabel, batch.ControllerUidLabel}
	for _, value := range controllerUidLabels {
		_, found := obj.Spec.Template.Labels[value]
		if found {
			// User asked us to automatically generate a selector, but set manual labels.
			// If there is a conflict, we will reject in validation.
		} else {
			obj.Spec.Template.Labels[value] = string(obj.ObjectMeta.UID)
		}
	}
	// Select the controller-uid label.  This is sufficient for uniqueness.
	if obj.Spec.Selector == nil {
		obj.Spec.Selector = &metav1.LabelSelector{}
	}
	if obj.Spec.Selector.MatchLabels == nil {
		obj.Spec.Selector.MatchLabels = make(map[string]string)
	}

	if _, found := obj.Spec.Selector.MatchLabels[batch.ControllerUidLabel]; !found {
		obj.Spec.Selector.MatchLabels[batch.ControllerUidLabel] = string(obj.ObjectMeta.UID)
	}
	// If the user specified matchLabel controller-uid=$WRONGUID, then it should fail
	// in validation, either because the selector does not match the pod template
	// (controller-uid=$WRONGUID does not match controller-uid=$UID, which we applied
	// above, or we will reject in validation because the template has the wrong
	// labels.
}

// TODO: generalize generateSelector so it can work for other controller
// objects such as ReplicaSet.  Can use pkg/api/meta to generically get the
// UID, but need some way to generically access the selector and pod labels
// fields.

// Canonicalize normalizes the object after validation.
func (jobStrategy) Canonicalize(obj runtime.Object) {
}

func (jobStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// AllowCreateOnUpdate is false for jobs; this means a POST is needed to create one.
func (jobStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (jobStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	job := obj.(*batch.Job)
	oldJob := old.(*batch.Job)

	opts := validationOptionsForJob(job, oldJob)
	validationErrorList := batchvalidation.ValidateJob(job, opts)
	updateErrorList := batchvalidation.ValidateJobUpdate(job, oldJob, opts)
	return append(validationErrorList, updateErrorList...)
}

// WarningsOnUpdate returns warnings for the given update.
func (jobStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	var warnings []string
	newJob := obj.(*batch.Job)
	oldJob := old.(*batch.Job)
	if newJob.Generation != oldJob.Generation {
		warnings = job.WarningsForJobSpec(ctx, field.NewPath("spec"), &newJob.Spec, &oldJob.Spec)
	}
	return warnings
}

type jobStatusStrategy struct {
	jobStrategy
}

var StatusStrategy = jobStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (jobStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return map[fieldpath.APIVersion]*fieldpath.Set{
		"batch/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}
}

func (jobStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newJob := obj.(*batch.Job)
	oldJob := old.(*batch.Job)
	newJob.Spec = oldJob.Spec
}

func (jobStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newJob := obj.(*batch.Job)
	oldJob := old.(*batch.Job)

	opts := getStatusValidationOptions(newJob, oldJob)
	return batchvalidation.ValidateJobUpdateStatus(newJob, oldJob, opts)
}

// getStatusValidationOptions returns validation options for Job status
func getStatusValidationOptions(newJob, oldJob *batch.Job) batchvalidation.JobStatusValidationOptions {
	if utilfeature.DefaultFeatureGate.Enabled(features.JobManagedBy) {
		// A strengthened validation of the Job status transitions is needed since the
		// Job managedBy field let's the Job object be controlled by external
		// controllers. We want to make sure the transitions done by the external
		// controllers meet the expectations of the clients of the Job API.
		// For example, we verify that a Job in terminal state (Failed or Complete)
		// does not flip to a non-terminal state.
		//
		// In the checks below we fail validation for Job status fields (or conditions) only if they change their values
		// (compared to the oldJob). This allows proceeding with status updates unrelated to the fields violating the
		// checks, while blocking bad status updates for jobs with correct status.
		//
		// Also note, there is another reason we run the validation rules only
		// if the associated status fields changed. We do it also because some of
		// the validation rules might be temporarily violated just after a user
		// updating the spec. In that case we want to give time to the Job
		// controller to "fix" the status in the following sync. For example, the
		// rule for checking the format of completedIndexes expects them to be
		// below .spec.completions, however, this it is ok if the
		// status.completedIndexes go beyond completions just after a user scales
		// down a Job.
		isIndexed := ptr.Deref(newJob.Spec.CompletionMode, batch.NonIndexedCompletion) == batch.IndexedCompletion

		isJobFinishedChanged := batchvalidation.IsJobFinished(oldJob) != batchvalidation.IsJobFinished(newJob)
		isJobCompleteChanged := batchvalidation.IsJobComplete(oldJob) != batchvalidation.IsJobComplete(newJob)
		isJobFailedChanged := batchvalidation.IsJobFailed(oldJob) != batchvalidation.IsJobFailed(newJob)
		isJobFailureTargetChanged := batchvalidation.IsConditionTrue(oldJob.Status.Conditions, batch.JobFailureTarget) != batchvalidation.IsConditionTrue(newJob.Status.Conditions, batch.JobFailureTarget)
		isCompletedIndexesChanged := oldJob.Status.CompletedIndexes != newJob.Status.CompletedIndexes
		isFailedIndexesChanged := !ptr.Equal(oldJob.Status.FailedIndexes, newJob.Status.FailedIndexes)
		isActiveChanged := oldJob.Status.Active != newJob.Status.Active
		isStartTimeChanged := !ptr.Equal(oldJob.Status.StartTime, newJob.Status.StartTime)
		isCompletionTimeChanged := !ptr.Equal(oldJob.Status.CompletionTime, newJob.Status.CompletionTime)
		isUncountedTerminatedPodsChanged := !apiequality.Semantic.DeepEqual(oldJob.Status.UncountedTerminatedPods, newJob.Status.UncountedTerminatedPods)

		return batchvalidation.JobStatusValidationOptions{
			// We allow to decrease the counter for succeeded pods for jobs which
			// have equal parallelism and completions, as they can be scaled-down.
			RejectDecreasingSucceededCounter:             !isIndexed || !ptr.Equal(newJob.Spec.Completions, newJob.Spec.Parallelism),
			RejectDecreasingFailedCounter:                true,
			RejectDisablingTerminalCondition:             true,
			RejectInvalidCompletedIndexes:                isCompletedIndexesChanged,
			RejectInvalidFailedIndexes:                   isFailedIndexesChanged,
			RejectCompletedIndexesForNonIndexedJob:       isCompletedIndexesChanged,
			RejectFailedIndexesForNoBackoffLimitPerIndex: isFailedIndexesChanged,
			RejectFailedIndexesOverlappingCompleted:      isFailedIndexesChanged || isCompletedIndexesChanged,
			RejectFinishedJobWithActivePods:              isJobFinishedChanged || isActiveChanged,
			RejectFinishedJobWithoutStartTime:            isJobFinishedChanged || isStartTimeChanged,
			RejectFinishedJobWithUncountedTerminatedPods: isJobFinishedChanged || isUncountedTerminatedPodsChanged,
			RejectStartTimeUpdateForUnsuspendedJob:       isStartTimeChanged,
			RejectCompletionTimeBeforeStartTime:          isStartTimeChanged || isCompletionTimeChanged,
			RejectMutatingCompletionTime:                 true,
			RejectNotCompleteJobWithCompletionTime:       isJobCompleteChanged || isCompletionTimeChanged,
			RejectCompleteJobWithoutCompletionTime:       isJobCompleteChanged || isCompletionTimeChanged,
			RejectCompleteJobWithFailedCondition:         isJobCompleteChanged || isJobFailedChanged,
			RejectCompleteJobWithFailureTargetCondition:  isJobCompleteChanged || isJobFailureTargetChanged,
			AllowForSuccessCriteriaMetInExtendedScope:    true,
		}
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.JobPodReplacementPolicy) {
		return batchvalidation.JobStatusValidationOptions{
			AllowForSuccessCriteriaMetInExtendedScope: true,
		}
	}
	return batchvalidation.JobStatusValidationOptions{
		AllowForSuccessCriteriaMetInExtendedScope: batchvalidation.IsConditionTrue(oldJob.Status.Conditions, batch.JobSuccessCriteriaMet),
	}
}

// WarningsOnUpdate returns warnings for the given update.
func (jobStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// JobSelectableFields returns a field set that represents the object for matching purposes.
func JobToSelectableFields(job *batch.Job) fields.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(&job.ObjectMeta, true)
	specificFieldsSet := fields.Set{
		"status.successful": strconv.Itoa(int(job.Status.Succeeded)),
	}
	return generic.MergeFieldsSets(objectMetaFieldsSet, specificFieldsSet)
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	job, ok := obj.(*batch.Job)
	if !ok {
		return nil, nil, fmt.Errorf("given object is not a job.")
	}
	return labels.Set(job.ObjectMeta.Labels), JobToSelectableFields(job), nil
}

// MatchJob is the filter used by the generic etcd backend to route
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func MatchJob(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}
