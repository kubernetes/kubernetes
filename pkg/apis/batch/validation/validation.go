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
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/robfig/cron/v3"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	apimachineryvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/utils/pointer"
	"k8s.io/utils/ptr"
)

// maxParallelismForIndexJob is the maximum parallelism that an Indexed Job
// is allowed to have. This threshold allows to cap the length of
// .status.completedIndexes.
const maxParallelismForIndexedJob = 100000

// maxFailedIndexesForIndexedJob is the maximum number of failed indexes that
// an Indexed Job is allowed to have. This threshold allows to cap the length of
// .status.completedIndexes and .status.failedIndexes.
const maxFailedIndexesForIndexedJob = 100_000

const (
	completionsSoftLimit                    = 100_000
	parallelismLimitForHighCompletions      = 10_000
	maxFailedIndexesLimitForHighCompletions = 10_000

	// maximum number of rules in pod failure policy
	maxPodFailurePolicyRules = 20

	// maximum number of values for a OnExitCodes requirement in pod failure policy
	maxPodFailurePolicyOnExitCodesValues = 255

	// maximum number of patterns for a OnPodConditions requirement in pod failure policy
	maxPodFailurePolicyOnPodConditionsPatterns = 20

	// maximum length of the value of the managedBy field
	maxManagedByLength = 63

	// maximum length of succeededIndexes in JobSuccessPolicy.
	maxJobSuccessPolicySucceededIndexesLimit = 64 * 1024
	// maximum number of rules in successPolicy.
	maxSuccessPolicyRule = 20
)

var (
	supportedPodFailurePolicyActions = sets.New(
		batch.PodFailurePolicyActionCount,
		batch.PodFailurePolicyActionFailIndex,
		batch.PodFailurePolicyActionFailJob,
		batch.PodFailurePolicyActionIgnore)

	supportedPodFailurePolicyOnExitCodesOperator = sets.New(
		batch.PodFailurePolicyOnExitCodesOpIn,
		batch.PodFailurePolicyOnExitCodesOpNotIn)

	supportedPodFailurePolicyOnPodConditionsStatus = sets.New(
		api.ConditionFalse,
		api.ConditionTrue,
		api.ConditionUnknown)

	supportedPodReplacementPolicy = sets.New(
		batch.Failed,
		batch.TerminatingOrFailed)
)

// validateGeneratedSelector validates that the generated selector on a controller object match the controller object
// metadata, and the labels on the pod template are as generated.
//
// TODO: generalize for other controller objects that will follow the same pattern, such as ReplicaSet and DaemonSet, and
// move to new location.  Replace batch.Job with an interface.
func validateGeneratedSelector(obj *batch.Job, validateBatchLabels bool) field.ErrorList {
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
	// the user added conflicting labels.  Validate that the expected
	// generated ones are there.
	allErrs = append(allErrs, apivalidation.ValidateHasLabel(obj.Spec.Template.ObjectMeta, field.NewPath("spec").Child("template").Child("metadata"), batch.LegacyControllerUidLabel, string(obj.UID))...)
	allErrs = append(allErrs, apivalidation.ValidateHasLabel(obj.Spec.Template.ObjectMeta, field.NewPath("spec").Child("template").Child("metadata"), batch.LegacyJobNameLabel, string(obj.Name))...)
	expectedLabels := make(map[string]string)
	if validateBatchLabels {
		allErrs = append(allErrs, apivalidation.ValidateHasLabel(obj.Spec.Template.ObjectMeta, field.NewPath("spec").Child("template").Child("metadata"), batch.ControllerUidLabel, string(obj.UID))...)
		allErrs = append(allErrs, apivalidation.ValidateHasLabel(obj.Spec.Template.ObjectMeta, field.NewPath("spec").Child("template").Child("metadata"), batch.JobNameLabel, string(obj.Name))...)
		expectedLabels[batch.ControllerUidLabel] = string(obj.UID)
		expectedLabels[batch.JobNameLabel] = string(obj.Name)
	}
	// Labels created by the Kubernetes project should have a Kubernetes prefix.
	// These labels are set due to legacy reasons.

	expectedLabels[batch.LegacyControllerUidLabel] = string(obj.UID)
	expectedLabels[batch.LegacyJobNameLabel] = string(obj.Name)
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
	allErrs = append(allErrs, validateGeneratedSelector(job, opts.RequirePrefixedLabels)...)
	allErrs = append(allErrs, ValidateJobSpec(&job.Spec, field.NewPath("spec"), opts.PodValidationOptions)...)
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

// ValidateJobSpec validates a JobSpec and returns an ErrorList with any errors.
func ValidateJobSpec(spec *batch.JobSpec, fldPath *field.Path, opts apivalidation.PodValidationOptions) field.ErrorList {
	allErrs := validateJobSpec(spec, fldPath, opts)
	if spec.Selector == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("selector"), ""))
	} else {
		labelSelectorValidationOpts := unversionedvalidation.LabelSelectorValidationOptions{
			AllowInvalidLabelValueInSelector: opts.AllowInvalidLabelValueInSelector,
		}
		allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(spec.Selector, labelSelectorValidationOpts, fldPath.Child("selector"))...)
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
	if spec.BackoffLimitPerIndex != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*spec.BackoffLimitPerIndex), fldPath.Child("backoffLimitPerIndex"))...)
	}
	if spec.MaxFailedIndexes != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*spec.MaxFailedIndexes), fldPath.Child("maxFailedIndexes"))...)
		if spec.BackoffLimitPerIndex == nil {
			allErrs = append(allErrs, field.Required(fldPath.Child("backoffLimitPerIndex"), fmt.Sprintf("when maxFailedIndexes is specified")))
		}
	}
	if spec.ManagedBy != nil {
		allErrs = append(allErrs, apimachineryvalidation.IsDomainPrefixedPath(fldPath.Child("managedBy"), *spec.ManagedBy)...)
		if len(*spec.ManagedBy) > maxManagedByLength {
			allErrs = append(allErrs, field.TooLongMaxLength(fldPath.Child("managedBy"), *spec.ManagedBy, maxManagedByLength))
		}
	}
	if spec.CompletionMode != nil {
		if *spec.CompletionMode != batch.NonIndexedCompletion && *spec.CompletionMode != batch.IndexedCompletion {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("completionMode"), spec.CompletionMode, []batch.CompletionMode{batch.NonIndexedCompletion, batch.IndexedCompletion}))
		}
		if *spec.CompletionMode == batch.IndexedCompletion {
			if spec.Completions == nil {
				allErrs = append(allErrs, field.Required(fldPath.Child("completions"), fmt.Sprintf("when completion mode is %s", batch.IndexedCompletion)))
			}
			if spec.Parallelism != nil && *spec.Parallelism > maxParallelismForIndexedJob {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("parallelism"), *spec.Parallelism, fmt.Sprintf("must be less than or equal to %d when completion mode is %s", maxParallelismForIndexedJob, batch.IndexedCompletion)))
			}
			if spec.Completions != nil && spec.MaxFailedIndexes != nil && *spec.MaxFailedIndexes > *spec.Completions {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("maxFailedIndexes"), *spec.MaxFailedIndexes, "must be less than or equal to completions"))
			}
			if spec.MaxFailedIndexes != nil && *spec.MaxFailedIndexes > maxFailedIndexesForIndexedJob {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("maxFailedIndexes"), *spec.MaxFailedIndexes, fmt.Sprintf("must be less than or equal to %d", maxFailedIndexesForIndexedJob)))
			}
			if spec.Completions != nil && *spec.Completions > completionsSoftLimit && spec.BackoffLimitPerIndex != nil {
				if spec.MaxFailedIndexes == nil {
					allErrs = append(allErrs, field.Required(fldPath.Child("maxFailedIndexes"), fmt.Sprintf("must be specified when completions is above %d", completionsSoftLimit)))
				}
				if spec.Parallelism != nil && *spec.Parallelism > parallelismLimitForHighCompletions {
					allErrs = append(allErrs, field.Invalid(fldPath.Child("parallelism"), *spec.Parallelism, fmt.Sprintf("must be less than or equal to %d when completions are above %d and used with backoff limit per index", parallelismLimitForHighCompletions, completionsSoftLimit)))
				}
				if spec.MaxFailedIndexes != nil && *spec.MaxFailedIndexes > maxFailedIndexesLimitForHighCompletions {
					allErrs = append(allErrs, field.Invalid(fldPath.Child("maxFailedIndexes"), *spec.MaxFailedIndexes, fmt.Sprintf("must be less than or equal to %d when completions are above %d and used with backoff limit per index", maxFailedIndexesLimitForHighCompletions, completionsSoftLimit)))
				}
			}
		}
	}
	if spec.CompletionMode == nil || *spec.CompletionMode == batch.NonIndexedCompletion {
		if spec.BackoffLimitPerIndex != nil {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("backoffLimitPerIndex"), *spec.BackoffLimitPerIndex, "requires indexed completion mode"))
		}
		if spec.MaxFailedIndexes != nil {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("maxFailedIndexes"), *spec.MaxFailedIndexes, "requires indexed completion mode"))
		}
	}

	if spec.PodFailurePolicy != nil {
		allErrs = append(allErrs, validatePodFailurePolicy(spec, fldPath.Child("podFailurePolicy"))...)
	}
	if spec.SuccessPolicy != nil {
		if ptr.Deref(spec.CompletionMode, batch.NonIndexedCompletion) != batch.IndexedCompletion {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("successPolicy"), *spec.SuccessPolicy, "requires indexed completion mode"))
		} else {
			allErrs = append(allErrs, validateSuccessPolicy(spec, fldPath.Child("successPolicy"))...)
		}
	}

	allErrs = append(allErrs, validatePodReplacementPolicy(spec, fldPath.Child("podReplacementPolicy"))...)

	allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpec(&spec.Template, fldPath.Child("template"), opts)...)

	// spec.Template.Spec.RestartPolicy can be defaulted as RestartPolicyAlways
	// by SetDefaults_PodSpec function when the user does not explicitly specify a value for it,
	// so we check both empty and RestartPolicyAlways cases here
	if spec.Template.Spec.RestartPolicy == api.RestartPolicyAlways || spec.Template.Spec.RestartPolicy == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("template", "spec", "restartPolicy"),
			fmt.Sprintf("valid values: %q, %q", api.RestartPolicyOnFailure, api.RestartPolicyNever)))
	} else if spec.Template.Spec.RestartPolicy != api.RestartPolicyOnFailure && spec.Template.Spec.RestartPolicy != api.RestartPolicyNever {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("template", "spec", "restartPolicy"),
			spec.Template.Spec.RestartPolicy, []api.RestartPolicy{api.RestartPolicyOnFailure, api.RestartPolicyNever}))
	} else if spec.PodFailurePolicy != nil && spec.Template.Spec.RestartPolicy != api.RestartPolicyNever {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("template", "spec", "restartPolicy"),
			spec.Template.Spec.RestartPolicy, fmt.Sprintf("only %q is supported when podFailurePolicy is specified", api.RestartPolicyNever)))
	}
	return allErrs
}

func validatePodFailurePolicy(spec *batch.JobSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	rulesPath := fldPath.Child("rules")
	if len(spec.PodFailurePolicy.Rules) > maxPodFailurePolicyRules {
		allErrs = append(allErrs, field.TooMany(rulesPath, len(spec.PodFailurePolicy.Rules), maxPodFailurePolicyRules))
	}
	containerNames := sets.NewString()
	for _, containerSpec := range spec.Template.Spec.Containers {
		containerNames.Insert(containerSpec.Name)
	}
	for _, containerSpec := range spec.Template.Spec.InitContainers {
		containerNames.Insert(containerSpec.Name)
	}
	for i, rule := range spec.PodFailurePolicy.Rules {
		allErrs = append(allErrs, validatePodFailurePolicyRule(spec, &rule, rulesPath.Index(i), containerNames)...)
	}
	return allErrs
}

func validatePodReplacementPolicy(spec *batch.JobSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if spec.PodReplacementPolicy != nil {
		// If PodFailurePolicy is specified then we only allow Failed.
		if spec.PodFailurePolicy != nil {
			if *spec.PodReplacementPolicy != batch.Failed {
				allErrs = append(allErrs, field.NotSupported(fldPath, *spec.PodReplacementPolicy, []batch.PodReplacementPolicy{batch.Failed}))
			}
			// If PodFailurePolicy not specified we allow values in supportedPodReplacementPolicy.
		} else if !supportedPodReplacementPolicy.Has(*spec.PodReplacementPolicy) {
			allErrs = append(allErrs, field.NotSupported(fldPath, *spec.PodReplacementPolicy, sets.List(supportedPodReplacementPolicy)))
		}
	}
	return allErrs
}

func validatePodFailurePolicyRule(spec *batch.JobSpec, rule *batch.PodFailurePolicyRule, rulePath *field.Path, containerNames sets.String) field.ErrorList {
	var allErrs field.ErrorList
	actionPath := rulePath.Child("action")
	if rule.Action == "" {
		allErrs = append(allErrs, field.Required(actionPath, fmt.Sprintf("valid values: %q", sets.List(supportedPodFailurePolicyActions))))
	} else if rule.Action == batch.PodFailurePolicyActionFailIndex {
		if spec.BackoffLimitPerIndex == nil {
			allErrs = append(allErrs, field.Invalid(actionPath, rule.Action, "requires the backoffLimitPerIndex to be set"))
		}
	} else if !supportedPodFailurePolicyActions.Has(rule.Action) {
		allErrs = append(allErrs, field.NotSupported(actionPath, rule.Action, sets.List(supportedPodFailurePolicyActions)))
	}
	if rule.OnExitCodes != nil {
		allErrs = append(allErrs, validatePodFailurePolicyRuleOnExitCodes(rule.OnExitCodes, rulePath.Child("onExitCodes"), containerNames)...)
	}
	if len(rule.OnPodConditions) > 0 {
		allErrs = append(allErrs, validatePodFailurePolicyRuleOnPodConditions(rule.OnPodConditions, rulePath.Child("onPodConditions"))...)
	}
	if rule.OnExitCodes != nil && len(rule.OnPodConditions) > 0 {
		allErrs = append(allErrs, field.Invalid(rulePath, field.OmitValueType{}, "specifying both OnExitCodes and OnPodConditions is not supported"))
	}
	if rule.OnExitCodes == nil && len(rule.OnPodConditions) == 0 {
		allErrs = append(allErrs, field.Invalid(rulePath, field.OmitValueType{}, "specifying one of OnExitCodes and OnPodConditions is required"))
	}
	return allErrs
}

func validatePodFailurePolicyRuleOnPodConditions(onPodConditions []batch.PodFailurePolicyOnPodConditionsPattern, onPodConditionsPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if len(onPodConditions) > maxPodFailurePolicyOnPodConditionsPatterns {
		allErrs = append(allErrs, field.TooMany(onPodConditionsPath, len(onPodConditions), maxPodFailurePolicyOnPodConditionsPatterns))
	}
	for j, pattern := range onPodConditions {
		patternPath := onPodConditionsPath.Index(j)
		statusPath := patternPath.Child("status")
		allErrs = append(allErrs, apivalidation.ValidateQualifiedName(string(pattern.Type), patternPath.Child("type"))...)
		if pattern.Status == "" {
			allErrs = append(allErrs, field.Required(statusPath, fmt.Sprintf("valid values: %q", sets.List(supportedPodFailurePolicyOnPodConditionsStatus))))
		} else if !supportedPodFailurePolicyOnPodConditionsStatus.Has(pattern.Status) {
			allErrs = append(allErrs, field.NotSupported(statusPath, pattern.Status, sets.List(supportedPodFailurePolicyOnPodConditionsStatus)))
		}
	}
	return allErrs
}

func validatePodFailurePolicyRuleOnExitCodes(onExitCode *batch.PodFailurePolicyOnExitCodesRequirement, onExitCodesPath *field.Path, containerNames sets.String) field.ErrorList {
	var allErrs field.ErrorList
	operatorPath := onExitCodesPath.Child("operator")
	if onExitCode.Operator == "" {
		allErrs = append(allErrs, field.Required(operatorPath, fmt.Sprintf("valid values: %q", sets.List(supportedPodFailurePolicyOnExitCodesOperator))))
	} else if !supportedPodFailurePolicyOnExitCodesOperator.Has(onExitCode.Operator) {
		allErrs = append(allErrs, field.NotSupported(operatorPath, onExitCode.Operator, sets.List(supportedPodFailurePolicyOnExitCodesOperator)))
	}
	if onExitCode.ContainerName != nil && !containerNames.Has(*onExitCode.ContainerName) {
		allErrs = append(allErrs, field.Invalid(onExitCodesPath.Child("containerName"), *onExitCode.ContainerName, "must be one of the container or initContainer names in the pod template"))
	}
	valuesPath := onExitCodesPath.Child("values")
	if len(onExitCode.Values) == 0 {
		allErrs = append(allErrs, field.Invalid(valuesPath, onExitCode.Values, "at least one value is required"))
	} else if len(onExitCode.Values) > maxPodFailurePolicyOnExitCodesValues {
		allErrs = append(allErrs, field.TooMany(valuesPath, len(onExitCode.Values), maxPodFailurePolicyOnExitCodesValues))
	}
	isOrdered := true
	uniqueValues := sets.NewInt32()
	for j, exitCodeValue := range onExitCode.Values {
		valuePath := valuesPath.Index(j)
		if onExitCode.Operator == batch.PodFailurePolicyOnExitCodesOpIn && exitCodeValue == 0 {
			allErrs = append(allErrs, field.Invalid(valuePath, exitCodeValue, "must not be 0 for the In operator"))
		}
		if uniqueValues.Has(exitCodeValue) {
			allErrs = append(allErrs, field.Duplicate(valuePath, exitCodeValue))
		} else {
			uniqueValues.Insert(exitCodeValue)
		}
		if j > 0 && onExitCode.Values[j-1] > exitCodeValue {
			isOrdered = false
		}
	}
	if !isOrdered {
		allErrs = append(allErrs, field.Invalid(valuesPath, onExitCode.Values, "must be ordered"))
	}

	return allErrs
}

func validateSuccessPolicy(spec *batch.JobSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	rulesPath := fldPath.Child("rules")
	if len(spec.SuccessPolicy.Rules) == 0 {
		allErrs = append(allErrs, field.Required(rulesPath, "at least one rules must be specified when the successPolicy is specified"))
	}
	if len(spec.SuccessPolicy.Rules) > maxSuccessPolicyRule {
		allErrs = append(allErrs, field.TooMany(rulesPath, len(spec.SuccessPolicy.Rules), maxSuccessPolicyRule))
	}
	for i, rule := range spec.SuccessPolicy.Rules {
		allErrs = append(allErrs, validateSuccessPolicyRule(spec, &rule, rulesPath.Index(i))...)
	}
	return allErrs
}

func validateSuccessPolicyRule(spec *batch.JobSpec, rule *batch.SuccessPolicyRule, rulePath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if rule.SucceededCount == nil && rule.SucceededIndexes == nil {
		allErrs = append(allErrs, field.Required(rulePath, "at least one of succeededCount or succeededIndexes must be specified"))
	}
	var totalIndexes int32
	if rule.SucceededIndexes != nil {
		succeededIndexes := rulePath.Child("succeededIndexes")
		if len(*rule.SucceededIndexes) > maxJobSuccessPolicySucceededIndexesLimit {
			allErrs = append(allErrs, field.TooLong(succeededIndexes, *rule.SucceededIndexes, maxJobSuccessPolicySucceededIndexesLimit))
		}
		var err error
		if totalIndexes, err = validateIndexesFormat(*rule.SucceededIndexes, *spec.Completions); err != nil {
			allErrs = append(allErrs, field.Invalid(succeededIndexes, *rule.SucceededIndexes, fmt.Sprintf("error parsing succeededIndexes: %s", err.Error())))
		}
	}
	if rule.SucceededCount != nil {
		succeededCountPath := rulePath.Child("succeededCount")
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*rule.SucceededCount), succeededCountPath)...)
		if *rule.SucceededCount > *spec.Completions {
			allErrs = append(allErrs, field.Invalid(succeededCountPath, *rule.SucceededCount, fmt.Sprintf("must be less than or equal to %d (the number of specified completions)", *spec.Completions)))
		}
		if rule.SucceededIndexes != nil && *rule.SucceededCount > totalIndexes {
			allErrs = append(allErrs, field.Invalid(succeededCountPath, *rule.SucceededCount, fmt.Sprintf("must be less than or equal to %d (the number of indexes in the specified succeededIndexes field)", totalIndexes)))
		}
	}
	return allErrs
}

// validateJobStatus validates a JobStatus and returns an ErrorList with any errors.
func validateJobStatus(job *batch.Job, fldPath *field.Path, opts JobStatusValidationOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	status := job.Status
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.Active), fldPath.Child("active"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.Succeeded), fldPath.Child("succeeded"))...)
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(status.Failed), fldPath.Child("failed"))...)
	if status.Ready != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*status.Ready), fldPath.Child("ready"))...)
	}
	if status.Terminating != nil {
		allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(*status.Terminating), fldPath.Child("terminating"))...)
	}
	if status.UncountedTerminatedPods != nil {
		path := fldPath.Child("uncountedTerminatedPods")
		seen := sets.New[types.UID]()
		for i, k := range status.UncountedTerminatedPods.Succeeded {
			p := path.Child("succeeded").Index(i)
			if k == "" {
				allErrs = append(allErrs, field.Invalid(p, k, "must not be empty"))
			} else if seen.Has(k) {
				allErrs = append(allErrs, field.Duplicate(p, k))
			} else {
				seen.Insert(k)
			}
		}
		for i, k := range status.UncountedTerminatedPods.Failed {
			p := path.Child("failed").Index(i)
			if k == "" {
				allErrs = append(allErrs, field.Invalid(p, k, "must not be empty"))
			} else if seen.Has(k) {
				allErrs = append(allErrs, field.Duplicate(p, k))
			} else {
				seen.Insert(k)
			}
		}
	}
	if opts.RejectCompleteJobWithFailedCondition {
		if IsJobComplete(job) && IsJobFailed(job) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), field.OmitValueType{}, "cannot set Complete=True and Failed=true conditions"))
		}
	}
	if opts.RejectCompleteJobWithFailureTargetCondition {
		if IsJobComplete(job) && IsConditionTrue(status.Conditions, batch.JobFailureTarget) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), field.OmitValueType{}, "cannot set Complete=True and FailureTarget=true conditions"))
		}
	}
	if opts.RejectNotCompleteJobWithCompletionTime {
		if status.CompletionTime != nil && !IsJobComplete(job) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("completionTime"), status.CompletionTime, "cannot set completionTime when there is no Complete=True condition"))
		}
	}
	if opts.RejectCompleteJobWithoutCompletionTime {
		if status.CompletionTime == nil && IsJobComplete(job) {
			allErrs = append(allErrs, field.Required(fldPath.Child("completionTime"), "completionTime is required for Complete jobs"))
		}
	}
	if opts.RejectCompletionTimeBeforeStartTime {
		if status.StartTime != nil && status.CompletionTime != nil && status.CompletionTime.Before(status.StartTime) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("completionTime"), status.CompletionTime, "completionTime cannot be set before startTime"))
		}
	}
	if opts.RejectFailedJobWithoutFailureTarget {
		if IsJobFailed(job) && !isJobFailureTarget(job) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), field.OmitValueType{}, "cannot set Failed=True condition without the FailureTarget=true condition"))
		}
	}
	if opts.RejectCompleteJobWithoutSuccessCriteriaMet {
		if IsJobComplete(job) && !isJobSuccessCriteriaMet(job) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), field.OmitValueType{}, "cannot set Complete=True condition without the SuccessCriteriaMet=true condition"))
		}
	}
	isJobFinished := IsJobFinished(job)
	if opts.RejectFinishedJobWithActivePods {
		if status.Active > 0 && isJobFinished {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("active"), status.Active, "active>0 is invalid for finished job"))
		}
	}
	if opts.RejectFinishedJobWithoutStartTime {
		if status.StartTime == nil && isJobFinished {
			allErrs = append(allErrs, field.Required(fldPath.Child("startTime"), "startTime is required for finished job"))
		}
	}
	if opts.RejectFinishedJobWithUncountedTerminatedPods {
		if isJobFinished && status.UncountedTerminatedPods != nil && len(status.UncountedTerminatedPods.Failed)+len(status.UncountedTerminatedPods.Succeeded) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("uncountedTerminatedPods"), status.UncountedTerminatedPods, "uncountedTerminatedPods needs to be empty for finished job"))
		}
	}
	if opts.RejectInvalidCompletedIndexes {
		if job.Spec.Completions != nil {
			if _, err := validateIndexesFormat(status.CompletedIndexes, int32(*job.Spec.Completions)); err != nil {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("completedIndexes"), status.CompletedIndexes, fmt.Sprintf("error parsing completedIndexes: %s", err.Error())))
			}
		}
	}
	if opts.RejectInvalidFailedIndexes {
		if job.Spec.Completions != nil && job.Spec.BackoffLimitPerIndex != nil && status.FailedIndexes != nil {
			if _, err := validateIndexesFormat(*status.FailedIndexes, int32(*job.Spec.Completions)); err != nil {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("failedIndexes"), status.FailedIndexes, fmt.Sprintf("error parsing failedIndexes: %s", err.Error())))
			}
		}
	}
	isIndexed := ptr.Deref(job.Spec.CompletionMode, batch.NonIndexedCompletion) == batch.IndexedCompletion
	if opts.RejectCompletedIndexesForNonIndexedJob {
		if len(status.CompletedIndexes) != 0 && !isIndexed {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("completedIndexes"), status.CompletedIndexes, "cannot set non-empty completedIndexes when non-indexed completion mode"))
		}
	}
	if opts.RejectFailedIndexesForNoBackoffLimitPerIndex {
		// Note that this check also verifies that FailedIndexes are not used for
		// regular (non-indexed) jobs, because regular jobs have backoffLimitPerIndex = nil.
		if job.Spec.BackoffLimitPerIndex == nil && status.FailedIndexes != nil {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("failedIndexes"), *status.FailedIndexes, "cannot set non-null failedIndexes when backoffLimitPerIndex is null"))
		}
	}
	if opts.RejectFailedIndexesOverlappingCompleted {
		if job.Spec.Completions != nil && status.FailedIndexes != nil {
			if err := validateFailedIndexesNotOverlapCompleted(status.CompletedIndexes, *status.FailedIndexes, int32(*job.Spec.Completions)); err != nil {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("failedIndexes"), *status.FailedIndexes, err.Error()))
			}
		}
	}
	if opts.RejectFinishedJobWithTerminatingPods {
		if status.Terminating != nil && *status.Terminating > 0 && isJobFinished {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("terminating"), status.Terminating, "terminating>0 is invalid for finished job"))
		}
	}
	if opts.RejectMoreReadyThanActivePods {
		if status.Ready != nil && *status.Ready > status.Active {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("ready"), *status.Ready, "cannot set more ready pods than active"))
		}
	}
	if !opts.AllowForSuccessCriteriaMetInExtendedScope && ptr.Deref(job.Spec.CompletionMode, batch.NonIndexedCompletion) != batch.IndexedCompletion && isJobSuccessCriteriaMet(job) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), field.OmitValueType{}, "cannot set SuccessCriteriaMet to NonIndexed Job"))
	}
	if isJobSuccessCriteriaMet(job) && IsJobFailed(job) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), field.OmitValueType{}, "cannot set SuccessCriteriaMet=True and Failed=true conditions"))
	}
	if isJobSuccessCriteriaMet(job) && isJobFailureTarget(job) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), field.OmitValueType{}, "cannot set SuccessCriteriaMet=True and FailureTarget=true conditions"))
	}
	if !opts.AllowForSuccessCriteriaMetInExtendedScope && job.Spec.SuccessPolicy == nil && isJobSuccessCriteriaMet(job) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), field.OmitValueType{}, "cannot set SuccessCriteriaMet=True for Job without SuccessPolicy"))
	}
	if job.Spec.SuccessPolicy != nil && !isJobSuccessCriteriaMet(job) && IsJobComplete(job) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("conditions"), field.OmitValueType{}, "cannot set Complete=True for Job with SuccessPolicy unless SuccessCriteriaMet=True"))
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
func ValidateJobUpdateStatus(job, oldJob *batch.Job, opts JobStatusValidationOptions) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&job.ObjectMeta, &oldJob.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateJobStatusUpdate(job, oldJob, opts)...)
	return allErrs
}

// ValidateJobSpecUpdate validates an update to a JobSpec and returns an ErrorList with any errors.
func ValidateJobSpecUpdate(spec, oldSpec batch.JobSpec, fldPath *field.Path, opts JobValidationOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateJobSpec(&spec, fldPath, opts.PodValidationOptions)...)
	allErrs = append(allErrs, validateCompletions(spec, oldSpec, fldPath.Child("completions"), opts)...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Selector, oldSpec.Selector, fldPath.Child("selector"))...)
	allErrs = append(allErrs, validatePodTemplateUpdate(spec, oldSpec, fldPath, opts)...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.CompletionMode, oldSpec.CompletionMode, fldPath.Child("completionMode"))...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.PodFailurePolicy, oldSpec.PodFailurePolicy, fldPath.Child("podFailurePolicy"))...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.BackoffLimitPerIndex, oldSpec.BackoffLimitPerIndex, fldPath.Child("backoffLimitPerIndex"))...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.ManagedBy, oldSpec.ManagedBy, fldPath.Child("managedBy"))...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.SuccessPolicy, oldSpec.SuccessPolicy, fldPath.Child("successPolicy"))...)
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
		oldTemplate.Spec.NodeSelector = template.Spec.NodeSelector       // +k8s:verify-mutation:reason=clone
		oldTemplate.Spec.Tolerations = template.Spec.Tolerations         // +k8s:verify-mutation:reason=clone
		oldTemplate.Annotations = template.Annotations                   // +k8s:verify-mutation:reason=clone
		oldTemplate.Labels = template.Labels                             // +k8s:verify-mutation:reason=clone
		oldTemplate.Spec.SchedulingGates = template.Spec.SchedulingGates // +k8s:verify-mutation:reason=clone
	}
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(template, oldTemplate, fldPath.Child("template"))...)
	return allErrs
}

// ValidateJobStatusUpdate validates an update to a JobStatus and returns an ErrorList with any errors.
func ValidateJobStatusUpdate(job, oldJob *batch.Job, opts JobStatusValidationOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	statusFld := field.NewPath("status")
	allErrs = append(allErrs, validateJobStatus(job, statusFld, opts)...)

	if opts.RejectDisablingTerminalCondition {
		for _, cType := range []batch.JobConditionType{batch.JobFailed, batch.JobComplete, batch.JobFailureTarget} {
			if IsConditionTrue(oldJob.Status.Conditions, cType) && !IsConditionTrue(job.Status.Conditions, cType) {
				allErrs = append(allErrs, field.Invalid(statusFld.Child("conditions"), field.OmitValueType{}, fmt.Sprintf("cannot disable the terminal %s=True condition", string(cType))))
			}
		}
	}
	if opts.RejectDecreasingFailedCounter {
		if job.Status.Failed < oldJob.Status.Failed {
			allErrs = append(allErrs, field.Invalid(statusFld.Child("failed"), job.Status.Failed, "cannot decrease the failed counter"))
		}
	}
	if opts.RejectDecreasingSucceededCounter {
		if job.Status.Succeeded < oldJob.Status.Succeeded {
			allErrs = append(allErrs, field.Invalid(statusFld.Child("succeeded"), job.Status.Succeeded, "cannot decrease the succeeded counter"))
		}
	}
	if opts.RejectMutatingCompletionTime {
		// Note that we check the condition only when `job.Status.CompletionTime != nil`, this is because
		// we don't want to block transitions to completionTime = nil when the job is not finished yet.
		// Setting completionTime = nil for finished jobs is prevented in RejectCompleteJobWithoutCompletionTime.
		if job.Status.CompletionTime != nil && oldJob.Status.CompletionTime != nil && !ptr.Equal(job.Status.CompletionTime, oldJob.Status.CompletionTime) {
			allErrs = append(allErrs, field.Invalid(statusFld.Child("completionTime"), job.Status.CompletionTime, "completionTime cannot be mutated"))
		}
	}
	if opts.RejectStartTimeUpdateForUnsuspendedJob {
		// Note that we check `oldJob.Status.StartTime != nil` to allow transitioning from
		// startTime = nil to startTime != nil for unsuspended jobs, which is a desired transition.
		if oldJob.Status.StartTime != nil && !ptr.Equal(oldJob.Status.StartTime, job.Status.StartTime) && !ptr.Deref(job.Spec.Suspend, false) {
			allErrs = append(allErrs, field.Required(statusFld.Child("startTime"), "startTime cannot be removed for unsuspended job"))
		}
	}
	if isJobSuccessCriteriaMet(oldJob) && !isJobSuccessCriteriaMet(job) {
		allErrs = append(allErrs, field.Invalid(statusFld.Child("conditions"), field.OmitValueType{}, "cannot disable the SuccessCriteriaMet=True condition"))
	}
	if IsJobComplete(oldJob) && !isJobSuccessCriteriaMet(oldJob) && isJobSuccessCriteriaMet(job) {
		allErrs = append(allErrs, field.Invalid(statusFld.Child("conditions"), field.OmitValueType{}, "cannot set SuccessCriteriaMet=True for Job already has Complete=true conditions"))
	}
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
		allowTZInSchedule := false
		if oldSpec != nil {
			allowTZInSchedule = strings.Contains(oldSpec.Schedule, "TZ")
		}
		allErrs = append(allErrs, validateScheduleFormat(spec.Schedule, allowTZInSchedule, spec.TimeZone, fldPath.Child("schedule"))...)
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
		validValues := []batch.ConcurrencyPolicy{batch.AllowConcurrent, batch.ForbidConcurrent, batch.ReplaceConcurrent}
		allErrs = append(allErrs, field.NotSupported(fldPath, *concurrencyPolicy, validValues))
	}

	return allErrs
}

func validateScheduleFormat(schedule string, allowTZInSchedule bool, timeZone *string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if _, err := cron.ParseStandard(schedule); err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, schedule, err.Error()))
	}
	switch {
	case allowTZInSchedule && strings.Contains(schedule, "TZ") && timeZone != nil:
		allErrs = append(allErrs, field.Invalid(fldPath, schedule, "cannot use both timeZone field and TZ or CRON_TZ in schedule"))
	case !allowTZInSchedule && strings.Contains(schedule, "TZ"):
		allErrs = append(allErrs, field.Invalid(fldPath, schedule, "cannot use TZ or CRON_TZ in schedule, use timeZone field instead"))
	}

	return allErrs
}

// https://data.iana.org/time-zones/theory.html#naming
// * A name must not be empty, or contain '//', or start or end with '/'.
// * Do not use the file name components '.' and '..'.
// * Within a file name component, use only ASCII letters, '.', '-' and '_'.
// * Do not use digits, as that might create an ambiguity with POSIX TZ strings.
// * A file name component must not exceed 14 characters or start with '-'
//
// 0-9 and + characters are tolerated to accommodate legacy compatibility names
var validTimeZoneCharacters = regexp.MustCompile(`^[A-Za-z\.\-_0-9+]{1,14}$`)

func validateTimeZone(timeZone *string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if timeZone == nil {
		return allErrs
	}

	if len(*timeZone) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, timeZone, "timeZone must be nil or non-empty string"))
		return allErrs
	}

	for _, part := range strings.Split(*timeZone, "/") {
		if part == "." || part == ".." || strings.HasPrefix(part, "-") || !validTimeZoneCharacters.MatchString(part) {
			allErrs = append(allErrs, field.Invalid(fldPath, timeZone, fmt.Sprintf("unknown time zone %s", *timeZone)))
			return allErrs
		}
	}

	if strings.EqualFold(*timeZone, "Local") {
		allErrs = append(allErrs, field.Invalid(fldPath, timeZone, "timeZone must be an explicit time zone as defined in https://www.iana.org/time-zones"))
	}

	if _, err := time.LoadLocation(*timeZone); err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, timeZone, err.Error()))
	}

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

func validateCompletions(spec, oldSpec batch.JobSpec, fldPath *field.Path, opts JobValidationOptions) field.ErrorList {
	if !opts.AllowElasticIndexedJobs {
		return apivalidation.ValidateImmutableField(spec.Completions, oldSpec.Completions, fldPath)
	}

	// Completions is immutable for non-indexed jobs.
	// For Indexed Jobs, if ElasticIndexedJob feature gate is not enabled,
	// fall back to validating that spec.Completions is always immutable.
	isIndexedJob := spec.CompletionMode != nil && *spec.CompletionMode == batch.IndexedCompletion
	if !isIndexedJob {
		return apivalidation.ValidateImmutableField(spec.Completions, oldSpec.Completions, fldPath)
	}

	var allErrs field.ErrorList
	if apiequality.Semantic.DeepEqual(spec.Completions, oldSpec.Completions) {
		return allErrs
	}
	// Indexed Jobs cannot set completions to nil. The nil check
	// is already performed in validateJobSpec, no need to add another error.
	if spec.Completions == nil {
		return allErrs
	}

	if *spec.Completions != *spec.Parallelism {
		allErrs = append(allErrs, field.Invalid(fldPath, spec.Completions, fmt.Sprintf("can only be modified in tandem with %s", fldPath.Root().Child("parallelism").String())))
	}
	return allErrs
}

func IsJobFinished(job *batch.Job) bool {
	for _, c := range job.Status.Conditions {
		if (c.Type == batch.JobComplete || c.Type == batch.JobFailed) && c.Status == api.ConditionTrue {
			return true
		}
	}
	return false
}

func IsJobComplete(job *batch.Job) bool {
	return IsConditionTrue(job.Status.Conditions, batch.JobComplete)
}

func IsJobFailed(job *batch.Job) bool {
	return IsConditionTrue(job.Status.Conditions, batch.JobFailed)
}

func isJobSuccessCriteriaMet(job *batch.Job) bool {
	return IsConditionTrue(job.Status.Conditions, batch.JobSuccessCriteriaMet)
}

func isJobFailureTarget(job *batch.Job) bool {
	return IsConditionTrue(job.Status.Conditions, batch.JobFailureTarget)
}

func IsConditionTrue(list []batch.JobCondition, cType batch.JobConditionType) bool {
	for _, c := range list {
		if c.Type == cType && c.Status == api.ConditionTrue {
			return true
		}
	}
	return false
}

func validateFailedIndexesNotOverlapCompleted(completedIndexesStr string, failedIndexesStr string, completions int32) error {
	if len(completedIndexesStr) == 0 || len(failedIndexesStr) == 0 {
		return nil
	}
	completedIndexesIntervals := strings.Split(completedIndexesStr, ",")
	failedIndexesIntervals := strings.Split(failedIndexesStr, ",")
	var completedPos, failedPos int
	cX, cY, cErr := parseIndexInterval(completedIndexesIntervals[completedPos], completions)
	fX, fY, fErr := parseIndexInterval(failedIndexesIntervals[failedPos], completions)
	for completedPos < len(completedIndexesIntervals) && failedPos < len(failedIndexesIntervals) {
		if cErr != nil {
			// Failure to parse "completed" interval. We go to the next interval,
			// the error will be reported to the user when validating the format.
			completedPos++
			if completedPos < len(completedIndexesIntervals) {
				cX, cY, cErr = parseIndexInterval(completedIndexesIntervals[completedPos], completions)
			}
		} else if fErr != nil {
			// Failure to parse "failed" interval. We go to the next interval,
			// the error will be reported to the user when validating the format.
			failedPos++
			if failedPos < len(failedIndexesIntervals) {
				fX, fY, fErr = parseIndexInterval(failedIndexesIntervals[failedPos], completions)
			}
		} else {
			// We have one failed and one completed interval parsed.
			if cX <= fY && fX <= cY {
				return fmt.Errorf("failedIndexes and completedIndexes overlap at index: %d", max(cX, fX))
			}
			// No overlap, let's move to the next one.
			if cX <= fX {
				completedPos++
				if completedPos < len(completedIndexesIntervals) {
					cX, cY, cErr = parseIndexInterval(completedIndexesIntervals[completedPos], completions)
				}
			} else {
				failedPos++
				if failedPos < len(failedIndexesIntervals) {
					fX, fY, fErr = parseIndexInterval(failedIndexesIntervals[failedPos], completions)
				}
			}
		}
	}
	return nil
}

func validateIndexesFormat(indexesStr string, completions int32) (int32, error) {
	if len(indexesStr) == 0 {
		return 0, nil
	}
	var lastIndex *int32
	var total int32
	for _, intervalStr := range strings.Split(indexesStr, ",") {
		x, y, err := parseIndexInterval(intervalStr, completions)
		if err != nil {
			return 0, err
		}
		if lastIndex != nil && *lastIndex >= x {
			return 0, fmt.Errorf("non-increasing order, previous: %d, current: %d", *lastIndex, x)
		}
		total += y - x + 1
		lastIndex = &y
	}
	return total, nil
}

func parseIndexInterval(intervalStr string, completions int32) (int32, int32, error) {
	limitsStr := strings.Split(intervalStr, "-")
	if len(limitsStr) > 2 {
		return 0, 0, fmt.Errorf("the fragment %q violates the requirement that an index interval can have at most two parts separated by '-'", intervalStr)
	}
	x, err := strconv.Atoi(limitsStr[0])
	if err != nil {
		return 0, 0, fmt.Errorf("cannot convert string to integer for index: %q", limitsStr[0])
	}
	if x >= int(completions) {
		return 0, 0, fmt.Errorf("too large index: %q", limitsStr[0])
	}
	if len(limitsStr) > 1 {
		y, err := strconv.Atoi(limitsStr[1])
		if err != nil {
			return 0, 0, fmt.Errorf("cannot convert string to integer for index: %q", limitsStr[1])
		}
		if y >= int(completions) {
			return 0, 0, fmt.Errorf("too large index: %q", limitsStr[1])
		}
		if x >= y {
			return 0, 0, fmt.Errorf("non-increasing order, previous: %d, current: %d", x, y)
		}
		return int32(x), int32(y), nil
	}
	return int32(x), int32(x), nil
}

type JobValidationOptions struct {
	apivalidation.PodValidationOptions
	// Allow mutable node affinity, selector and tolerations of the template
	AllowMutableSchedulingDirectives bool
	// Allow elastic indexed jobs
	AllowElasticIndexedJobs bool
	// Require Job to have the label on batch.kubernetes.io/job-name and batch.kubernetes.io/controller-uid
	RequirePrefixedLabels bool
}

type JobStatusValidationOptions struct {
	RejectDecreasingSucceededCounter             bool
	RejectDecreasingFailedCounter                bool
	RejectDisablingTerminalCondition             bool
	RejectInvalidCompletedIndexes                bool
	RejectInvalidFailedIndexes                   bool
	RejectFailedIndexesOverlappingCompleted      bool
	RejectCompletedIndexesForNonIndexedJob       bool
	RejectFailedIndexesForNoBackoffLimitPerIndex bool
	RejectFailedJobWithoutFailureTarget          bool
	RejectCompleteJobWithoutSuccessCriteriaMet   bool
	RejectFinishedJobWithActivePods              bool
	RejectFinishedJobWithoutStartTime            bool
	RejectFinishedJobWithUncountedTerminatedPods bool
	RejectStartTimeUpdateForUnsuspendedJob       bool
	RejectCompletionTimeBeforeStartTime          bool
	RejectMutatingCompletionTime                 bool
	RejectCompleteJobWithoutCompletionTime       bool
	RejectNotCompleteJobWithCompletionTime       bool
	RejectCompleteJobWithFailedCondition         bool
	RejectCompleteJobWithFailureTargetCondition  bool
	AllowForSuccessCriteriaMetInExtendedScope    bool
	RejectMoreReadyThanActivePods                bool
	RejectFinishedJobWithTerminatingPods         bool
}
