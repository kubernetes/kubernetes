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

package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

const (
	// All Kubernetes labels need to be prefixed with Kubernetes to distinguish them from end-user labels
	// More info: https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#label-selector-and-annotation-conventions
	labelPrefix = "batch.kubernetes.io/"

	// CronJobScheduledTimestampAnnotation is the scheduled timestamp annotation for the Job.
	// It records the original/expected scheduled timestamp for the running job, represented in RFC3339.
	// The CronJob controller adds this annotation if the CronJobsScheduledAnnotation feature gate (beta in 1.28) is enabled.
	CronJobScheduledTimestampAnnotation = labelPrefix + "cronjob-scheduled-timestamp"

	JobCompletionIndexAnnotation = labelPrefix + "job-completion-index"
	// JobTrackingFinalizer is a finalizer for Job's pods. It prevents them from
	// being deleted before being accounted in the Job status.
	//
	// Additionally, the apiserver and job controller use this string as a Job
	// annotation, to mark Jobs that are being tracked using pod finalizers.
	// However, this behavior is deprecated in kubernetes 1.26. This means that, in
	// 1.27+, one release after JobTrackingWithFinalizers graduates to GA, the
	// apiserver and job controller will ignore this annotation and they will
	// always track jobs using finalizers.
	JobTrackingFinalizer = labelPrefix + "job-tracking"
	// The Job labels will use batch.kubernetes.io as a prefix for all labels
	// Historically the job controller uses unprefixed labels for job-name and controller-uid and
	// Kubernetes continutes to recognize those unprefixed labels for consistency.
	JobNameLabel = labelPrefix + "job-name"
	// ControllerUid is used to programatically get pods corresponding to a Job.
	// There is a corresponding label without the batch.kubernetes.io that we support for legacy reasons.
	ControllerUidLabel = labelPrefix + "controller-uid"
	// Annotation indicating the number of failures for the index corresponding
	// to the pod, which are counted towards the backoff limit.
	JobIndexFailureCountAnnotation = labelPrefix + "job-index-failure-count"
	// Annotation indicating the number of failures for the index corresponding
	// to the pod, which don't count towards the backoff limit, according to the
	// pod failure policy. When the annotation is absent zero is implied.
	JobIndexIgnoredFailureCountAnnotation = labelPrefix + "job-index-ignored-failure-count"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Job represents the configuration of a single job.
type Job struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Specification of the desired behavior of a job.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec JobSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// Current status of a job.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status JobStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// JobList is a collection of jobs.
type JobList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of Jobs.
	Items []Job `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// CompletionMode specifies how Pod completions of a Job are tracked.
// +enum
type CompletionMode string

const (
	// NonIndexedCompletion is a Job completion mode. In this mode, the Job is
	// considered complete when there have been .spec.completions
	// successfully completed Pods. Pod completions are homologous to each other.
	NonIndexedCompletion CompletionMode = "NonIndexed"

	// IndexedCompletion is a Job completion mode. In this mode, the Pods of a
	// Job get an associated completion index from 0 to (.spec.completions - 1).
	// The Job is  considered complete when a Pod completes for each completion
	// index.
	IndexedCompletion CompletionMode = "Indexed"
)

// PodFailurePolicyAction specifies how a Pod failure is handled.
// +enum
type PodFailurePolicyAction string

const (
	// This is an action which might be taken on a pod failure - mark the
	// pod's job as Failed and terminate all running pods.
	PodFailurePolicyActionFailJob PodFailurePolicyAction = "FailJob"

	// This is an action which might be taken on a pod failure - mark the
	// Job's index as failed to avoid restarts within this index. This action
	// can only be used when backoffLimitPerIndex is set.
	PodFailurePolicyActionFailIndex PodFailurePolicyAction = "FailIndex"

	// This is an action which might be taken on a pod failure - the counter towards
	// .backoffLimit, represented by the job's .status.failed field, is not
	// incremented and a replacement pod is created.
	PodFailurePolicyActionIgnore PodFailurePolicyAction = "Ignore"

	// This is an action which might be taken on a pod failure - the pod failure
	// is handled in the default way - the counter towards .backoffLimit,
	// represented by the job's .status.failed field, is incremented.
	PodFailurePolicyActionCount PodFailurePolicyAction = "Count"
)

// +enum
type PodFailurePolicyOnExitCodesOperator string

const (
	PodFailurePolicyOnExitCodesOpIn    PodFailurePolicyOnExitCodesOperator = "In"
	PodFailurePolicyOnExitCodesOpNotIn PodFailurePolicyOnExitCodesOperator = "NotIn"
)

// PodReplacementPolicy specifies the policy for creating pod replacements.
// +enum
type PodReplacementPolicy string

const (
	// TerminatingOrFailed means that we recreate pods
	// when they are terminating (has a metadata.deletionTimestamp) or failed.
	TerminatingOrFailed PodReplacementPolicy = "TerminatingOrFailed"
	// Failed means to wait until a previously created Pod is fully terminated (has phase
	// Failed or Succeeded) before creating a replacement Pod.
	Failed PodReplacementPolicy = "Failed"
)

// PodFailurePolicyOnExitCodesRequirement describes the requirement for handling
// a failed pod based on its container exit codes. In particular, it lookups the
// .state.terminated.exitCode for each app container and init container status,
// represented by the .status.containerStatuses and .status.initContainerStatuses
// fields in the Pod status, respectively. Containers completed with success
// (exit code 0) are excluded from the requirement check.
type PodFailurePolicyOnExitCodesRequirement struct {
	// Restricts the check for exit codes to the container with the
	// specified name. When null, the rule applies to all containers.
	// When specified, it should match one the container or initContainer
	// names in the pod template.
	// +optional
	ContainerName *string `json:"containerName" protobuf:"bytes,1,opt,name=containerName"`

	// Represents the relationship between the container exit code(s) and the
	// specified values. Containers completed with success (exit code 0) are
	// excluded from the requirement check. Possible values are:
	//
	// - In: the requirement is satisfied if at least one container exit code
	//   (might be multiple if there are multiple containers not restricted
	//   by the 'containerName' field) is in the set of specified values.
	// - NotIn: the requirement is satisfied if at least one container exit code
	//   (might be multiple if there are multiple containers not restricted
	//   by the 'containerName' field) is not in the set of specified values.
	// Additional values are considered to be added in the future. Clients should
	// react to an unknown operator by assuming the requirement is not satisfied.
	Operator PodFailurePolicyOnExitCodesOperator `json:"operator" protobuf:"bytes,2,req,name=operator"`

	// Specifies the set of values. Each returned container exit code (might be
	// multiple in case of multiple containers) is checked against this set of
	// values with respect to the operator. The list of values must be ordered
	// and must not contain duplicates. Value '0' cannot be used for the In operator.
	// At least one element is required. At most 255 elements are allowed.
	// +listType=set
	Values []int32 `json:"values" protobuf:"varint,3,rep,name=values"`
}

// PodFailurePolicyOnPodConditionsPattern describes a pattern for matching
// an actual pod condition type.
type PodFailurePolicyOnPodConditionsPattern struct {
	// Specifies the required Pod condition type. To match a pod condition
	// it is required that specified type equals the pod condition type.
	Type corev1.PodConditionType `json:"type" protobuf:"bytes,1,req,name=type"`

	// Specifies the required Pod condition status. To match a pod condition
	// it is required that the specified status equals the pod condition status.
	// Defaults to True.
	Status corev1.ConditionStatus `json:"status" protobuf:"bytes,2,req,name=status"`
}

// PodFailurePolicyRule describes how a pod failure is handled when the requirements are met.
// One of onExitCodes and onPodConditions, but not both, can be used in each rule.
type PodFailurePolicyRule struct {
	// Specifies the action taken on a pod failure when the requirements are satisfied.
	// Possible values are:
	//
	// - FailJob: indicates that the pod's job is marked as Failed and all
	//   running pods are terminated.
	// - FailIndex: indicates that the pod's index is marked as Failed and will
	//   not be restarted.
	//   This value is alpha-level. It can be used when the
	//   `JobBackoffLimitPerIndex` feature gate is enabled (disabled by default).
	// - Ignore: indicates that the counter towards the .backoffLimit is not
	//   incremented and a replacement pod is created.
	// - Count: indicates that the pod is handled in the default way - the
	//   counter towards the .backoffLimit is incremented.
	// Additional values are considered to be added in the future. Clients should
	// react to an unknown action by skipping the rule.
	Action PodFailurePolicyAction `json:"action" protobuf:"bytes,1,req,name=action"`

	// Represents the requirement on the container exit codes.
	// +optional
	OnExitCodes *PodFailurePolicyOnExitCodesRequirement `json:"onExitCodes" protobuf:"bytes,2,opt,name=onExitCodes"`

	// Represents the requirement on the pod conditions. The requirement is represented
	// as a list of pod condition patterns. The requirement is satisfied if at
	// least one pattern matches an actual pod condition. At most 20 elements are allowed.
	// +listType=atomic
	// +optional
	OnPodConditions []PodFailurePolicyOnPodConditionsPattern `json:"onPodConditions" protobuf:"bytes,3,opt,name=onPodConditions"`
}

// PodFailurePolicy describes how failed pods influence the backoffLimit.
type PodFailurePolicy struct {
	// A list of pod failure policy rules. The rules are evaluated in order.
	// Once a rule matches a Pod failure, the remaining of the rules are ignored.
	// When no rule matches the Pod failure, the default handling applies - the
	// counter of pod failures is incremented and it is checked against
	// the backoffLimit. At most 20 elements are allowed.
	// +listType=atomic
	Rules []PodFailurePolicyRule `json:"rules" protobuf:"bytes,1,opt,name=rules"`
}

// JobSpec describes how the job execution will look like.
type JobSpec struct {

	// Specifies the maximum desired number of pods the job should
	// run at any given time. The actual number of pods running in steady state will
	// be less than this number when ((.spec.completions - .status.successful) < .spec.parallelism),
	// i.e. when the work left to do is less than max parallelism.
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/
	// +optional
	Parallelism *int32 `json:"parallelism,omitempty" protobuf:"varint,1,opt,name=parallelism"`

	// Specifies the desired number of successfully finished pods the
	// job should be run with.  Setting to null means that the success of any
	// pod signals the success of all pods, and allows parallelism to have any positive
	// value.  Setting to 1 means that parallelism is limited to 1 and the success of that
	// pod signals the success of the job.
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/
	// +optional
	Completions *int32 `json:"completions,omitempty" protobuf:"varint,2,opt,name=completions"`

	// Specifies the duration in seconds relative to the startTime that the job
	// may be continuously active before the system tries to terminate it; value
	// must be positive integer. If a Job is suspended (at creation or through an
	// update), this timer will effectively be stopped and reset when the Job is
	// resumed again.
	// +optional
	ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty" protobuf:"varint,3,opt,name=activeDeadlineSeconds"`

	// Specifies the policy of handling failed pods. In particular, it allows to
	// specify the set of actions and conditions which need to be
	// satisfied to take the associated action.
	// If empty, the default behaviour applies - the counter of failed pods,
	// represented by the jobs's .status.failed field, is incremented and it is
	// checked against the backoffLimit. This field cannot be used in combination
	// with restartPolicy=OnFailure.
	//
	// This field is beta-level. It can be used when the `JobPodFailurePolicy`
	// feature gate is enabled (enabled by default).
	// +optional
	PodFailurePolicy *PodFailurePolicy `json:"podFailurePolicy,omitempty" protobuf:"bytes,11,opt,name=podFailurePolicy"`

	// Specifies the number of retries before marking this job failed.
	// Defaults to 6
	// +optional
	BackoffLimit *int32 `json:"backoffLimit,omitempty" protobuf:"varint,7,opt,name=backoffLimit"`

	// Specifies the limit for the number of retries within an
	// index before marking this index as failed. When enabled the number of
	// failures per index is kept in the pod's
	// batch.kubernetes.io/job-index-failure-count annotation. It can only
	// be set when Job's completionMode=Indexed, and the Pod's restart
	// policy is Never. The field is immutable.
	// This field is alpha-level. It can be used when the `JobBackoffLimitPerIndex`
	// feature gate is enabled (disabled by default).
	// +optional
	BackoffLimitPerIndex *int32 `json:"backoffLimitPerIndex,omitempty" protobuf:"varint,12,opt,name=backoffLimitPerIndex"`

	// Specifies the maximal number of failed indexes before marking the Job as
	// failed, when backoffLimitPerIndex is set. Once the number of failed
	// indexes exceeds this number the entire Job is marked as Failed and its
	// execution is terminated. When left as null the job continues execution of
	// all of its indexes and is marked with the `Complete` Job condition.
	// It can only be specified when backoffLimitPerIndex is set.
	// It can be null or up to completions. It is required and must be
	// less than or equal to 10^4 when is completions greater than 10^5.
	// This field is alpha-level. It can be used when the `JobBackoffLimitPerIndex`
	// feature gate is enabled (disabled by default).
	// +optional
	MaxFailedIndexes *int32 `json:"maxFailedIndexes,omitempty" protobuf:"varint,13,opt,name=maxFailedIndexes"`

	// TODO enabled it when https://github.com/kubernetes/kubernetes/issues/28486 has been fixed
	// Optional number of failed pods to retain.
	// +optional
	// FailedPodsLimit *int32 `json:"failedPodsLimit,omitempty" protobuf:"varint,9,opt,name=failedPodsLimit"`

	// A label query over pods that should match the pod count.
	// Normally, the system sets this field for you.
	// More info: https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#label-selectors
	// +optional
	Selector *metav1.LabelSelector `json:"selector,omitempty" protobuf:"bytes,4,opt,name=selector"`

	// manualSelector controls generation of pod labels and pod selectors.
	// Leave `manualSelector` unset unless you are certain what you are doing.
	// When false or unset, the system pick labels unique to this job
	// and appends those labels to the pod template.  When true,
	// the user is responsible for picking unique labels and specifying
	// the selector.  Failure to pick a unique label may cause this
	// and other jobs to not function correctly.  However, You may see
	// `manualSelector=true` in jobs that were created with the old `extensions/v1beta1`
	// API.
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/#specifying-your-own-pod-selector
	// +optional
	ManualSelector *bool `json:"manualSelector,omitempty" protobuf:"varint,5,opt,name=manualSelector"`

	// Describes the pod that will be created when executing a job.
	// The only allowed template.spec.restartPolicy values are "Never" or "OnFailure".
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/
	Template corev1.PodTemplateSpec `json:"template" protobuf:"bytes,6,opt,name=template"`

	// ttlSecondsAfterFinished limits the lifetime of a Job that has finished
	// execution (either Complete or Failed). If this field is set,
	// ttlSecondsAfterFinished after the Job finishes, it is eligible to be
	// automatically deleted. When the Job is being deleted, its lifecycle
	// guarantees (e.g. finalizers) will be honored. If this field is unset,
	// the Job won't be automatically deleted. If this field is set to zero,
	// the Job becomes eligible to be deleted immediately after it finishes.
	// +optional
	TTLSecondsAfterFinished *int32 `json:"ttlSecondsAfterFinished,omitempty" protobuf:"varint,8,opt,name=ttlSecondsAfterFinished"`

	// completionMode specifies how Pod completions are tracked. It can be
	// `NonIndexed` (default) or `Indexed`.
	//
	// `NonIndexed` means that the Job is considered complete when there have
	// been .spec.completions successfully completed Pods. Each Pod completion is
	// homologous to each other.
	//
	// `Indexed` means that the Pods of a
	// Job get an associated completion index from 0 to (.spec.completions - 1),
	// available in the annotation batch.kubernetes.io/job-completion-index.
	// The Job is considered complete when there is one successfully completed Pod
	// for each index.
	// When value is `Indexed`, .spec.completions must be specified and
	// `.spec.parallelism` must be less than or equal to 10^5.
	// In addition, The Pod name takes the form
	// `$(job-name)-$(index)-$(random-string)`,
	// the Pod hostname takes the form `$(job-name)-$(index)`.
	//
	// More completion modes can be added in the future.
	// If the Job controller observes a mode that it doesn't recognize, which
	// is possible during upgrades due to version skew, the controller
	// skips updates for the Job.
	// +optional
	CompletionMode *CompletionMode `json:"completionMode,omitempty" protobuf:"bytes,9,opt,name=completionMode,casttype=CompletionMode"`

	// suspend specifies whether the Job controller should create Pods or not. If
	// a Job is created with suspend set to true, no Pods are created by the Job
	// controller. If a Job is suspended after creation (i.e. the flag goes from
	// false to true), the Job controller will delete all active Pods associated
	// with this Job. Users must design their workload to gracefully handle this.
	// Suspending a Job will reset the StartTime field of the Job, effectively
	// resetting the ActiveDeadlineSeconds timer too. Defaults to false.
	//
	// +optional
	Suspend *bool `json:"suspend,omitempty" protobuf:"varint,10,opt,name=suspend"`

	// podReplacementPolicy specifies when to create replacement Pods.
	// Possible values are:
	// - TerminatingOrFailed means that we recreate pods
	//   when they are terminating (has a metadata.deletionTimestamp) or failed.
	// - Failed means to wait until a previously created Pod is fully terminated (has phase
	//   Failed or Succeeded) before creating a replacement Pod.
	//
	// When using podFailurePolicy, Failed is the the only allowed value.
	// TerminatingOrFailed and Failed are allowed values when podFailurePolicy is not in use.
	// This is an alpha field. Enable JobPodReplacementPolicy to be able to use this field.
	// +optional
	PodReplacementPolicy *PodReplacementPolicy `json:"podReplacementPolicy,omitempty" protobuf:"bytes,14,opt,name=podReplacementPolicy,casttype=podReplacementPolicy"`
}

// JobStatus represents the current state of a Job.
type JobStatus struct {
	// The latest available observations of an object's current state. When a Job
	// fails, one of the conditions will have type "Failed" and status true. When
	// a Job is suspended, one of the conditions will have type "Suspended" and
	// status true; when the Job is resumed, the status of this condition will
	// become false. When a Job is completed, one of the conditions will have
	// type "Complete" and status true.
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=atomic
	Conditions []JobCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,1,rep,name=conditions"`

	// Represents time when the job controller started processing a job. When a
	// Job is created in the suspended state, this field is not set until the
	// first time it is resumed. This field is reset every time a Job is resumed
	// from suspension. It is represented in RFC3339 form and is in UTC.
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty" protobuf:"bytes,2,opt,name=startTime"`

	// Represents time when the job was completed. It is not guaranteed to
	// be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	// The completion time is only set when the job finishes successfully.
	// +optional
	CompletionTime *metav1.Time `json:"completionTime,omitempty" protobuf:"bytes,3,opt,name=completionTime"`

	// The number of pending and running pods.
	// +optional
	Active int32 `json:"active,omitempty" protobuf:"varint,4,opt,name=active"`

	// The number of pods which reached phase Succeeded.
	// +optional
	Succeeded int32 `json:"succeeded,omitempty" protobuf:"varint,5,opt,name=succeeded"`

	// The number of pods which reached phase Failed.
	// +optional
	Failed int32 `json:"failed,omitempty" protobuf:"varint,6,opt,name=failed"`

	// The number of pods which are terminating (in phase Pending or Running
	// and have a deletionTimestamp).
	//
	// This field is alpha-level. The job controller populates the field when
	// the feature gate JobPodReplacementPolicy is enabled (disabled by default).
	// +optional
	Terminating *int32 `json:"terminating,omitempty" protobuf:"varint,11,opt,name=terminating"`

	// completedIndexes holds the completed indexes when .spec.completionMode =
	// "Indexed" in a text format. The indexes are represented as decimal integers
	// separated by commas. The numbers are listed in increasing order. Three or
	// more consecutive numbers are compressed and represented by the first and
	// last element of the series, separated by a hyphen.
	// For example, if the completed indexes are 1, 3, 4, 5 and 7, they are
	// represented as "1,3-5,7".
	// +optional
	CompletedIndexes string `json:"completedIndexes,omitempty" protobuf:"bytes,7,opt,name=completedIndexes"`

	// FailedIndexes holds the failed indexes when backoffLimitPerIndex=true.
	// The indexes are represented in the text format analogous as for the
	// `completedIndexes` field, ie. they are kept as decimal integers
	// separated by commas. The numbers are listed in increasing order. Three or
	// more consecutive numbers are compressed and represented by the first and
	// last element of the series, separated by a hyphen.
	// For example, if the failed indexes are 1, 3, 4, 5 and 7, they are
	// represented as "1,3-5,7".
	// This field is alpha-level. It can be used when the `JobBackoffLimitPerIndex`
	// feature gate is enabled (disabled by default).
	// +optional
	FailedIndexes *string `json:"failedIndexes,omitempty" protobuf:"bytes,10,opt,name=failedIndexes"`

	// uncountedTerminatedPods holds the UIDs of Pods that have terminated but
	// the job controller hasn't yet accounted for in the status counters.
	//
	// The job controller creates pods with a finalizer. When a pod terminates
	// (succeeded or failed), the controller does three steps to account for it
	// in the job status:
	//
	// 1. Add the pod UID to the arrays in this field.
	// 2. Remove the pod finalizer.
	// 3. Remove the pod UID from the arrays while increasing the corresponding
	//     counter.
	//
	// Old jobs might not be tracked using this field, in which case the field
	// remains null.
	// +optional
	UncountedTerminatedPods *UncountedTerminatedPods `json:"uncountedTerminatedPods,omitempty" protobuf:"bytes,8,opt,name=uncountedTerminatedPods"`

	// The number of pods which have a Ready condition.
	//
	// This field is beta-level. The job controller populates the field when
	// the feature gate JobReadyPods is enabled (enabled by default).
	// +optional
	Ready *int32 `json:"ready,omitempty" protobuf:"varint,9,opt,name=ready"`
}

// UncountedTerminatedPods holds UIDs of Pods that have terminated but haven't
// been accounted in Job status counters.
type UncountedTerminatedPods struct {
	// succeeded holds UIDs of succeeded Pods.
	// +listType=set
	// +optional
	Succeeded []types.UID `json:"succeeded,omitempty" protobuf:"bytes,1,rep,name=succeeded,casttype=k8s.io/apimachinery/pkg/types.UID"`

	// failed holds UIDs of failed Pods.
	// +listType=set
	// +optional
	Failed []types.UID `json:"failed,omitempty" protobuf:"bytes,2,rep,name=failed,casttype=k8s.io/apimachinery/pkg/types.UID"`
}

type JobConditionType string

// These are built-in conditions of a job.
const (
	// JobSuspended means the job has been suspended.
	JobSuspended JobConditionType = "Suspended"
	// JobComplete means the job has completed its execution.
	JobComplete JobConditionType = "Complete"
	// JobFailed means the job has failed its execution.
	JobFailed JobConditionType = "Failed"
	// FailureTarget means the job is about to fail its execution.
	JobFailureTarget JobConditionType = "FailureTarget"
)

// JobCondition describes current state of a job.
type JobCondition struct {
	// Type of job condition, Complete or Failed.
	Type JobConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=JobConditionType"`
	// Status of the condition, one of True, False, Unknown.
	Status corev1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=k8s.io/api/core/v1.ConditionStatus"`
	// Last time the condition was checked.
	// +optional
	LastProbeTime metav1.Time `json:"lastProbeTime,omitempty" protobuf:"bytes,3,opt,name=lastProbeTime"`
	// Last time the condition transit from one status to another.
	// +optional
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty" protobuf:"bytes,4,opt,name=lastTransitionTime"`
	// (brief) reason for the condition's last transition.
	// +optional
	Reason string `json:"reason,omitempty" protobuf:"bytes,5,opt,name=reason"`
	// Human readable message indicating details about last transition.
	// +optional
	Message string `json:"message,omitempty" protobuf:"bytes,6,opt,name=message"`
}

// JobTemplateSpec describes the data a Job should have when created from a template
type JobTemplateSpec struct {
	// Standard object's metadata of the jobs created from this template.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Specification of the desired behavior of the job.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec JobSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CronJob represents the configuration of a single cron job.
type CronJob struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Specification of the desired behavior of a cron job, including the schedule.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec CronJobSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// Current status of a cron job.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status CronJobStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CronJobList is a collection of cron jobs.
type CronJobList struct {
	metav1.TypeMeta `json:",inline"`

	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of CronJobs.
	Items []CronJob `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// CronJobSpec describes how the job execution will look like and when it will actually run.
type CronJobSpec struct {

	// The schedule in Cron format, see https://en.wikipedia.org/wiki/Cron.
	Schedule string `json:"schedule" protobuf:"bytes,1,opt,name=schedule"`

	// The time zone name for the given schedule, see https://en.wikipedia.org/wiki/List_of_tz_database_time_zones.
	// If not specified, this will default to the time zone of the kube-controller-manager process.
	// The set of valid time zone names and the time zone offset is loaded from the system-wide time zone
	// database by the API server during CronJob validation and the controller manager during execution.
	// If no system-wide time zone database can be found a bundled version of the database is used instead.
	// If the time zone name becomes invalid during the lifetime of a CronJob or due to a change in host
	// configuration, the controller will stop creating new new Jobs and will create a system event with the
	// reason UnknownTimeZone.
	// More information can be found in https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/#time-zones
	// +optional
	TimeZone *string `json:"timeZone,omitempty" protobuf:"bytes,8,opt,name=timeZone"`

	// Optional deadline in seconds for starting the job if it misses scheduled
	// time for any reason.  Missed jobs executions will be counted as failed ones.
	// +optional
	StartingDeadlineSeconds *int64 `json:"startingDeadlineSeconds,omitempty" protobuf:"varint,2,opt,name=startingDeadlineSeconds"`

	// Specifies how to treat concurrent executions of a Job.
	// Valid values are:
	//
	// - "Allow" (default): allows CronJobs to run concurrently;
	// - "Forbid": forbids concurrent runs, skipping next run if previous run hasn't finished yet;
	// - "Replace": cancels currently running job and replaces it with a new one
	// +optional
	ConcurrencyPolicy ConcurrencyPolicy `json:"concurrencyPolicy,omitempty" protobuf:"bytes,3,opt,name=concurrencyPolicy,casttype=ConcurrencyPolicy"`

	// This flag tells the controller to suspend subsequent executions, it does
	// not apply to already started executions.  Defaults to false.
	// +optional
	Suspend *bool `json:"suspend,omitempty" protobuf:"varint,4,opt,name=suspend"`

	// Specifies the job that will be created when executing a CronJob.
	JobTemplate JobTemplateSpec `json:"jobTemplate" protobuf:"bytes,5,opt,name=jobTemplate"`

	// The number of successful finished jobs to retain. Value must be non-negative integer.
	// Defaults to 3.
	// +optional
	SuccessfulJobsHistoryLimit *int32 `json:"successfulJobsHistoryLimit,omitempty" protobuf:"varint,6,opt,name=successfulJobsHistoryLimit"`

	// The number of failed finished jobs to retain. Value must be non-negative integer.
	// Defaults to 1.
	// +optional
	FailedJobsHistoryLimit *int32 `json:"failedJobsHistoryLimit,omitempty" protobuf:"varint,7,opt,name=failedJobsHistoryLimit"`
}

// ConcurrencyPolicy describes how the job will be handled.
// Only one of the following concurrent policies may be specified.
// If none of the following policies is specified, the default one
// is AllowConcurrent.
// +enum
type ConcurrencyPolicy string

const (
	// AllowConcurrent allows CronJobs to run concurrently.
	AllowConcurrent ConcurrencyPolicy = "Allow"

	// ForbidConcurrent forbids concurrent runs, skipping next run if previous
	// hasn't finished yet.
	ForbidConcurrent ConcurrencyPolicy = "Forbid"

	// ReplaceConcurrent cancels currently running job and replaces it with a new one.
	ReplaceConcurrent ConcurrencyPolicy = "Replace"
)

// CronJobStatus represents the current state of a cron job.
type CronJobStatus struct {
	// A list of pointers to currently running jobs.
	// +optional
	// +listType=atomic
	Active []corev1.ObjectReference `json:"active,omitempty" protobuf:"bytes,1,rep,name=active"`

	// Information when was the last time the job was successfully scheduled.
	// +optional
	LastScheduleTime *metav1.Time `json:"lastScheduleTime,omitempty" protobuf:"bytes,4,opt,name=lastScheduleTime"`

	// Information when was the last time the job successfully completed.
	// +optional
	LastSuccessfulTime *metav1.Time `json:"lastSuccessfulTime,omitempty" protobuf:"bytes,5,opt,name=lastSuccessfulTime"`
}
