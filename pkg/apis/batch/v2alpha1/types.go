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

package v2alpha1

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
)

// Job represents the configuration of a single job.
type Job struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	v1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec is a structure defining the expected behavior of a job.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Spec JobSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// Status is a structure describing current status of a job.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Status JobStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// JobList is a collection of jobs.
type JobList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	unversioned.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of Job.
	Items []Job `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// JobTemplate describes a template for creating copies of a predefined pod.
type JobTemplate struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	v1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Template defines jobs that will be created from this template
	// http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Template JobTemplateSpec `json:"template,omitempty" protobuf:"bytes,2,opt,name=template"`
}

// JobTemplateSpec describes the data a Job should have when created from a template
type JobTemplateSpec struct {
	// Standard object's metadata of the jobs created from this template.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	v1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Specification of the desired behavior of the job.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Spec JobSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
}

// JobSpec describes how the job execution will look like.
type JobSpec struct {

	// Parallelism specifies the maximum desired number of pods the job should
	// run at any given time. The actual number of pods running in steady state will
	// be less than this number when ((.spec.completions - .status.successful) < .spec.parallelism),
	// i.e. when the work left to do is less than max parallelism.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/jobs.md
	Parallelism *int32 `json:"parallelism,omitempty" protobuf:"varint,1,opt,name=parallelism"`

	// Completions specifies the desired number of successfully finished pods the
	// job should be run with.  Setting to nil means that the success of any
	// pod signals the success of all pods, and allows parallelism to have any positive
	// value.  Setting to 1 means that parallelism is limited to 1 and the success of that
	// pod signals the success of the job.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/jobs.md
	Completions *int32 `json:"completions,omitempty" protobuf:"varint,2,opt,name=completions"`

	// Optional duration in seconds relative to the startTime that the job may be active
	// before the system tries to terminate it; value must be positive integer
	ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty" protobuf:"varint,3,opt,name=activeDeadlineSeconds"`

	// Selector is a label query over pods that should match the pod count.
	// Normally, the system sets this field for you.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/labels.md#label-selectors
	Selector *LabelSelector `json:"selector,omitempty" protobuf:"bytes,4,opt,name=selector"`

	// ManualSelector controls generation of pod labels and pod selectors.
	// Leave `manualSelector` unset unless you are certain what you are doing.
	// When false or unset, the system pick labels unique to this job
	// and appends those labels to the pod template.  When true,
	// the user is responsible for picking unique labels and specifying
	// the selector.  Failure to pick a unique label may cause this
	// and other jobs to not function correctly.  However, You may see
	// `manualSelector=true` in jobs that were created with the old `extensions/v1beta1`
	// API.
	// More info: http://releases.k8s.io/HEAD/docs/design/selector-generation.md
	ManualSelector *bool `json:"manualSelector,omitempty" protobuf:"varint,5,opt,name=manualSelector"`

	// Template is the object that describes the pod that will be created when
	// executing a job.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/jobs.md
	Template v1.PodTemplateSpec `json:"template" protobuf:"bytes,6,opt,name=template"`
}

// JobStatus represents the current state of a Job.
type JobStatus struct {

	// Conditions represent the latest available observations of an object's current state.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/jobs.md
	Conditions []JobCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,1,rep,name=conditions"`

	// StartTime represents time when the job was acknowledged by the Job Manager.
	// It is not guaranteed to be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	StartTime *unversioned.Time `json:"startTime,omitempty" protobuf:"bytes,2,opt,name=startTime"`

	// CompletionTime represents time when the job was completed. It is not guaranteed to
	// be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	CompletionTime *unversioned.Time `json:"completionTime,omitempty" protobuf:"bytes,3,opt,name=completionTime"`

	// Active is the number of actively running pods.
	Active int32 `json:"active,omitempty" protobuf:"varint,4,opt,name=active"`

	// Succeeded is the number of pods which reached Phase Succeeded.
	Succeeded int32 `json:"succeeded,omitempty" protobuf:"varint,5,opt,name=succeeded"`

	// Failed is the number of pods which reached Phase Failed.
	Failed int32 `json:"failed,omitempty" protobuf:"varint,6,opt,name=failed"`
}

type JobConditionType string

// These are valid conditions of a job.
const (
	// JobComplete means the job has completed its execution.
	JobComplete JobConditionType = "Complete"
	// JobFailed means the job has failed its execution.
	JobFailed JobConditionType = "Failed"
)

// JobCondition describes current state of a job.
type JobCondition struct {
	// Type of job condition, Complete or Failed.
	Type JobConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=JobConditionType"`
	// Status of the condition, one of True, False, Unknown.
	Status v1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=k8s.io/kubernetes/pkg/api/v1.ConditionStatus"`
	// Last time the condition was checked.
	LastProbeTime unversioned.Time `json:"lastProbeTime,omitempty" protobuf:"bytes,3,opt,name=lastProbeTime"`
	// Last time the condition transit from one status to another.
	LastTransitionTime unversioned.Time `json:"lastTransitionTime,omitempty" protobuf:"bytes,4,opt,name=lastTransitionTime"`
	// (brief) reason for the condition's last transition.
	Reason string `json:"reason,omitempty" protobuf:"bytes,5,opt,name=reason"`
	// Human readable message indicating details about last transition.
	Message string `json:"message,omitempty" protobuf:"bytes,6,opt,name=message"`
}

// ScheduledJob represents the configuration of a single scheduled job.
type ScheduledJob struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	v1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec is a structure defining the expected behavior of a job, including the schedule.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Spec ScheduledJobSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// Status is a structure describing current status of a job.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Status ScheduledJobStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// ScheduledJobList is a collection of scheduled jobs.
type ScheduledJobList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	unversioned.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of ScheduledJob.
	Items []ScheduledJob `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// ScheduledJobSpec describes how the job execution will look like and when it will actually run.
type ScheduledJobSpec struct {

	// Schedule contains the schedule in Cron format, see https://en.wikipedia.org/wiki/Cron.
	Schedule string `json:"schedule" protobuf:"bytes,1,opt,name=schedule"`

	// Optional deadline in seconds for starting the job if it misses scheduled
	// time for any reason.  Missed jobs executions will be counted as failed ones.
	StartingDeadlineSeconds *int64 `json:"startingDeadlineSeconds,omitempty" protobuf:"varint,2,opt,name=startingDeadlineSeconds"`

	// ConcurrencyPolicy specifies how to treat concurrent executions of a Job.
	ConcurrencyPolicy ConcurrencyPolicy `json:"concurrencyPolicy,omitempty" protobuf:"bytes,3,opt,name=concurrencyPolicy,casttype=ConcurrencyPolicy"`

	// Suspend flag tells the controller to suspend subsequent executions, it does
	// not apply to already started executions.  Defaults to false.
	Suspend *bool `json:"suspend" protobuf:"varint,4,opt,name=suspend"`

	// JobTemplate is the object that describes the job that will be created when
	// executing a ScheduledJob.
	JobTemplate JobTemplateSpec `json:"jobTemplate" protobuf:"bytes,5,opt,name=jobTemplate"`
}

// ConcurrencyPolicy describes how the job will be handled.
// Only one of the following concurrent policies may be specified.
// If none of the following policies is specified, the default one
// is AllowConcurrent.
type ConcurrencyPolicy string

const (
	// AllowConcurrent allows ScheduledJobs to run concurrently.
	AllowConcurrent ConcurrencyPolicy = "Allow"

	// ForbidConcurrent forbids concurrent runs, skipping next run if previous
	// hasn't finished yet.
	ForbidConcurrent ConcurrencyPolicy = "Forbid"

	// ReplaceConcurrent cancels currently running job and replaces it with a new one.
	ReplaceConcurrent ConcurrencyPolicy = "Replace"
)

// ScheduledJobStatus represents the current state of a Job.
type ScheduledJobStatus struct {
	// Active holds pointers to currently running jobs.
	Active []v1.ObjectReference `json:"active,omitempty" protobuf:"bytes,1,rep,name=active"`

	// LastScheduleTime keeps information of when was the last time the job was successfully scheduled.
	LastScheduleTime *unversioned.Time `json:"lastScheduleTime,omitempty" protobuf:"bytes,4,opt,name=lastScheduleTime"`
}

// A label selector is a label query over a set of resources. The result of matchLabels and
// matchExpressions are ANDed. An empty label selector matches all objects. A null
// label selector matches no objects.
type LabelSelector struct {
	// matchLabels is a map of {key,value} pairs. A single {key,value} in the matchLabels
	// map is equivalent to an element of matchExpressions, whose key field is "key", the
	// operator is "In", and the values array contains only "value". The requirements are ANDed.
	MatchLabels map[string]string `json:"matchLabels,omitempty" protobuf:"bytes,1,rep,name=matchLabels"`
	// matchExpressions is a list of label selector requirements. The requirements are ANDed.
	MatchExpressions []LabelSelectorRequirement `json:"matchExpressions,omitempty" protobuf:"bytes,2,rep,name=matchExpressions"`
}

// A label selector requirement is a selector that contains values, a key, and an operator that
// relates the key and values.
type LabelSelectorRequirement struct {
	// key is the label key that the selector applies to.
	Key string `json:"key" patchStrategy:"merge" patchMergeKey:"key" protobuf:"bytes,1,opt,name=key"`
	// operator represents a key's relationship to a set of values.
	// Valid operators ard In, NotIn, Exists and DoesNotExist.
	Operator LabelSelectorOperator `json:"operator" protobuf:"bytes,2,opt,name=operator,casttype=LabelSelectorOperator"`
	// values is an array of string values. If the operator is In or NotIn,
	// the values array must be non-empty. If the operator is Exists or DoesNotExist,
	// the values array must be empty. This array is replaced during a strategic
	// merge patch.
	Values []string `json:"values,omitempty" protobuf:"bytes,3,rep,name=values"`
}

// A label selector operator is the set of operators that can be used in a selector requirement.
type LabelSelectorOperator string

const (
	LabelSelectorOpIn           LabelSelectorOperator = "In"
	LabelSelectorOpNotIn        LabelSelectorOperator = "NotIn"
	LabelSelectorOpExists       LabelSelectorOperator = "Exists"
	LabelSelectorOpDoesNotExist LabelSelectorOperator = "DoesNotExist"
)
