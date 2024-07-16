/*
Copyright 2021 The Kubernetes Authors.

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

package metrics

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// JobControllerSubsystem - subsystem name used for this controller.
const JobControllerSubsystem = "job_controller"

var (
	// JobSyncDurationSeconds tracks the latency of Job syncs. Possible label
	// values:
	//   completion_mode: Indexed, NonIndexed
	//   result:          success, error
	//   action:          reconciling, tracking, pods_created, pods_deleted
	JobSyncDurationSeconds = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      JobControllerSubsystem,
			Name:           "job_sync_duration_seconds",
			Help:           "The time it took to sync a job",
			StabilityLevel: metrics.STABLE,
			Buckets:        metrics.ExponentialBuckets(0.004, 2, 15),
		},
		[]string{"completion_mode", "result", "action"},
	)
	// JobSyncNum tracks the number of Job syncs. Possible label values:
	//   completion_mode: Indexed, NonIndexed
	//   result:          success, error
	//   action:          reconciling, tracking, pods_created, pods_deleted
	JobSyncNum = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      JobControllerSubsystem,
			Name:           "job_syncs_total",
			Help:           "The number of job syncs",
			StabilityLevel: metrics.STABLE,
		},
		[]string{"completion_mode", "result", "action"},
	)
	// JobFinishedNum tracks the number of Jobs that finish. Empty reason label
	// is used to count successful jobs.
	// Possible label values:
	//   completion_mode: Indexed, NonIndexed
	//   result:          failed, succeeded
	//   reason:          "BackoffLimitExceeded", "DeadlineExceeded", "PodFailurePolicy", "FailedIndexes", "MaxFailedIndexesExceeded", ""
	JobFinishedNum = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      JobControllerSubsystem,
			Name:           "jobs_finished_total",
			Help:           "The number of finished jobs",
			StabilityLevel: metrics.STABLE,
		},
		[]string{"completion_mode", "result", "reason"},
	)

	// JobByExternalControllerTotal tracks the number of Jobs that were created
	// as managed by an external controller.
	// The value of the label controller_name corresponds to the value of the
	// managedBy field.
	JobByExternalControllerTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      JobControllerSubsystem,
			Name:           "jobs_by_external_controller_total",
			Help:           "The number of Jobs managed by an external controller",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"controller_name"},
	)

	// JobPodsFinished records the number of finished Pods that the job controller
	// finished tracking.
	// It only applies to Jobs that were created while the feature gate
	// JobTrackingWithFinalizers was enabled.
	// Possible label values:
	//   completion_mode: Indexed, NonIndexed
	//   result:          failed, succeeded
	JobPodsFinished = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      JobControllerSubsystem,
			Name:           "job_pods_finished_total",
			Help:           "The number of finished Pods that are fully tracked",
			StabilityLevel: metrics.STABLE,
		},
		[]string{"completion_mode", "result"})

	// PodFailuresHandledByFailurePolicy records the number of finished Pods
	// handled by pod failure policy.
	// Possible label values:
	//   action: FailJob, Ignore, Count
	PodFailuresHandledByFailurePolicy = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem: JobControllerSubsystem,
			Name:      "pod_failures_handled_by_failure_policy_total",
			Help: `The number of failed Pods handled by failure policy with
			respect to the failure policy action applied based on the matched
			rule. Possible values of the action label correspond to the
			possible values for the failure policy rule action, which are:
			"FailJob", "Ignore" and "Count".`,
		},
		[]string{"action"})

	// TerminatedPodsTrackingFinalizerTotal records the addition and removal of
	// terminated pods that have the finalizer batch.kubernetes.io/job-tracking,
	// regardless of whether they are owned by a Job.
	TerminatedPodsTrackingFinalizerTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem: JobControllerSubsystem,
			Name:      "terminated_pods_tracking_finalizer_total",
			Help: `The number of terminated pods (phase=Failed|Succeeded)
that have the finalizer batch.kubernetes.io/job-tracking
The event label can be "add" or "delete".`,
		}, []string{"event"})

	// JobFinishedIndexesTotal records the number of finished indexes.
	JobFinishedIndexesTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem: JobControllerSubsystem,
			Name:      "job_finished_indexes_total",
			Help: `The number of finished indexes. Possible values for the
			status label are: "succeeded", "failed". Possible values for the
			backoffLimit label are: "perIndex" and "global"`,
		},
		[]string{"status", "backoffLimit"})

	// JobPodsCreationTotal records the number of pods created by the job controller
	// based on the reason for their creation (i.e. if PodReplacementPolicy was specified)
	// and the status of the creation (i.e. if the Pod creation succeeded or failed).
	// Possible label values:
	//   reason: new, recreate_terminating_or_failed, recreate_failed
	//   status: succeeded, failed
	JobPodsCreationTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem: JobControllerSubsystem,
			Name:      "job_pods_creation_total",
			Help: `The number of Pods created by the Job controller labelled with a reason for the Pod creation.
This metric also distinguishes between Pods created using different PodReplacementPolicy settings.
Possible values of the "reason" label are:
"new", "recreate_terminating_or_failed", "recreate_failed".
Possible values of the "status" label are:
"succeeded", "failed".`,
		}, []string{"reason", "status"})
)

const (
	// Possible values for the "action" label in the above metrics.

	// JobSyncActionReconciling when the Job's pod creation/deletion expectations
	// are unsatisfied and the controller is waiting for issued Pod
	// creation/deletions to complete.
	JobSyncActionReconciling = "reconciling"
	// JobSyncActionTracking when the Job's pod creation/deletion expectations
	// are satisfied and the number of active Pods matches expectations (i.e. no
	// pod creation/deletions issued in this sync). This is expected to be the
	// action in most of the syncs.
	JobSyncActionTracking = "tracking"
	// JobSyncActionPodsCreated when the controller creates Pods. This can happen
	// when the number of active Pods is less than the wanted Job parallelism.
	JobSyncActionPodsCreated = "pods_created"
	// JobSyncActionPodsDeleted when the controller deletes Pods. This can happen
	// if a Job is suspended or if the number of active Pods is more than
	// parallelism.
	JobSyncActionPodsDeleted = "pods_deleted"

	// Possible values for "result" and "status" (job_pods_creation_total) labels in the above metrics.

	Succeeded = "succeeded"
	Failed    = "failed"

	// Possible values for "event"  label in the terminated_pods_tracking_finalizer
	// metric.
	Add    = "add"
	Delete = "delete"

	// Possible values for "reason" label in the job_pods_creation_total metric.

	PodCreateNew                   = "new"
	PodRecreateTerminatingOrFailed = "recreate_terminating_or_failed"
	PodRecreateFailed              = "recreate_failed"
)

var registerMetrics sync.Once

// Register registers Job controller metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(JobSyncDurationSeconds)
		legacyregistry.MustRegister(JobSyncNum)
		legacyregistry.MustRegister(JobFinishedNum)
		legacyregistry.MustRegister(JobPodsFinished)
		legacyregistry.MustRegister(PodFailuresHandledByFailurePolicy)
		legacyregistry.MustRegister(TerminatedPodsTrackingFinalizerTotal)
		legacyregistry.MustRegister(JobFinishedIndexesTotal)
		legacyregistry.MustRegister(JobPodsCreationTotal)
		legacyregistry.MustRegister(JobByExternalControllerTotal)
	})
}
