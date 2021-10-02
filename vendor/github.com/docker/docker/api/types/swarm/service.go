package swarm // import "github.com/docker/docker/api/types/swarm"

import "time"

// Service represents a service.
type Service struct {
	ID string
	Meta
	Spec         ServiceSpec   `json:",omitempty"`
	PreviousSpec *ServiceSpec  `json:",omitempty"`
	Endpoint     Endpoint      `json:",omitempty"`
	UpdateStatus *UpdateStatus `json:",omitempty"`

	// ServiceStatus is an optional, extra field indicating the number of
	// desired and running tasks. It is provided primarily as a shortcut to
	// calculating these values client-side, which otherwise would require
	// listing all tasks for a service, an operation that could be
	// computation and network expensive.
	ServiceStatus *ServiceStatus `json:",omitempty"`

	// JobStatus is the status of a Service which is in one of ReplicatedJob or
	// GlobalJob modes. It is absent on Replicated and Global services.
	JobStatus *JobStatus `json:",omitempty"`
}

// ServiceSpec represents the spec of a service.
type ServiceSpec struct {
	Annotations

	// TaskTemplate defines how the service should construct new tasks when
	// orchestrating this service.
	TaskTemplate   TaskSpec      `json:",omitempty"`
	Mode           ServiceMode   `json:",omitempty"`
	UpdateConfig   *UpdateConfig `json:",omitempty"`
	RollbackConfig *UpdateConfig `json:",omitempty"`

	// Networks field in ServiceSpec is deprecated. The
	// same field in TaskSpec should be used instead.
	// This field will be removed in a future release.
	Networks     []NetworkAttachmentConfig `json:",omitempty"`
	EndpointSpec *EndpointSpec             `json:",omitempty"`
}

// ServiceMode represents the mode of a service.
type ServiceMode struct {
	Replicated    *ReplicatedService `json:",omitempty"`
	Global        *GlobalService     `json:",omitempty"`
	ReplicatedJob *ReplicatedJob     `json:",omitempty"`
	GlobalJob     *GlobalJob         `json:",omitempty"`
}

// UpdateState is the state of a service update.
type UpdateState string

const (
	// UpdateStateUpdating is the updating state.
	UpdateStateUpdating UpdateState = "updating"
	// UpdateStatePaused is the paused state.
	UpdateStatePaused UpdateState = "paused"
	// UpdateStateCompleted is the completed state.
	UpdateStateCompleted UpdateState = "completed"
	// UpdateStateRollbackStarted is the state with a rollback in progress.
	UpdateStateRollbackStarted UpdateState = "rollback_started"
	// UpdateStateRollbackPaused is the state with a rollback in progress.
	UpdateStateRollbackPaused UpdateState = "rollback_paused"
	// UpdateStateRollbackCompleted is the state with a rollback in progress.
	UpdateStateRollbackCompleted UpdateState = "rollback_completed"
)

// UpdateStatus reports the status of a service update.
type UpdateStatus struct {
	State       UpdateState `json:",omitempty"`
	StartedAt   *time.Time  `json:",omitempty"`
	CompletedAt *time.Time  `json:",omitempty"`
	Message     string      `json:",omitempty"`
}

// ReplicatedService is a kind of ServiceMode.
type ReplicatedService struct {
	Replicas *uint64 `json:",omitempty"`
}

// GlobalService is a kind of ServiceMode.
type GlobalService struct{}

// ReplicatedJob is the a type of Service which executes a defined Tasks
// in parallel until the specified number of Tasks have succeeded.
type ReplicatedJob struct {
	// MaxConcurrent indicates the maximum number of Tasks that should be
	// executing simultaneously for this job at any given time. There may be
	// fewer Tasks that MaxConcurrent executing simultaneously; for example, if
	// there are fewer than MaxConcurrent tasks needed to reach
	// TotalCompletions.
	//
	// If this field is empty, it will default to a max concurrency of 1.
	MaxConcurrent *uint64 `json:",omitempty"`

	// TotalCompletions is the total number of Tasks desired to run to
	// completion.
	//
	// If this field is empty, the value of MaxConcurrent will be used.
	TotalCompletions *uint64 `json:",omitempty"`
}

// GlobalJob is the type of a Service which executes a Task on every Node
// matching the Service's placement constraints. These tasks run to completion
// and then exit.
//
// This type is deliberately empty.
type GlobalJob struct{}

const (
	// UpdateFailureActionPause PAUSE
	UpdateFailureActionPause = "pause"
	// UpdateFailureActionContinue CONTINUE
	UpdateFailureActionContinue = "continue"
	// UpdateFailureActionRollback ROLLBACK
	UpdateFailureActionRollback = "rollback"

	// UpdateOrderStopFirst STOP_FIRST
	UpdateOrderStopFirst = "stop-first"
	// UpdateOrderStartFirst START_FIRST
	UpdateOrderStartFirst = "start-first"
)

// UpdateConfig represents the update configuration.
type UpdateConfig struct {
	// Maximum number of tasks to be updated in one iteration.
	// 0 means unlimited parallelism.
	Parallelism uint64

	// Amount of time between updates.
	Delay time.Duration `json:",omitempty"`

	// FailureAction is the action to take when an update failures.
	FailureAction string `json:",omitempty"`

	// Monitor indicates how long to monitor a task for failure after it is
	// created. If the task fails by ending up in one of the states
	// REJECTED, COMPLETED, or FAILED, within Monitor from its creation,
	// this counts as a failure. If it fails after Monitor, it does not
	// count as a failure. If Monitor is unspecified, a default value will
	// be used.
	Monitor time.Duration `json:",omitempty"`

	// MaxFailureRatio is the fraction of tasks that may fail during
	// an update before the failure action is invoked. Any task created by
	// the current update which ends up in one of the states REJECTED,
	// COMPLETED or FAILED within Monitor from its creation counts as a
	// failure. The number of failures is divided by the number of tasks
	// being updated, and if this fraction is greater than
	// MaxFailureRatio, the failure action is invoked.
	//
	// If the failure action is CONTINUE, there is no effect.
	// If the failure action is PAUSE, no more tasks will be updated until
	// another update is started.
	MaxFailureRatio float32

	// Order indicates the order of operations when rolling out an updated
	// task. Either the old task is shut down before the new task is
	// started, or the new task is started before the old task is shut down.
	Order string
}

// ServiceStatus represents the number of running tasks in a service and the
// number of tasks desired to be running.
type ServiceStatus struct {
	// RunningTasks is the number of tasks for the service actually in the
	// Running state
	RunningTasks uint64

	// DesiredTasks is the number of tasks desired to be running by the
	// service. For replicated services, this is the replica count. For global
	// services, this is computed by taking the number of tasks with desired
	// state of not-Shutdown.
	DesiredTasks uint64

	// CompletedTasks is the number of tasks in the state Completed, if this
	// service is in ReplicatedJob or GlobalJob mode. This field must be
	// cross-referenced with the service type, because the default value of 0
	// may mean that a service is not in a job mode, or it may mean that the
	// job has yet to complete any tasks.
	CompletedTasks uint64
}

// JobStatus is the status of a job-type service.
type JobStatus struct {
	// JobIteration is a value increased each time a Job is executed,
	// successfully or otherwise. "Executed", in this case, means the job as a
	// whole has been started, not that an individual Task has been launched. A
	// job is "Executed" when its ServiceSpec is updated. JobIteration can be
	// used to disambiguate Tasks belonging to different executions of a job.
	//
	// Though JobIteration will increase with each subsequent execution, it may
	// not necessarily increase by 1, and so JobIteration should not be used to
	// keep track of the number of times a job has been executed.
	JobIteration Version

	// LastExecution is the time that the job was last executed, as observed by
	// Swarm manager.
	LastExecution time.Time `json:",omitempty"`
}
