package swarm

import "time"

// Service represents a service.
type Service struct {
	ID string
	Meta
	Spec         ServiceSpec  `json:",omitempty"`
	PreviousSpec *ServiceSpec `json:",omitempty"`
	Endpoint     Endpoint     `json:",omitempty"`
	UpdateStatus UpdateStatus `json:",omitempty"`
}

// ServiceSpec represents the spec of a service.
type ServiceSpec struct {
	Annotations

	// TaskTemplate defines how the service should construct new tasks when
	// orchestrating this service.
	TaskTemplate TaskSpec      `json:",omitempty"`
	Mode         ServiceMode   `json:",omitempty"`
	UpdateConfig *UpdateConfig `json:",omitempty"`

	// Networks field in ServiceSpec is deprecated. The
	// same field in TaskSpec should be used instead.
	// This field will be removed in a future release.
	Networks     []NetworkAttachmentConfig `json:",omitempty"`
	EndpointSpec *EndpointSpec             `json:",omitempty"`
}

// ServiceMode represents the mode of a service.
type ServiceMode struct {
	Replicated *ReplicatedService `json:",omitempty"`
	Global     *GlobalService     `json:",omitempty"`
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
)

// UpdateStatus reports the status of a service update.
type UpdateStatus struct {
	State       UpdateState `json:",omitempty"`
	StartedAt   time.Time   `json:",omitempty"`
	CompletedAt time.Time   `json:",omitempty"`
	Message     string      `json:",omitempty"`
}

// ReplicatedService is a kind of ServiceMode.
type ReplicatedService struct {
	Replicas *uint64 `json:",omitempty"`
}

// GlobalService is a kind of ServiceMode.
type GlobalService struct{}

const (
	// UpdateFailureActionPause PAUSE
	UpdateFailureActionPause = "pause"
	// UpdateFailureActionContinue CONTINUE
	UpdateFailureActionContinue = "continue"
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
}
