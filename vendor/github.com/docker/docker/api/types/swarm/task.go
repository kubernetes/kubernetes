package swarm // import "github.com/docker/docker/api/types/swarm"

import (
	"time"

	"github.com/docker/docker/api/types/swarm/runtime"
)

// TaskState represents the state of a task.
type TaskState string

const (
	// TaskStateNew NEW
	TaskStateNew TaskState = "new"
	// TaskStateAllocated ALLOCATED
	TaskStateAllocated TaskState = "allocated"
	// TaskStatePending PENDING
	TaskStatePending TaskState = "pending"
	// TaskStateAssigned ASSIGNED
	TaskStateAssigned TaskState = "assigned"
	// TaskStateAccepted ACCEPTED
	TaskStateAccepted TaskState = "accepted"
	// TaskStatePreparing PREPARING
	TaskStatePreparing TaskState = "preparing"
	// TaskStateReady READY
	TaskStateReady TaskState = "ready"
	// TaskStateStarting STARTING
	TaskStateStarting TaskState = "starting"
	// TaskStateRunning RUNNING
	TaskStateRunning TaskState = "running"
	// TaskStateComplete COMPLETE
	TaskStateComplete TaskState = "complete"
	// TaskStateShutdown SHUTDOWN
	TaskStateShutdown TaskState = "shutdown"
	// TaskStateFailed FAILED
	TaskStateFailed TaskState = "failed"
	// TaskStateRejected REJECTED
	TaskStateRejected TaskState = "rejected"
	// TaskStateRemove REMOVE
	TaskStateRemove TaskState = "remove"
	// TaskStateOrphaned ORPHANED
	TaskStateOrphaned TaskState = "orphaned"
)

// Task represents a task.
type Task struct {
	ID string
	Meta
	Annotations

	Spec                TaskSpec            `json:",omitempty"`
	ServiceID           string              `json:",omitempty"`
	Slot                int                 `json:",omitempty"`
	NodeID              string              `json:",omitempty"`
	Status              TaskStatus          `json:",omitempty"`
	DesiredState        TaskState           `json:",omitempty"`
	NetworksAttachments []NetworkAttachment `json:",omitempty"`
	GenericResources    []GenericResource   `json:",omitempty"`

	// JobIteration is the JobIteration of the Service that this Task was
	// spawned from, if the Service is a ReplicatedJob or GlobalJob. This is
	// used to determine which Tasks belong to which run of the job. This field
	// is absent if the Service mode is Replicated or Global.
	JobIteration *Version `json:",omitempty"`
}

// TaskSpec represents the spec of a task.
type TaskSpec struct {
	// ContainerSpec, NetworkAttachmentSpec, and PluginSpec are mutually exclusive.
	// PluginSpec is only used when the `Runtime` field is set to `plugin`
	// NetworkAttachmentSpec is used if the `Runtime` field is set to
	// `attachment`.
	ContainerSpec         *ContainerSpec         `json:",omitempty"`
	PluginSpec            *runtime.PluginSpec    `json:",omitempty"`
	NetworkAttachmentSpec *NetworkAttachmentSpec `json:",omitempty"`

	Resources     *ResourceRequirements     `json:",omitempty"`
	RestartPolicy *RestartPolicy            `json:",omitempty"`
	Placement     *Placement                `json:",omitempty"`
	Networks      []NetworkAttachmentConfig `json:",omitempty"`

	// LogDriver specifies the LogDriver to use for tasks created from this
	// spec. If not present, the one on cluster default on swarm.Spec will be
	// used, finally falling back to the engine default if not specified.
	LogDriver *Driver `json:",omitempty"`

	// ForceUpdate is a counter that triggers an update even if no relevant
	// parameters have been changed.
	ForceUpdate uint64

	Runtime RuntimeType `json:",omitempty"`
}

// Resources represents resources (CPU/Memory).
type Resources struct {
	NanoCPUs         int64             `json:",omitempty"`
	MemoryBytes      int64             `json:",omitempty"`
	GenericResources []GenericResource `json:",omitempty"`
}

// GenericResource represents a "user defined" resource which can
// be either an integer (e.g: SSD=3) or a string (e.g: SSD=sda1)
type GenericResource struct {
	NamedResourceSpec    *NamedGenericResource    `json:",omitempty"`
	DiscreteResourceSpec *DiscreteGenericResource `json:",omitempty"`
}

// NamedGenericResource represents a "user defined" resource which is defined
// as a string.
// "Kind" is used to describe the Kind of a resource (e.g: "GPU", "FPGA", "SSD", ...)
// Value is used to identify the resource (GPU="UUID-1", FPGA="/dev/sdb5", ...)
type NamedGenericResource struct {
	Kind  string `json:",omitempty"`
	Value string `json:",omitempty"`
}

// DiscreteGenericResource represents a "user defined" resource which is defined
// as an integer
// "Kind" is used to describe the Kind of a resource (e.g: "GPU", "FPGA", "SSD", ...)
// Value is used to count the resource (SSD=5, HDD=3, ...)
type DiscreteGenericResource struct {
	Kind  string `json:",omitempty"`
	Value int64  `json:",omitempty"`
}

// ResourceRequirements represents resources requirements.
type ResourceRequirements struct {
	Limits       *Resources `json:",omitempty"`
	Reservations *Resources `json:",omitempty"`
}

// Placement represents orchestration parameters.
type Placement struct {
	Constraints []string              `json:",omitempty"`
	Preferences []PlacementPreference `json:",omitempty"`
	MaxReplicas uint64                `json:",omitempty"`

	// Platforms stores all the platforms that the image can run on.
	// This field is used in the platform filter for scheduling. If empty,
	// then the platform filter is off, meaning there are no scheduling restrictions.
	Platforms []Platform `json:",omitempty"`
}

// PlacementPreference provides a way to make the scheduler aware of factors
// such as topology.
type PlacementPreference struct {
	Spread *SpreadOver
}

// SpreadOver is a scheduling preference that instructs the scheduler to spread
// tasks evenly over groups of nodes identified by labels.
type SpreadOver struct {
	// label descriptor, such as engine.labels.az
	SpreadDescriptor string
}

// RestartPolicy represents the restart policy.
type RestartPolicy struct {
	Condition   RestartPolicyCondition `json:",omitempty"`
	Delay       *time.Duration         `json:",omitempty"`
	MaxAttempts *uint64                `json:",omitempty"`
	Window      *time.Duration         `json:",omitempty"`
}

// RestartPolicyCondition represents when to restart.
type RestartPolicyCondition string

const (
	// RestartPolicyConditionNone NONE
	RestartPolicyConditionNone RestartPolicyCondition = "none"
	// RestartPolicyConditionOnFailure ON_FAILURE
	RestartPolicyConditionOnFailure RestartPolicyCondition = "on-failure"
	// RestartPolicyConditionAny ANY
	RestartPolicyConditionAny RestartPolicyCondition = "any"
)

// TaskStatus represents the status of a task.
type TaskStatus struct {
	Timestamp       time.Time        `json:",omitempty"`
	State           TaskState        `json:",omitempty"`
	Message         string           `json:",omitempty"`
	Err             string           `json:",omitempty"`
	ContainerStatus *ContainerStatus `json:",omitempty"`
	PortStatus      PortStatus       `json:",omitempty"`
}

// ContainerStatus represents the status of a container.
type ContainerStatus struct {
	ContainerID string
	PID         int
	ExitCode    int
}

// PortStatus represents the port status of a task's host ports whose
// service has published host ports
type PortStatus struct {
	Ports []PortConfig `json:",omitempty"`
}
