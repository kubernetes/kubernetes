// Libcontainer provides a native Go implementation for creating containers
// with namespaces, cgroups, capabilities, and filesystem access controls.
// It allows you to manage the lifecycle of the container performing additional operations
// after the container is created.
package libcontainer

import (
	"github.com/docker/libcontainer/configs"
)

// The status of a container.
type Status int

const (
	// The container exists and is running.
	Running Status = iota + 1

	// The container exists, it is in the process of being paused.
	Pausing

	// The container exists, but all its processes are paused.
	Paused

	// The container exists, but its state is saved on disk
	Checkpointed

	// The container does not exist.
	Destroyed
)

// State represents a running container's state
type State struct {
	// ID is the container ID.
	ID string `json:"id"`

	// InitProcessPid is the init process id in the parent namespace.
	InitProcessPid int `json:"init_process_pid"`

	// InitProcessStartTime is the init process start time.
	InitProcessStartTime string `json:"init_process_start"`

	// Path to all the cgroups setup for a container. Key is cgroup subsystem name
	// with the value as the path.
	CgroupPaths map[string]string `json:"cgroup_paths"`

	// NamespacePaths are filepaths to the container's namespaces. Key is the namespace type
	// with the value as the path.
	NamespacePaths map[configs.NamespaceType]string `json:"namespace_paths"`

	// Config is the container's configuration.
	Config configs.Config `json:"config"`

	// Container's standard descriptors (std{in,out,err}), needed for checkpoint and restore
	ExternalDescriptors []string `json:"external_descriptors,omitempty"`
}

// A libcontainer container object.
//
// Each container is thread-safe within the same process. Since a container can
// be destroyed by a separate process, any function may return that the container
// was not found.
type Container interface {
	// Returns the ID of the container
	ID() string

	// Returns the current status of the container.
	//
	// errors:
	// ContainerDestroyed - Container no longer exists,
	// Systemerror - System error.
	Status() (Status, error)

	// State returns the current container's state information.
	//
	// errors:
	// Systemerror - System error.
	State() (*State, error)

	// Returns the current config of the container.
	Config() configs.Config

	// Returns the PIDs inside this container. The PIDs are in the namespace of the calling process.
	//
	// errors:
	// ContainerDestroyed - Container no longer exists,
	// Systemerror - System error.
	//
	// Some of the returned PIDs may no longer refer to processes in the Container, unless
	// the Container state is PAUSED in which case every PID in the slice is valid.
	Processes() ([]int, error)

	// Returns statistics for the container.
	//
	// errors:
	// ContainerDestroyed - Container no longer exists,
	// Systemerror - System error.
	Stats() (*Stats, error)

	// Set cgroup resources of container as configured
	//
	// We can use this to change resources when containers are running.
	//
	// errors:
	// Systemerror - System error.
	Set(config configs.Config) error

	// Start a process inside the container. Returns error if process fails to
	// start. You can track process lifecycle with passed Process structure.
	//
	// errors:
	// ContainerDestroyed - Container no longer exists,
	// ConfigInvalid - config is invalid,
	// ContainerPaused - Container is paused,
	// Systemerror - System error.
	Start(process *Process) (err error)

	// Checkpoint checkpoints the running container's state to disk using the criu(8) utility.
	//
	// errors:
	// Systemerror - System error.
	Checkpoint(criuOpts *CriuOpts) error

	// Restore restores the checkpointed container to a running state using the criu(8) utiity.
	//
	// errors:
	// Systemerror - System error.
	Restore(process *Process, criuOpts *CriuOpts) error

	// Destroys the container after killing all running processes.
	//
	// Any event registrations are removed before the container is destroyed.
	// No error is returned if the container is already destroyed.
	//
	// errors:
	// Systemerror - System error.
	Destroy() error

	// If the Container state is RUNNING or PAUSING, sets the Container state to PAUSING and pauses
	// the execution of any user processes. Asynchronously, when the container finished being paused the
	// state is changed to PAUSED.
	// If the Container state is PAUSED, do nothing.
	//
	// errors:
	// ContainerDestroyed - Container no longer exists,
	// Systemerror - System error.
	Pause() error

	// If the Container state is PAUSED, resumes the execution of any user processes in the
	// Container before setting the Container state to RUNNING.
	// If the Container state is RUNNING, do nothing.
	//
	// errors:
	// ContainerDestroyed - Container no longer exists,
	// Systemerror - System error.
	Resume() error

	// NotifyOOM returns a read-only channel signaling when the container receives an OOM notification.
	//
	// errors:
	// Systemerror - System error.
	NotifyOOM() (<-chan struct{}, error)
}
