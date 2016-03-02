// Libcontainer provides a native Go implementation for creating containers
// with namespaces, cgroups, capabilities, and filesystem access controls.
// It allows you to manage the lifecycle of the container performing additional operations
// after the container is created.
package libcontainer

import (
	"os"

	"github.com/opencontainers/runc/libcontainer/configs"
)

// The status of a container.
type Status int

const (
	// The container exists but has not been run yet
	Created Status = iota

	// The container exists and is running.
	Running

	// The container exists, it is in the process of being paused.
	Pausing

	// The container exists, but all its processes are paused.
	Paused

	// The container exists, but its state is saved on disk
	Checkpointed

	// The container does not exist.
	Destroyed
)

func (s Status) String() string {
	switch s {
	case Created:
		return "created"
	case Running:
		return "running"
	case Pausing:
		return "pausing"
	case Paused:
		return "paused"
	case Checkpointed:
		return "checkpointed"
	case Destroyed:
		return "destroyed"
	default:
		return "unknown"
	}
}

// BaseState represents the platform agnostic pieces relating to a
// running container's state
type BaseState struct {
	// ID is the container ID.
	ID string `json:"id"`

	// InitProcessPid is the init process id in the parent namespace.
	InitProcessPid int `json:"init_process_pid"`

	// InitProcessStartTime is the init process start time.
	InitProcessStartTime string `json:"init_process_start"`

	// Config is the container's configuration.
	Config configs.Config `json:"config"`
}

// A libcontainer container object.
//
// Each container is thread-safe within the same process. Since a container can
// be destroyed by a separate process, any function may return that the container
// was not found. BaseContainer includes methods that are platform agnostic.
type BaseContainer interface {
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

	// Set resources of container as configured
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

	// Destroys the container after killing all running processes.
	//
	// Any event registrations are removed before the container is destroyed.
	// No error is returned if the container is already destroyed.
	//
	// errors:
	// Systemerror - System error.
	Destroy() error

	// Signal sends the provided signal code to the container's initial process.
	//
	// errors:
	// Systemerror - System error.
	Signal(s os.Signal) error
}
