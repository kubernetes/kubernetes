// Package libcontainer provides a native Go implementation for creating containers
// with namespaces, cgroups, capabilities, and filesystem access controls.
// It allows you to manage the lifecycle of the container performing additional operations
// after the container is created.
package libcontainer

import (
	"os"
	"time"

	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runtime-spec/specs-go"
)

// Status is the status of a container.
type Status int

const (
	// Created is the status that denotes the container exists but has not been run yet.
	Created Status = iota
	// Running is the status that denotes the container exists and is running.
	Running
	// Pausing is the status that denotes the container exists, it is in the process of being paused.
	Pausing
	// Paused is the status that denotes the container exists, but all its processes are paused.
	Paused
	// Stopped is the status that denotes the container does not have a created or running process.
	Stopped
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
	case Stopped:
		return "stopped"
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

	// InitProcessStartTime is the init process start time in clock cycles since boot time.
	InitProcessStartTime uint64 `json:"init_process_start"`

	// Created is the unix timestamp for the creation time of the container in UTC
	Created time.Time `json:"created"`

	// Config is the container's configuration.
	Config configs.Config `json:"config"`
}

// BaseContainer is a libcontainer container object.
//
// Each container is thread-safe within the same process. Since a container can
// be destroyed by a separate process, any function may return that the container
// was not found. BaseContainer includes methods that are platform agnostic.
type BaseContainer interface {
	// Returns the ID of the container
	ID() string

	// Returns the current status of the container.
	Status() (Status, error)

	// State returns the current container's state information.
	State() (*State, error)

	// OCIState returns the current container's state information.
	OCIState() (*specs.State, error)

	// Returns the current config of the container.
	Config() configs.Config

	// Returns the PIDs inside this container. The PIDs are in the namespace of the calling process.
	//
	// Some of the returned PIDs may no longer refer to processes in the Container, unless
	// the Container state is PAUSED in which case every PID in the slice is valid.
	Processes() ([]int, error)

	// Returns statistics for the container.
	Stats() (*Stats, error)

	// Set resources of container as configured
	//
	// We can use this to change resources when containers are running.
	//
	Set(config configs.Config) error

	// Start a process inside the container. Returns error if process fails to
	// start. You can track process lifecycle with passed Process structure.
	Start(process *Process) (err error)

	// Run immediately starts the process inside the container.  Returns error if process
	// fails to start.  It does not block waiting for the exec fifo  after start returns but
	// opens the fifo after start returns.
	Run(process *Process) (err error)

	// Destroys the container, if its in a valid state, after killing any
	// remaining running processes.
	//
	// Any event registrations are removed before the container is destroyed.
	// No error is returned if the container is already destroyed.
	//
	// Running containers must first be stopped using Signal(..).
	// Paused containers must first be resumed using Resume(..).
	Destroy() error

	// Signal sends the provided signal code to the container's initial process.
	//
	// If all is specified the signal is sent to all processes in the container
	// including the initial process.
	Signal(s os.Signal, all bool) error

	// Exec signals the container to exec the users process at the end of the init.
	Exec() error
}
