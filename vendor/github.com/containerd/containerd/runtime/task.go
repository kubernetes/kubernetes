package runtime

import (
	"context"
	"time"

	"github.com/gogo/protobuf/types"
)

// TaskInfo provides task specific information
type TaskInfo struct {
	ID        string
	Runtime   string
	Spec      []byte
	Namespace string
}

// Process is a runtime object for an executing process inside a container
type Process interface {
	ID() string
	// State returns the process state
	State(context.Context) (State, error)
	// Kill signals a container
	Kill(context.Context, uint32, bool) error
	// Pty resizes the processes pty/console
	ResizePty(context.Context, ConsoleSize) error
	// CloseStdin closes the processes stdin
	CloseIO(context.Context) error
	// Start the container's user defined process
	Start(context.Context) error
	// Wait for the process to exit
	Wait(context.Context) (*Exit, error)
}

// Task is the runtime object for an executing container
type Task interface {
	Process

	// Information of the container
	Info() TaskInfo
	// Pause pauses the container process
	Pause(context.Context) error
	// Resume unpauses the container process
	Resume(context.Context) error
	// Exec adds a process into the container
	Exec(context.Context, string, ExecOpts) (Process, error)
	// Pids returns all pids
	Pids(context.Context) ([]ProcessInfo, error)
	// Checkpoint checkpoints a container to an image with live system data
	Checkpoint(context.Context, string, *types.Any) error
	// DeleteProcess deletes a specific exec process via its id
	DeleteProcess(context.Context, string) (*Exit, error)
	// Update sets the provided resources to a running task
	Update(context.Context, *types.Any) error
	// Process returns a process within the task for the provided id
	Process(context.Context, string) (Process, error)
	// Metrics returns runtime specific metrics for a task
	Metrics(context.Context) (interface{}, error)
}

// ExecOpts provides additional options for additional processes running in a task
type ExecOpts struct {
	Spec *types.Any
	IO   IO
}

// ConsoleSize of a pty or windows terminal
type ConsoleSize struct {
	Width  uint32
	Height uint32
}

// Status is the runtime status of a task and/or process
type Status int

const (
	// CreatedStatus when a process has been created
	CreatedStatus Status = iota + 1
	// RunningStatus when a process is running
	RunningStatus
	// StoppedStatus when a process has stopped
	StoppedStatus
	// DeletedStatus when a process has been deleted
	DeletedStatus
	// PausedStatus when a process is paused
	PausedStatus
	// PausingStatus when a process is currently pausing
	PausingStatus
)

// State information for a process
type State struct {
	// Status is the current status of the container
	Status Status
	// Pid is the main process id for the container
	Pid uint32
	// ExitStatus of the process
	// Only valid if the Status is Stopped
	ExitStatus uint32
	// ExitedAt is the time at which the process exited
	// Only valid if the Status is Stopped
	ExitedAt time.Time
	Stdin    string
	Stdout   string
	Stderr   string
	Terminal bool
}

// ProcessInfo holds platform specific process information
type ProcessInfo struct {
	// Pid is the process ID
	Pid uint32
	// Info includes additional process information
	// Info varies by platform
	Info interface{}
}
