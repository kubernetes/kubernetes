// +build !windows

package shim

import (
	"context"
	"io"
	"time"

	"github.com/containerd/console"
	"github.com/pkg/errors"
)

type stdio struct {
	stdin    string
	stdout   string
	stderr   string
	terminal bool
}

func (s stdio) isNull() bool {
	return s.stdin == "" && s.stdout == "" && s.stderr == ""
}

type process interface {
	processState

	// ID returns the id for the process
	ID() string
	// Pid returns the pid for the process
	Pid() int
	// ExitStatus returns the exit status
	ExitStatus() int
	// ExitedAt is the time the process exited
	ExitedAt() time.Time
	// Stdin returns the process STDIN
	Stdin() io.Closer
	// Stdio returns io information for the container
	Stdio() stdio
	// Status returns the process status
	Status(context.Context) (string, error)
	// Wait blocks until the process has exited
	Wait()
}

type processState interface {
	// Resize resizes the process console
	Resize(ws console.WinSize) error
	// Start execution of the process
	Start(context.Context) error
	// Delete deletes the process and its resourcess
	Delete(context.Context) error
	// Kill kills the process
	Kill(context.Context, uint32, bool) error
	// SetExited sets the exit status for the process
	SetExited(status int)
}

func stateName(v interface{}) string {
	switch v.(type) {
	case *runningState, *execRunningState:
		return "running"
	case *createdState, *execCreatedState, *createdCheckpointState:
		return "created"
	case *pausedState:
		return "paused"
	case *deletedState:
		return "deleted"
	case *stoppedState:
		return "stopped"
	}
	panic(errors.Errorf("invalid state %v", v))
}
