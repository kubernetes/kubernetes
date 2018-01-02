// +build linux

package linux

import (
	"context"

	"github.com/containerd/containerd/api/types/task"
	"github.com/containerd/containerd/errdefs"
	shim "github.com/containerd/containerd/linux/shim/v1"
	"github.com/containerd/containerd/runtime"
)

// Process implements a linux process
type Process struct {
	id string
	t  *Task
}

// ID of the process
func (p *Process) ID() string {
	return p.id
}

// Kill sends the provided signal to the underlying process
//
// Unable to kill all processes in the task using this method on a process
func (p *Process) Kill(ctx context.Context, signal uint32, _ bool) error {
	_, err := p.t.shim.Kill(ctx, &shim.KillRequest{
		Signal: signal,
		ID:     p.id,
	})
	if err != nil {
		return errdefs.FromGRPC(err)
	}
	return err
}

// State of process
func (p *Process) State(ctx context.Context) (runtime.State, error) {
	// use the container status for the status of the process
	response, err := p.t.shim.State(ctx, &shim.StateRequest{
		ID: p.id,
	})
	if err != nil {
		return runtime.State{}, errdefs.FromGRPC(err)
	}
	var status runtime.Status
	switch response.Status {
	case task.StatusCreated:
		status = runtime.CreatedStatus
	case task.StatusRunning:
		status = runtime.RunningStatus
	case task.StatusStopped:
		status = runtime.StoppedStatus
	case task.StatusPaused:
		status = runtime.PausedStatus
	case task.StatusPausing:
		status = runtime.PausingStatus
	}
	return runtime.State{
		Pid:        response.Pid,
		Status:     status,
		Stdin:      response.Stdin,
		Stdout:     response.Stdout,
		Stderr:     response.Stderr,
		Terminal:   response.Terminal,
		ExitStatus: response.ExitStatus,
	}, nil
}

// ResizePty changes the side of the process's PTY to the provided width and height
func (p *Process) ResizePty(ctx context.Context, size runtime.ConsoleSize) error {
	_, err := p.t.shim.ResizePty(ctx, &shim.ResizePtyRequest{
		ID:     p.id,
		Width:  size.Width,
		Height: size.Height,
	})
	if err != nil {
		err = errdefs.FromGRPC(err)
	}
	return err
}

// CloseIO closes the provided IO pipe for the process
func (p *Process) CloseIO(ctx context.Context) error {
	_, err := p.t.shim.CloseIO(ctx, &shim.CloseIORequest{
		ID:    p.id,
		Stdin: true,
	})
	if err != nil {
		return errdefs.FromGRPC(err)
	}
	return nil
}

// Start the process
func (p *Process) Start(ctx context.Context) error {
	_, err := p.t.shim.Start(ctx, &shim.StartRequest{
		ID: p.id,
	})
	if err != nil {
		return errdefs.FromGRPC(err)
	}
	return nil
}

// Wait on the process to exit and return the exit status and timestamp
func (p *Process) Wait(ctx context.Context) (*runtime.Exit, error) {
	r, err := p.t.shim.Wait(ctx, &shim.WaitRequest{
		ID: p.id,
	})
	if err != nil {
		return nil, err
	}
	return &runtime.Exit{
		Timestamp: r.ExitedAt,
		Status:    r.ExitStatus,
	}, nil
}
