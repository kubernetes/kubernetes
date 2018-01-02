package containerd

import (
	"context"
	"syscall"

	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/linux/runcopts"
	"github.com/containerd/containerd/mount"
)

// NewTaskOpts allows the caller to set options on a new task
type NewTaskOpts func(context.Context, *Client, *TaskInfo) error

// WithRootFS allows a task to be created without a snapshot being allocated to its container
func WithRootFS(mounts []mount.Mount) NewTaskOpts {
	return func(ctx context.Context, c *Client, ti *TaskInfo) error {
		ti.RootFS = mounts
		return nil
	}
}

// WithExit causes the task to exit after a successful checkpoint
func WithExit(r *CheckpointTaskInfo) error {
	r.Options = &runcopts.CheckpointOptions{
		Exit: true,
	}
	return nil
}

// WithCheckpointName sets the image name for the checkpoint
func WithCheckpointName(name string) CheckpointTaskOpts {
	return func(r *CheckpointTaskInfo) error {
		r.Name = name
		return nil
	}
}

// ProcessDeleteOpts allows the caller to set options for the deletion of a task
type ProcessDeleteOpts func(context.Context, Process) error

// WithProcessKill will forcefully kill and delete a process
func WithProcessKill(ctx context.Context, p Process) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	// ignore errors to wait and kill as we are forcefully killing
	// the process and don't care about the exit status
	s, err := p.Wait(ctx)
	if err != nil {
		return err
	}
	if err := p.Kill(ctx, syscall.SIGKILL, WithKillAll); err != nil {
		if errdefs.IsFailedPrecondition(err) || errdefs.IsNotFound(err) {
			return nil
		}
		return err
	}
	// wait for the process to fully stop before letting the rest of the deletion complete
	<-s
	return nil
}

// KillInfo contains information on how to process a Kill action
type KillInfo struct {
	// All kills all processes inside the task
	// only valid on tasks, ignored on processes
	All bool
}

// KillOpts allows options to be set for the killing of a process
type KillOpts func(context.Context, Process, *KillInfo) error

// WithKillAll kills all processes for a task
func WithKillAll(ctx context.Context, p Process, i *KillInfo) error {
	i.All = true
	return nil
}
