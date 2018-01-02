// +build linux

package linux

import (
	"context"

	"github.com/pkg/errors"
	"google.golang.org/grpc"

	"github.com/containerd/cgroups"
	"github.com/containerd/containerd/api/types/task"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/linux/shim/client"
	shim "github.com/containerd/containerd/linux/shim/v1"
	"github.com/containerd/containerd/runtime"
	"github.com/gogo/protobuf/types"
)

// Task on a linux based system
type Task struct {
	id        string
	pid       int
	shim      *client.Client
	namespace string
	cg        cgroups.Cgroup
	monitor   runtime.TaskMonitor
}

func newTask(id, namespace string, pid int, shim *client.Client, monitor runtime.TaskMonitor) (*Task, error) {
	var (
		err error
		cg  cgroups.Cgroup
	)
	if pid > 0 {
		cg, err = cgroups.Load(cgroups.V1, cgroups.PidPath(pid))
		if err != nil && err != cgroups.ErrCgroupDeleted {
			return nil, err
		}
	}
	return &Task{
		id:        id,
		pid:       pid,
		shim:      shim,
		namespace: namespace,
		cg:        cg,
		monitor:   monitor,
	}, nil
}

// ID of the task
func (t *Task) ID() string {
	return t.id
}

// Info returns task information about the runtime and namespace
func (t *Task) Info() runtime.TaskInfo {
	return runtime.TaskInfo{
		ID:        t.id,
		Runtime:   pluginID,
		Namespace: t.namespace,
	}
}

// Start the task
func (t *Task) Start(ctx context.Context) error {
	hasCgroup := t.cg != nil
	r, err := t.shim.Start(ctx, &shim.StartRequest{
		ID: t.id,
	})
	if err != nil {
		return errdefs.FromGRPC(err)
	}
	t.pid = int(r.Pid)
	if !hasCgroup {
		cg, err := cgroups.Load(cgroups.V1, cgroups.PidPath(t.pid))
		if err != nil {
			return err
		}
		t.cg = cg
		if err := t.monitor.Monitor(t); err != nil {
			return err
		}
	}
	return nil
}

// State returns runtime information for the task
func (t *Task) State(ctx context.Context) (runtime.State, error) {
	response, err := t.shim.State(ctx, &shim.StateRequest{
		ID: t.id,
	})
	if err != nil {
		if err != grpc.ErrServerStopped {
			return runtime.State{}, errdefs.FromGRPC(err)
		}
		return runtime.State{}, errdefs.ErrNotFound
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
		ExitedAt:   response.ExitedAt,
	}, nil
}

// Pause the task and all processes
func (t *Task) Pause(ctx context.Context) error {
	_, err := t.shim.Pause(ctx, empty)
	if err != nil {
		err = errdefs.FromGRPC(err)
	}
	return err
}

// Resume the task and all processes
func (t *Task) Resume(ctx context.Context) error {
	if _, err := t.shim.Resume(ctx, empty); err != nil {
		return errdefs.FromGRPC(err)
	}
	return nil
}

// Kill the task using the provided signal
//
// Optionally send the signal to all processes that are a child of the task
func (t *Task) Kill(ctx context.Context, signal uint32, all bool) error {
	if _, err := t.shim.Kill(ctx, &shim.KillRequest{
		ID:     t.id,
		Signal: signal,
		All:    all,
	}); err != nil {
		return errdefs.FromGRPC(err)
	}
	return nil
}

// Exec creates a new process inside the task
func (t *Task) Exec(ctx context.Context, id string, opts runtime.ExecOpts) (runtime.Process, error) {
	request := &shim.ExecProcessRequest{
		ID:       id,
		Stdin:    opts.IO.Stdin,
		Stdout:   opts.IO.Stdout,
		Stderr:   opts.IO.Stderr,
		Terminal: opts.IO.Terminal,
		Spec:     opts.Spec,
	}
	if _, err := t.shim.Exec(ctx, request); err != nil {
		return nil, errdefs.FromGRPC(err)
	}
	return &Process{
		id: id,
		t:  t,
	}, nil
}

// Pids returns all system level process ids running inside the task
func (t *Task) Pids(ctx context.Context) ([]runtime.ProcessInfo, error) {
	resp, err := t.shim.ListPids(ctx, &shim.ListPidsRequest{
		ID: t.id,
	})
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}
	var processList []runtime.ProcessInfo
	for _, p := range resp.Processes {
		processList = append(processList, runtime.ProcessInfo{
			Pid: p.Pid,
		})
	}
	return processList, nil
}

// ResizePty changes the side of the task's PTY to the provided width and height
func (t *Task) ResizePty(ctx context.Context, size runtime.ConsoleSize) error {
	_, err := t.shim.ResizePty(ctx, &shim.ResizePtyRequest{
		ID:     t.id,
		Width:  size.Width,
		Height: size.Height,
	})
	if err != nil {
		err = errdefs.FromGRPC(err)
	}
	return err
}

// CloseIO closes the provided IO on the task
func (t *Task) CloseIO(ctx context.Context) error {
	_, err := t.shim.CloseIO(ctx, &shim.CloseIORequest{
		ID:    t.id,
		Stdin: true,
	})
	if err != nil {
		err = errdefs.FromGRPC(err)
	}
	return err
}

// Checkpoint creates a system level dump of the task and process information that can be later restored
func (t *Task) Checkpoint(ctx context.Context, path string, options *types.Any) error {
	r := &shim.CheckpointTaskRequest{
		Path:    path,
		Options: options,
	}
	if _, err := t.shim.Checkpoint(ctx, r); err != nil {
		return errdefs.FromGRPC(err)
	}
	return nil
}

// DeleteProcess removes the provided process from the task and deletes all on disk state
func (t *Task) DeleteProcess(ctx context.Context, id string) (*runtime.Exit, error) {
	r, err := t.shim.DeleteProcess(ctx, &shim.DeleteProcessRequest{
		ID: id,
	})
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}
	return &runtime.Exit{
		Status:    r.ExitStatus,
		Timestamp: r.ExitedAt,
		Pid:       r.Pid,
	}, nil
}

// Update changes runtime information of a running task
func (t *Task) Update(ctx context.Context, resources *types.Any) error {
	if _, err := t.shim.Update(ctx, &shim.UpdateTaskRequest{
		Resources: resources,
	}); err != nil {
		return errdefs.FromGRPC(err)
	}
	return nil
}

// Process returns a specific process inside the task by the process id
func (t *Task) Process(ctx context.Context, id string) (runtime.Process, error) {
	// TODO: verify process exists for container
	return &Process{
		id: id,
		t:  t,
	}, nil
}

// Metrics returns runtime specific system level metric information for the task
func (t *Task) Metrics(ctx context.Context) (interface{}, error) {
	if t.cg == nil {
		return nil, errors.Wrap(errdefs.ErrNotFound, "cgroup does not exist")
	}
	stats, err := t.cg.Stat(cgroups.IgnoreNotExist)
	if err != nil {
		return nil, err
	}
	return stats, nil
}

// Cgroup returns the underlying cgroup for a linux task
func (t *Task) Cgroup() (cgroups.Cgroup, error) {
	if t.cg == nil {
		return nil, errors.Wrap(errdefs.ErrNotFound, "cgroup does not exist")
	}
	return t.cg, nil
}

// Wait for the task to exit returning the status and timestamp
func (t *Task) Wait(ctx context.Context) (*runtime.Exit, error) {
	r, err := t.shim.Wait(ctx, &shim.WaitRequest{
		ID: t.id,
	})
	if err != nil {
		return nil, err
	}
	return &runtime.Exit{
		Timestamp: r.ExitedAt,
		Status:    r.ExitStatus,
	}, nil
}
