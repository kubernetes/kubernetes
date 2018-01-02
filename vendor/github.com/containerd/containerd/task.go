package containerd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	goruntime "runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/containerd/containerd/api/services/tasks/v1"
	"github.com/containerd/containerd/api/types"
	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/diff"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/images"
	"github.com/containerd/containerd/mount"
	"github.com/containerd/containerd/plugin"
	"github.com/containerd/containerd/rootfs"
	"github.com/containerd/typeurl"
	google_protobuf "github.com/gogo/protobuf/types"
	digest "github.com/opencontainers/go-digest"
	"github.com/opencontainers/image-spec/specs-go/v1"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"
)

// UnknownExitStatus is returned when containerd is unable to
// determine the exit status of a process. This can happen if the process never starts
// or if an error was encountered when obtaining the exit status, it is set to 255.
const UnknownExitStatus = 255

const (
	checkpointDateFormat = "01-02-2006-15:04:05"
	checkpointNameFormat = "containerd.io/checkpoint/%s:%s"
)

// Status returns process status and exit information
type Status struct {
	// Status of the process
	Status ProcessStatus
	// ExitStatus returned by the process
	ExitStatus uint32
	// ExitedTime is the time at which the process died
	ExitTime time.Time
}

// ProcessInfo provides platform specific process information
type ProcessInfo struct {
	// Pid is the process ID
	Pid uint32
	// Info includes additional process information
	// Info varies by platform
	Info *google_protobuf.Any
}

// ProcessStatus returns a human readable status for the Process representing its current status
type ProcessStatus string

const (
	// Running indicates the process is currently executing
	Running ProcessStatus = "running"
	// Created indicates the process has been created within containerd but the
	// user's defined process has not started
	Created ProcessStatus = "created"
	// Stopped indicates that the process has ran and exited
	Stopped ProcessStatus = "stopped"
	// Paused indicates that the process is currently paused
	Paused ProcessStatus = "paused"
	// Pausing indicates that the process is currently switching from a
	// running state into a paused state
	Pausing ProcessStatus = "pausing"
	// Unknown indicates that we could not determine the status from the runtime
	Unknown ProcessStatus = "unknown"
)

// IOCloseInfo allows specific io pipes to be closed on a process
type IOCloseInfo struct {
	Stdin bool
}

// IOCloserOpts allows the caller to set specific pipes as closed on a process
type IOCloserOpts func(*IOCloseInfo)

// WithStdinCloser closes the stdin of a process
func WithStdinCloser(r *IOCloseInfo) {
	r.Stdin = true
}

// CheckpointTaskInfo allows specific checkpoint information to be set for the task
type CheckpointTaskInfo struct {
	Name string
	// ParentCheckpoint is the digest of a parent checkpoint
	ParentCheckpoint digest.Digest
	// Options hold runtime specific settings for checkpointing a task
	Options interface{}
}

// CheckpointTaskOpts allows the caller to set checkpoint options
type CheckpointTaskOpts func(*CheckpointTaskInfo) error

// TaskInfo sets options for task creation
type TaskInfo struct {
	// Checkpoint is the Descriptor for an existing checkpoint that can be used
	// to restore a task's runtime and memory state
	Checkpoint *types.Descriptor
	// RootFS is a list of mounts to use as the task's root filesystem
	RootFS []mount.Mount
	// Options hold runtime specific settings for task creation
	Options interface{}
}

// Task is the executable object within containerd
type Task interface {
	Process

	// Pause suspends the execution of the task
	Pause(context.Context) error
	// Resume the execution of the task
	Resume(context.Context) error
	// Exec creates a new process inside the task
	Exec(context.Context, string, *specs.Process, IOCreation) (Process, error)
	// Pids returns a list of system specific process ids inside the task
	Pids(context.Context) ([]ProcessInfo, error)
	// Checkpoint serializes the runtime and memory information of a task into an
	// OCI Index that can be push and pulled from a remote resource.
	//
	// Additional software like CRIU maybe required to checkpoint and restore tasks
	Checkpoint(context.Context, ...CheckpointTaskOpts) (Image, error)
	// Update modifies executing tasks with updated settings
	Update(context.Context, ...UpdateTaskOpts) error
	// LoadProcess loads a previously created exec'd process
	LoadProcess(context.Context, string, IOAttach) (Process, error)
	// Metrics returns task metrics for runtime specific metrics
	//
	// The metric types are generic to containerd and change depending on the runtime
	// For the built in Linux runtime, github.com/containerd/cgroups.Metrics
	// are returned in protobuf format
	Metrics(context.Context) (*types.Metric, error)
}

var _ = (Task)(&task{})

type task struct {
	client *Client

	io  IO
	id  string
	pid uint32

	mu sync.Mutex
}

// Pid returns the pid or process id for the task
func (t *task) Pid() uint32 {
	return t.pid
}

func (t *task) Start(ctx context.Context) error {
	r, err := t.client.TaskService().Start(ctx, &tasks.StartRequest{
		ContainerID: t.id,
	})
	if err != nil {
		t.io.Close()
		return errdefs.FromGRPC(err)
	}
	t.pid = r.Pid
	return nil
}

func (t *task) Kill(ctx context.Context, s syscall.Signal, opts ...KillOpts) error {
	var i KillInfo
	for _, o := range opts {
		if err := o(ctx, t, &i); err != nil {
			return err
		}
	}
	_, err := t.client.TaskService().Kill(ctx, &tasks.KillRequest{
		Signal:      uint32(s),
		ContainerID: t.id,
		All:         i.All,
	})
	if err != nil {
		return errdefs.FromGRPC(err)
	}
	return nil
}

func (t *task) Pause(ctx context.Context) error {
	_, err := t.client.TaskService().Pause(ctx, &tasks.PauseTaskRequest{
		ContainerID: t.id,
	})
	return errdefs.FromGRPC(err)
}

func (t *task) Resume(ctx context.Context) error {
	_, err := t.client.TaskService().Resume(ctx, &tasks.ResumeTaskRequest{
		ContainerID: t.id,
	})
	return errdefs.FromGRPC(err)
}

func (t *task) Status(ctx context.Context) (Status, error) {
	r, err := t.client.TaskService().Get(ctx, &tasks.GetRequest{
		ContainerID: t.id,
	})
	if err != nil {
		return Status{}, errdefs.FromGRPC(err)
	}
	return Status{
		Status:     ProcessStatus(strings.ToLower(r.Process.Status.String())),
		ExitStatus: r.Process.ExitStatus,
		ExitTime:   r.Process.ExitedAt,
	}, nil
}

func (t *task) Wait(ctx context.Context) (<-chan ExitStatus, error) {
	c := make(chan ExitStatus, 1)
	go func() {
		defer close(c)
		r, err := t.client.TaskService().Wait(ctx, &tasks.WaitRequest{
			ContainerID: t.id,
		})
		if err != nil {
			c <- ExitStatus{
				code: UnknownExitStatus,
				err:  err,
			}
			return
		}
		c <- ExitStatus{
			code:     r.ExitStatus,
			exitedAt: r.ExitedAt,
		}
	}()
	return c, nil
}

// Delete deletes the task and its runtime state
// it returns the exit status of the task and any errors that were encountered
// during cleanup
func (t *task) Delete(ctx context.Context, opts ...ProcessDeleteOpts) (*ExitStatus, error) {
	for _, o := range opts {
		if err := o(ctx, t); err != nil {
			return nil, err
		}
	}
	status, err := t.Status(ctx)
	if err != nil && errdefs.IsNotFound(err) {
		return nil, err
	}
	switch status.Status {
	case Stopped, Unknown, "":
	case Created:
		if t.client.runtime == fmt.Sprintf("%s.%s", plugin.RuntimePlugin, "windows") {
			// On windows Created is akin to Stopped
			break
		}
		fallthrough
	default:
		return nil, errors.Wrapf(errdefs.ErrFailedPrecondition, "task must be stopped before deletion: %s", status.Status)
	}
	if t.io != nil {
		t.io.Cancel()
		t.io.Wait()
		t.io.Close()
	}
	r, err := t.client.TaskService().Delete(ctx, &tasks.DeleteTaskRequest{
		ContainerID: t.id,
	})
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}
	return &ExitStatus{code: r.ExitStatus, exitedAt: r.ExitedAt}, nil
}

func (t *task) Exec(ctx context.Context, id string, spec *specs.Process, ioCreate IOCreation) (Process, error) {
	if id == "" {
		return nil, errors.Wrapf(errdefs.ErrInvalidArgument, "exec id must not be empty")
	}
	i, err := ioCreate(id)
	if err != nil {
		return nil, err
	}
	any, err := typeurl.MarshalAny(spec)
	if err != nil {
		return nil, err
	}
	cfg := i.Config()
	request := &tasks.ExecProcessRequest{
		ContainerID: t.id,
		ExecID:      id,
		Terminal:    cfg.Terminal,
		Stdin:       cfg.Stdin,
		Stdout:      cfg.Stdout,
		Stderr:      cfg.Stderr,
		Spec:        any,
	}
	if _, err := t.client.TaskService().Exec(ctx, request); err != nil {
		i.Cancel()
		i.Wait()
		i.Close()
		return nil, errdefs.FromGRPC(err)
	}
	return &process{
		id:   id,
		task: t,
		io:   i,
	}, nil
}

func (t *task) Pids(ctx context.Context) ([]ProcessInfo, error) {
	response, err := t.client.TaskService().ListPids(ctx, &tasks.ListPidsRequest{
		ContainerID: t.id,
	})
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}
	var processList []ProcessInfo
	for _, p := range response.Processes {
		processList = append(processList, ProcessInfo{
			Pid:  p.Pid,
			Info: p.Info,
		})
	}
	return processList, nil
}

func (t *task) CloseIO(ctx context.Context, opts ...IOCloserOpts) error {
	r := &tasks.CloseIORequest{
		ContainerID: t.id,
	}
	var i IOCloseInfo
	for _, o := range opts {
		o(&i)
	}
	r.Stdin = i.Stdin
	_, err := t.client.TaskService().CloseIO(ctx, r)
	return errdefs.FromGRPC(err)
}

func (t *task) IO() IO {
	return t.io
}

func (t *task) Resize(ctx context.Context, w, h uint32) error {
	_, err := t.client.TaskService().ResizePty(ctx, &tasks.ResizePtyRequest{
		ContainerID: t.id,
		Width:       w,
		Height:      h,
	})
	return errdefs.FromGRPC(err)
}

func (t *task) Checkpoint(ctx context.Context, opts ...CheckpointTaskOpts) (Image, error) {
	ctx, done, err := t.client.withLease(ctx)
	if err != nil {
		return nil, err
	}
	defer done()

	request := &tasks.CheckpointTaskRequest{
		ContainerID: t.id,
	}
	var i CheckpointTaskInfo
	for _, o := range opts {
		if err := o(&i); err != nil {
			return nil, err
		}
	}
	// set a default name
	if i.Name == "" {
		i.Name = fmt.Sprintf(checkpointNameFormat, t.id, time.Now().Format(checkpointDateFormat))
	}
	request.ParentCheckpoint = i.ParentCheckpoint
	if i.Options != nil {
		any, err := typeurl.MarshalAny(i.Options)
		if err != nil {
			return nil, err
		}
		request.Options = any
	}
	// make sure we pause it and resume after all other filesystem operations are completed
	if err := t.Pause(ctx); err != nil {
		return nil, err
	}
	defer t.Resume(ctx)
	cr, err := t.client.ContainerService().Get(ctx, t.id)
	if err != nil {
		return nil, err
	}
	index := v1.Index{
		Annotations: make(map[string]string),
	}
	if err := t.checkpointTask(ctx, &index, request); err != nil {
		return nil, err
	}
	if cr.Image != "" {
		if err := t.checkpointImage(ctx, &index, cr.Image); err != nil {
			return nil, err
		}
		index.Annotations["image.name"] = cr.Image
	}
	if cr.SnapshotKey != "" {
		if err := t.checkpointRWSnapshot(ctx, &index, cr.Snapshotter, cr.SnapshotKey); err != nil {
			return nil, err
		}
	}
	desc, err := t.writeIndex(ctx, &index)
	if err != nil {
		return nil, err
	}
	im := images.Image{
		Name:   i.Name,
		Target: desc,
		Labels: map[string]string{
			"containerd.io/checkpoint": "true",
		},
	}
	if im, err = t.client.ImageService().Create(ctx, im); err != nil {
		return nil, err
	}
	return &image{
		client: t.client,
		i:      im,
	}, nil
}

// UpdateTaskInfo allows updated specific settings to be changed on a task
type UpdateTaskInfo struct {
	// Resources updates a tasks resource constraints
	Resources interface{}
}

// UpdateTaskOpts allows a caller to update task settings
type UpdateTaskOpts func(context.Context, *Client, *UpdateTaskInfo) error

func (t *task) Update(ctx context.Context, opts ...UpdateTaskOpts) error {
	request := &tasks.UpdateTaskRequest{
		ContainerID: t.id,
	}
	var i UpdateTaskInfo
	for _, o := range opts {
		if err := o(ctx, t.client, &i); err != nil {
			return err
		}
	}
	if i.Resources != nil {
		any, err := typeurl.MarshalAny(i.Resources)
		if err != nil {
			return err
		}
		request.Resources = any
	}
	_, err := t.client.TaskService().Update(ctx, request)
	return errdefs.FromGRPC(err)
}

func (t *task) LoadProcess(ctx context.Context, id string, ioAttach IOAttach) (Process, error) {
	response, err := t.client.TaskService().Get(ctx, &tasks.GetRequest{
		ContainerID: t.id,
		ExecID:      id,
	})
	if err != nil {
		err = errdefs.FromGRPC(err)
		if errdefs.IsNotFound(err) {
			return nil, errors.Wrapf(err, "no running process found")
		}
		return nil, err
	}
	var i IO
	if ioAttach != nil {
		if i, err = attachExistingIO(response, ioAttach); err != nil {
			return nil, err
		}
	}
	return &process{
		id:   id,
		task: t,
		io:   i,
	}, nil
}

func (t *task) Metrics(ctx context.Context) (*types.Metric, error) {
	response, err := t.client.TaskService().Metrics(ctx, &tasks.MetricsRequest{
		Filters: []string{
			"id==" + t.id,
		},
	})
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}

	if response.Metrics == nil {
		_, err := t.Status(ctx)
		if err != nil && errdefs.IsNotFound(err) {
			return nil, err
		}
		return nil, errors.New("no metrics received")
	}

	return response.Metrics[0], nil
}

func (t *task) checkpointTask(ctx context.Context, index *v1.Index, request *tasks.CheckpointTaskRequest) error {
	response, err := t.client.TaskService().Checkpoint(ctx, request)
	if err != nil {
		return errdefs.FromGRPC(err)
	}
	// add the checkpoint descriptors to the index
	for _, d := range response.Descriptors {
		index.Manifests = append(index.Manifests, v1.Descriptor{
			MediaType: d.MediaType,
			Size:      d.Size_,
			Digest:    d.Digest,
			Platform: &v1.Platform{
				OS:           goruntime.GOOS,
				Architecture: goruntime.GOARCH,
			},
		})
	}
	return nil
}

func (t *task) checkpointRWSnapshot(ctx context.Context, index *v1.Index, snapshotterName string, id string) error {
	opts := []diff.Opt{
		diff.WithReference(fmt.Sprintf("checkpoint-rw-%s", id)),
	}
	rw, err := rootfs.Diff(ctx, id, t.client.SnapshotService(snapshotterName), t.client.DiffService(), opts...)
	if err != nil {
		return err
	}
	rw.Platform = &v1.Platform{
		OS:           goruntime.GOOS,
		Architecture: goruntime.GOARCH,
	}
	index.Manifests = append(index.Manifests, rw)
	return nil
}

func (t *task) checkpointImage(ctx context.Context, index *v1.Index, image string) error {
	if image == "" {
		return fmt.Errorf("cannot checkpoint image with empty name")
	}
	ir, err := t.client.ImageService().Get(ctx, image)
	if err != nil {
		return err
	}
	index.Manifests = append(index.Manifests, ir.Target)
	return nil
}

func (t *task) writeIndex(ctx context.Context, index *v1.Index) (d v1.Descriptor, err error) {
	labels := map[string]string{}
	for i, m := range index.Manifests {
		labels[fmt.Sprintf("containerd.io/gc.ref.content.%d", i)] = m.Digest.String()
	}
	buf := bytes.NewBuffer(nil)
	if err := json.NewEncoder(buf).Encode(index); err != nil {
		return v1.Descriptor{}, err
	}
	return writeContent(ctx, t.client.ContentStore(), v1.MediaTypeImageIndex, t.id, buf, content.WithLabels(labels))
}

func writeContent(ctx context.Context, store content.Store, mediaType, ref string, r io.Reader, opts ...content.Opt) (d v1.Descriptor, err error) {
	writer, err := store.Writer(ctx, ref, 0, "")
	if err != nil {
		return d, err
	}
	defer writer.Close()
	size, err := io.Copy(writer, r)
	if err != nil {
		return d, err
	}
	if err := writer.Commit(ctx, size, "", opts...); err != nil {
		return d, err
	}
	return v1.Descriptor{
		MediaType: mediaType,
		Digest:    writer.Digest(),
		Size:      size,
	}, nil
}
