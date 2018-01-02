// +build windows

package windows

import (
	"context"
	"sync"
	"time"

	"github.com/Microsoft/hcsshim"
	eventsapi "github.com/containerd/containerd/api/services/events/v1"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/events"
	"github.com/containerd/containerd/runtime"
	"github.com/containerd/containerd/windows/hcsshimtypes"
	"github.com/containerd/typeurl"
	"github.com/gogo/protobuf/types"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"
)

type task struct {
	sync.Mutex

	id        string
	namespace string
	pid       uint32
	io        *pipeSet
	status    runtime.Status
	spec      *specs.Spec
	processes map[string]*process
	hyperV    bool

	publisher events.Publisher
	rwLayer   string

	pidPool           *pidPool
	hcsContainer      hcsshim.Container
	terminateDuration time.Duration
	servicing         bool
}

func (t *task) ID() string {
	return t.id
}

func (t *task) State(ctx context.Context) (runtime.State, error) {
	var (
		status     runtime.Status
		exitStatus uint32
		exitedAt   time.Time
	)

	if p := t.getProcess(t.id); p != nil {
		status = p.Status()
		exitStatus = p.exitCode
		exitedAt = p.exitTime
	} else {
		status = t.getStatus()
	}

	return runtime.State{
		Status:     status,
		Pid:        t.pid,
		Stdin:      t.io.src.Stdin,
		Stdout:     t.io.src.Stdout,
		Stderr:     t.io.src.Stderr,
		Terminal:   t.io.src.Terminal,
		ExitStatus: exitStatus,
		ExitedAt:   exitedAt,
	}, nil
}

func (t *task) Kill(ctx context.Context, signal uint32, all bool) error {
	p := t.getProcess(t.id)
	if p == nil {
		return errors.Wrapf(errdefs.ErrFailedPrecondition, "task is not running")
	}

	if p.Status() == runtime.StoppedStatus {
		return errors.Wrapf(errdefs.ErrNotFound, "process is stopped")
	}

	return p.Kill(ctx, signal, all)
}

func (t *task) ResizePty(ctx context.Context, size runtime.ConsoleSize) error {
	p := t.getProcess(t.id)
	if p == nil {
		return errors.Wrap(errdefs.ErrFailedPrecondition, "task not started")
	}

	return p.ResizePty(ctx, size)
}

func (t *task) CloseIO(ctx context.Context) error {
	p := t.getProcess(t.id)
	if p == nil {
		return errors.Wrap(errdefs.ErrFailedPrecondition, "task not started")
	}

	return p.hcs.CloseStdin()
}

func (t *task) Info() runtime.TaskInfo {
	return runtime.TaskInfo{
		ID:        t.id,
		Runtime:   pluginID,
		Namespace: t.namespace,
		// TODO(mlaventure): what about Spec? I think this could be removed from the info, the id is enough since it matches the one from the container
	}
}

func (t *task) Start(ctx context.Context) error {
	p := t.getProcess(t.id)
	if p == nil {
		panic("init process is missing")
	}

	if p.Status() != runtime.CreatedStatus {
		return errors.Wrap(errdefs.ErrFailedPrecondition, "process was already started")
	}

	if err := p.Start(ctx); err != nil {
		return err
	}
	t.publisher.Publish(ctx,
		runtime.TaskStartEventTopic,
		&eventsapi.TaskStart{
			ContainerID: t.id,
			Pid:         t.pid,
		})
	return nil
}

func (t *task) Pause(ctx context.Context) error {
	if t.hyperV {
		err := t.hcsContainer.Pause()
		if err == nil {
			t.Lock()
			t.status = runtime.PausedStatus
			t.Unlock()

			t.publisher.Publish(ctx,
				runtime.TaskPausedEventTopic,
				&eventsapi.TaskPaused{
					ContainerID: t.id,
				})
			return nil
		}
		return errors.Wrap(err, "hcsshim failed to pause task")
	}

	return errors.Wrap(errdefs.ErrFailedPrecondition, "not an hyperV task")
}

func (t *task) Resume(ctx context.Context) error {
	if t.hyperV {
		err := t.hcsContainer.Resume()
		if err == nil {
			t.Lock()
			t.status = runtime.RunningStatus
			t.Unlock()

			t.publisher.Publish(ctx,
				runtime.TaskResumedEventTopic,
				&eventsapi.TaskResumed{
					ContainerID: t.id,
				})
			return nil
		}
		return errors.Wrap(err, "hcsshim failed to resume task")
	}

	return errors.Wrap(errdefs.ErrFailedPrecondition, "not an hyperV task")
}

func (t *task) Exec(ctx context.Context, id string, opts runtime.ExecOpts) (runtime.Process, error) {
	if p := t.getProcess(t.id); p == nil {
		return nil, errors.Wrap(errdefs.ErrFailedPrecondition, "task not started")
	}

	if p := t.getProcess(id); p != nil {
		return nil, errors.Wrap(errdefs.ErrAlreadyExists, "id already in use")
	}

	s, err := typeurl.UnmarshalAny(opts.Spec)
	if err != nil {
		return nil, err
	}
	spec := s.(*specs.Process)
	if spec.Cwd == "" {
		spec.Cwd = t.spec.Process.Cwd
	}

	var pset *pipeSet
	if pset, err = newPipeSet(ctx, opts.IO); err != nil {
		return nil, err
	}

	conf := newWindowsProcessConfig(spec, pset)
	p, err := t.newProcess(ctx, id, conf, pset)
	if err != nil {
		return nil, err
	}

	t.publisher.Publish(ctx,
		runtime.TaskExecAddedEventTopic,
		&eventsapi.TaskExecAdded{
			ContainerID: t.id,
			ExecID:      id,
		})

	return p, nil
}

func (t *task) Pids(ctx context.Context) ([]runtime.ProcessInfo, error) {
	t.Lock()
	defer t.Unlock()

	var infoList []runtime.ProcessInfo
	hcsProcessList, err := t.hcsContainer.ProcessList()
	if err != nil {
		return nil, err
	}

	for _, process := range hcsProcessList {
		info, err := t.convertToProcessDetails(process)
		if err != nil {
			return nil, err
		}
		infoList = append(infoList, runtime.ProcessInfo{
			Pid:  process.ProcessId,
			Info: info,
		})
	}

	return infoList, nil
}

func (t *task) Checkpoint(_ context.Context, _ string, _ *types.Any) error {
	return errors.Wrap(errdefs.ErrUnavailable, "not supported")
}

func (t *task) DeleteProcess(ctx context.Context, id string) (*runtime.Exit, error) {
	if id == t.id {
		return nil, errors.Wrapf(errdefs.ErrInvalidArgument,
			"cannot delete init process")
	}
	if p := t.getProcess(id); p != nil {
		ec, ea, err := p.ExitCode()
		if err != nil {
			return nil, err
		}

		// If we never started the process close the pipes
		if p.Status() == runtime.CreatedStatus {
			p.io.Close()
			ea = time.Now()
		}

		t.removeProcess(id)
		return &runtime.Exit{
			Pid:       p.pid,
			Status:    ec,
			Timestamp: ea,
		}, nil
	}
	return nil, errors.Wrapf(errdefs.ErrNotFound, "no such process %s", id)
}

func (t *task) Update(ctx context.Context, resources *types.Any) error {
	return errors.Wrap(errdefs.ErrUnavailable, "not supported")
}

func (t *task) Process(ctx context.Context, id string) (p runtime.Process, err error) {
	p = t.getProcess(id)
	if p == nil {
		err = errors.Wrapf(errdefs.ErrNotFound, "no such process %d", id)
	}

	return p, err
}

func (t *task) Metrics(ctx context.Context) (interface{}, error) {
	return nil, errors.Wrap(errdefs.ErrUnavailable, "not supported")
}

func (t *task) Wait(ctx context.Context) (*runtime.Exit, error) {
	p := t.getProcess(t.id)
	if p == nil {
		return nil, errors.Wrapf(errdefs.ErrNotFound, "no such process %d", t.id)
	}
	return p.Wait(ctx)
}

func (t *task) newProcess(ctx context.Context, id string, conf *hcsshim.ProcessConfig, pset *pipeSet) (*process, error) {
	var (
		err error
		pid uint32
	)

	// If we fail, close the io right now
	defer func() {
		if err != nil {
			pset.Close()
		}
	}()

	t.Lock()
	if len(t.processes) == 0 {
		pid = t.pid
	} else {
		if pid, err = t.pidPool.Get(); err != nil {
			t.Unlock()
			return nil, err
		}
		defer func() {
			if err != nil {
				t.pidPool.Put(pid)
			}
		}()
	}
	wp := &process{
		id:     id,
		pid:    pid,
		io:     pset,
		task:   t,
		exitCh: make(chan struct{}),
		conf:   conf,
	}
	t.processes[id] = wp
	t.Unlock()

	return wp, nil
}

func (t *task) getProcess(id string) *process {
	t.Lock()
	p := t.processes[id]
	t.Unlock()

	return p
}

func (t *task) removeProcessNL(id string) {
	if p, ok := t.processes[id]; ok {
		if p.io != nil {
			p.io.Close()
		}
		t.pidPool.Put(p.pid)
		delete(t.processes, id)
	}
}

func (t *task) removeProcess(id string) {
	t.Lock()
	t.removeProcessNL(id)
	t.Unlock()
}

func (t *task) getStatus() runtime.Status {
	t.Lock()
	status := t.status
	t.Unlock()

	return status
}

// stop tries to shutdown the task.
// It will do so by first calling Shutdown on the hcsshim.Container and if
// that fails, by resorting to caling Terminate
func (t *task) stop(ctx context.Context) error {
	if err := t.hcsStop(ctx, t.hcsContainer.Shutdown); err != nil {
		return t.hcsStop(ctx, t.hcsContainer.Terminate)
	}
	t.hcsContainer.Close()
	return nil
}

func (t *task) hcsStop(ctx context.Context, stop func() error) error {
	err := stop()
	switch {
	case hcsshim.IsPending(err):
		err = t.hcsContainer.WaitTimeout(t.terminateDuration)
	case hcsshim.IsAlreadyStopped(err):
		err = nil
	}
	return err
}

func (t *task) cleanup() {
	t.Lock()
	for _, p := range t.processes {
		t.removeProcessNL(p.id)
	}
	removeLayer(context.Background(), t.rwLayer)
	t.Unlock()
}

// convertToProcessDetails converts a given hcsshim ProcessListItem to proto ProcessDetails
func (t *task) convertToProcessDetails(p hcsshim.ProcessListItem) (*hcsshimtypes.ProcessDetails, error) {
	return &hcsshimtypes.ProcessDetails{
		ImageName:                    p.ImageName,
		CreatedAt:                    p.CreateTimestamp,
		KernelTime_100Ns:             p.KernelTime100ns,
		MemoryCommitBytes:            p.MemoryCommitBytes,
		MemoryWorkingSetPrivateBytes: p.MemoryWorkingSetPrivateBytes,
		MemoryWorkingSetSharedBytes:  p.MemoryWorkingSetSharedBytes,
		ProcessID:                    p.ProcessId,
		UserTime_100Ns:               p.UserTime100ns,
	}, nil
}
