package tasks

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	"github.com/boltdb/bolt"
	api "github.com/containerd/containerd/api/services/tasks/v1"
	"github.com/containerd/containerd/api/types"
	"github.com/containerd/containerd/api/types/task"
	"github.com/containerd/containerd/archive"
	"github.com/containerd/containerd/containers"
	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/events"
	"github.com/containerd/containerd/filters"
	"github.com/containerd/containerd/images"
	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/metadata"
	"github.com/containerd/containerd/mount"
	"github.com/containerd/containerd/plugin"
	"github.com/containerd/containerd/runtime"
	"github.com/containerd/typeurl"
	google_protobuf "github.com/golang/protobuf/ptypes/empty"
	"github.com/pkg/errors"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

var (
	_     = (api.TasksServer)(&service{})
	empty = &google_protobuf.Empty{}
)

func init() {
	plugin.Register(&plugin.Registration{
		Type: plugin.GRPCPlugin,
		ID:   "tasks",
		Requires: []plugin.Type{
			plugin.RuntimePlugin,
			plugin.MetadataPlugin,
		},
		InitFn: initFunc,
	})
}

func initFunc(ic *plugin.InitContext) (interface{}, error) {
	rt, err := ic.GetByType(plugin.RuntimePlugin)
	if err != nil {
		return nil, err
	}

	m, err := ic.Get(plugin.MetadataPlugin)
	if err != nil {
		return nil, err
	}
	cs := m.(*metadata.DB).ContentStore()
	runtimes := make(map[string]runtime.Runtime)
	for _, rr := range rt {
		ri, err := rr.Instance()
		if err != nil {
			log.G(ic.Context).WithError(err).Warn("could not load runtime instance due to initialization error")
			continue
		}
		r := ri.(runtime.Runtime)
		runtimes[r.ID()] = r
	}

	if len(runtimes) == 0 {
		return nil, errors.New("no runtimes available to create task service")
	}
	return &service{
		runtimes:  runtimes,
		db:        m.(*metadata.DB),
		store:     cs,
		publisher: ic.Events,
	}, nil
}

type service struct {
	runtimes  map[string]runtime.Runtime
	db        *metadata.DB
	store     content.Store
	publisher events.Publisher
}

func (s *service) Register(server *grpc.Server) error {
	api.RegisterTasksServer(server, s)
	return nil
}

func (s *service) Create(ctx context.Context, r *api.CreateTaskRequest) (*api.CreateTaskResponse, error) {
	var (
		checkpointPath string
		err            error
	)
	if r.Checkpoint != nil {
		checkpointPath, err = ioutil.TempDir("", "ctrd-checkpoint")
		if err != nil {
			return nil, err
		}
		if r.Checkpoint.MediaType != images.MediaTypeContainerd1Checkpoint {
			return nil, fmt.Errorf("unsupported checkpoint type %q", r.Checkpoint.MediaType)
		}
		reader, err := s.store.ReaderAt(ctx, r.Checkpoint.Digest)
		if err != nil {
			return nil, err
		}
		_, err = archive.Apply(ctx, checkpointPath, content.NewReader(reader))
		reader.Close()
		if err != nil {
			return nil, err
		}
	}

	container, err := s.getContainer(ctx, r.ContainerID)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	opts := runtime.CreateOpts{
		Spec: container.Spec,
		IO: runtime.IO{
			Stdin:    r.Stdin,
			Stdout:   r.Stdout,
			Stderr:   r.Stderr,
			Terminal: r.Terminal,
		},
		Checkpoint: checkpointPath,
		Options:    r.Options,
	}
	for _, m := range r.Rootfs {
		opts.Rootfs = append(opts.Rootfs, mount.Mount{
			Type:    m.Type,
			Source:  m.Source,
			Options: m.Options,
		})
	}
	runtime, err := s.getRuntime(container.Runtime.Name)
	if err != nil {
		return nil, err
	}
	c, err := runtime.Create(ctx, r.ContainerID, opts)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	state, err := c.State(ctx)
	if err != nil {
		log.G(ctx).Error(err)
	}

	return &api.CreateTaskResponse{
		ContainerID: r.ContainerID,
		Pid:         state.Pid,
	}, nil
}

func (s *service) Start(ctx context.Context, r *api.StartRequest) (*api.StartResponse, error) {
	t, err := s.getTask(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	p := runtime.Process(t)
	if r.ExecID != "" {
		if p, err = t.Process(ctx, r.ExecID); err != nil {
			return nil, errdefs.ToGRPC(err)
		}
	}
	if err := p.Start(ctx); err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	state, err := p.State(ctx)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return &api.StartResponse{
		Pid: state.Pid,
	}, nil
}

func (s *service) Delete(ctx context.Context, r *api.DeleteTaskRequest) (*api.DeleteResponse, error) {
	t, err := s.getTask(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	runtime, err := s.getRuntime(t.Info().Runtime)
	if err != nil {
		return nil, err
	}
	exit, err := runtime.Delete(ctx, t)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return &api.DeleteResponse{
		ExitStatus: exit.Status,
		ExitedAt:   exit.Timestamp,
		Pid:        exit.Pid,
	}, nil
}

func (s *service) DeleteProcess(ctx context.Context, r *api.DeleteProcessRequest) (*api.DeleteResponse, error) {
	t, err := s.getTask(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	exit, err := t.DeleteProcess(ctx, r.ExecID)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return &api.DeleteResponse{
		ID:         r.ExecID,
		ExitStatus: exit.Status,
		ExitedAt:   exit.Timestamp,
		Pid:        exit.Pid,
	}, nil
}

func processFromContainerd(ctx context.Context, p runtime.Process) (*task.Process, error) {
	state, err := p.State(ctx)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	var status task.Status
	switch state.Status {
	case runtime.CreatedStatus:
		status = task.StatusCreated
	case runtime.RunningStatus:
		status = task.StatusRunning
	case runtime.StoppedStatus:
		status = task.StatusStopped
	case runtime.PausedStatus:
		status = task.StatusPaused
	case runtime.PausingStatus:
		status = task.StatusPausing
	default:
		log.G(ctx).WithField("status", state.Status).Warn("unknown status")
	}
	return &task.Process{
		ID:         p.ID(),
		Pid:        state.Pid,
		Status:     status,
		Stdin:      state.Stdin,
		Stdout:     state.Stdout,
		Stderr:     state.Stderr,
		Terminal:   state.Terminal,
		ExitStatus: state.ExitStatus,
		ExitedAt:   state.ExitedAt,
	}, nil
}

func (s *service) Get(ctx context.Context, r *api.GetRequest) (*api.GetResponse, error) {
	task, err := s.getTask(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	p := runtime.Process(task)
	if r.ExecID != "" {
		if p, err = task.Process(ctx, r.ExecID); err != nil {
			return nil, errdefs.ToGRPC(err)
		}
	}
	t, err := processFromContainerd(ctx, p)
	if err != nil {
		return nil, err
	}
	return &api.GetResponse{
		Process: t,
	}, nil
}

func (s *service) List(ctx context.Context, r *api.ListTasksRequest) (*api.ListTasksResponse, error) {
	resp := &api.ListTasksResponse{}
	for _, r := range s.runtimes {
		tasks, err := r.Tasks(ctx)
		if err != nil {
			return nil, errdefs.ToGRPC(err)
		}
		addTasks(ctx, resp, tasks)
	}
	return resp, nil
}

func addTasks(ctx context.Context, r *api.ListTasksResponse, tasks []runtime.Task) {
	for _, t := range tasks {
		tt, err := processFromContainerd(ctx, t)
		if err != nil {
			log.G(ctx).WithError(err).WithField("id", t.ID()).Error("converting task to protobuf")
			continue
		}
		r.Tasks = append(r.Tasks, tt)
	}
}

func (s *service) Pause(ctx context.Context, r *api.PauseTaskRequest) (*google_protobuf.Empty, error) {
	t, err := s.getTask(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	err = t.Pause(ctx)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return empty, nil
}

func (s *service) Resume(ctx context.Context, r *api.ResumeTaskRequest) (*google_protobuf.Empty, error) {
	t, err := s.getTask(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	err = t.Resume(ctx)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return empty, nil
}

func (s *service) Kill(ctx context.Context, r *api.KillRequest) (*google_protobuf.Empty, error) {
	t, err := s.getTask(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	p := runtime.Process(t)
	if r.ExecID != "" {
		if p, err = t.Process(ctx, r.ExecID); err != nil {
			return nil, errdefs.ToGRPC(err)
		}
	}
	if err := p.Kill(ctx, r.Signal, r.All); err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return empty, nil
}

func (s *service) ListPids(ctx context.Context, r *api.ListPidsRequest) (*api.ListPidsResponse, error) {
	t, err := s.getTask(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	processList, err := t.Pids(ctx)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	var processes []*task.ProcessInfo
	for _, p := range processList {
		processInfo := task.ProcessInfo{
			Pid: p.Pid,
		}
		if p.Info != nil {
			a, err := typeurl.MarshalAny(p.Info)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to marshal process %d info", p.Pid)
			}
			processInfo.Info = a
		}
		processes = append(processes, &processInfo)
	}
	return &api.ListPidsResponse{
		Processes: processes,
	}, nil
}

func (s *service) Exec(ctx context.Context, r *api.ExecProcessRequest) (*google_protobuf.Empty, error) {
	if r.ExecID == "" {
		return nil, grpc.Errorf(codes.InvalidArgument, "exec id cannot be empty")
	}
	t, err := s.getTask(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	if _, err := t.Exec(ctx, r.ExecID, runtime.ExecOpts{
		Spec: r.Spec,
		IO: runtime.IO{
			Stdin:    r.Stdin,
			Stdout:   r.Stdout,
			Stderr:   r.Stderr,
			Terminal: r.Terminal,
		},
	}); err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return empty, nil
}

func (s *service) ResizePty(ctx context.Context, r *api.ResizePtyRequest) (*google_protobuf.Empty, error) {
	t, err := s.getTask(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	p := runtime.Process(t)
	if r.ExecID != "" {
		if p, err = t.Process(ctx, r.ExecID); err != nil {
			return nil, errdefs.ToGRPC(err)
		}
	}
	if err := p.ResizePty(ctx, runtime.ConsoleSize{
		Width:  r.Width,
		Height: r.Height,
	}); err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return empty, nil
}

func (s *service) CloseIO(ctx context.Context, r *api.CloseIORequest) (*google_protobuf.Empty, error) {
	t, err := s.getTask(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	p := runtime.Process(t)
	if r.ExecID != "" {
		if p, err = t.Process(ctx, r.ExecID); err != nil {
			return nil, errdefs.ToGRPC(err)
		}
	}
	if r.Stdin {
		if err := p.CloseIO(ctx); err != nil {
			return nil, err
		}
	}
	return empty, nil
}

func (s *service) Checkpoint(ctx context.Context, r *api.CheckpointTaskRequest) (*api.CheckpointTaskResponse, error) {
	container, err := s.getContainer(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	t, err := s.getTaskFromContainer(ctx, container)
	if err != nil {
		return nil, err
	}
	image, err := ioutil.TempDir("", "ctd-checkpoint")
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	defer os.RemoveAll(image)
	if err := t.Checkpoint(ctx, image, r.Options); err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	// write checkpoint to the content store
	tar := archive.Diff(ctx, "", image)
	cp, err := s.writeContent(ctx, images.MediaTypeContainerd1Checkpoint, image, tar)
	// close tar first after write
	if err := tar.Close(); err != nil {
		return nil, err
	}
	if err != nil {
		return nil, err
	}
	// write the config to the content store
	data, err := container.Spec.Marshal()
	if err != nil {
		return nil, err
	}
	spec := bytes.NewReader(data)
	specD, err := s.writeContent(ctx, images.MediaTypeContainerd1CheckpointConfig, filepath.Join(image, "spec"), spec)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return &api.CheckpointTaskResponse{
		Descriptors: []*types.Descriptor{
			cp,
			specD,
		},
	}, nil
}

func (s *service) Update(ctx context.Context, r *api.UpdateTaskRequest) (*google_protobuf.Empty, error) {
	t, err := s.getTask(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	if err := t.Update(ctx, r.Resources); err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return empty, nil
}

func (s *service) Metrics(ctx context.Context, r *api.MetricsRequest) (*api.MetricsResponse, error) {
	filter, err := filters.ParseAll(r.Filters...)
	if err != nil {
		return nil, err
	}
	var resp api.MetricsResponse
	for _, r := range s.runtimes {
		tasks, err := r.Tasks(ctx)
		if err != nil {
			return nil, err
		}
		getTasksMetrics(ctx, filter, tasks, &resp)
	}
	return &resp, nil
}

func (s *service) Wait(ctx context.Context, r *api.WaitRequest) (*api.WaitResponse, error) {
	t, err := s.getTask(ctx, r.ContainerID)
	if err != nil {
		return nil, err
	}
	p := runtime.Process(t)
	if r.ExecID != "" {
		if p, err = t.Process(ctx, r.ExecID); err != nil {
			return nil, errdefs.ToGRPC(err)
		}
	}
	exit, err := p.Wait(ctx)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return &api.WaitResponse{
		ExitStatus: exit.Status,
		ExitedAt:   exit.Timestamp,
	}, nil
}

func getTasksMetrics(ctx context.Context, filter filters.Filter, tasks []runtime.Task, r *api.MetricsResponse) {
	for _, tk := range tasks {
		if !filter.Match(filters.AdapterFunc(func(fieldpath []string) (string, bool) {
			t := tk
			switch fieldpath[0] {
			case "id":
				return t.ID(), true
			case "namespace":
				return t.Info().Namespace, true
			case "runtime":
				return t.Info().Runtime, true
			}
			return "", false
		})) {
			continue
		}

		collected := time.Now()
		metrics, err := tk.Metrics(ctx)
		if err != nil {
			if !errdefs.IsNotFound(err) {
				log.G(ctx).WithError(err).Errorf("collecting metrics for %s", tk.ID())
			}
			continue
		}
		data, err := typeurl.MarshalAny(metrics)
		if err != nil {
			log.G(ctx).WithError(err).Errorf("marshal metrics for %s", tk.ID())
			continue
		}
		r.Metrics = append(r.Metrics, &types.Metric{
			ID:        tk.ID(),
			Timestamp: collected,
			Data:      data,
		})
	}
}

func (s *service) writeContent(ctx context.Context, mediaType, ref string, r io.Reader) (*types.Descriptor, error) {
	writer, err := s.store.Writer(ctx, ref, 0, "")
	if err != nil {
		return nil, err
	}
	defer writer.Close()
	size, err := io.Copy(writer, r)
	if err != nil {
		return nil, err
	}
	if err := writer.Commit(ctx, 0, ""); err != nil {
		return nil, err
	}
	return &types.Descriptor{
		MediaType: mediaType,
		Digest:    writer.Digest(),
		Size_:     size,
	}, nil
}

func (s *service) getContainer(ctx context.Context, id string) (*containers.Container, error) {
	var container containers.Container
	if err := s.db.View(func(tx *bolt.Tx) error {
		store := metadata.NewContainerStore(tx)
		var err error
		container, err = store.Get(ctx, id)
		return err
	}); err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return &container, nil
}

func (s *service) getTask(ctx context.Context, id string) (runtime.Task, error) {
	container, err := s.getContainer(ctx, id)
	if err != nil {
		return nil, err
	}
	return s.getTaskFromContainer(ctx, container)
}

func (s *service) getTaskFromContainer(ctx context.Context, container *containers.Container) (runtime.Task, error) {
	runtime, err := s.getRuntime(container.Runtime.Name)
	if err != nil {
		return nil, errdefs.ToGRPCf(err, "runtime for task %s", container.Runtime.Name)
	}
	t, err := runtime.Get(ctx, container.ID)
	if err != nil {
		return nil, grpc.Errorf(codes.NotFound, "task %v not found", container.ID)
	}
	return t, nil
}

func (s *service) getRuntime(name string) (runtime.Runtime, error) {
	runtime, ok := s.runtimes[name]
	if !ok {
		return nil, grpc.Errorf(codes.NotFound, "unknown runtime %q", name)
	}
	return runtime, nil
}
