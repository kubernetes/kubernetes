// +build !windows

package shim

import (
	"fmt"
	"os"
	"sync"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"

	"github.com/containerd/console"
	eventsapi "github.com/containerd/containerd/api/services/events/v1"
	"github.com/containerd/containerd/api/types/task"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/events"
	shimapi "github.com/containerd/containerd/linux/shim/v1"
	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/namespaces"
	"github.com/containerd/containerd/reaper"
	"github.com/containerd/containerd/runtime"
	runc "github.com/containerd/go-runc"
	google_protobuf "github.com/golang/protobuf/ptypes/empty"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

var empty = &google_protobuf.Empty{}

// RuncRoot is the path to the root runc state directory
const RuncRoot = "/run/containerd/runc"

// Config contains shim specific configuration
type Config struct {
	Path          string
	Namespace     string
	WorkDir       string
	Criu          string
	RuntimeRoot   string
	SystemdCgroup bool
}

// NewService returns a new shim service that can be used via GRPC
func NewService(config Config, publisher events.Publisher) (*Service, error) {
	if config.Namespace == "" {
		return nil, fmt.Errorf("shim namespace cannot be empty")
	}
	context := namespaces.WithNamespace(context.Background(), config.Namespace)
	context = log.WithLogger(context, logrus.WithFields(logrus.Fields{
		"namespace": config.Namespace,
		"path":      config.Path,
		"pid":       os.Getpid(),
	}))
	s := &Service{
		config:    config,
		context:   context,
		processes: make(map[string]process),
		events:    make(chan interface{}, 128),
		ec:        reaper.Default.Subscribe(),
	}
	go s.processExits()
	if err := s.initPlatform(); err != nil {
		return nil, errors.Wrap(err, "failed to initialized platform behavior")
	}
	go s.forward(publisher)
	return s, nil
}

// platform handles platform-specific behavior that may differs across
// platform implementations
type platform interface {
	copyConsole(ctx context.Context, console console.Console, stdin, stdout, stderr string, wg, cwg *sync.WaitGroup) (console.Console, error)
	shutdownConsole(ctx context.Context, console console.Console) error
	close() error
}

// Service is the shim implementation of a remote shim over GRPC
type Service struct {
	mu sync.Mutex

	config    Config
	context   context.Context
	processes map[string]process
	events    chan interface{}
	platform  platform
	ec        chan runc.Exit

	// Filled by Create()
	id     string
	bundle string
}

// Create a new initial process and container with the underlying OCI runtime
func (s *Service) Create(ctx context.Context, r *shimapi.CreateTaskRequest) (*shimapi.CreateTaskResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	process, err := s.newInitProcess(ctx, r)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	// save the main task id and bundle to the shim for additional requests
	s.id = r.ID
	s.bundle = r.Bundle
	pid := process.Pid()
	s.processes[r.ID] = process
	s.events <- &eventsapi.TaskCreate{
		ContainerID: r.ID,
		Bundle:      r.Bundle,
		Rootfs:      r.Rootfs,
		IO: &eventsapi.TaskIO{
			Stdin:    r.Stdin,
			Stdout:   r.Stdout,
			Stderr:   r.Stderr,
			Terminal: r.Terminal,
		},
		Checkpoint: r.Checkpoint,
		Pid:        uint32(pid),
	}
	return &shimapi.CreateTaskResponse{
		Pid: uint32(pid),
	}, nil
}

// Start a process
func (s *Service) Start(ctx context.Context, r *shimapi.StartRequest) (*shimapi.StartResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p := s.processes[r.ID]
	if p == nil {
		return nil, errdefs.ToGRPCf(errdefs.ErrNotFound, "process %s not found", r.ID)
	}
	if err := p.Start(ctx); err != nil {
		return nil, err
	}
	if r.ID == s.id {
		s.events <- &eventsapi.TaskStart{
			ContainerID: s.id,
			Pid:         uint32(p.Pid()),
		}
	} else {
		pid := p.Pid()
		s.events <- &eventsapi.TaskExecStarted{
			ContainerID: s.id,
			ExecID:      r.ID,
			Pid:         uint32(pid),
		}
	}
	return &shimapi.StartResponse{
		ID:  p.ID(),
		Pid: uint32(p.Pid()),
	}, nil
}

// Delete the initial process and container
func (s *Service) Delete(ctx context.Context, r *google_protobuf.Empty) (*shimapi.DeleteResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p := s.processes[s.id]
	if p == nil {
		return nil, errdefs.ToGRPCf(errdefs.ErrFailedPrecondition, "container must be created")
	}

	if err := p.Delete(ctx); err != nil {
		return nil, err
	}
	delete(s.processes, s.id)
	s.platform.close()
	s.events <- &eventsapi.TaskDelete{
		ContainerID: s.id,
		ExitStatus:  uint32(p.ExitStatus()),
		ExitedAt:    p.ExitedAt(),
		Pid:         uint32(p.Pid()),
	}
	return &shimapi.DeleteResponse{
		ExitStatus: uint32(p.ExitStatus()),
		ExitedAt:   p.ExitedAt(),
		Pid:        uint32(p.Pid()),
	}, nil
}

// DeleteProcess deletes an exec'd process
func (s *Service) DeleteProcess(ctx context.Context, r *shimapi.DeleteProcessRequest) (*shimapi.DeleteResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if r.ID == s.id {
		return nil, grpc.Errorf(codes.InvalidArgument, "cannot delete init process with DeleteProcess")
	}
	p := s.processes[r.ID]
	if p == nil {
		return nil, errors.Wrapf(errdefs.ErrNotFound, "process %s", r.ID)
	}
	if err := p.Delete(ctx); err != nil {
		return nil, err
	}
	delete(s.processes, r.ID)
	return &shimapi.DeleteResponse{
		ExitStatus: uint32(p.ExitStatus()),
		ExitedAt:   p.ExitedAt(),
		Pid:        uint32(p.Pid()),
	}, nil
}

// Exec an additional process inside the container
func (s *Service) Exec(ctx context.Context, r *shimapi.ExecProcessRequest) (*google_protobuf.Empty, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if p := s.processes[r.ID]; p != nil {
		return nil, errdefs.ToGRPCf(errdefs.ErrAlreadyExists, "id %s", r.ID)
	}

	p := s.processes[s.id]
	if p == nil {
		return nil, errdefs.ToGRPCf(errdefs.ErrFailedPrecondition, "container must be created")
	}

	process, err := newExecProcess(ctx, s.config.Path, r, p.(*initProcess), r.ID)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	s.processes[r.ID] = process

	s.events <- &eventsapi.TaskExecAdded{
		ContainerID: s.id,
		ExecID:      r.ID,
	}
	return empty, nil
}

// ResizePty of a process
func (s *Service) ResizePty(ctx context.Context, r *shimapi.ResizePtyRequest) (*google_protobuf.Empty, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if r.ID == "" {
		return nil, errdefs.ToGRPCf(errdefs.ErrInvalidArgument, "id not provided")
	}
	ws := console.WinSize{
		Width:  uint16(r.Width),
		Height: uint16(r.Height),
	}
	p := s.processes[r.ID]
	if p == nil {
		return nil, errors.Errorf("process does not exist %s", r.ID)
	}
	if err := p.Resize(ws); err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return empty, nil
}

// State returns runtime state information for a process
func (s *Service) State(ctx context.Context, r *shimapi.StateRequest) (*shimapi.StateResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p := s.processes[r.ID]
	if p == nil {
		return nil, errdefs.ToGRPCf(errdefs.ErrNotFound, "process id %s not found", r.ID)
	}
	st, err := p.Status(ctx)
	if err != nil {
		return nil, err
	}
	status := task.StatusUnknown
	switch st {
	case "created":
		status = task.StatusCreated
	case "running":
		status = task.StatusRunning
	case "stopped":
		status = task.StatusStopped
	case "paused":
		status = task.StatusPaused
	case "pausing":
		status = task.StatusPausing
	}
	sio := p.Stdio()
	return &shimapi.StateResponse{
		ID:         p.ID(),
		Bundle:     s.bundle,
		Pid:        uint32(p.Pid()),
		Status:     status,
		Stdin:      sio.stdin,
		Stdout:     sio.stdout,
		Stderr:     sio.stderr,
		Terminal:   sio.terminal,
		ExitStatus: uint32(p.ExitStatus()),
		ExitedAt:   p.ExitedAt(),
	}, nil
}

// Pause the container
func (s *Service) Pause(ctx context.Context, r *google_protobuf.Empty) (*google_protobuf.Empty, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p := s.processes[s.id]
	if p == nil {
		return nil, errdefs.ToGRPCf(errdefs.ErrFailedPrecondition, "container must be created")
	}
	if err := p.(*initProcess).Pause(ctx); err != nil {
		return nil, err
	}
	s.events <- &eventsapi.TaskPaused{
		ContainerID: s.id,
	}
	return empty, nil
}

// Resume the container
func (s *Service) Resume(ctx context.Context, r *google_protobuf.Empty) (*google_protobuf.Empty, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p := s.processes[s.id]
	if p == nil {
		return nil, errdefs.ToGRPCf(errdefs.ErrFailedPrecondition, "container must be created")
	}
	if err := p.(*initProcess).Resume(ctx); err != nil {
		return nil, err
	}
	s.events <- &eventsapi.TaskResumed{
		ContainerID: s.id,
	}
	return empty, nil
}

// Kill a process with the provided signal
func (s *Service) Kill(ctx context.Context, r *shimapi.KillRequest) (*google_protobuf.Empty, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if r.ID == "" {
		p := s.processes[s.id]
		if p == nil {
			return nil, errdefs.ToGRPCf(errdefs.ErrFailedPrecondition, "container must be created")
		}
		if err := p.Kill(ctx, r.Signal, r.All); err != nil {
			return nil, errdefs.ToGRPC(err)
		}
		return empty, nil
	}

	p := s.processes[r.ID]
	if p == nil {
		return nil, errdefs.ToGRPCf(errdefs.ErrNotFound, "process id %s not found", r.ID)
	}
	if err := p.Kill(ctx, r.Signal, r.All); err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return empty, nil
}

// ListPids returns all pids inside the container
func (s *Service) ListPids(ctx context.Context, r *shimapi.ListPidsRequest) (*shimapi.ListPidsResponse, error) {
	pids, err := s.getContainerPids(ctx, r.ID)
	if err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	var processes []*task.ProcessInfo
	for _, pid := range pids {
		processes = append(processes, &task.ProcessInfo{
			Pid: pid,
		})
	}
	return &shimapi.ListPidsResponse{
		Processes: processes,
	}, nil
}

// CloseIO of a process
func (s *Service) CloseIO(ctx context.Context, r *shimapi.CloseIORequest) (*google_protobuf.Empty, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p := s.processes[r.ID]
	if p == nil {
		return nil, errdefs.ToGRPCf(errdefs.ErrNotFound, "process does not exist %s", r.ID)
	}
	if stdin := p.Stdin(); stdin != nil {
		if err := stdin.Close(); err != nil {
			return nil, errors.Wrap(err, "close stdin")
		}
	}
	return empty, nil
}

// Checkpoint the container
func (s *Service) Checkpoint(ctx context.Context, r *shimapi.CheckpointTaskRequest) (*google_protobuf.Empty, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p := s.processes[s.id]
	if p == nil {
		return nil, errdefs.ToGRPCf(errdefs.ErrFailedPrecondition, "container must be created")
	}
	if err := p.(*initProcess).Checkpoint(ctx, r); err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	s.events <- &eventsapi.TaskCheckpointed{
		ContainerID: s.id,
	}
	return empty, nil
}

// ShimInfo returns shim information such as the shim's pid
func (s *Service) ShimInfo(ctx context.Context, r *google_protobuf.Empty) (*shimapi.ShimInfoResponse, error) {
	return &shimapi.ShimInfoResponse{
		ShimPid: uint32(os.Getpid()),
	}, nil
}

// Update a running container
func (s *Service) Update(ctx context.Context, r *shimapi.UpdateTaskRequest) (*google_protobuf.Empty, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p := s.processes[s.id]
	if p == nil {
		return nil, errdefs.ToGRPCf(errdefs.ErrFailedPrecondition, "container must be created")
	}
	if err := p.(*initProcess).Update(ctx, r); err != nil {
		return nil, errdefs.ToGRPC(err)
	}
	return empty, nil
}

// Wait for a process to exit
func (s *Service) Wait(ctx context.Context, r *shimapi.WaitRequest) (*shimapi.WaitResponse, error) {
	s.mu.Lock()
	p := s.processes[r.ID]
	s.mu.Unlock()
	if p == nil {
		return nil, errdefs.ToGRPCf(errdefs.ErrFailedPrecondition, "container must be created")
	}
	p.Wait()

	return &shimapi.WaitResponse{
		ExitStatus: uint32(p.ExitStatus()),
		ExitedAt:   p.ExitedAt(),
	}, nil
}

func (s *Service) processExits() {
	for e := range s.ec {
		s.checkProcesses(e)
	}
}

func (s *Service) checkProcesses(e runc.Exit) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, p := range s.processes {
		if p.Pid() == e.Pid {
			if ip, ok := p.(*initProcess); ok {
				// Ensure all children are killed
				if err := ip.killAll(s.context); err != nil {
					log.G(s.context).WithError(err).WithField("id", ip.ID()).
						Error("failed to kill init's children")
				}
			}
			p.SetExited(e.Status)
			s.events <- &eventsapi.TaskExit{
				ContainerID: s.id,
				ID:          p.ID(),
				Pid:         uint32(e.Pid),
				ExitStatus:  uint32(e.Status),
				ExitedAt:    p.ExitedAt(),
			}
			return
		}
	}
}

func (s *Service) getContainerPids(ctx context.Context, id string) ([]uint32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p := s.processes[s.id]
	if p == nil {
		return nil, errors.Wrapf(errdefs.ErrFailedPrecondition, "container must be created")
	}

	ps, err := p.(*initProcess).runtime.Ps(ctx, id)
	if err != nil {
		return nil, err
	}
	pids := make([]uint32, 0, len(ps))
	for _, pid := range ps {
		pids = append(pids, uint32(pid))
	}
	return pids, nil
}

func (s *Service) forward(publisher events.Publisher) {
	for e := range s.events {
		if err := publisher.Publish(s.context, getTopic(s.context, e), e); err != nil {
			logrus.WithError(err).Error("post event")
		}
	}
}

func getTopic(ctx context.Context, e interface{}) string {
	switch e.(type) {
	case *eventsapi.TaskCreate:
		return runtime.TaskCreateEventTopic
	case *eventsapi.TaskStart:
		return runtime.TaskStartEventTopic
	case *eventsapi.TaskOOM:
		return runtime.TaskOOMEventTopic
	case *eventsapi.TaskExit:
		return runtime.TaskExitEventTopic
	case *eventsapi.TaskDelete:
		return runtime.TaskDeleteEventTopic
	case *eventsapi.TaskExecAdded:
		return runtime.TaskExecAddedEventTopic
	case *eventsapi.TaskExecStarted:
		return runtime.TaskExecStartedEventTopic
	case *eventsapi.TaskPaused:
		return runtime.TaskPausedEventTopic
	case *eventsapi.TaskResumed:
		return runtime.TaskResumedEventTopic
	case *eventsapi.TaskCheckpointed:
		return runtime.TaskCheckpointedEventTopic
	default:
		logrus.Warnf("no topic for type %#v", e)
	}
	return runtime.TaskUnknownTopic
}
