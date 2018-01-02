package libcontainerd

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	containerd "github.com/containerd/containerd/api/grpc/types"
	containerd_runtime_types "github.com/containerd/containerd/runtime"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/mount"
	"github.com/golang/protobuf/ptypes"
	"github.com/golang/protobuf/ptypes/timestamp"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
	"golang.org/x/sys/unix"
)

type client struct {
	clientCommon

	// Platform specific properties below here.
	remote        *remote
	q             queue
	exitNotifiers map[string]*exitNotifier
	liveRestore   bool
}

// GetServerVersion returns the connected server version information
func (clnt *client) GetServerVersion(ctx context.Context) (*ServerVersion, error) {
	resp, err := clnt.remote.apiClient.GetServerVersion(ctx, &containerd.GetServerVersionRequest{})
	if err != nil {
		return nil, err
	}

	sv := &ServerVersion{
		GetServerVersionResponse: *resp,
	}

	return sv, nil
}

// AddProcess is the handler for adding a process to an already running
// container. It's called through docker exec. It returns the system pid of the
// exec'd process.
func (clnt *client) AddProcess(ctx context.Context, containerID, processFriendlyName string, specp Process, attachStdio StdioCallback) (pid int, err error) {
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	container, err := clnt.getContainer(containerID)
	if err != nil {
		return -1, err
	}

	spec, err := container.spec()
	if err != nil {
		return -1, err
	}
	sp := spec.Process
	sp.Args = specp.Args
	sp.Terminal = specp.Terminal
	if len(specp.Env) > 0 {
		sp.Env = specp.Env
	}
	if specp.Cwd != nil {
		sp.Cwd = *specp.Cwd
	}
	if specp.User != nil {
		sp.User = specs.User{
			UID:            specp.User.UID,
			GID:            specp.User.GID,
			AdditionalGids: specp.User.AdditionalGids,
		}
	}
	if specp.Capabilities != nil {
		sp.Capabilities.Bounding = specp.Capabilities
		sp.Capabilities.Effective = specp.Capabilities
		sp.Capabilities.Inheritable = specp.Capabilities
		sp.Capabilities.Permitted = specp.Capabilities
	}

	p := container.newProcess(processFriendlyName)

	r := &containerd.AddProcessRequest{
		Args:     sp.Args,
		Cwd:      sp.Cwd,
		Terminal: sp.Terminal,
		Id:       containerID,
		Env:      sp.Env,
		User: &containerd.User{
			Uid:            sp.User.UID,
			Gid:            sp.User.GID,
			AdditionalGids: sp.User.AdditionalGids,
		},
		Pid:             processFriendlyName,
		Stdin:           p.fifo(unix.Stdin),
		Stdout:          p.fifo(unix.Stdout),
		Stderr:          p.fifo(unix.Stderr),
		Capabilities:    sp.Capabilities.Effective,
		ApparmorProfile: sp.ApparmorProfile,
		SelinuxLabel:    sp.SelinuxLabel,
		NoNewPrivileges: sp.NoNewPrivileges,
		Rlimits:         convertRlimits(sp.Rlimits),
	}

	fifoCtx, cancel := context.WithCancel(context.Background())
	defer func() {
		if err != nil {
			cancel()
		}
	}()

	iopipe, err := p.openFifos(fifoCtx, sp.Terminal)
	if err != nil {
		return -1, err
	}

	resp, err := clnt.remote.apiClient.AddProcess(ctx, r)
	if err != nil {
		p.closeFifos(iopipe)
		return -1, err
	}

	var stdinOnce sync.Once
	stdin := iopipe.Stdin
	iopipe.Stdin = ioutils.NewWriteCloserWrapper(stdin, func() error {
		var err error
		stdinOnce.Do(func() { // on error from attach we don't know if stdin was already closed
			err = stdin.Close()
			if err2 := p.sendCloseStdin(); err == nil {
				err = err2
			}
		})
		return err
	})

	container.processes[processFriendlyName] = p

	if err := attachStdio(*iopipe); err != nil {
		p.closeFifos(iopipe)
		return -1, err
	}

	return int(resp.SystemPid), nil
}

func (clnt *client) SignalProcess(containerID string, pid string, sig int) error {
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	_, err := clnt.remote.apiClient.Signal(context.Background(), &containerd.SignalRequest{
		Id:     containerID,
		Pid:    pid,
		Signal: uint32(sig),
	})
	return err
}

func (clnt *client) Resize(containerID, processFriendlyName string, width, height int) error {
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	if _, err := clnt.getContainer(containerID); err != nil {
		return err
	}
	_, err := clnt.remote.apiClient.UpdateProcess(context.Background(), &containerd.UpdateProcessRequest{
		Id:     containerID,
		Pid:    processFriendlyName,
		Width:  uint32(width),
		Height: uint32(height),
	})
	return err
}

func (clnt *client) Pause(containerID string) error {
	return clnt.setState(containerID, StatePause)
}

func (clnt *client) setState(containerID, state string) error {
	clnt.lock(containerID)
	container, err := clnt.getContainer(containerID)
	if err != nil {
		clnt.unlock(containerID)
		return err
	}
	if container.systemPid == 0 {
		clnt.unlock(containerID)
		return fmt.Errorf("No active process for container %s", containerID)
	}
	st := "running"
	if state == StatePause {
		st = "paused"
	}
	chstate := make(chan struct{})
	_, err = clnt.remote.apiClient.UpdateContainer(context.Background(), &containerd.UpdateContainerRequest{
		Id:     containerID,
		Pid:    InitFriendlyName,
		Status: st,
	})
	if err != nil {
		clnt.unlock(containerID)
		return err
	}
	container.pauseMonitor.append(state, chstate)
	clnt.unlock(containerID)
	<-chstate
	return nil
}

func (clnt *client) Resume(containerID string) error {
	return clnt.setState(containerID, StateResume)
}

func (clnt *client) Stats(containerID string) (*Stats, error) {
	resp, err := clnt.remote.apiClient.Stats(context.Background(), &containerd.StatsRequest{containerID})
	if err != nil {
		return nil, err
	}
	return (*Stats)(resp), nil
}

// Take care of the old 1.11.0 behavior in case the version upgrade
// happened without a clean daemon shutdown
func (clnt *client) cleanupOldRootfs(containerID string) {
	// Unmount and delete the bundle folder
	if mts, err := mount.GetMounts(); err == nil {
		for _, mts := range mts {
			if strings.HasSuffix(mts.Mountpoint, containerID+"/rootfs") {
				if err := unix.Unmount(mts.Mountpoint, unix.MNT_DETACH); err == nil {
					os.RemoveAll(strings.TrimSuffix(mts.Mountpoint, "/rootfs"))
				}
				break
			}
		}
	}
}

func (clnt *client) setExited(containerID string, exitCode uint32) error {
	clnt.lock(containerID)
	defer clnt.unlock(containerID)

	err := clnt.backend.StateChanged(containerID, StateInfo{
		CommonStateInfo: CommonStateInfo{
			State:    StateExit,
			ExitCode: exitCode,
		}})

	clnt.cleanupOldRootfs(containerID)

	return err
}

func (clnt *client) GetPidsForContainer(containerID string) ([]int, error) {
	cont, err := clnt.getContainerdContainer(containerID)
	if err != nil {
		return nil, err
	}
	pids := make([]int, len(cont.Pids))
	for i, p := range cont.Pids {
		pids[i] = int(p)
	}
	return pids, nil
}

// Summary returns a summary of the processes running in a container.
// This is a no-op on Linux.
func (clnt *client) Summary(containerID string) ([]Summary, error) {
	return nil, nil
}

func (clnt *client) getContainerdContainer(containerID string) (*containerd.Container, error) {
	resp, err := clnt.remote.apiClient.State(context.Background(), &containerd.StateRequest{Id: containerID})
	if err != nil {
		return nil, err
	}
	for _, cont := range resp.Containers {
		if cont.Id == containerID {
			return cont, nil
		}
	}
	return nil, fmt.Errorf("invalid state response")
}

func (clnt *client) UpdateResources(containerID string, resources Resources) error {
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	container, err := clnt.getContainer(containerID)
	if err != nil {
		return err
	}
	if container.systemPid == 0 {
		return fmt.Errorf("No active process for container %s", containerID)
	}
	_, err = clnt.remote.apiClient.UpdateContainer(context.Background(), &containerd.UpdateContainerRequest{
		Id:        containerID,
		Pid:       InitFriendlyName,
		Resources: (*containerd.UpdateResource)(&resources),
	})
	if err != nil {
		return err
	}
	return nil
}

func (clnt *client) getExitNotifier(containerID string) *exitNotifier {
	clnt.mapMutex.RLock()
	defer clnt.mapMutex.RUnlock()
	return clnt.exitNotifiers[containerID]
}

func (clnt *client) getOrCreateExitNotifier(containerID string) *exitNotifier {
	clnt.mapMutex.Lock()
	w, ok := clnt.exitNotifiers[containerID]
	defer clnt.mapMutex.Unlock()
	if !ok {
		w = &exitNotifier{c: make(chan struct{}), client: clnt}
		clnt.exitNotifiers[containerID] = w
	}
	return w
}

func (clnt *client) restore(cont *containerd.Container, lastEvent *containerd.Event, attachStdio StdioCallback, options ...CreateOption) (err error) {
	clnt.lock(cont.Id)
	defer clnt.unlock(cont.Id)

	logrus.Debugf("libcontainerd: restore container %s state %s", cont.Id, cont.Status)

	containerID := cont.Id
	if _, err := clnt.getContainer(containerID); err == nil {
		return fmt.Errorf("container %s is already active", containerID)
	}

	defer func() {
		if err != nil {
			clnt.deleteContainer(cont.Id)
		}
	}()

	container := clnt.newContainer(cont.BundlePath, options...)
	container.systemPid = systemPid(cont)

	var terminal bool
	for _, p := range cont.Processes {
		if p.Pid == InitFriendlyName {
			terminal = p.Terminal
		}
	}

	fifoCtx, cancel := context.WithCancel(context.Background())
	defer func() {
		if err != nil {
			cancel()
		}
	}()

	iopipe, err := container.openFifos(fifoCtx, terminal)
	if err != nil {
		return err
	}
	var stdinOnce sync.Once
	stdin := iopipe.Stdin
	iopipe.Stdin = ioutils.NewWriteCloserWrapper(stdin, func() error {
		var err error
		stdinOnce.Do(func() { // on error from attach we don't know if stdin was already closed
			err = stdin.Close()
		})
		return err
	})

	if err := attachStdio(*iopipe); err != nil {
		container.closeFifos(iopipe)
		return err
	}

	clnt.appendContainer(container)

	err = clnt.backend.StateChanged(containerID, StateInfo{
		CommonStateInfo: CommonStateInfo{
			State: StateRestore,
			Pid:   container.systemPid,
		}})

	if err != nil {
		container.closeFifos(iopipe)
		return err
	}

	if lastEvent != nil {
		// This should only be a pause or resume event
		if lastEvent.Type == StatePause || lastEvent.Type == StateResume {
			return clnt.backend.StateChanged(containerID, StateInfo{
				CommonStateInfo: CommonStateInfo{
					State: lastEvent.Type,
					Pid:   container.systemPid,
				}})
		}

		logrus.Warnf("libcontainerd: unexpected backlog event: %#v", lastEvent)
	}

	return nil
}

func (clnt *client) getContainerLastEventSinceTime(id string, tsp *timestamp.Timestamp) (*containerd.Event, error) {
	er := &containerd.EventsRequest{
		Timestamp:  tsp,
		StoredOnly: true,
		Id:         id,
	}
	events, err := clnt.remote.apiClient.Events(context.Background(), er)
	if err != nil {
		logrus.Errorf("libcontainerd: failed to get container events stream for %s: %q", er.Id, err)
		return nil, err
	}

	var ev *containerd.Event
	for {
		e, err := events.Recv()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			logrus.Errorf("libcontainerd: failed to get container event for %s: %q", id, err)
			return nil, err
		}
		ev = e
		logrus.Debugf("libcontainerd: received past event %#v", ev)
	}

	return ev, nil
}

func (clnt *client) getContainerLastEvent(id string) (*containerd.Event, error) {
	ev, err := clnt.getContainerLastEventSinceTime(id, clnt.remote.restoreFromTimestamp)
	if err == nil && ev == nil {
		// If ev is nil and the container is running in containerd,
		// we already consumed all the event of the
		// container, included the "exit" one.
		// Thus, we request all events containerd has in memory for
		// this container in order to get the last one (which should
		// be an exit event)
		logrus.Warnf("libcontainerd: client is out of sync, restore was called on a fully synced container (%s).", id)
		// Request all events since beginning of time
		t := time.Unix(0, 0)
		tsp, err := ptypes.TimestampProto(t)
		if err != nil {
			logrus.Errorf("libcontainerd: getLastEventSinceTime() failed to convert timestamp: %q", err)
			return nil, err
		}

		return clnt.getContainerLastEventSinceTime(id, tsp)
	}

	return ev, err
}

func (clnt *client) Restore(containerID string, attachStdio StdioCallback, options ...CreateOption) error {
	// Synchronize with live events
	clnt.remote.Lock()
	defer clnt.remote.Unlock()
	// Check that containerd still knows this container.
	//
	// In the unlikely event that Restore for this container process
	// the its past event before the main loop, the event will be
	// processed twice. However, this is not an issue as all those
	// events will do is change the state of the container to be
	// exactly the same.
	cont, err := clnt.getContainerdContainer(containerID)
	// Get its last event
	ev, eerr := clnt.getContainerLastEvent(containerID)
	if err != nil || containerd_runtime_types.State(cont.Status) == containerd_runtime_types.Stopped {
		if err != nil {
			logrus.Warnf("libcontainerd: failed to retrieve container %s state: %v", containerID, err)
		}
		if ev != nil && (ev.Pid != InitFriendlyName || ev.Type != StateExit) {
			// Wait a while for the exit event
			timeout := time.NewTimer(10 * time.Second)
			tick := time.NewTicker(100 * time.Millisecond)
		stop:
			for {
				select {
				case <-timeout.C:
					break stop
				case <-tick.C:
					ev, eerr = clnt.getContainerLastEvent(containerID)
					if eerr != nil {
						break stop
					}
					if ev != nil && ev.Pid == InitFriendlyName && ev.Type == StateExit {
						break stop
					}
				}
			}
			timeout.Stop()
			tick.Stop()
		}

		// get the exit status for this container, if we don't have
		// one, indicate an error
		ec := uint32(255)
		if eerr == nil && ev != nil && ev.Pid == InitFriendlyName && ev.Type == StateExit {
			ec = ev.Status
		}
		clnt.setExited(containerID, ec)

		return nil
	}

	// container is still alive
	if clnt.liveRestore {
		if err := clnt.restore(cont, ev, attachStdio, options...); err != nil {
			logrus.Errorf("libcontainerd: error restoring %s: %v", containerID, err)
		}
		return nil
	}

	// Kill the container if liveRestore == false
	w := clnt.getOrCreateExitNotifier(containerID)
	clnt.lock(cont.Id)
	container := clnt.newContainer(cont.BundlePath)
	container.systemPid = systemPid(cont)
	clnt.appendContainer(container)
	clnt.unlock(cont.Id)

	container.discardFifos()

	if err := clnt.Signal(containerID, int(unix.SIGTERM)); err != nil {
		logrus.Errorf("libcontainerd: error sending sigterm to %v: %v", containerID, err)
	}

	// Let the main loop handle the exit event
	clnt.remote.Unlock()

	if ev != nil && ev.Type == StatePause {
		// resume container, it depends on the main loop, so we do it after Unlock()
		logrus.Debugf("libcontainerd: %s was paused, resuming it so it can die", containerID)
		if err := clnt.Resume(containerID); err != nil {
			return fmt.Errorf("failed to resume container: %v", err)
		}
	}

	select {
	case <-time.After(10 * time.Second):
		if err := clnt.Signal(containerID, int(unix.SIGKILL)); err != nil {
			logrus.Errorf("libcontainerd: error sending sigkill to %v: %v", containerID, err)
		}
		select {
		case <-time.After(2 * time.Second):
		case <-w.wait():
			// relock because of the defer
			clnt.remote.Lock()
			return nil
		}
	case <-w.wait():
		// relock because of the defer
		clnt.remote.Lock()
		return nil
	}
	// relock because of the defer
	clnt.remote.Lock()

	clnt.deleteContainer(containerID)

	return clnt.setExited(containerID, uint32(255))
}

func (clnt *client) CreateCheckpoint(containerID string, checkpointID string, checkpointDir string, exit bool) error {
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	if _, err := clnt.getContainer(containerID); err != nil {
		return err
	}

	_, err := clnt.remote.apiClient.CreateCheckpoint(context.Background(), &containerd.CreateCheckpointRequest{
		Id: containerID,
		Checkpoint: &containerd.Checkpoint{
			Name:        checkpointID,
			Exit:        exit,
			Tcp:         true,
			UnixSockets: true,
			Shell:       false,
			EmptyNS:     []string{"network"},
		},
		CheckpointDir: checkpointDir,
	})
	return err
}

func (clnt *client) DeleteCheckpoint(containerID string, checkpointID string, checkpointDir string) error {
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	if _, err := clnt.getContainer(containerID); err != nil {
		return err
	}

	_, err := clnt.remote.apiClient.DeleteCheckpoint(context.Background(), &containerd.DeleteCheckpointRequest{
		Id:            containerID,
		Name:          checkpointID,
		CheckpointDir: checkpointDir,
	})
	return err
}

func (clnt *client) ListCheckpoints(containerID string, checkpointDir string) (*Checkpoints, error) {
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	if _, err := clnt.getContainer(containerID); err != nil {
		return nil, err
	}

	resp, err := clnt.remote.apiClient.ListCheckpoint(context.Background(), &containerd.ListCheckpointRequest{
		Id:            containerID,
		CheckpointDir: checkpointDir,
	})
	if err != nil {
		return nil, err
	}
	return (*Checkpoints)(resp), nil
}
