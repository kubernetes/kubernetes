package libcontainer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/opencontainers/runtime-spec/specs-go"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fs2"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/intelrdt"
	"github.com/opencontainers/runc/libcontainer/internal/userns"
	"github.com/opencontainers/runc/libcontainer/logs"
	"github.com/opencontainers/runc/libcontainer/system"
	"github.com/opencontainers/runc/libcontainer/utils"
)

type parentProcess interface {
	// pid returns the pid for the running process.
	pid() int

	// start starts the process execution.
	start() error

	// send a SIGKILL to the process and wait for the exit.
	terminate() error

	// wait waits on the process returning the process state.
	wait() (*os.ProcessState, error)

	// startTime returns the process start time.
	startTime() (uint64, error)
	signal(os.Signal) error
	externalDescriptors() []string
	setExternalDescriptors(fds []string)
	forwardChildLogs() chan error
}

type processComm struct {
	// Used to send initial configuration to "runc init" and for "runc init" to
	// indicate that it is ready.
	initSockParent *os.File
	initSockChild  *os.File
	// Used for control messages between parent and "runc init".
	syncSockParent *syncSocket
	syncSockChild  *syncSocket
	// Used for log forwarding from "runc init" to the parent.
	logPipeParent *os.File
	logPipeChild  *os.File
}

func newProcessComm() (*processComm, error) {
	var (
		comm processComm
		err  error
	)
	comm.initSockParent, comm.initSockChild, err = utils.NewSockPair("init")
	if err != nil {
		return nil, fmt.Errorf("unable to create init pipe: %w", err)
	}
	comm.syncSockParent, comm.syncSockChild, err = newSyncSockpair("sync")
	if err != nil {
		return nil, fmt.Errorf("unable to create sync pipe: %w", err)
	}
	comm.logPipeParent, comm.logPipeChild, err = os.Pipe()
	if err != nil {
		return nil, fmt.Errorf("unable to create log pipe: %w", err)
	}
	return &comm, nil
}

func (c *processComm) closeChild() {
	_ = c.initSockChild.Close()
	_ = c.syncSockChild.Close()
	_ = c.logPipeChild.Close()
}

func (c *processComm) closeParent() {
	_ = c.initSockParent.Close()
	_ = c.syncSockParent.Close()
	// c.logPipeParent is kept alive for ForwardLogs
}

type setnsProcess struct {
	cmd             *exec.Cmd
	comm            *processComm
	cgroupPaths     map[string]string
	rootlessCgroups bool
	manager         cgroups.Manager
	intelRdtPath    string
	config          *initConfig
	fds             []string
	process         *Process
	bootstrapData   io.Reader
	initProcessPid  int
}

func (p *setnsProcess) startTime() (uint64, error) {
	stat, err := system.Stat(p.pid())
	return stat.StartTime, err
}

func (p *setnsProcess) signal(sig os.Signal) error {
	s, ok := sig.(unix.Signal)
	if !ok {
		return errors.New("os: unsupported signal type")
	}
	return unix.Kill(p.pid(), s)
}

func (p *setnsProcess) start() (retErr error) {
	defer p.comm.closeParent()

	if p.process.IOPriority != nil {
		if err := setIOPriority(p.process.IOPriority); err != nil {
			return err
		}
	}

	// get the "before" value of oom kill count
	oom, _ := p.manager.OOMKillCount()
	err := p.cmd.Start()
	// close the child-side of the pipes (controlled by child)
	p.comm.closeChild()
	if err != nil {
		return fmt.Errorf("error starting setns process: %w", err)
	}

	defer func() {
		if retErr != nil {
			if newOom, err := p.manager.OOMKillCount(); err == nil && newOom != oom {
				// Someone in this cgroup was killed, this _might_ be us.
				retErr = fmt.Errorf("%w (possibly OOM-killed)", retErr)
			}
			err := ignoreTerminateErrors(p.terminate())
			if err != nil {
				logrus.WithError(err).Warn("unable to terminate setnsProcess")
			}
		}
	}()

	if p.bootstrapData != nil {
		if _, err := io.Copy(p.comm.initSockParent, p.bootstrapData); err != nil {
			return fmt.Errorf("error copying bootstrap data to pipe: %w", err)
		}
	}
	if err := p.execSetns(); err != nil {
		return fmt.Errorf("error executing setns process: %w", err)
	}
	for _, path := range p.cgroupPaths {
		if err := cgroups.WriteCgroupProc(path, p.pid()); err != nil && !p.rootlessCgroups {
			// On cgroup v2 + nesting + domain controllers, WriteCgroupProc may fail with EBUSY.
			// https://github.com/opencontainers/runc/issues/2356#issuecomment-621277643
			// Try to join the cgroup of InitProcessPid.
			if cgroups.IsCgroup2UnifiedMode() && p.initProcessPid != 0 {
				initProcCgroupFile := fmt.Sprintf("/proc/%d/cgroup", p.initProcessPid)
				initCg, initCgErr := cgroups.ParseCgroupFile(initProcCgroupFile)
				if initCgErr == nil {
					if initCgPath, ok := initCg[""]; ok {
						initCgDirpath := filepath.Join(fs2.UnifiedMountpoint, initCgPath)
						logrus.Debugf("adding pid %d to cgroups %v failed (%v), attempting to join %q (obtained from %s)",
							p.pid(), p.cgroupPaths, err, initCg, initCgDirpath)
						// NOTE: initCgDirPath is not guaranteed to exist because we didn't pause the container.
						err = cgroups.WriteCgroupProc(initCgDirpath, p.pid())
					}
				}
			}
			if err != nil {
				return fmt.Errorf("error adding pid %d to cgroups: %w", p.pid(), err)
			}
		}
	}
	if p.intelRdtPath != "" {
		// if Intel RDT "resource control" filesystem path exists
		_, err := os.Stat(p.intelRdtPath)
		if err == nil {
			if err := intelrdt.WriteIntelRdtTasks(p.intelRdtPath, p.pid()); err != nil {
				return fmt.Errorf("error adding pid %d to Intel RDT: %w", p.pid(), err)
			}
		}
	}

	if err := utils.WriteJSON(p.comm.initSockParent, p.config); err != nil {
		return fmt.Errorf("error writing config to pipe: %w", err)
	}

	var seenProcReady bool
	ierr := parseSync(p.comm.syncSockParent, func(sync *syncT) error {
		switch sync.Type {
		case procReady:
			seenProcReady = true
			// Set rlimits, this has to be done here because we lose permissions
			// to raise the limits once we enter a user-namespace
			if err := setupRlimits(p.config.Rlimits, p.pid()); err != nil {
				return fmt.Errorf("error setting rlimits for ready process: %w", err)
			}

			// Sync with child.
			if err := writeSync(p.comm.syncSockParent, procRun); err != nil {
				return err
			}
		case procHooks:
			// This shouldn't happen.
			panic("unexpected procHooks in setns")
		case procMountPlease:
			// This shouldn't happen.
			panic("unexpected procMountPlease in setns")
		case procSeccomp:
			if p.config.Config.Seccomp.ListenerPath == "" {
				return errors.New("seccomp listenerPath is not set")
			}
			if sync.Arg == nil {
				return fmt.Errorf("sync %q is missing an argument", sync.Type)
			}
			var srcFd int
			if err := json.Unmarshal(*sync.Arg, &srcFd); err != nil {
				return fmt.Errorf("sync %q passed invalid fd arg: %w", sync.Type, err)
			}
			seccompFd, err := pidGetFd(p.pid(), srcFd)
			if err != nil {
				return fmt.Errorf("sync %q get fd %d from child failed: %w", sync.Type, srcFd, err)
			}
			defer seccompFd.Close()
			// We have a copy, the child can keep working. We don't need to
			// wait for the seccomp notify listener to get the fd before we
			// permit the child to continue because the child will happily wait
			// for the listener if it hits SCMP_ACT_NOTIFY.
			if err := writeSync(p.comm.syncSockParent, procSeccompDone); err != nil {
				return err
			}

			bundle, annotations := utils.Annotations(p.config.Config.Labels)
			containerProcessState := &specs.ContainerProcessState{
				Version:  specs.Version,
				Fds:      []string{specs.SeccompFdName},
				Pid:      p.cmd.Process.Pid,
				Metadata: p.config.Config.Seccomp.ListenerMetadata,
				State: specs.State{
					Version:     specs.Version,
					ID:          p.config.ContainerID,
					Status:      specs.StateRunning,
					Pid:         p.initProcessPid,
					Bundle:      bundle,
					Annotations: annotations,
				},
			}
			if err := sendContainerProcessState(p.config.Config.Seccomp.ListenerPath,
				containerProcessState, seccompFd); err != nil {
				return err
			}
		default:
			return errors.New("invalid JSON payload from child")
		}
		return nil
	})

	if err := p.comm.syncSockParent.Shutdown(unix.SHUT_WR); err != nil && ierr == nil {
		return err
	}
	if !seenProcReady && ierr == nil {
		ierr = errors.New("procReady not received")
	}
	// Must be done after Shutdown so the child will exit and we can wait for it.
	if ierr != nil {
		_, _ = p.wait()
		return ierr
	}
	return nil
}

// execSetns runs the process that executes C code to perform the setns calls
// because setns support requires the C process to fork off a child and perform the setns
// before the go runtime boots, we wait on the process to die and receive the child's pid
// over the provided pipe.
func (p *setnsProcess) execSetns() error {
	status, err := p.cmd.Process.Wait()
	if err != nil {
		_ = p.cmd.Wait()
		return fmt.Errorf("error waiting on setns process to finish: %w", err)
	}
	if !status.Success() {
		_ = p.cmd.Wait()
		return &exec.ExitError{ProcessState: status}
	}
	var pid *pid
	if err := json.NewDecoder(p.comm.initSockParent).Decode(&pid); err != nil {
		_ = p.cmd.Wait()
		return fmt.Errorf("error reading pid from init pipe: %w", err)
	}

	// Clean up the zombie parent process
	// On Unix systems FindProcess always succeeds.
	firstChildProcess, _ := os.FindProcess(pid.PidFirstChild)

	// Ignore the error in case the child has already been reaped for any reason
	_, _ = firstChildProcess.Wait()

	process, err := os.FindProcess(pid.Pid)
	if err != nil {
		return err
	}
	p.cmd.Process = process
	p.process.ops = p
	return nil
}

// terminate sends a SIGKILL to the forked process for the setns routine then waits to
// avoid the process becoming a zombie.
func (p *setnsProcess) terminate() error {
	if p.cmd.Process == nil {
		return nil
	}
	err := p.cmd.Process.Kill()
	if _, werr := p.wait(); err == nil {
		err = werr
	}
	return err
}

func (p *setnsProcess) wait() (*os.ProcessState, error) {
	err := p.cmd.Wait()

	// Return actual ProcessState even on Wait error
	return p.cmd.ProcessState, err
}

func (p *setnsProcess) pid() int {
	return p.cmd.Process.Pid
}

func (p *setnsProcess) externalDescriptors() []string {
	return p.fds
}

func (p *setnsProcess) setExternalDescriptors(newFds []string) {
	p.fds = newFds
}

func (p *setnsProcess) forwardChildLogs() chan error {
	return logs.ForwardLogs(p.comm.logPipeParent)
}

type initProcess struct {
	cmd             *exec.Cmd
	comm            *processComm
	config          *initConfig
	manager         cgroups.Manager
	intelRdtManager *intelrdt.Manager
	container       *Container
	fds             []string
	process         *Process
	bootstrapData   io.Reader
}

func (p *initProcess) pid() int {
	return p.cmd.Process.Pid
}

func (p *initProcess) externalDescriptors() []string {
	return p.fds
}

// getChildPid receives the final child's pid over the provided pipe.
func (p *initProcess) getChildPid() (int, error) {
	var pid pid
	if err := json.NewDecoder(p.comm.initSockParent).Decode(&pid); err != nil {
		_ = p.cmd.Wait()
		return -1, err
	}

	// Clean up the zombie parent process
	// On Unix systems FindProcess always succeeds.
	firstChildProcess, _ := os.FindProcess(pid.PidFirstChild)

	// Ignore the error in case the child has already been reaped for any reason
	_, _ = firstChildProcess.Wait()

	return pid.Pid, nil
}

func (p *initProcess) waitForChildExit(childPid int) error {
	status, err := p.cmd.Process.Wait()
	if err != nil {
		_ = p.cmd.Wait()
		return err
	}
	if !status.Success() {
		_ = p.cmd.Wait()
		return &exec.ExitError{ProcessState: status}
	}

	process, err := os.FindProcess(childPid)
	if err != nil {
		return err
	}
	p.cmd.Process = process
	p.process.ops = p
	return nil
}

type mountSourceRequestFn func(*configs.Mount) (*mountSource, error)

// goCreateMountSources spawns a goroutine which creates open_tree(2)-style
// mountfds based on the requested configs.Mount configuration. The returned
// requestFn and cancelFn are used to interact with the goroutine.
//
// The caller of the returned mountSourceRequestFn is responsible for closing
// the returned file.
func (p *initProcess) goCreateMountSources(ctx context.Context) (mountSourceRequestFn, context.CancelFunc, error) {
	type response struct {
		src *mountSource
		err error
	}

	errCh := make(chan error, 1)
	requestCh := make(chan *configs.Mount)
	responseCh := make(chan response)

	ctx, cancelFn := context.WithTimeout(ctx, 1*time.Minute)
	go func() {
		// We lock this thread because we need to setns(2) here. There is no
		// UnlockOSThread() here, to ensure that the Go runtime will kill this
		// thread once this goroutine returns (ensuring no other goroutines run
		// in this context).
		runtime.LockOSThread()

		// Detach from the shared fs of the rest of the Go process in order to
		// be able to CLONE_NEWNS.
		if err := unix.Unshare(unix.CLONE_FS); err != nil {
			err = os.NewSyscallError("unshare(CLONE_FS)", err)
			errCh <- fmt.Errorf("mount source thread: %w", err)
			return
		}

		// Attach to the container's mount namespace.
		nsFd, err := os.Open(fmt.Sprintf("/proc/%d/ns/mnt", p.pid()))
		if err != nil {
			errCh <- fmt.Errorf("mount source thread: open container mntns: %w", err)
			return
		}
		defer nsFd.Close()
		if err := unix.Setns(int(nsFd.Fd()), unix.CLONE_NEWNS); err != nil {
			err = os.NewSyscallError("setns", err)
			errCh <- fmt.Errorf("mount source thread: join container mntns: %w", err)
			return
		}

		// No errors during setup!
		close(errCh)
		logrus.Debugf("mount source thread: successfully running in container mntns")

		nsHandles := new(userns.Handles)
		defer nsHandles.Release()
	loop:
		for {
			select {
			case m, ok := <-requestCh:
				if !ok {
					break loop
				}
				src, err := mountFd(nsHandles, m)
				logrus.Debugf("mount source thread: handling request for %q: %v %v", m.Source, src, err)
				responseCh <- response{
					src: src,
					err: err,
				}
			case <-ctx.Done():
				break loop
			}
		}
		logrus.Debugf("mount source thread: closing thread: %v", ctx.Err())
		close(responseCh)
	}()

	// Check for setup errors.
	err := <-errCh
	if err != nil {
		cancelFn()
		return nil, nil, err
	}

	// TODO: Switch to context.AfterFunc when we switch to Go 1.21.
	var requestChCloseOnce sync.Once
	requestFn := func(m *configs.Mount) (*mountSource, error) {
		var err error
		select {
		case requestCh <- m:
			select {
			case resp, ok := <-responseCh:
				if ok {
					return resp.src, resp.err
				}
			case <-ctx.Done():
				err = fmt.Errorf("receive mount source context cancelled: %w", ctx.Err())
			}
		case <-ctx.Done():
			err = fmt.Errorf("send mount request cancelled: %w", ctx.Err())
		}
		requestChCloseOnce.Do(func() { close(requestCh) })
		return nil, err
	}
	return requestFn, cancelFn, nil
}

func (p *initProcess) start() (retErr error) {
	defer p.comm.closeParent()
	err := p.cmd.Start()
	p.process.ops = p
	// close the child-side of the pipes (controlled by child)
	p.comm.closeChild()
	if err != nil {
		p.process.ops = nil
		return fmt.Errorf("unable to start init: %w", err)
	}

	defer func() {
		if retErr != nil {
			// Find out if init is killed by the kernel's OOM killer.
			// Get the count before killing init as otherwise cgroup
			// might be removed by systemd.
			oom, err := p.manager.OOMKillCount()
			if err != nil {
				logrus.WithError(err).Warn("unable to get oom kill count")
			} else if oom > 0 {
				// Does not matter what the particular error was,
				// its cause is most probably OOM, so report that.
				const oomError = "container init was OOM-killed (memory limit too low?)"

				if logrus.GetLevel() >= logrus.DebugLevel {
					// Only show the original error if debug is set,
					// as it is not generally very useful.
					retErr = fmt.Errorf(oomError+": %w", retErr)
				} else {
					retErr = errors.New(oomError)
				}
			}

			// Terminate the process to ensure we can remove cgroups.
			if err := ignoreTerminateErrors(p.terminate()); err != nil {
				logrus.WithError(err).Warn("unable to terminate initProcess")
			}

			_ = p.manager.Destroy()
			if p.intelRdtManager != nil {
				_ = p.intelRdtManager.Destroy()
			}
		}
	}()

	// Do this before syncing with child so that no children can escape the
	// cgroup. We don't need to worry about not doing this and not being root
	// because we'd be using the rootless cgroup manager in that case.
	if err := p.manager.Apply(p.pid()); err != nil {
		if errors.Is(err, cgroups.ErrRootless) {
			// ErrRootless is to be ignored except when
			// the container doesn't have private pidns.
			if !p.config.Config.Namespaces.IsPrivate(configs.NEWPID) {
				// TODO: make this an error in runc 1.3.
				logrus.Warn("Creating a rootless container with no cgroup and no private pid namespace. " +
					"Such configuration is strongly discouraged (as it is impossible to properly kill all container's processes) " +
					"and will result in an error in a future runc version.")
			}
		} else {
			return fmt.Errorf("unable to apply cgroup configuration: %w", err)
		}
	}
	if p.intelRdtManager != nil {
		if err := p.intelRdtManager.Apply(p.pid()); err != nil {
			return fmt.Errorf("unable to apply Intel RDT configuration: %w", err)
		}
	}
	if _, err := io.Copy(p.comm.initSockParent, p.bootstrapData); err != nil {
		return fmt.Errorf("can't copy bootstrap data to pipe: %w", err)
	}

	childPid, err := p.getChildPid()
	if err != nil {
		return fmt.Errorf("can't get final child's PID from pipe: %w", err)
	}

	// Save the standard descriptor names before the container process
	// can potentially move them (e.g., via dup2()).  If we don't do this now,
	// we won't know at checkpoint time which file descriptor to look up.
	fds, err := getPipeFds(childPid)
	if err != nil {
		return fmt.Errorf("error getting pipe fds for pid %d: %w", childPid, err)
	}
	p.setExternalDescriptors(fds)

	// Wait for our first child to exit
	if err := p.waitForChildExit(childPid); err != nil {
		return fmt.Errorf("error waiting for our first child to exit: %w", err)
	}

	// Spin up a goroutine to handle remapping mount requests by runc init.
	// There is no point doing this for rootless containers because they cannot
	// configure MOUNT_ATTR_IDMAP, nor do OPEN_TREE_CLONE. We could just
	// service plain-open requests for plain bind-mounts but there's no need
	// (rootless containers will never have permission issues on a source mount
	// that the parent process can help with -- they are the same user).
	var mountRequest mountSourceRequestFn
	if !p.container.config.RootlessEUID {
		request, cancel, err := p.goCreateMountSources(context.Background())
		if err != nil {
			return fmt.Errorf("error spawning mount remapping thread: %w", err)
		}
		defer cancel()
		mountRequest = request
	}

	if err := p.createNetworkInterfaces(); err != nil {
		return fmt.Errorf("error creating network interfaces: %w", err)
	}
	if err := p.updateSpecState(); err != nil {
		return fmt.Errorf("error updating spec state: %w", err)
	}
	if err := utils.WriteJSON(p.comm.initSockParent, p.config); err != nil {
		return fmt.Errorf("error sending config to init process: %w", err)
	}

	var seenProcReady bool
	ierr := parseSync(p.comm.syncSockParent, func(sync *syncT) error {
		switch sync.Type {
		case procMountPlease:
			if mountRequest == nil {
				return fmt.Errorf("cannot fulfil mount requests as a rootless user")
			}
			var m *configs.Mount
			if sync.Arg == nil {
				return fmt.Errorf("sync %q is missing an argument", sync.Type)
			}
			if err := json.Unmarshal(*sync.Arg, &m); err != nil {
				return fmt.Errorf("sync %q passed invalid mount arg: %w", sync.Type, err)
			}
			mnt, err := mountRequest(m)
			if err != nil {
				return fmt.Errorf("failed to fulfil mount request: %w", err)
			}
			defer mnt.file.Close()

			arg, err := json.Marshal(mnt)
			if err != nil {
				return fmt.Errorf("sync %q failed to marshal mountSource: %w", sync.Type, err)
			}
			argMsg := json.RawMessage(arg)
			if err := doWriteSync(p.comm.syncSockParent, syncT{
				Type: procMountFd,
				Arg:  &argMsg,
				File: mnt.file,
			}); err != nil {
				return err
			}
		case procSeccomp:
			if p.config.Config.Seccomp.ListenerPath == "" {
				return errors.New("seccomp listenerPath is not set")
			}
			var srcFd int
			if sync.Arg == nil {
				return fmt.Errorf("sync %q is missing an argument", sync.Type)
			}
			if err := json.Unmarshal(*sync.Arg, &srcFd); err != nil {
				return fmt.Errorf("sync %q passed invalid fd arg: %w", sync.Type, err)
			}
			seccompFd, err := pidGetFd(p.pid(), srcFd)
			if err != nil {
				return fmt.Errorf("sync %q get fd %d from child failed: %w", sync.Type, srcFd, err)
			}
			defer seccompFd.Close()
			// We have a copy, the child can keep working. We don't need to
			// wait for the seccomp notify listener to get the fd before we
			// permit the child to continue because the child will happily wait
			// for the listener if it hits SCMP_ACT_NOTIFY.
			if err := writeSync(p.comm.syncSockParent, procSeccompDone); err != nil {
				return err
			}

			s, err := p.container.currentOCIState()
			if err != nil {
				return err
			}

			// initProcessStartTime hasn't been set yet.
			s.Pid = p.cmd.Process.Pid
			s.Status = specs.StateCreating
			containerProcessState := &specs.ContainerProcessState{
				Version:  specs.Version,
				Fds:      []string{specs.SeccompFdName},
				Pid:      s.Pid,
				Metadata: p.config.Config.Seccomp.ListenerMetadata,
				State:    *s,
			}
			if err := sendContainerProcessState(p.config.Config.Seccomp.ListenerPath,
				containerProcessState, seccompFd); err != nil {
				return err
			}
		case procReady:
			seenProcReady = true
			// Set rlimits, this has to be done here because we lose permissions
			// to raise the limits once we enter a user-namespace
			if err := setupRlimits(p.config.Rlimits, p.pid()); err != nil {
				return fmt.Errorf("error setting rlimits for ready process: %w", err)
			}

			// generate a timestamp indicating when the container was started
			p.container.created = time.Now().UTC()
			p.container.state = &createdState{
				c: p.container,
			}

			// NOTE: If the procRun state has been synced and the
			// runc-create process has been killed for some reason,
			// the runc-init[2:stage] process will be leaky. And
			// the runc command also fails to parse root directory
			// because the container doesn't have state.json.
			//
			// In order to cleanup the runc-init[2:stage] by
			// runc-delete/stop, we should store the status before
			// procRun sync.
			state, uerr := p.container.updateState(p)
			if uerr != nil {
				return fmt.Errorf("unable to store init state: %w", uerr)
			}
			p.container.initProcessStartTime = state.InitProcessStartTime

			// Sync with child.
			if err := writeSync(p.comm.syncSockParent, procRun); err != nil {
				return err
			}
		case procHooks:
			// Setup cgroup before prestart hook, so that the prestart hook could apply cgroup permissions.
			if err := p.manager.Set(p.config.Config.Cgroups.Resources); err != nil {
				return fmt.Errorf("error setting cgroup config for procHooks process: %w", err)
			}
			if p.intelRdtManager != nil {
				if err := p.intelRdtManager.Set(p.config.Config); err != nil {
					return fmt.Errorf("error setting Intel RDT config for procHooks process: %w", err)
				}
			}
			if len(p.config.Config.Hooks) != 0 {
				s, err := p.container.currentOCIState()
				if err != nil {
					return err
				}
				// initProcessStartTime hasn't been set yet.
				s.Pid = p.cmd.Process.Pid
				s.Status = specs.StateCreating
				hooks := p.config.Config.Hooks

				if err := hooks.Run(configs.Prestart, s); err != nil {
					return err
				}
				if err := hooks.Run(configs.CreateRuntime, s); err != nil {
					return err
				}
			}
			// Sync with child.
			if err := writeSync(p.comm.syncSockParent, procHooksDone); err != nil {
				return err
			}
		default:
			return errors.New("invalid JSON payload from child")
		}
		return nil
	})

	if err := p.comm.syncSockParent.Shutdown(unix.SHUT_WR); err != nil && ierr == nil {
		return err
	}
	if !seenProcReady && ierr == nil {
		ierr = errors.New("procReady not received")
	}
	if ierr != nil {
		return fmt.Errorf("error during container init: %w", ierr)
	}
	return nil
}

func (p *initProcess) wait() (*os.ProcessState, error) {
	err := p.cmd.Wait()
	return p.cmd.ProcessState, err
}

func (p *initProcess) terminate() error {
	if p.cmd.Process == nil {
		return nil
	}
	err := p.cmd.Process.Kill()
	if _, werr := p.wait(); err == nil {
		err = werr
	}
	return err
}

func (p *initProcess) startTime() (uint64, error) {
	stat, err := system.Stat(p.pid())
	return stat.StartTime, err
}

func (p *initProcess) updateSpecState() error {
	s, err := p.container.currentOCIState()
	if err != nil {
		return err
	}

	p.config.SpecState = s
	return nil
}

func (p *initProcess) createNetworkInterfaces() error {
	for _, config := range p.config.Config.Networks {
		strategy, err := getStrategy(config.Type)
		if err != nil {
			return err
		}
		n := &network{
			Network: *config,
		}
		if err := strategy.create(n, p.pid()); err != nil {
			return err
		}
		p.config.Networks = append(p.config.Networks, n)
	}
	return nil
}

func (p *initProcess) signal(sig os.Signal) error {
	s, ok := sig.(unix.Signal)
	if !ok {
		return errors.New("os: unsupported signal type")
	}
	return unix.Kill(p.pid(), s)
}

func (p *initProcess) setExternalDescriptors(newFds []string) {
	p.fds = newFds
}

func (p *initProcess) forwardChildLogs() chan error {
	return logs.ForwardLogs(p.comm.logPipeParent)
}

func pidGetFd(pid, srcFd int) (*os.File, error) {
	pidFd, err := unix.PidfdOpen(pid, 0)
	if err != nil {
		return nil, os.NewSyscallError("pidfd_open", err)
	}
	defer unix.Close(pidFd)
	fd, err := unix.PidfdGetfd(pidFd, srcFd, 0)
	if err != nil {
		return nil, os.NewSyscallError("pidfd_getfd", err)
	}
	return os.NewFile(uintptr(fd), "[pidfd_getfd]"), nil
}

func sendContainerProcessState(listenerPath string, state *specs.ContainerProcessState, file *os.File) error {
	conn, err := net.Dial("unix", listenerPath)
	if err != nil {
		return fmt.Errorf("failed to connect with seccomp agent specified in the seccomp profile: %w", err)
	}

	socket, err := conn.(*net.UnixConn).File()
	if err != nil {
		return fmt.Errorf("cannot get seccomp socket: %w", err)
	}
	defer socket.Close()

	b, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("cannot marshall seccomp state: %w", err)
	}

	if err := utils.SendRawFd(socket, string(b), file.Fd()); err != nil {
		return fmt.Errorf("cannot send seccomp fd to %s: %w", listenerPath, err)
	}
	runtime.KeepAlive(file)
	return nil
}

func getPipeFds(pid int) ([]string, error) {
	fds := make([]string, 3)

	dirPath := filepath.Join("/proc", strconv.Itoa(pid), "/fd")
	for i := 0; i < 3; i++ {
		// XXX: This breaks if the path is not a valid symlink (which can
		//      happen in certain particularly unlucky mount namespace setups).
		f := filepath.Join(dirPath, strconv.Itoa(i))
		target, err := os.Readlink(f)
		if err != nil {
			// Ignore permission errors, for rootless containers and other
			// non-dumpable processes. if we can't get the fd for a particular
			// file, there's not much we can do.
			if os.IsPermission(err) {
				continue
			}
			return fds, err
		}
		fds[i] = target
	}
	return fds, nil
}

// InitializeIO creates pipes for use with the process's stdio and returns the
// opposite side for each. Do not use this if you want to have a pseudoterminal
// set up for you by libcontainer (TODO: fix that too).
// TODO: This is mostly unnecessary, and should be handled by clients.
func (p *Process) InitializeIO(rootuid, rootgid int) (i *IO, err error) {
	var fds []uintptr
	i = &IO{}
	// cleanup in case of an error
	defer func() {
		if err != nil {
			for _, fd := range fds {
				_ = unix.Close(int(fd))
			}
		}
	}()
	// STDIN
	r, w, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	fds = append(fds, r.Fd(), w.Fd())
	p.Stdin, i.Stdin = r, w
	// STDOUT
	if r, w, err = os.Pipe(); err != nil {
		return nil, err
	}
	fds = append(fds, r.Fd(), w.Fd())
	p.Stdout, i.Stdout = w, r
	// STDERR
	if r, w, err = os.Pipe(); err != nil {
		return nil, err
	}
	fds = append(fds, r.Fd(), w.Fd())
	p.Stderr, i.Stderr = w, r
	// change ownership of the pipes in case we are in a user namespace
	for _, fd := range fds {
		if err := unix.Fchown(int(fd), rootuid, rootgid); err != nil {
			return nil, &os.PathError{Op: "fchown", Path: "fd " + strconv.Itoa(int(fd)), Err: err}
		}
	}
	return i, nil
}

func setIOPriority(ioprio *configs.IOPriority) error {
	const ioprioWhoPgrp = 1

	class, ok := configs.IOPrioClassMapping[ioprio.Class]
	if !ok {
		return fmt.Errorf("invalid io priority class: %s", ioprio.Class)
	}

	// Combine class and priority into a single value
	// https://github.com/torvalds/linux/blob/v5.18/include/uapi/linux/ioprio.h#L5-L17
	iop := (class << 13) | ioprio.Priority
	_, _, errno := unix.RawSyscall(unix.SYS_IOPRIO_SET, ioprioWhoPgrp, 0, uintptr(iop))
	if errno != 0 {
		return fmt.Errorf("failed to set io priority: %w", errno)
	}
	return nil
}
