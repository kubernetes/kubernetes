package libcontainer

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fs2"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/intelrdt"
	"github.com/opencontainers/runc/libcontainer/logs"
	"github.com/opencontainers/runc/libcontainer/system"
	"github.com/opencontainers/runc/libcontainer/utils"
	"github.com/opencontainers/runtime-spec/specs-go"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
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

type filePair struct {
	parent *os.File
	child  *os.File
}

type setnsProcess struct {
	cmd             *exec.Cmd
	messageSockPair filePair
	logFilePair     filePair
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
	defer p.messageSockPair.parent.Close()
	// get the "before" value of oom kill count
	oom, _ := p.manager.OOMKillCount()
	err := p.cmd.Start()
	// close the write-side of the pipes (controlled by child)
	p.messageSockPair.child.Close()
	p.logFilePair.child.Close()
	if err != nil {
		return fmt.Errorf("error starting setns process: %w", err)
	}

	waitInit := initWaiter(p.messageSockPair.parent)
	defer func() {
		if retErr != nil {
			if newOom, err := p.manager.OOMKillCount(); err == nil && newOom != oom {
				// Someone in this cgroup was killed, this _might_ be us.
				retErr = fmt.Errorf("%w (possibly OOM-killed)", retErr)
			}
			werr := <-waitInit
			if werr != nil {
				logrus.WithError(werr).Warn()
			}
			err := ignoreTerminateErrors(p.terminate())
			if err != nil {
				logrus.WithError(err).Warn("unable to terminate setnsProcess")
			}
		}
	}()

	if p.bootstrapData != nil {
		if _, err := io.Copy(p.messageSockPair.parent, p.bootstrapData); err != nil {
			return fmt.Errorf("error copying bootstrap data to pipe: %w", err)
		}
	}
	err = <-waitInit
	if err != nil {
		return err
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
	// set rlimits, this has to be done here because we lose permissions
	// to raise the limits once we enter a user-namespace
	if err := setupRlimits(p.config.Rlimits, p.pid()); err != nil {
		return fmt.Errorf("error setting rlimits for process: %w", err)
	}
	if err := utils.WriteJSON(p.messageSockPair.parent, p.config); err != nil {
		return fmt.Errorf("error writing config to pipe: %w", err)
	}

	ierr := parseSync(p.messageSockPair.parent, func(sync *syncT) error {
		switch sync.Type {
		case procReady:
			// This shouldn't happen.
			panic("unexpected procReady in setns")
		case procHooks:
			// This shouldn't happen.
			panic("unexpected procHooks in setns")
		case procSeccomp:
			if p.config.Config.Seccomp.ListenerPath == "" {
				return errors.New("listenerPath is not set")
			}

			seccompFd, err := recvSeccompFd(uintptr(p.pid()), uintptr(sync.Fd))
			if err != nil {
				return err
			}
			defer unix.Close(seccompFd)

			bundle, annotations := utils.Annotations(p.config.Config.Labels)
			containerProcessState := &specs.ContainerProcessState{
				Version:  specs.Version,
				Fds:      []string{specs.SeccompFdName},
				Pid:      p.cmd.Process.Pid,
				Metadata: p.config.Config.Seccomp.ListenerMetadata,
				State: specs.State{
					Version:     specs.Version,
					ID:          p.config.ContainerId,
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

			// Sync with child.
			if err := writeSync(p.messageSockPair.parent, procSeccompDone); err != nil {
				return err
			}
			return nil
		default:
			return errors.New("invalid JSON payload from child")
		}
	})

	if err := unix.Shutdown(int(p.messageSockPair.parent.Fd()), unix.SHUT_WR); err != nil {
		return &os.PathError{Op: "shutdown", Path: "(init pipe)", Err: err}
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
	if err := json.NewDecoder(p.messageSockPair.parent).Decode(&pid); err != nil {
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
	return logs.ForwardLogs(p.logFilePair.parent)
}

type initProcess struct {
	cmd             *exec.Cmd
	messageSockPair filePair
	logFilePair     filePair
	config          *initConfig
	manager         cgroups.Manager
	intelRdtManager *intelrdt.Manager
	container       *linuxContainer
	fds             []string
	process         *Process
	bootstrapData   io.Reader
	sharePidns      bool
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
	if err := json.NewDecoder(p.messageSockPair.parent).Decode(&pid); err != nil {
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

func (p *initProcess) start() (retErr error) {
	defer p.messageSockPair.parent.Close() //nolint: errcheck
	err := p.cmd.Start()
	p.process.ops = p
	// close the write-side of the pipes (controlled by child)
	_ = p.messageSockPair.child.Close()
	_ = p.logFilePair.child.Close()
	if err != nil {
		p.process.ops = nil
		return fmt.Errorf("unable to start init: %w", err)
	}

	waitInit := initWaiter(p.messageSockPair.parent)
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

			werr := <-waitInit
			if werr != nil {
				logrus.WithError(werr).Warn()
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
		return fmt.Errorf("unable to apply cgroup configuration: %w", err)
	}
	if p.intelRdtManager != nil {
		if err := p.intelRdtManager.Apply(p.pid()); err != nil {
			return fmt.Errorf("unable to apply Intel RDT configuration: %w", err)
		}
	}
	if _, err := io.Copy(p.messageSockPair.parent, p.bootstrapData); err != nil {
		return fmt.Errorf("can't copy bootstrap data to pipe: %w", err)
	}
	err = <-waitInit
	if err != nil {
		return err
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

	if err := p.createNetworkInterfaces(); err != nil {
		return fmt.Errorf("error creating network interfaces: %w", err)
	}
	if err := p.updateSpecState(); err != nil {
		return fmt.Errorf("error updating spec state: %w", err)
	}
	if err := p.sendConfig(); err != nil {
		return fmt.Errorf("error sending config to init process: %w", err)
	}
	var (
		sentRun    bool
		sentResume bool
	)

	ierr := parseSync(p.messageSockPair.parent, func(sync *syncT) error {
		switch sync.Type {
		case procSeccomp:
			if p.config.Config.Seccomp.ListenerPath == "" {
				return errors.New("listenerPath is not set")
			}

			seccompFd, err := recvSeccompFd(uintptr(childPid), uintptr(sync.Fd))
			if err != nil {
				return err
			}
			defer unix.Close(seccompFd)

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

			// Sync with child.
			if err := writeSync(p.messageSockPair.parent, procSeccompDone); err != nil {
				return err
			}
		case procReady:
			// set rlimits, this has to be done here because we lose permissions
			// to raise the limits once we enter a user-namespace
			if err := setupRlimits(p.config.Rlimits, p.pid()); err != nil {
				return fmt.Errorf("error setting rlimits for ready process: %w", err)
			}
			// call prestart and CreateRuntime hooks
			if !p.config.Config.Namespaces.Contains(configs.NEWNS) {
				// Setup cgroup before the hook, so that the prestart and CreateRuntime hook could apply cgroup permissions.
				if err := p.manager.Set(p.config.Config.Cgroups.Resources); err != nil {
					return fmt.Errorf("error setting cgroup config for ready process: %w", err)
				}
				if p.intelRdtManager != nil {
					if err := p.intelRdtManager.Set(p.config.Config); err != nil {
						return fmt.Errorf("error setting Intel RDT config for ready process: %w", err)
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

					if err := hooks[configs.Prestart].RunHooks(s); err != nil {
						return err
					}
					if err := hooks[configs.CreateRuntime].RunHooks(s); err != nil {
						return err
					}
				}
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
				return fmt.Errorf("unable to store init state: %w", err)
			}
			p.container.initProcessStartTime = state.InitProcessStartTime

			// Sync with child.
			if err := writeSync(p.messageSockPair.parent, procRun); err != nil {
				return err
			}
			sentRun = true
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

				if err := hooks[configs.Prestart].RunHooks(s); err != nil {
					return err
				}
				if err := hooks[configs.CreateRuntime].RunHooks(s); err != nil {
					return err
				}
			}
			// Sync with child.
			if err := writeSync(p.messageSockPair.parent, procResume); err != nil {
				return err
			}
			sentResume = true
		default:
			return errors.New("invalid JSON payload from child")
		}

		return nil
	})

	if !sentRun {
		return fmt.Errorf("error during container init: %w", ierr)
	}
	if p.config.Config.Namespaces.Contains(configs.NEWNS) && !sentResume {
		return errors.New("could not synchronise after executing prestart and CreateRuntime hooks with container process")
	}
	if err := unix.Shutdown(int(p.messageSockPair.parent.Fd()), unix.SHUT_WR); err != nil {
		return &os.PathError{Op: "shutdown", Path: "(init pipe)", Err: err}
	}

	// Must be done after Shutdown so the child will exit and we can wait for it.
	if ierr != nil {
		_, _ = p.wait()
		return ierr
	}
	return nil
}

func (p *initProcess) wait() (*os.ProcessState, error) {
	err := p.cmd.Wait()
	// we should kill all processes in cgroup when init is died if we use host PID namespace
	if p.sharePidns {
		_ = signalAllProcesses(p.manager, unix.SIGKILL)
	}
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

func (p *initProcess) sendConfig() error {
	// send the config to the container's init process, we don't use JSON Encode
	// here because there might be a problem in JSON decoder in some cases, see:
	// https://github.com/docker/docker/issues/14203#issuecomment-174177790
	return utils.WriteJSON(p.messageSockPair.parent, p.config)
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
	return logs.ForwardLogs(p.logFilePair.parent)
}

func recvSeccompFd(childPid, childFd uintptr) (int, error) {
	pidfd, _, errno := unix.Syscall(unix.SYS_PIDFD_OPEN, childPid, 0, 0)
	if errno != 0 {
		return -1, fmt.Errorf("performing SYS_PIDFD_OPEN syscall: %w", errno)
	}
	defer unix.Close(int(pidfd))

	seccompFd, _, errno := unix.Syscall(unix.SYS_PIDFD_GETFD, pidfd, childFd, 0)
	if errno != 0 {
		return -1, fmt.Errorf("performing SYS_PIDFD_GETFD syscall: %w", errno)
	}

	return int(seccompFd), nil
}

func sendContainerProcessState(listenerPath string, state *specs.ContainerProcessState, fd int) error {
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

	err = utils.SendFds(socket, b, fd)
	if err != nil {
		return fmt.Errorf("cannot send seccomp fd to %s: %w", listenerPath, err)
	}

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

// initWaiter returns a channel to wait on for making sure
// runc init has finished the initial setup.
func initWaiter(r io.Reader) chan error {
	ch := make(chan error, 1)
	go func() {
		defer close(ch)

		inited := make([]byte, 1)
		n, err := r.Read(inited)
		if err == nil {
			if n < 1 {
				err = errors.New("short read")
			} else if inited[0] != 0 {
				err = fmt.Errorf("unexpected %d != 0", inited[0])
			} else {
				ch <- nil
				return
			}
		}
		ch <- fmt.Errorf("waiting for init preliminary setup: %w", err)
	}()

	return ch
}
