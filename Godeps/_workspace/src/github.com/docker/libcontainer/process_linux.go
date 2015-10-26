// +build linux

package libcontainer

import (
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"syscall"

	"github.com/docker/libcontainer/cgroups"
	"github.com/docker/libcontainer/system"
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

	// startTime return's the process start time.
	startTime() (string, error)

	signal(os.Signal) error
}

type setnsProcess struct {
	cmd         *exec.Cmd
	parentPipe  *os.File
	childPipe   *os.File
	cgroupPaths map[string]string
	config      *initConfig
}

func (p *setnsProcess) startTime() (string, error) {
	return system.GetProcessStartTime(p.pid())
}

func (p *setnsProcess) signal(sig os.Signal) error {
	s, ok := sig.(syscall.Signal)
	if !ok {
		return errors.New("os: unsupported signal type")
	}
	return syscall.Kill(p.cmd.Process.Pid, s)
}

func (p *setnsProcess) start() (err error) {
	defer p.parentPipe.Close()
	if err = p.execSetns(); err != nil {
		return newSystemError(err)
	}
	if len(p.cgroupPaths) > 0 {
		if err := cgroups.EnterPid(p.cgroupPaths, p.cmd.Process.Pid); err != nil {
			return newSystemError(err)
		}
	}
	if err := json.NewEncoder(p.parentPipe).Encode(p.config); err != nil {
		return newSystemError(err)
	}
	if err := syscall.Shutdown(int(p.parentPipe.Fd()), syscall.SHUT_WR); err != nil {
		return newSystemError(err)
	}
	// wait for the child process to fully complete and receive an error message
	// if one was encoutered
	var ierr *genericError
	if err := json.NewDecoder(p.parentPipe).Decode(&ierr); err != nil && err != io.EOF {
		return newSystemError(err)
	}
	if ierr != nil {
		return newSystemError(ierr)
	}

	return nil
}

// execSetns runs the process that executes C code to perform the setns calls
// because setns support requires the C process to fork off a child and perform the setns
// before the go runtime boots, we wait on the process to die and receive the child's pid
// over the provided pipe.
func (p *setnsProcess) execSetns() error {
	err := p.cmd.Start()
	p.childPipe.Close()
	if err != nil {
		return newSystemError(err)
	}
	status, err := p.cmd.Process.Wait()
	if err != nil {
		p.cmd.Wait()
		return newSystemError(err)
	}
	if !status.Success() {
		p.cmd.Wait()
		return newSystemError(&exec.ExitError{ProcessState: status})
	}
	var pid *pid
	if err := json.NewDecoder(p.parentPipe).Decode(&pid); err != nil {
		p.cmd.Wait()
		return newSystemError(err)
	}

	process, err := os.FindProcess(pid.Pid)
	if err != nil {
		return err
	}

	p.cmd.Process = process
	return nil
}

// terminate sends a SIGKILL to the forked process for the setns routine then waits to
// avoid the process becomming a zombie.
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
	if err != nil {
		return p.cmd.ProcessState, err
	}

	return p.cmd.ProcessState, nil
}

func (p *setnsProcess) pid() int {
	return p.cmd.Process.Pid
}

type initProcess struct {
	cmd        *exec.Cmd
	parentPipe *os.File
	childPipe  *os.File
	config     *initConfig
	manager    cgroups.Manager
}

func (p *initProcess) pid() int {
	return p.cmd.Process.Pid
}

func (p *initProcess) start() error {
	defer p.parentPipe.Close()
	err := p.cmd.Start()
	p.childPipe.Close()
	if err != nil {
		return newSystemError(err)
	}
	// Do this before syncing with child so that no children
	// can escape the cgroup
	if err := p.manager.Apply(p.pid()); err != nil {
		return newSystemError(err)
	}
	defer func() {
		if err != nil {
			// TODO: should not be the responsibility to call here
			p.manager.Destroy()
		}
	}()
	if err := p.createNetworkInterfaces(); err != nil {
		return newSystemError(err)
	}
	if err := p.sendConfig(); err != nil {
		return newSystemError(err)
	}
	// wait for the child process to fully complete and receive an error message
	// if one was encoutered
	var ierr *genericError
	if err := json.NewDecoder(p.parentPipe).Decode(&ierr); err != nil && err != io.EOF {
		return newSystemError(err)
	}
	if ierr != nil {
		return newSystemError(ierr)
	}
	return nil
}

func (p *initProcess) wait() (*os.ProcessState, error) {
	err := p.cmd.Wait()
	if err != nil {
		return p.cmd.ProcessState, err
	}
	// we should kill all processes in cgroup when init is died if we use host PID namespace
	if p.cmd.SysProcAttr.Cloneflags&syscall.CLONE_NEWPID == 0 {
		killCgroupProcesses(p.manager)
	}
	return p.cmd.ProcessState, nil
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

func (p *initProcess) startTime() (string, error) {
	return system.GetProcessStartTime(p.pid())
}

func (p *initProcess) sendConfig() error {
	// send the state to the container's init process then shutdown writes for the parent
	if err := json.NewEncoder(p.parentPipe).Encode(p.config); err != nil {
		return err
	}
	// shutdown writes for the parent side of the pipe
	return syscall.Shutdown(int(p.parentPipe.Fd()), syscall.SHUT_WR)
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
	s, ok := sig.(syscall.Signal)
	if !ok {
		return errors.New("os: unsupported signal type")
	}
	return syscall.Kill(p.cmd.Process.Pid, s)
}
