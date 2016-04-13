// +build linux,cgo

package native

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/execdriver"
	"github.com/docker/docker/pkg/parsers"
	"github.com/docker/docker/pkg/pools"
	"github.com/docker/docker/pkg/reexec"
	sysinfo "github.com/docker/docker/pkg/system"
	"github.com/docker/docker/pkg/term"
	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/cgroups/systemd"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/system"
	"github.com/opencontainers/runc/libcontainer/utils"
)

const (
	DriverName = "native"
	Version    = "0.2"
)

type driver struct {
	root             string
	initPath         string
	activeContainers map[string]libcontainer.Container
	machineMemory    int64
	factory          libcontainer.Factory
	sync.Mutex
}

func NewDriver(root, initPath string, options []string) (*driver, error) {
	meminfo, err := sysinfo.ReadMemInfo()
	if err != nil {
		return nil, err
	}

	if err := sysinfo.MkdirAll(root, 0700); err != nil {
		return nil, err
	}

	// choose cgroup manager
	// this makes sure there are no breaking changes to people
	// who upgrade from versions without native.cgroupdriver opt
	cgm := libcontainer.Cgroupfs
	if systemd.UseSystemd() {
		cgm = libcontainer.SystemdCgroups
	}

	// parse the options
	for _, option := range options {
		key, val, err := parsers.ParseKeyValueOpt(option)
		if err != nil {
			return nil, err
		}
		key = strings.ToLower(key)
		switch key {
		case "native.cgroupdriver":
			// override the default if they set options
			switch val {
			case "systemd":
				if systemd.UseSystemd() {
					cgm = libcontainer.SystemdCgroups
				} else {
					// warn them that they chose the wrong driver
					logrus.Warn("You cannot use systemd as native.cgroupdriver, using cgroupfs instead")
				}
			case "cgroupfs":
				cgm = libcontainer.Cgroupfs
			default:
				return nil, fmt.Errorf("Unknown native.cgroupdriver given %q. try cgroupfs or systemd", val)
			}
		default:
			return nil, fmt.Errorf("Unknown option %s\n", key)
		}
	}

	f, err := libcontainer.New(
		root,
		cgm,
		libcontainer.InitPath(reexec.Self(), DriverName),
	)
	if err != nil {
		return nil, err
	}

	return &driver{
		root:             root,
		initPath:         initPath,
		activeContainers: make(map[string]libcontainer.Container),
		machineMemory:    meminfo.MemTotal,
		factory:          f,
	}, nil
}

type execOutput struct {
	exitCode int
	err      error
}

func (d *driver) Run(c *execdriver.Command, pipes *execdriver.Pipes, startCallback execdriver.StartCallback) (execdriver.ExitStatus, error) {
	// take the Command and populate the libcontainer.Config from it
	container, err := d.createContainer(c)
	if err != nil {
		return execdriver.ExitStatus{ExitCode: -1}, err
	}

	p := &libcontainer.Process{
		Args: append([]string{c.ProcessConfig.Entrypoint}, c.ProcessConfig.Arguments...),
		Env:  c.ProcessConfig.Env,
		Cwd:  c.WorkingDir,
		User: c.ProcessConfig.User,
	}

	if err := setupPipes(container, &c.ProcessConfig, p, pipes); err != nil {
		return execdriver.ExitStatus{ExitCode: -1}, err
	}

	cont, err := d.factory.Create(c.ID, container)
	if err != nil {
		return execdriver.ExitStatus{ExitCode: -1}, err
	}
	d.Lock()
	d.activeContainers[c.ID] = cont
	d.Unlock()
	defer func() {
		cont.Destroy()
		d.cleanContainer(c.ID)
	}()

	if err := cont.Start(p); err != nil {
		return execdriver.ExitStatus{ExitCode: -1}, err
	}

	if startCallback != nil {
		pid, err := p.Pid()
		if err != nil {
			p.Signal(os.Kill)
			p.Wait()
			return execdriver.ExitStatus{ExitCode: -1}, err
		}
		startCallback(&c.ProcessConfig, pid)
	}

	oom := notifyOnOOM(cont)
	waitF := p.Wait
	if nss := cont.Config().Namespaces; !nss.Contains(configs.NEWPID) {
		// we need such hack for tracking processes with inherited fds,
		// because cmd.Wait() waiting for all streams to be copied
		waitF = waitInPIDHost(p, cont)
	}
	ps, err := waitF()
	if err != nil {
		execErr, ok := err.(*exec.ExitError)
		if !ok {
			return execdriver.ExitStatus{ExitCode: -1}, err
		}
		ps = execErr.ProcessState
	}
	cont.Destroy()
	_, oomKill := <-oom
	return execdriver.ExitStatus{ExitCode: utils.ExitStatus(ps.Sys().(syscall.WaitStatus)), OOMKilled: oomKill}, nil
}

// notifyOnOOM returns a channel that signals if the container received an OOM notification
// for any process.  If it is unable to subscribe to OOM notifications then a closed
// channel is returned as it will be non-blocking and return the correct result when read.
func notifyOnOOM(container libcontainer.Container) <-chan struct{} {
	oom, err := container.NotifyOOM()
	if err != nil {
		logrus.Warnf("Your kernel does not support OOM notifications: %s", err)
		c := make(chan struct{})
		close(c)
		return c
	}
	return oom
}

func killCgroupProcs(c libcontainer.Container) {
	var procs []*os.Process
	if err := c.Pause(); err != nil {
		logrus.Warn(err)
	}
	pids, err := c.Processes()
	if err != nil {
		// don't care about childs if we can't get them, this is mostly because cgroup already deleted
		logrus.Warnf("Failed to get processes from container %s: %v", c.ID(), err)
	}
	for _, pid := range pids {
		if p, err := os.FindProcess(pid); err == nil {
			procs = append(procs, p)
			if err := p.Kill(); err != nil {
				logrus.Warn(err)
			}
		}
	}
	if err := c.Resume(); err != nil {
		logrus.Warn(err)
	}
	for _, p := range procs {
		if _, err := p.Wait(); err != nil {
			logrus.Warn(err)
		}
	}
}

func waitInPIDHost(p *libcontainer.Process, c libcontainer.Container) func() (*os.ProcessState, error) {
	return func() (*os.ProcessState, error) {
		pid, err := p.Pid()
		if err != nil {
			return nil, err
		}

		process, err := os.FindProcess(pid)
		s, err := process.Wait()
		if err != nil {
			execErr, ok := err.(*exec.ExitError)
			if !ok {
				return s, err
			}
			s = execErr.ProcessState
		}
		killCgroupProcs(c)
		p.Wait()
		return s, err
	}
}

func (d *driver) Kill(c *execdriver.Command, sig int) error {
	d.Lock()
	active := d.activeContainers[c.ID]
	d.Unlock()
	if active == nil {
		return fmt.Errorf("active container for %s does not exist", c.ID)
	}
	state, err := active.State()
	if err != nil {
		return err
	}
	return syscall.Kill(state.InitProcessPid, syscall.Signal(sig))
}

func (d *driver) Pause(c *execdriver.Command) error {
	d.Lock()
	active := d.activeContainers[c.ID]
	d.Unlock()
	if active == nil {
		return fmt.Errorf("active container for %s does not exist", c.ID)
	}
	return active.Pause()
}

func (d *driver) Unpause(c *execdriver.Command) error {
	d.Lock()
	active := d.activeContainers[c.ID]
	d.Unlock()
	if active == nil {
		return fmt.Errorf("active container for %s does not exist", c.ID)
	}
	return active.Resume()
}

func (d *driver) Terminate(c *execdriver.Command) error {
	defer d.cleanContainer(c.ID)
	container, err := d.factory.Load(c.ID)
	if err != nil {
		return err
	}
	defer container.Destroy()
	state, err := container.State()
	if err != nil {
		return err
	}
	pid := state.InitProcessPid
	currentStartTime, err := system.GetProcessStartTime(pid)
	if err != nil {
		return err
	}
	if state.InitProcessStartTime == currentStartTime {
		err = syscall.Kill(pid, 9)
		syscall.Wait4(pid, nil, 0, nil)
	}
	return err
}

func (d *driver) Info(id string) execdriver.Info {
	return &info{
		ID:     id,
		driver: d,
	}
}

func (d *driver) Name() string {
	return fmt.Sprintf("%s-%s", DriverName, Version)
}

func (d *driver) GetPidsForContainer(id string) ([]int, error) {
	d.Lock()
	active := d.activeContainers[id]
	d.Unlock()

	if active == nil {
		return nil, fmt.Errorf("active container for %s does not exist", id)
	}
	return active.Processes()
}

func (d *driver) cleanContainer(id string) error {
	d.Lock()
	delete(d.activeContainers, id)
	d.Unlock()
	return os.RemoveAll(filepath.Join(d.root, id))
}

func (d *driver) createContainerRoot(id string) error {
	return os.MkdirAll(filepath.Join(d.root, id), 0655)
}

func (d *driver) Clean(id string) error {
	return os.RemoveAll(filepath.Join(d.root, id))
}

func (d *driver) Stats(id string) (*execdriver.ResourceStats, error) {
	d.Lock()
	c := d.activeContainers[id]
	d.Unlock()
	if c == nil {
		return nil, execdriver.ErrNotRunning
	}
	now := time.Now()
	stats, err := c.Stats()
	if err != nil {
		return nil, err
	}
	memoryLimit := c.Config().Cgroups.Memory
	// if the container does not have any memory limit specified set the
	// limit to the machines memory
	if memoryLimit == 0 {
		memoryLimit = d.machineMemory
	}
	return &execdriver.ResourceStats{
		Stats:       stats,
		Read:        now,
		MemoryLimit: memoryLimit,
	}, nil
}

type TtyConsole struct {
	console libcontainer.Console
}

func NewTtyConsole(console libcontainer.Console, pipes *execdriver.Pipes) (*TtyConsole, error) {
	tty := &TtyConsole{
		console: console,
	}

	if err := tty.AttachPipes(pipes); err != nil {
		tty.Close()
		return nil, err
	}

	return tty, nil
}

func (t *TtyConsole) Resize(h, w int) error {
	return term.SetWinsize(t.console.Fd(), &term.Winsize{Height: uint16(h), Width: uint16(w)})
}

func (t *TtyConsole) AttachPipes(pipes *execdriver.Pipes) error {
	go func() {
		if wb, ok := pipes.Stdout.(interface {
			CloseWriters() error
		}); ok {
			defer wb.CloseWriters()
		}

		pools.Copy(pipes.Stdout, t.console)
	}()

	if pipes.Stdin != nil {
		go func() {
			pools.Copy(t.console, pipes.Stdin)

			pipes.Stdin.Close()
		}()
	}

	return nil
}

func (t *TtyConsole) Close() error {
	return t.console.Close()
}

func setupPipes(container *configs.Config, processConfig *execdriver.ProcessConfig, p *libcontainer.Process, pipes *execdriver.Pipes) error {
	var term execdriver.Terminal
	var err error

	if processConfig.Tty {
		rootuid, err := container.HostUID()
		if err != nil {
			return err
		}
		cons, err := p.NewConsole(rootuid)
		if err != nil {
			return err
		}
		term, err = NewTtyConsole(cons, pipes)
	} else {
		p.Stdout = pipes.Stdout
		p.Stderr = pipes.Stderr
		r, w, err := os.Pipe()
		if err != nil {
			return err
		}
		if pipes.Stdin != nil {
			go func() {
				io.Copy(w, pipes.Stdin)
				w.Close()
			}()
			p.Stdin = r
		}
		term = &execdriver.StdConsole{}
	}
	if err != nil {
		return err
	}
	processConfig.Terminal = term
	return nil
}
