package libcontainer

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runtime-spec/specs-go"
	"golang.org/x/sys/unix"
)

func newStateTransitionError(from, to containerState) error {
	return &stateTransitionError{
		From: from.status().String(),
		To:   to.status().String(),
	}
}

// stateTransitionError is returned when an invalid state transition happens from one
// state to another.
type stateTransitionError struct {
	From string
	To   string
}

func (s *stateTransitionError) Error() string {
	return fmt.Sprintf("invalid state transition from %s to %s", s.From, s.To)
}

type containerState interface {
	transition(containerState) error
	destroy() error
	status() Status
}

func destroy(c *Container) error {
	// Usually, when a container init is gone, all other processes in its
	// cgroup are killed by the kernel. This is not the case for a shared
	// PID namespace container, which may have some processes left after
	// its init is killed or exited.
	//
	// As the container without init process running is considered stopped,
	// and destroy is supposed to remove all the container resources, we need
	// to kill those processes here.
	if !c.config.Namespaces.IsPrivate(configs.NEWPID) {
		_ = signalAllProcesses(c.cgroupManager, unix.SIGKILL)
	}
	if err := c.cgroupManager.Destroy(); err != nil {
		return fmt.Errorf("unable to remove container's cgroup: %w", err)
	}
	if c.intelRdtManager != nil {
		if err := c.intelRdtManager.Destroy(); err != nil {
			return fmt.Errorf("unable to remove container's IntelRDT group: %w", err)
		}
	}
	if err := os.RemoveAll(c.stateDir); err != nil {
		return fmt.Errorf("unable to remove container state dir: %w", err)
	}
	c.initProcess = nil
	err := runPoststopHooks(c)
	c.state = &stoppedState{c: c}
	return err
}

func runPoststopHooks(c *Container) error {
	hooks := c.config.Hooks
	if hooks == nil {
		return nil
	}

	s, err := c.currentOCIState()
	if err != nil {
		return err
	}
	s.Status = specs.StateStopped

	return hooks.Run(configs.Poststop, s)
}

// stoppedState represents a container is a stopped/destroyed state.
type stoppedState struct {
	c *Container
}

func (b *stoppedState) status() Status {
	return Stopped
}

func (b *stoppedState) transition(s containerState) error {
	switch s.(type) {
	case *runningState, *restoredState:
		b.c.state = s
		return nil
	case *stoppedState:
		return nil
	}
	return newStateTransitionError(b, s)
}

func (b *stoppedState) destroy() error {
	return destroy(b.c)
}

// runningState represents a container that is currently running.
type runningState struct {
	c *Container
}

func (r *runningState) status() Status {
	return Running
}

func (r *runningState) transition(s containerState) error {
	switch s.(type) {
	case *stoppedState:
		if r.c.hasInit() {
			return ErrRunning
		}
		r.c.state = s
		return nil
	case *pausedState:
		r.c.state = s
		return nil
	case *runningState:
		return nil
	}
	return newStateTransitionError(r, s)
}

func (r *runningState) destroy() error {
	if r.c.hasInit() {
		return ErrRunning
	}
	return destroy(r.c)
}

type createdState struct {
	c *Container
}

func (i *createdState) status() Status {
	return Created
}

func (i *createdState) transition(s containerState) error {
	switch s.(type) {
	case *runningState, *pausedState, *stoppedState:
		i.c.state = s
		return nil
	case *createdState:
		return nil
	}
	return newStateTransitionError(i, s)
}

func (i *createdState) destroy() error {
	_ = i.c.initProcess.signal(unix.SIGKILL)
	return destroy(i.c)
}

// pausedState represents a container that is currently pause.  It cannot be destroyed in a
// paused state and must transition back to running first.
type pausedState struct {
	c *Container
}

func (p *pausedState) status() Status {
	return Paused
}

func (p *pausedState) transition(s containerState) error {
	switch s.(type) {
	case *runningState, *stoppedState:
		p.c.state = s
		return nil
	case *pausedState:
		return nil
	}
	return newStateTransitionError(p, s)
}

func (p *pausedState) destroy() error {
	if p.c.hasInit() {
		return ErrPaused
	}
	if err := p.c.cgroupManager.Freeze(configs.Thawed); err != nil {
		return err
	}
	return destroy(p.c)
}

// restoredState is the same as the running state but also has associated checkpoint
// information that maybe need destroyed when the container is stopped and destroy is called.
type restoredState struct {
	imageDir string
	c        *Container
}

func (r *restoredState) status() Status {
	return Running
}

func (r *restoredState) transition(s containerState) error {
	switch s.(type) {
	case *stoppedState, *runningState:
		return nil
	}
	return newStateTransitionError(r, s)
}

func (r *restoredState) destroy() error {
	if _, err := os.Stat(filepath.Join(r.c.stateDir, "checkpoint")); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
	}
	return destroy(r.c)
}

// loadedState is used whenever a container is restored, loaded, or setting additional
// processes inside and it should not be destroyed when it is exiting.
type loadedState struct {
	c *Container
	s Status
}

func (n *loadedState) status() Status {
	return n.s
}

func (n *loadedState) transition(s containerState) error {
	n.c.state = s
	return nil
}

func (n *loadedState) destroy() error {
	if err := n.c.refreshState(); err != nil {
		return err
	}
	return n.c.state.destroy()
}
