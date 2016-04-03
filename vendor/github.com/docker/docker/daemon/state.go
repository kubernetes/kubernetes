package daemon

import (
	"fmt"
	"sync"
	"time"

	"github.com/docker/docker/daemon/execdriver"
	"github.com/docker/docker/pkg/units"
)

type State struct {
	sync.Mutex
	Running           bool
	Paused            bool
	Restarting        bool
	OOMKilled         bool
	removalInProgress bool // Not need for this to be persistent on disk.
	Dead              bool
	Pid               int
	ExitCode          int
	Error             string // contains last known error when starting the container
	StartedAt         time.Time
	FinishedAt        time.Time
	waitChan          chan struct{}
}

func NewState() *State {
	return &State{
		waitChan: make(chan struct{}),
	}
}

// String returns a human-readable description of the state
func (s *State) String() string {
	if s.Running {
		if s.Paused {
			return fmt.Sprintf("Up %s (Paused)", units.HumanDuration(time.Now().UTC().Sub(s.StartedAt)))
		}
		if s.Restarting {
			return fmt.Sprintf("Restarting (%d) %s ago", s.ExitCode, units.HumanDuration(time.Now().UTC().Sub(s.FinishedAt)))
		}

		return fmt.Sprintf("Up %s", units.HumanDuration(time.Now().UTC().Sub(s.StartedAt)))
	}

	if s.removalInProgress {
		return "Removal In Progress"
	}

	if s.Dead {
		return "Dead"
	}

	if s.StartedAt.IsZero() {
		return "Created"
	}

	if s.FinishedAt.IsZero() {
		return ""
	}

	return fmt.Sprintf("Exited (%d) %s ago", s.ExitCode, units.HumanDuration(time.Now().UTC().Sub(s.FinishedAt)))
}

// StateString returns a single string to describe state
func (s *State) StateString() string {
	if s.Running {
		if s.Paused {
			return "paused"
		}
		if s.Restarting {
			return "restarting"
		}
		return "running"
	}

	if s.Dead {
		return "dead"
	}

	if s.StartedAt.IsZero() {
		return "created"
	}

	return "exited"
}

func isValidStateString(s string) bool {
	if s != "paused" &&
		s != "restarting" &&
		s != "running" &&
		s != "dead" &&
		s != "created" &&
		s != "exited" {
		return false
	}
	return true
}

func wait(waitChan <-chan struct{}, timeout time.Duration) error {
	if timeout < 0 {
		<-waitChan
		return nil
	}
	select {
	case <-time.After(timeout):
		return fmt.Errorf("Timed out: %v", timeout)
	case <-waitChan:
		return nil
	}
}

// WaitRunning waits until state is running. If state already running it returns
// immediately. If you want wait forever you must supply negative timeout.
// Returns pid, that was passed to SetRunning
func (s *State) WaitRunning(timeout time.Duration) (int, error) {
	s.Lock()
	if s.Running {
		pid := s.Pid
		s.Unlock()
		return pid, nil
	}
	waitChan := s.waitChan
	s.Unlock()
	if err := wait(waitChan, timeout); err != nil {
		return -1, err
	}
	return s.GetPid(), nil
}

// WaitStop waits until state is stopped. If state already stopped it returns
// immediately. If you want wait forever you must supply negative timeout.
// Returns exit code, that was passed to SetStopped
func (s *State) WaitStop(timeout time.Duration) (int, error) {
	s.Lock()
	if !s.Running {
		exitCode := s.ExitCode
		s.Unlock()
		return exitCode, nil
	}
	waitChan := s.waitChan
	s.Unlock()
	if err := wait(waitChan, timeout); err != nil {
		return -1, err
	}
	return s.GetExitCode(), nil
}

func (s *State) IsRunning() bool {
	s.Lock()
	res := s.Running
	s.Unlock()
	return res
}

func (s *State) GetPid() int {
	s.Lock()
	res := s.Pid
	s.Unlock()
	return res
}

func (s *State) GetExitCode() int {
	s.Lock()
	res := s.ExitCode
	s.Unlock()
	return res
}

func (s *State) SetRunning(pid int) {
	s.Lock()
	s.setRunning(pid)
	s.Unlock()
}

func (s *State) setRunning(pid int) {
	s.Error = ""
	s.Running = true
	s.Paused = false
	s.Restarting = false
	s.ExitCode = 0
	s.Pid = pid
	s.StartedAt = time.Now().UTC()
	close(s.waitChan) // fire waiters for start
	s.waitChan = make(chan struct{})
}

func (s *State) SetStopped(exitStatus *execdriver.ExitStatus) {
	s.Lock()
	s.setStopped(exitStatus)
	s.Unlock()
}

func (s *State) setStopped(exitStatus *execdriver.ExitStatus) {
	s.Running = false
	s.Restarting = false
	s.Pid = 0
	s.FinishedAt = time.Now().UTC()
	s.ExitCode = exitStatus.ExitCode
	s.OOMKilled = exitStatus.OOMKilled
	close(s.waitChan) // fire waiters for stop
	s.waitChan = make(chan struct{})
}

// SetRestarting is when docker handles the auto restart of containers when they are
// in the middle of a stop and being restarted again
func (s *State) SetRestarting(exitStatus *execdriver.ExitStatus) {
	s.Lock()
	// we should consider the container running when it is restarting because of
	// all the checks in docker around rm/stop/etc
	s.Running = true
	s.Restarting = true
	s.Pid = 0
	s.FinishedAt = time.Now().UTC()
	s.ExitCode = exitStatus.ExitCode
	s.OOMKilled = exitStatus.OOMKilled
	close(s.waitChan) // fire waiters for stop
	s.waitChan = make(chan struct{})
	s.Unlock()
}

// setError sets the container's error state. This is useful when we want to
// know the error that occurred when container transits to another state
// when inspecting it
func (s *State) setError(err error) {
	s.Error = err.Error()
}

func (s *State) IsRestarting() bool {
	s.Lock()
	res := s.Restarting
	s.Unlock()
	return res
}

func (s *State) SetPaused() {
	s.Lock()
	s.Paused = true
	s.Unlock()
}

func (s *State) SetUnpaused() {
	s.Lock()
	s.Paused = false
	s.Unlock()
}

func (s *State) IsPaused() bool {
	s.Lock()
	res := s.Paused
	s.Unlock()
	return res
}

func (s *State) SetRemovalInProgress() error {
	s.Lock()
	defer s.Unlock()
	if s.removalInProgress {
		return fmt.Errorf("Status is already RemovalInProgress")
	}
	s.removalInProgress = true
	return nil
}

func (s *State) ResetRemovalInProgress() {
	s.Lock()
	s.removalInProgress = false
	s.Unlock()
}

func (s *State) SetDead() {
	s.Lock()
	s.Dead = true
	s.Unlock()
}
