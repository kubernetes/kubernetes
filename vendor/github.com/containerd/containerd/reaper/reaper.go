// +build !windows

package reaper

import (
	"os/exec"
	"sync"
	"time"

	"github.com/containerd/containerd/sys"
	runc "github.com/containerd/go-runc"
	"github.com/pkg/errors"
)

// ErrNoSuchProcess is returned when the process no longer exists
var ErrNoSuchProcess = errors.New("no such process")

const bufferSize = 2048

// Reap should be called when the process receives an SIGCHLD.  Reap will reap
// all exited processes and close their wait channels
func Reap() error {
	now := time.Now()
	exits, err := sys.Reap(false)
	Default.Lock()
	for c := range Default.subscribers {
		for _, e := range exits {
			c <- runc.Exit{
				Timestamp: now,
				Pid:       e.Pid,
				Status:    e.Status,
			}
		}

	}
	Default.Unlock()
	return err
}

// Default is the default monitor initialized for the package
var Default = &Monitor{
	subscribers: make(map[chan runc.Exit]struct{}),
}

// Monitor monitors the underlying system for process status changes
type Monitor struct {
	sync.Mutex

	subscribers map[chan runc.Exit]struct{}
}

// Start starts the command a registers the process with the reaper
func (m *Monitor) Start(c *exec.Cmd) (chan runc.Exit, error) {
	ec := m.Subscribe()
	if err := c.Start(); err != nil {
		m.Unsubscribe(ec)
		return nil, err
	}
	return ec, nil
}

// Wait blocks until a process is signal as dead.
// User should rely on the value of the exit status to determine if the
// command was successful or not.
func (m *Monitor) Wait(c *exec.Cmd, ec chan runc.Exit) (int, error) {
	for e := range ec {
		if e.Pid == c.Process.Pid {
			// make sure we flush all IO
			c.Wait()
			m.Unsubscribe(ec)
			return e.Status, nil
		}
	}
	// return no such process if the ec channel is closed and no more exit
	// events will be sent
	return -1, ErrNoSuchProcess
}

// Subscribe to process exit changes
func (m *Monitor) Subscribe() chan runc.Exit {
	c := make(chan runc.Exit, bufferSize)
	m.Lock()
	m.subscribers[c] = struct{}{}
	m.Unlock()
	return c
}

// Unsubscribe to process exit changes
func (m *Monitor) Unsubscribe(c chan runc.Exit) {
	m.Lock()
	delete(m.subscribers, c)
	close(c)
	m.Unlock()
}
