package restartmanager

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/docker/docker/api/types/container"
)

const (
	backoffMultiplier = 2
	defaultTimeout    = 100 * time.Millisecond
	maxRestartTimeout = 1 * time.Minute
)

// ErrRestartCanceled is returned when the restart manager has been
// canceled and will no longer restart the container.
var ErrRestartCanceled = errors.New("restart canceled")

// RestartManager defines object that controls container restarting rules.
type RestartManager interface {
	Cancel() error
	ShouldRestart(exitCode uint32, hasBeenManuallyStopped bool, executionDuration time.Duration) (bool, chan error, error)
}

type restartManager struct {
	sync.Mutex
	sync.Once
	policy       container.RestartPolicy
	restartCount int
	timeout      time.Duration
	active       bool
	cancel       chan struct{}
	canceled     bool
}

// New returns a new restartManager based on a policy.
func New(policy container.RestartPolicy, restartCount int) RestartManager {
	return &restartManager{policy: policy, restartCount: restartCount, cancel: make(chan struct{})}
}

func (rm *restartManager) SetPolicy(policy container.RestartPolicy) {
	rm.Lock()
	rm.policy = policy
	rm.Unlock()
}

func (rm *restartManager) ShouldRestart(exitCode uint32, hasBeenManuallyStopped bool, executionDuration time.Duration) (bool, chan error, error) {
	if rm.policy.IsNone() {
		return false, nil, nil
	}
	rm.Lock()
	unlockOnExit := true
	defer func() {
		if unlockOnExit {
			rm.Unlock()
		}
	}()

	if rm.canceled {
		return false, nil, ErrRestartCanceled
	}

	if rm.active {
		return false, nil, fmt.Errorf("invalid call on an active restart manager")
	}
	// if the container ran for more than 10s, regardless of status and policy reset the
	// the timeout back to the default.
	if executionDuration.Seconds() >= 10 {
		rm.timeout = 0
	}
	switch {
	case rm.timeout == 0:
		rm.timeout = defaultTimeout
	case rm.timeout < maxRestartTimeout:
		rm.timeout *= backoffMultiplier
	}
	if rm.timeout > maxRestartTimeout {
		rm.timeout = maxRestartTimeout
	}

	var restart bool
	switch {
	case rm.policy.IsAlways():
		restart = true
	case rm.policy.IsUnlessStopped() && !hasBeenManuallyStopped:
		restart = true
	case rm.policy.IsOnFailure():
		// the default value of 0 for MaximumRetryCount means that we will not enforce a maximum count
		if max := rm.policy.MaximumRetryCount; max == 0 || rm.restartCount < max {
			restart = exitCode != 0
		}
	}

	if !restart {
		rm.active = false
		return false, nil, nil
	}

	rm.restartCount++

	unlockOnExit = false
	rm.active = true
	rm.Unlock()

	ch := make(chan error)
	go func() {
		select {
		case <-rm.cancel:
			ch <- ErrRestartCanceled
			close(ch)
		case <-time.After(rm.timeout):
			rm.Lock()
			close(ch)
			rm.active = false
			rm.Unlock()
		}
	}()

	return true, ch, nil
}

func (rm *restartManager) Cancel() error {
	rm.Do(func() {
		rm.Lock()
		rm.canceled = true
		close(rm.cancel)
		rm.Unlock()
	})
	return nil
}
