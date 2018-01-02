// +build !windows

package shim

import (
	"context"

	"github.com/containerd/console"
	"github.com/pkg/errors"
)

type execCreatedState struct {
	p *execProcess
}

func (s *execCreatedState) transition(name string) error {
	switch name {
	case "running":
		s.p.processState = &execRunningState{p: s.p}
	case "stopped":
		s.p.processState = &execStoppedState{p: s.p}
	case "deleted":
		s.p.processState = &deletedState{}
	default:
		return errors.Errorf("invalid state transition %q to %q", stateName(s), name)
	}
	return nil
}

func (s *execCreatedState) Resize(ws console.WinSize) error {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()

	return s.p.resize(ws)
}

func (s *execCreatedState) Start(ctx context.Context) error {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()
	if err := s.p.start(ctx); err != nil {
		return err
	}
	return s.transition("running")
}

func (s *execCreatedState) Delete(ctx context.Context) error {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()
	if err := s.p.delete(ctx); err != nil {
		return err
	}
	return s.transition("deleted")
}

func (s *execCreatedState) Kill(ctx context.Context, sig uint32, all bool) error {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()

	return s.p.kill(ctx, sig, all)
}

func (s *execCreatedState) SetExited(status int) {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()

	s.p.setExited(status)

	if err := s.transition("stopped"); err != nil {
		panic(err)
	}
}

type execRunningState struct {
	p *execProcess
}

func (s *execRunningState) transition(name string) error {
	switch name {
	case "stopped":
		s.p.processState = &execStoppedState{p: s.p}
	default:
		return errors.Errorf("invalid state transition %q to %q", stateName(s), name)
	}
	return nil
}

func (s *execRunningState) Resize(ws console.WinSize) error {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()

	return s.p.resize(ws)
}

func (s *execRunningState) Start(ctx context.Context) error {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()

	return errors.Errorf("cannot start a running process")
}

func (s *execRunningState) Delete(ctx context.Context) error {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()

	return errors.Errorf("cannot delete a running process")
}

func (s *execRunningState) Kill(ctx context.Context, sig uint32, all bool) error {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()

	return s.p.kill(ctx, sig, all)
}

func (s *execRunningState) SetExited(status int) {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()

	s.p.setExited(status)

	if err := s.transition("stopped"); err != nil {
		panic(err)
	}
}

type execStoppedState struct {
	p *execProcess
}

func (s *execStoppedState) transition(name string) error {
	switch name {
	case "deleted":
		s.p.processState = &deletedState{}
	default:
		return errors.Errorf("invalid state transition %q to %q", stateName(s), name)
	}
	return nil
}

func (s *execStoppedState) Resize(ws console.WinSize) error {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()

	return errors.Errorf("cannot resize a stopped container")
}

func (s *execStoppedState) Start(ctx context.Context) error {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()

	return errors.Errorf("cannot start a stopped process")
}

func (s *execStoppedState) Delete(ctx context.Context) error {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()
	if err := s.p.delete(ctx); err != nil {
		return err
	}
	return s.transition("deleted")
}

func (s *execStoppedState) Kill(ctx context.Context, sig uint32, all bool) error {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()

	return s.p.kill(ctx, sig, all)
}

func (s *execStoppedState) SetExited(status int) {
	// no op
}
