// +build linux

package libcontainer

import "testing"

func TestStateStatus(t *testing.T) {
	states := map[containerState]Status{
		&stoppedState{}:  Destroyed,
		&runningState{}:  Running,
		&restoredState{}: Running,
		&pausedState{}:   Paused,
	}
	for s, status := range states {
		if s.status() != status {
			t.Fatalf("state returned %s but expected %s", s.status(), status)
		}
	}
}

func isStateTransitionError(err error) bool {
	_, ok := err.(*stateTransitionError)
	return ok
}

func TestStoppedStateTransition(t *testing.T) {
	s := &stoppedState{c: &linuxContainer{}}
	valid := []containerState{
		&stoppedState{},
		&runningState{},
		&restoredState{},
	}
	for _, v := range valid {
		if err := s.transition(v); err != nil {
			t.Fatal(err)
		}
	}
	err := s.transition(&pausedState{})
	if err == nil {
		t.Fatal("transition to paused state should fail")
	}
	if !isStateTransitionError(err) {
		t.Fatal("expected stateTransitionError")
	}
}

func TestPausedStateTransition(t *testing.T) {
	s := &pausedState{c: &linuxContainer{}}
	valid := []containerState{
		&pausedState{},
		&runningState{},
		&stoppedState{},
	}
	for _, v := range valid {
		if err := s.transition(v); err != nil {
			t.Fatal(err)
		}
	}
}

func TestRestoredStateTransition(t *testing.T) {
	s := &restoredState{c: &linuxContainer{}}
	valid := []containerState{
		&stoppedState{},
		&runningState{},
	}
	for _, v := range valid {
		if err := s.transition(v); err != nil {
			t.Fatal(err)
		}
	}
	err := s.transition(&createdState{})
	if err == nil {
		t.Fatal("transition to created state should fail")
	}
	if !isStateTransitionError(err) {
		t.Fatal("expected stateTransitionError")
	}
}
