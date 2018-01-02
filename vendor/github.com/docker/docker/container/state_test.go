package container

import (
	"context"
	"testing"
	"time"

	"github.com/docker/docker/api/types"
)

func TestIsValidHealthString(t *testing.T) {
	contexts := []struct {
		Health   string
		Expected bool
	}{
		{types.Healthy, true},
		{types.Unhealthy, true},
		{types.Starting, true},
		{types.NoHealthcheck, true},
		{"fail", false},
	}

	for _, c := range contexts {
		v := IsValidHealthString(c.Health)
		if v != c.Expected {
			t.Fatalf("Expected %t, but got %t", c.Expected, v)
		}
	}
}

func TestStateRunStop(t *testing.T) {
	s := NewState()

	// Begin another wait with WaitConditionRemoved. It should complete
	// within 200 milliseconds.
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	removalWait := s.Wait(ctx, WaitConditionRemoved)

	// Full lifecycle two times.
	for i := 1; i <= 2; i++ {
		// A wait with WaitConditionNotRunning should return
		// immediately since the state is now either "created" (on the
		// first iteration) or "exited" (on the second iteration). It
		// shouldn't take more than 50 milliseconds.
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()
		// Expectx exit code to be i-1 since it should be the exit
		// code from the previous loop or 0 for the created state.
		if status := <-s.Wait(ctx, WaitConditionNotRunning); status.ExitCode() != i-1 {
			t.Fatalf("ExitCode %v, expected %v, err %q", status.ExitCode(), i-1, status.Err())
		}

		// A wait with WaitConditionNextExit should block until the
		// container has started and exited. It shouldn't take more
		// than 100 milliseconds.
		ctx, cancel = context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		initialWait := s.Wait(ctx, WaitConditionNextExit)

		// Set the state to "Running".
		s.Lock()
		s.SetRunning(i, true)
		s.Unlock()

		// Assert desired state.
		if !s.IsRunning() {
			t.Fatal("State not running")
		}
		if s.Pid != i {
			t.Fatalf("Pid %v, expected %v", s.Pid, i)
		}
		if s.ExitCode() != 0 {
			t.Fatalf("ExitCode %v, expected 0", s.ExitCode())
		}

		// Now that it's running, a wait with WaitConditionNotRunning
		// should block until we stop the container. It shouldn't take
		// more than 100 milliseconds.
		ctx, cancel = context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		exitWait := s.Wait(ctx, WaitConditionNotRunning)

		// Set the state to "Exited".
		s.Lock()
		s.SetStopped(&ExitStatus{ExitCode: i})
		s.Unlock()

		// Assert desired state.
		if s.IsRunning() {
			t.Fatal("State is running")
		}
		if s.ExitCode() != i {
			t.Fatalf("ExitCode %v, expected %v", s.ExitCode(), i)
		}
		if s.Pid != 0 {
			t.Fatalf("Pid %v, expected 0", s.Pid)
		}

		// Receive the initialWait result.
		if status := <-initialWait; status.ExitCode() != i {
			t.Fatalf("ExitCode %v, expected %v, err %q", status.ExitCode(), i, status.Err())
		}

		// Receive the exitWait result.
		if status := <-exitWait; status.ExitCode() != i {
			t.Fatalf("ExitCode %v, expected %v, err %q", status.ExitCode(), i, status.Err())
		}
	}

	// Set the state to dead and removed.
	s.SetDead()
	s.SetRemoved()

	// Wait for removed status or timeout.
	if status := <-removalWait; status.ExitCode() != 2 {
		// Should have the final exit code from the loop.
		t.Fatalf("Removal wait exitCode %v, expected %v, err %q", status.ExitCode(), 2, status.Err())
	}
}

func TestStateTimeoutWait(t *testing.T) {
	s := NewState()

	s.Lock()
	s.SetRunning(0, true)
	s.Unlock()

	// Start a wait with a timeout.
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	waitC := s.Wait(ctx, WaitConditionNotRunning)

	// It should timeout *before* this 200ms timer does.
	select {
	case <-time.After(200 * time.Millisecond):
		t.Fatal("Stop callback doesn't fire in 200 milliseconds")
	case status := <-waitC:
		t.Log("Stop callback fired")
		// Should be a timeout error.
		if status.Err() == nil {
			t.Fatal("expected timeout error, got nil")
		}
		if status.ExitCode() != -1 {
			t.Fatalf("expected exit code %v, got %v", -1, status.ExitCode())
		}
	}

	s.Lock()
	s.SetStopped(&ExitStatus{ExitCode: 0})
	s.Unlock()

	// Start another wait with a timeout. This one should return
	// immediately.
	ctx, cancel = context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	waitC = s.Wait(ctx, WaitConditionNotRunning)

	select {
	case <-time.After(200 * time.Millisecond):
		t.Fatal("Stop callback doesn't fire in 200 milliseconds")
	case status := <-waitC:
		t.Log("Stop callback fired")
		if status.ExitCode() != 0 {
			t.Fatalf("expected exit code %v, got %v, err %q", 0, status.ExitCode(), status.Err())
		}
	}
}
