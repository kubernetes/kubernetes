package daemon

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/docker/docker/daemon/execdriver"
)

func TestStateRunStop(t *testing.T) {
	s := NewState()
	for i := 1; i < 3; i++ { // full lifecycle two times
		started := make(chan struct{})
		var pid int64
		go func() {
			runPid, _ := s.WaitRunning(-1 * time.Second)
			atomic.StoreInt64(&pid, int64(runPid))
			close(started)
		}()
		s.SetRunning(i + 100)
		if !s.IsRunning() {
			t.Fatal("State not running")
		}
		if s.Pid != i+100 {
			t.Fatalf("Pid %v, expected %v", s.Pid, i+100)
		}
		if s.ExitCode != 0 {
			t.Fatalf("ExitCode %v, expected 0", s.ExitCode)
		}
		select {
		case <-time.After(100 * time.Millisecond):
			t.Fatal("Start callback doesn't fire in 100 milliseconds")
		case <-started:
			t.Log("Start callback fired")
		}
		runPid := int(atomic.LoadInt64(&pid))
		if runPid != i+100 {
			t.Fatalf("Pid %v, expected %v", runPid, i+100)
		}
		if pid, err := s.WaitRunning(-1 * time.Second); err != nil || pid != i+100 {
			t.Fatalf("WaitRunning returned pid: %v, err: %v, expected pid: %v, err: %v", pid, err, i+100, nil)
		}

		stopped := make(chan struct{})
		var exit int64
		go func() {
			exitCode, _ := s.WaitStop(-1 * time.Second)
			atomic.StoreInt64(&exit, int64(exitCode))
			close(stopped)
		}()
		s.SetStopped(&execdriver.ExitStatus{ExitCode: i})
		if s.IsRunning() {
			t.Fatal("State is running")
		}
		if s.ExitCode != i {
			t.Fatalf("ExitCode %v, expected %v", s.ExitCode, i)
		}
		if s.Pid != 0 {
			t.Fatalf("Pid %v, expected 0", s.Pid)
		}
		select {
		case <-time.After(100 * time.Millisecond):
			t.Fatal("Stop callback doesn't fire in 100 milliseconds")
		case <-stopped:
			t.Log("Stop callback fired")
		}
		exitCode := int(atomic.LoadInt64(&exit))
		if exitCode != i {
			t.Fatalf("ExitCode %v, expected %v", exitCode, i)
		}
		if exitCode, err := s.WaitStop(-1 * time.Second); err != nil || exitCode != i {
			t.Fatalf("WaitStop returned exitCode: %v, err: %v, expected exitCode: %v, err: %v", exitCode, err, i, nil)
		}
	}
}

func TestStateTimeoutWait(t *testing.T) {
	s := NewState()
	started := make(chan struct{})
	go func() {
		s.WaitRunning(100 * time.Millisecond)
		close(started)
	}()
	select {
	case <-time.After(200 * time.Millisecond):
		t.Fatal("Start callback doesn't fire in 100 milliseconds")
	case <-started:
		t.Log("Start callback fired")
	}
	s.SetRunning(42)
	stopped := make(chan struct{})
	go func() {
		s.WaitRunning(100 * time.Millisecond)
		close(stopped)
	}()
	select {
	case <-time.After(200 * time.Millisecond):
		t.Fatal("Start callback doesn't fire in 100 milliseconds")
	case <-stopped:
		t.Log("Start callback fired")
	}

}
