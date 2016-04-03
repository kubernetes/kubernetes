package metrics

import (
	"bytes"
	"os"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestInmemSignal(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	inm := NewInmemSink(10*time.Millisecond, 50*time.Millisecond)
	sig := NewInmemSignal(inm, syscall.SIGUSR1, buf)
	defer sig.Stop()

	inm.SetGauge([]string{"foo"}, 42)
	inm.EmitKey([]string{"bar"}, 42)
	inm.IncrCounter([]string{"baz"}, 42)
	inm.AddSample([]string{"wow"}, 42)

	// Wait for period to end
	time.Sleep(15 * time.Millisecond)

	// Send signal!
	syscall.Kill(os.Getpid(), syscall.SIGUSR1)

	// Wait for flush
	time.Sleep(10 * time.Millisecond)

	// Check the output
	out := string(buf.Bytes())
	if !strings.Contains(out, "[G] 'foo': 42") {
		t.Fatalf("bad: %v", out)
	}
	if !strings.Contains(out, "[P] 'bar': 42") {
		t.Fatalf("bad: %v", out)
	}
	if !strings.Contains(out, "[C] 'baz': Count: 1 Sum: 42") {
		t.Fatalf("bad: %v", out)
	}
	if !strings.Contains(out, "[S] 'wow': Count: 1 Sum: 42") {
		t.Fatalf("bad: %v", out)
	}
}
