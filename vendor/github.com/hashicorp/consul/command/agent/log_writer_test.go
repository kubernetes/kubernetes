package agent

import (
	"testing"
)

type MockLogHandler struct {
	logs []string
}

func (m *MockLogHandler) HandleLog(l string) {
	m.logs = append(m.logs, l)
}

func TestLogWriter(t *testing.T) {
	h := &MockLogHandler{}
	w := NewLogWriter(4)

	// Write some logs
	w.Write([]byte("one")) // Gets dropped!
	w.Write([]byte("two"))
	w.Write([]byte("three"))
	w.Write([]byte("four"))
	w.Write([]byte("five"))

	// Register a handler, sends old!
	w.RegisterHandler(h)

	w.Write([]byte("six"))
	w.Write([]byte("seven"))

	// Deregister
	w.DeregisterHandler(h)

	w.Write([]byte("eight"))
	w.Write([]byte("nine"))

	out := []string{
		"two",
		"three",
		"four",
		"five",
		"six",
		"seven",
	}
	for idx := range out {
		if out[idx] != h.logs[idx] {
			t.Fatalf("mismatch %v", h.logs)
		}
	}
}
