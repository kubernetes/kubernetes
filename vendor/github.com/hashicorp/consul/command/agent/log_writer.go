package agent

import (
	"sync"
)

// LogHandler interface is used for clients that want to subscribe
// to logs, for example to stream them over an IPC mechanism
type LogHandler interface {
	HandleLog(string)
}

// logWriter implements io.Writer so it can be used as a log sink.
// It maintains a circular buffer of logs, and a set of handlers to
// which it can stream the logs to.
type logWriter struct {
	sync.Mutex
	logs     []string
	index    int
	handlers map[LogHandler]struct{}
}

// NewLogWriter creates a logWriter with the given buffer capacity
func NewLogWriter(buf int) *logWriter {
	return &logWriter{
		logs:     make([]string, buf),
		index:    0,
		handlers: make(map[LogHandler]struct{}),
	}
}

// RegisterHandler adds a log handler to receive logs, and sends
// the last buffered logs to the handler
func (l *logWriter) RegisterHandler(lh LogHandler) {
	l.Lock()
	defer l.Unlock()

	// Do nothing if already registered
	if _, ok := l.handlers[lh]; ok {
		return
	}

	// Register
	l.handlers[lh] = struct{}{}

	// Send the old logs
	if l.logs[l.index] != "" {
		for i := l.index; i < len(l.logs); i++ {
			lh.HandleLog(l.logs[i])
		}
	}
	for i := 0; i < l.index; i++ {
		lh.HandleLog(l.logs[i])
	}
}

// DeregisterHandler removes a LogHandler and prevents more invocations
func (l *logWriter) DeregisterHandler(lh LogHandler) {
	l.Lock()
	defer l.Unlock()
	delete(l.handlers, lh)
}

// Write is used to accumulate new logs
func (l *logWriter) Write(p []byte) (n int, err error) {
	l.Lock()
	defer l.Unlock()

	// Strip off newlines at the end if there are any since we store
	// individual log lines in the agent.
	n = len(p)
	if p[n-1] == '\n' {
		p = p[:n-1]
	}

	l.logs[l.index] = string(p)
	l.index = (l.index + 1) % len(l.logs)

	for lh, _ := range l.handlers {
		lh.HandleLog(string(p))
	}
	return
}
