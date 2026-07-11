package fsnotify

import "sync"

type shared struct {
	Events chan Event
	Errors chan error
	done   chan struct{}
	mu     sync.Mutex
}

func newShared(ev chan Event, errs chan error) *shared {
	return &shared{
		Events: ev,
		Errors: errs,
		done:   make(chan struct{}),
	}
}

// Returns true if the event was sent, or false if watcher is closed.
func (w *shared) sendEvent(e Event) bool {
	if e.Op == 0 {
		return true
	}
	select {
	case <-w.done:
		return false
	case w.Events <- e:
		return true
	}
}

// Returns true if the error was sent, or false if watcher is closed.
func (w *shared) sendError(err error) bool {
	if err == nil {
		return true
	}
	select {
	case <-w.done:
		return false
	case w.Errors <- err:
		return true
	}
}

func (w *shared) isClosed() bool {
	select {
	case <-w.done:
		return true
	default:
		return false
	}
}

// Mark as closed; returns true if it was already closed.
func (w *shared) close() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.isClosed() {
		return true
	}
	close(w.done)
	return false
}
