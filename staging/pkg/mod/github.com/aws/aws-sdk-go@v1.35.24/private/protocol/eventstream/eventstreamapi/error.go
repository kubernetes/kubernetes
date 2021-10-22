package eventstreamapi

import (
	"fmt"
	"sync"
)

type messageError struct {
	code string
	msg  string
}

func (e messageError) Code() string {
	return e.code
}

func (e messageError) Message() string {
	return e.msg
}

func (e messageError) Error() string {
	return fmt.Sprintf("%s: %s", e.code, e.msg)
}

func (e messageError) OrigErr() error {
	return nil
}

// OnceError wraps the behavior of recording an error
// once and signal on a channel when this has occurred.
// Signaling is done by closing of the channel.
//
// Type is safe for concurrent usage.
type OnceError struct {
	mu  sync.RWMutex
	err error
	ch  chan struct{}
}

// NewOnceError return a new OnceError
func NewOnceError() *OnceError {
	return &OnceError{
		ch: make(chan struct{}, 1),
	}
}

// Err acquires a read-lock and returns an
// error if one has been set.
func (e *OnceError) Err() error {
	e.mu.RLock()
	err := e.err
	e.mu.RUnlock()

	return err
}

// SetError acquires a write-lock and will set
// the underlying error value if one has not been set.
func (e *OnceError) SetError(err error) {
	if err == nil {
		return
	}

	e.mu.Lock()
	if e.err == nil {
		e.err = err
		close(e.ch)
	}
	e.mu.Unlock()
}

// ErrorSet returns a channel that will be used to signal
// that an error has been set. This channel will be closed
// when the error value has been set for OnceError.
func (e *OnceError) ErrorSet() <-chan struct{} {
	return e.ch
}
