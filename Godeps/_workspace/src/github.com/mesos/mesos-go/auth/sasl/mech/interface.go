package mech

import (
	"errors"

	"github.com/mesos/mesos-go/auth/callback"
)

var (
	IllegalStateErr = errors.New("illegal mechanism state")
)

type Interface interface {
	Handler() callback.Handler
	Discard() // clean up resources or sensitive information; idempotent
}

// return a mechanism and it's initialization step (may be a noop that returns
// a nil data blob and handle to the first "real" challenge step).
type Factory func(h callback.Handler) (Interface, StepFunc, error)

// StepFunc implementations should never return a nil StepFunc result. This
// helps keep the logic in the SASL authticatee simpler: step functions are
// never nil. Mechanisms that end up an error state (for example, some decoding
// logic fails...) should return a StepFunc that represents an error state.
// Some mechanisms may be able to recover from such.
type StepFunc func(m Interface, data []byte) (StepFunc, []byte, error)

// reflects an unrecoverable, illegal mechanism state; always returns IllegalState
// as the next step along with an IllegalStateErr
func IllegalState(m Interface, data []byte) (StepFunc, []byte, error) {
	return IllegalState, nil, IllegalStateErr
}
