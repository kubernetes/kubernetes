package callback

import (
	"fmt"
)

type Unsupported struct {
	Callback Interface
}

func (uc *Unsupported) Error() string {
	return fmt.Sprintf("Unsupported callback <%T>: %v", uc.Callback, uc.Callback)
}

type Interface interface {
	// marker interface
}

type Handler interface {
	// may return an Unsupported error on failure
	Handle(callbacks ...Interface) error
}

type HandlerFunc func(callbacks ...Interface) error

func (f HandlerFunc) Handle(callbacks ...Interface) error {
	return f(callbacks...)
}
