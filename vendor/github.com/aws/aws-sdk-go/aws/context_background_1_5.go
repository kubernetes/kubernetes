// +build !go1.7

package aws

import "time"

// An emptyCtx is a copy of the Go 1.7 context.emptyCtx type. This is copied to
// provide a 1.6 and 1.5 safe version of context that is compatible with Go
// 1.7's Context.
//
// An emptyCtx is never canceled, has no values, and has no deadline. It is not
// struct{}, since vars of this type must have distinct addresses.
type emptyCtx int

func (*emptyCtx) Deadline() (deadline time.Time, ok bool) {
	return
}

func (*emptyCtx) Done() <-chan struct{} {
	return nil
}

func (*emptyCtx) Err() error {
	return nil
}

func (*emptyCtx) Value(key interface{}) interface{} {
	return nil
}

func (e *emptyCtx) String() string {
	switch e {
	case backgroundCtx:
		return "aws.BackgroundContext"
	}
	return "unknown empty Context"
}

var (
	backgroundCtx = new(emptyCtx)
)

// BackgroundContext returns a context that will never be canceled, has no
// values, and no deadline. This context is used by the SDK to provide
// backwards compatibility with non-context API operations and functionality.
//
// Go 1.6 and before:
// This context function is equivalent to context.Background in the Go stdlib.
//
// Go 1.7 and later:
// The context returned will be the value returned by context.Background()
//
// See https://golang.org/pkg/context for more information on Contexts.
func BackgroundContext() Context {
	return backgroundCtx
}
