// +build !go1.9

package aws

import "time"

// Context is an copy of the Go v1.7 stdlib's context.Context interface.
// It is represented as a SDK interface to enable you to use the "WithContext"
// API methods with Go v1.6 and a Context type such as golang.org/x/net/context.
//
// See https://golang.org/pkg/context on how to use contexts.
type Context interface {
	// Deadline returns the time when work done on behalf of this context
	// should be canceled. Deadline returns ok==false when no deadline is
	// set. Successive calls to Deadline return the same results.
	Deadline() (deadline time.Time, ok bool)

	// Done returns a channel that's closed when work done on behalf of this
	// context should be canceled. Done may return nil if this context can
	// never be canceled. Successive calls to Done return the same value.
	Done() <-chan struct{}

	// Err returns a non-nil error value after Done is closed. Err returns
	// Canceled if the context was canceled or DeadlineExceeded if the
	// context's deadline passed. No other values for Err are defined.
	// After Done is closed, successive calls to Err return the same value.
	Err() error

	// Value returns the value associated with this context for key, or nil
	// if no value is associated with key. Successive calls to Value with
	// the same key returns the same result.
	//
	// Use context values only for request-scoped data that transits
	// processes and API boundaries, not for passing optional parameters to
	// functions.
	Value(key interface{}) interface{}
}
