package dbus

import (
	"context"
)

// Call represents a pending or completed method call.
type Call struct {
	Destination string
	Path        ObjectPath
	Method      string
	Args        []any

	// Strobes when the call is complete.
	Done chan *Call

	// After completion, the error status. If this is non-nil, it may be an
	// error message from the peer (with Error as its type) or some other error.
	Err error

	// Holds the response once the call is done.
	Body []any

	// ResponseSequence stores the sequence number of the DBus message containing
	// the call response (or error). This can be compared to the sequence number
	// of other call responses and signals on this connection to determine their
	// relative ordering on the underlying DBus connection.
	// For errors, ResponseSequence is populated only if the error came from a
	// DBusMessage that was received or if there was an error receiving. In case of
	// failure to make the call, ResponseSequence will be NoSequence.
	ResponseSequence Sequence

	// tracks context and canceler
	ctx         context.Context
	ctxCanceler context.CancelFunc
}

func (c *Call) Context() context.Context {
	if c.ctx == nil {
		return context.Background()
	}

	return c.ctx
}

func (c *Call) ContextCancel() {
	if c.ctxCanceler != nil {
		c.ctxCanceler()
	}
}

// Store stores the body of the reply into the provided pointers. It returns
// an error if the signatures of the body and retvalues don't match, or if
// the error status is not nil.
func (c *Call) Store(retvalues ...any) error {
	if c.Err != nil {
		return c.Err
	}

	return Store(c.Body, retvalues...)
}

func (c *Call) done() {
	c.Done <- c
	c.ContextCancel()
}
