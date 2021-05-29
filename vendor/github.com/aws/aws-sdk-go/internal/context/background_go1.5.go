// +build !go1.7

package context

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
	case BackgroundCtx:
		return "aws.BackgroundContext"
	}
	return "unknown empty Context"
}

// BackgroundCtx is the common base context.
var BackgroundCtx = new(emptyCtx)
