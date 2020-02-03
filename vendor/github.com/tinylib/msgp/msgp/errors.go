package msgp

import (
	"fmt"
	"reflect"
)

const resumableDefault = false

var (
	// ErrShortBytes is returned when the
	// slice being decoded is too short to
	// contain the contents of the message
	ErrShortBytes error = errShort{}

	// this error is only returned
	// if we reach code that should
	// be unreachable
	fatal error = errFatal{}
)

// Error is the interface satisfied
// by all of the errors that originate
// from this package.
type Error interface {
	error

	// Resumable returns whether
	// or not the error means that
	// the stream of data is malformed
	// and the information is unrecoverable.
	Resumable() bool
}

// contextError allows msgp Error instances to be enhanced with additional
// context about their origin.
type contextError interface {
	Error

	// withContext must not modify the error instance - it must clone and
	// return a new error with the context added.
	withContext(ctx string) error
}

// Cause returns the underlying cause of an error that has been wrapped
// with additional context.
func Cause(e error) error {
	out := e
	if e, ok := e.(errWrapped); ok && e.cause != nil {
		out = e.cause
	}
	return out
}

// Resumable returns whether or not the error means that the stream of data is
// malformed and the information is unrecoverable.
func Resumable(e error) bool {
	if e, ok := e.(Error); ok {
		return e.Resumable()
	}
	return resumableDefault
}

// WrapError wraps an error with additional context that allows the part of the
// serialized type that caused the problem to be identified. Underlying errors
// can be retrieved using Cause()
//
// The input error is not modified - a new error should be returned.
//
// ErrShortBytes is not wrapped with any context due to backward compatibility
// issues with the public API.
//
func WrapError(err error, ctx ...interface{}) error {
	switch e := err.(type) {
	case errShort:
		return e
	case contextError:
		return e.withContext(ctxString(ctx))
	default:
		return errWrapped{cause: err, ctx: ctxString(ctx)}
	}
}

// ctxString converts the incoming interface{} slice into a single string.
func ctxString(ctx []interface{}) string {
	out := ""
	for idx, cv := range ctx {
		if idx > 0 {
			out += "/"
		}
		out += fmt.Sprintf("%v", cv)
	}
	return out
}

func addCtx(ctx, add string) string {
	if ctx != "" {
		return add + "/" + ctx
	} else {
		return add
	}
}

// errWrapped allows arbitrary errors passed to WrapError to be enhanced with
// context and unwrapped with Cause()
type errWrapped struct {
	cause error
	ctx   string
}

func (e errWrapped) Error() string {
	if e.ctx != "" {
		return fmt.Sprintf("%s at %s", e.cause, e.ctx)
	} else {
		return e.cause.Error()
	}
}

func (e errWrapped) Resumable() bool {
	if e, ok := e.cause.(Error); ok {
		return e.Resumable()
	}
	return resumableDefault
}

type errShort struct{}

func (e errShort) Error() string   { return "msgp: too few bytes left to read object" }
func (e errShort) Resumable() bool { return false }

type errFatal struct {
	ctx string
}

func (f errFatal) Error() string {
	out := "msgp: fatal decoding error (unreachable code)"
	if f.ctx != "" {
		out += " at " + f.ctx
	}
	return out
}

func (f errFatal) Resumable() bool { return false }

func (f errFatal) withContext(ctx string) error { f.ctx = addCtx(f.ctx, ctx); return f }

// ArrayError is an error returned
// when decoding a fix-sized array
// of the wrong size
type ArrayError struct {
	Wanted uint32
	Got    uint32
	ctx    string
}

// Error implements the error interface
func (a ArrayError) Error() string {
	out := fmt.Sprintf("msgp: wanted array of size %d; got %d", a.Wanted, a.Got)
	if a.ctx != "" {
		out += " at " + a.ctx
	}
	return out
}

// Resumable is always 'true' for ArrayErrors
func (a ArrayError) Resumable() bool { return true }

func (a ArrayError) withContext(ctx string) error { a.ctx = addCtx(a.ctx, ctx); return a }

// IntOverflow is returned when a call
// would downcast an integer to a type
// with too few bits to hold its value.
type IntOverflow struct {
	Value         int64 // the value of the integer
	FailedBitsize int   // the bit size that the int64 could not fit into
	ctx           string
}

// Error implements the error interface
func (i IntOverflow) Error() string {
	str := fmt.Sprintf("msgp: %d overflows int%d", i.Value, i.FailedBitsize)
	if i.ctx != "" {
		str += " at " + i.ctx
	}
	return str
}

// Resumable is always 'true' for overflows
func (i IntOverflow) Resumable() bool { return true }

func (i IntOverflow) withContext(ctx string) error { i.ctx = addCtx(i.ctx, ctx); return i }

// UintOverflow is returned when a call
// would downcast an unsigned integer to a type
// with too few bits to hold its value
type UintOverflow struct {
	Value         uint64 // value of the uint
	FailedBitsize int    // the bit size that couldn't fit the value
	ctx           string
}

// Error implements the error interface
func (u UintOverflow) Error() string {
	str := fmt.Sprintf("msgp: %d overflows uint%d", u.Value, u.FailedBitsize)
	if u.ctx != "" {
		str += " at " + u.ctx
	}
	return str
}

// Resumable is always 'true' for overflows
func (u UintOverflow) Resumable() bool { return true }

func (u UintOverflow) withContext(ctx string) error { u.ctx = addCtx(u.ctx, ctx); return u }

// UintBelowZero is returned when a call
// would cast a signed integer below zero
// to an unsigned integer.
type UintBelowZero struct {
	Value int64 // value of the incoming int
	ctx   string
}

// Error implements the error interface
func (u UintBelowZero) Error() string {
	str := fmt.Sprintf("msgp: attempted to cast int %d to unsigned", u.Value)
	if u.ctx != "" {
		str += " at " + u.ctx
	}
	return str
}

// Resumable is always 'true' for overflows
func (u UintBelowZero) Resumable() bool { return true }

func (u UintBelowZero) withContext(ctx string) error {
	u.ctx = ctx
	return u
}

// A TypeError is returned when a particular
// decoding method is unsuitable for decoding
// a particular MessagePack value.
type TypeError struct {
	Method  Type // Type expected by method
	Encoded Type // Type actually encoded

	ctx string
}

// Error implements the error interface
func (t TypeError) Error() string {
	out := fmt.Sprintf("msgp: attempted to decode type %q with method for %q", t.Encoded, t.Method)
	if t.ctx != "" {
		out += " at " + t.ctx
	}
	return out
}

// Resumable returns 'true' for TypeErrors
func (t TypeError) Resumable() bool { return true }

func (t TypeError) withContext(ctx string) error { t.ctx = addCtx(t.ctx, ctx); return t }

// returns either InvalidPrefixError or
// TypeError depending on whether or not
// the prefix is recognized
func badPrefix(want Type, lead byte) error {
	t := sizes[lead].typ
	if t == InvalidType {
		return InvalidPrefixError(lead)
	}
	return TypeError{Method: want, Encoded: t}
}

// InvalidPrefixError is returned when a bad encoding
// uses a prefix that is not recognized in the MessagePack standard.
// This kind of error is unrecoverable.
type InvalidPrefixError byte

// Error implements the error interface
func (i InvalidPrefixError) Error() string {
	return fmt.Sprintf("msgp: unrecognized type prefix 0x%x", byte(i))
}

// Resumable returns 'false' for InvalidPrefixErrors
func (i InvalidPrefixError) Resumable() bool { return false }

// ErrUnsupportedType is returned
// when a bad argument is supplied
// to a function that takes `interface{}`.
type ErrUnsupportedType struct {
	T reflect.Type

	ctx string
}

// Error implements error
func (e *ErrUnsupportedType) Error() string {
	out := fmt.Sprintf("msgp: type %q not supported", e.T)
	if e.ctx != "" {
		out += " at " + e.ctx
	}
	return out
}

// Resumable returns 'true' for ErrUnsupportedType
func (e *ErrUnsupportedType) Resumable() bool { return true }

func (e *ErrUnsupportedType) withContext(ctx string) error {
	o := *e
	o.ctx = addCtx(o.ctx, ctx)
	return &o
}
