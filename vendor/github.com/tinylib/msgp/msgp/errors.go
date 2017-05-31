package msgp

import (
	"fmt"
	"reflect"
)

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
	// and  the information is unrecoverable.
	Resumable() bool
}

type errShort struct{}

func (e errShort) Error() string   { return "msgp: too few bytes left to read object" }
func (e errShort) Resumable() bool { return false }

type errFatal struct{}

func (f errFatal) Error() string   { return "msgp: fatal decoding error (unreachable code)" }
func (f errFatal) Resumable() bool { return false }

// ArrayError is an error returned
// when decoding a fix-sized array
// of the wrong size
type ArrayError struct {
	Wanted uint32
	Got    uint32
}

// Error implements the error interface
func (a ArrayError) Error() string {
	return fmt.Sprintf("msgp: wanted array of size %d; got %d", a.Wanted, a.Got)
}

// Resumable is always 'true' for ArrayErrors
func (a ArrayError) Resumable() bool { return true }

// IntOverflow is returned when a call
// would downcast an integer to a type
// with too few bits to hold its value.
type IntOverflow struct {
	Value         int64 // the value of the integer
	FailedBitsize int   // the bit size that the int64 could not fit into
}

// Error implements the error interface
func (i IntOverflow) Error() string {
	return fmt.Sprintf("msgp: %d overflows int%d", i.Value, i.FailedBitsize)
}

// Resumable is always 'true' for overflows
func (i IntOverflow) Resumable() bool { return true }

// UintOverflow is returned when a call
// would downcast an unsigned integer to a type
// with too few bits to hold its value
type UintOverflow struct {
	Value         uint64 // value of the uint
	FailedBitsize int    // the bit size that couldn't fit the value
}

// Error implements the error interface
func (u UintOverflow) Error() string {
	return fmt.Sprintf("msgp: %d overflows uint%d", u.Value, u.FailedBitsize)
}

// Resumable is always 'true' for overflows
func (u UintOverflow) Resumable() bool { return true }

// A TypeError is returned when a particular
// decoding method is unsuitable for decoding
// a particular MessagePack value.
type TypeError struct {
	Method  Type // Type expected by method
	Encoded Type // Type actually encoded
}

// Error implements the error interface
func (t TypeError) Error() string {
	return fmt.Sprintf("msgp: attempted to decode type %q with method for %q", t.Encoded, t.Method)
}

// Resumable returns 'true' for TypeErrors
func (t TypeError) Resumable() bool { return true }

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
}

// Error implements error
func (e *ErrUnsupportedType) Error() string { return fmt.Sprintf("msgp: type %q not supported", e.T) }

// Resumable returns 'true' for ErrUnsupportedType
func (e *ErrUnsupportedType) Resumable() bool { return true }
