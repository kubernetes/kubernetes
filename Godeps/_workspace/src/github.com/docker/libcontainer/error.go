package libcontainer

// API error code type.
type ErrorCode int

// API error codes.
const (
	// Factory errors
	IdInUse ErrorCode = iota
	InvalidIdFormat
	// TODO: add Load errors

	// Container errors
	ContainerDestroyed
	ContainerPaused

	// Common errors
	ConfigInvalid
	SystemError
)

// API Error type.
type Error interface {
	error

	// Returns the stack trace, if any, which identifies the
	// point at which the error occurred.
	Stack() []byte

	// Returns a verbose string including the error message
	// and a representation of the stack trace suitable for
	// printing.
	Detail() string

	// Returns the error code for this error.
	Code() ErrorCode
}
