// Package apierr represents API error types.
package apierr

import "fmt"

// A BaseError wraps the code and message which defines an error. It also
// can be used to wrap an original error object.
//
// Should be used as the root for errors satisfying the awserr.Error. Also
// for any error which does not fit into a specific error wrapper type.
type BaseError struct {
	// Classification of error
	code string

	// Detailed information about error
	message string

	// Optional original error this error is based off of. Allows building
	// chained errors.
	origErr error
}

// New returns an error object for the code, message, and err.
//
// code is a short no whitespace phrase depicting the classification of
// the error that is being created.
//
// message is the free flow string containing detailed information about the error.
//
// origErr is the error object which will be nested under the new error to be returned.
func New(code, message string, origErr error) *BaseError {
	return &BaseError{
		code:    code,
		message: message,
		origErr: origErr,
	}
}

// Error returns the string representation of the error.
//
// See ErrorWithExtra for formatting.
//
// Satisfies the error interface.
func (b *BaseError) Error() string {
	return b.ErrorWithExtra("")
}

// String returns the string representation of the error.
// Alias for Error to satisfy the stringer interface.
func (b *BaseError) String() string {
	return b.Error()
}

// Code returns the short phrase depicting the classification of the error.
func (b *BaseError) Code() string {
	return b.code
}

// Message returns the error details message.
func (b *BaseError) Message() string {
	return b.message
}

// OrigErr returns the original error if one was set. Nil is returned if no error
// was set.
func (b *BaseError) OrigErr() error {
	return b.origErr
}

// ErrorWithExtra is a helper method to add an extra string to the stratified
// error message. The extra message will be added on the next line below the
// error message like the following:
//
//     <error code>: <error message>
//         <extra message>
//
// If there is a original error the error will be included on a new line.
//
//     <error code>: <error message>
//         <extra message>
//     caused by: <original error>
func (b *BaseError) ErrorWithExtra(extra string) string {
	msg := fmt.Sprintf("%s: %s", b.code, b.message)
	if extra != "" {
		msg = fmt.Sprintf("%s\n\t%s", msg, extra)
	}
	if b.origErr != nil {
		msg = fmt.Sprintf("%s\ncaused by: %s", msg, b.origErr.Error())
	}
	return msg
}

// A RequestError wraps a request or service error.
//
// Composed of BaseError for code, message, and original error.
type RequestError struct {
	*BaseError
	statusCode int
	requestID  string
}

// NewRequestError returns a wrapped error with additional information for request
// status code, and service requestID.
//
// Should be used to wrap all request which involve service requests. Even if
// the request failed without a service response, but had an HTTP status code
// that may be meaningful.
//
// Also wraps original errors via the BaseError.
func NewRequestError(base *BaseError, statusCode int, requestID string) *RequestError {
	return &RequestError{
		BaseError:  base,
		statusCode: statusCode,
		requestID:  requestID,
	}
}

// Error returns the string representation of the error.
// Satisfies the error interface.
func (r *RequestError) Error() string {
	return r.ErrorWithExtra(fmt.Sprintf("status code: %d, request id: [%s]",
		r.statusCode, r.requestID))
}

// String returns the string representation of the error.
// Alias for Error to satisfy the stringer interface.
func (r *RequestError) String() string {
	return r.Error()
}

// StatusCode returns the wrapped status code for the error
func (r *RequestError) StatusCode() int {
	return r.statusCode
}

// RequestID returns the wrapped requestID
func (r *RequestError) RequestID() string {
	return r.requestID
}
