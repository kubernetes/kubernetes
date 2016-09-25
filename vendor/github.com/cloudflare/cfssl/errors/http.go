package errors

import (
	"errors"
	"net/http"
)

// HTTPError is an augmented error with a HTTP status code.
type HTTPError struct {
	StatusCode int
	error
}

// Error implements the error interface.
func (e *HTTPError) Error() string {
	return e.error.Error()
}

// NewMethodNotAllowed returns an appropriate error in the case that
// an HTTP client uses an invalid method (i.e. a GET in place of a POST)
// on an API endpoint.
func NewMethodNotAllowed(method string) *HTTPError {
	return &HTTPError{http.StatusMethodNotAllowed, errors.New(`Method is not allowed:"` + method + `"`)}
}

// NewBadRequest creates a HttpError with the given error and error code 400.
func NewBadRequest(err error) *HTTPError {
	return &HTTPError{http.StatusBadRequest, err}
}

// NewBadRequestString returns a HttpError with the supplied message
// and error code 400.
func NewBadRequestString(s string) *HTTPError {
	return NewBadRequest(errors.New(s))
}

// NewBadRequestMissingParameter returns a 400 HttpError as a required
// parameter is missing in the HTTP request.
func NewBadRequestMissingParameter(s string) *HTTPError {
	return NewBadRequestString(`Missing parameter "` + s + `"`)
}

// NewBadRequestUnwantedParameter returns a 400 HttpError as a unnecessary
// parameter is present in the HTTP request.
func NewBadRequestUnwantedParameter(s string) *HTTPError {
	return NewBadRequestString(`Unwanted parameter "` + s + `"`)
}
