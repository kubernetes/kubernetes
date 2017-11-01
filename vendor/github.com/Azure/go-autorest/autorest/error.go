package autorest

import (
	"fmt"
	"net/http"
)

const (
	// UndefinedStatusCode is used when HTTP status code is not available for an error.
	UndefinedStatusCode = 0
)

// DetailedError encloses a error with details of the package, method, and associated HTTP
// status code (if any).
type DetailedError struct {
	Original error

	// PackageType is the package type of the object emitting the error. For types, the value
	// matches that produced the the '%T' format specifier of the fmt package. For other elements,
	// such as functions, it is just the package name (e.g., "autorest").
	PackageType string

	// Method is the name of the method raising the error.
	Method string

	// StatusCode is the HTTP Response StatusCode (if non-zero) that led to the error.
	StatusCode interface{}

	// Message is the error message.
	Message string

	// Service Error is the response body of failed API in bytes
	ServiceError []byte

	// Response is the response object that was returned during failure if applicable.
	Response *http.Response
}

// NewError creates a new Error conforming object from the passed packageType, method, and
// message. message is treated as a format string to which the optional args apply.
func NewError(packageType string, method string, message string, args ...interface{}) DetailedError {
	return NewErrorWithError(nil, packageType, method, nil, message, args...)
}

// NewErrorWithResponse creates a new Error conforming object from the passed
// packageType, method, statusCode of the given resp (UndefinedStatusCode if
// resp is nil), and message. message is treated as a format string to which the
// optional args apply.
func NewErrorWithResponse(packageType string, method string, resp *http.Response, message string, args ...interface{}) DetailedError {
	return NewErrorWithError(nil, packageType, method, resp, message, args...)
}

// NewErrorWithError creates a new Error conforming object from the
// passed packageType, method, statusCode of the given resp (UndefinedStatusCode
// if resp is nil), message, and original error. message is treated as a format
// string to which the optional args apply.
func NewErrorWithError(original error, packageType string, method string, resp *http.Response, message string, args ...interface{}) DetailedError {
	if v, ok := original.(DetailedError); ok {
		return v
	}

	statusCode := UndefinedStatusCode
	if resp != nil {
		statusCode = resp.StatusCode
	}

	return DetailedError{
		Original:    original,
		PackageType: packageType,
		Method:      method,
		StatusCode:  statusCode,
		Message:     fmt.Sprintf(message, args...),
		Response:    resp,
	}
}

// Error returns a formatted containing all available details (i.e., PackageType, Method,
// StatusCode, Message, and original error (if any)).
func (e DetailedError) Error() string {
	if e.Original == nil {
		return fmt.Sprintf("%s#%s: %s: StatusCode=%d", e.PackageType, e.Method, e.Message, e.StatusCode)
	}
	return fmt.Sprintf("%s#%s: %s: StatusCode=%d -- Original Error: %v", e.PackageType, e.Method, e.Message, e.StatusCode, e.Original)
}
