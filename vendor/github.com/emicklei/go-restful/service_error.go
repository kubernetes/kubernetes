package restful

// Copyright 2013 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

import (
	"fmt"
	"net/http"
)

// ServiceError is a transport object to pass information about a non-Http error occurred in a WebService while processing a request.
type ServiceError struct {
	Code    int
	Message string
	Header  http.Header
}

// NewError returns a ServiceError using the code and reason
func NewError(code int, message string) ServiceError {
	return ServiceError{Code: code, Message: message}
}

// NewErrorWithHeader returns a ServiceError using the code, reason and header
func NewErrorWithHeader(code int, message string, header http.Header) ServiceError {
	return ServiceError{Code: code, Message: message, Header: header}
}

// Error returns a text representation of the service error
func (s ServiceError) Error() string {
	return fmt.Sprintf("[ServiceError:%v] %v", s.Code, s.Message)
}
