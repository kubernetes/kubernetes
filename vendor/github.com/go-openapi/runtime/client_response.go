// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package runtime

import (
	"fmt"
	"io"
)

// A ClientResponse represents a client response
// This bridges between responses obtained from different transports
type ClientResponse interface {
	Code() int
	Message() string
	GetHeader(string) string
	Body() io.ReadCloser
}

// A ClientResponseReaderFunc turns a function into a ClientResponseReader interface implementation
type ClientResponseReaderFunc func(ClientResponse, Consumer) (interface{}, error)

// ReadResponse reads the response
func (read ClientResponseReaderFunc) ReadResponse(resp ClientResponse, consumer Consumer) (interface{}, error) {
	return read(resp, consumer)
}

// A ClientResponseReader is an interface for things want to read a response.
// An application of this is to create structs from response values
type ClientResponseReader interface {
	ReadResponse(ClientResponse, Consumer) (interface{}, error)
}

// NewAPIError creates a new API error
func NewAPIError(opName string, payload interface{}, code int) *APIError {
	return &APIError{
		OperationName: opName,
		Response:      payload,
		Code:          code,
	}
}

// APIError wraps an error model and captures the status code
type APIError struct {
	OperationName string
	Response      interface{}
	Code          int
}

func (a *APIError) Error() string {
	return fmt.Sprintf("%s (status %d): %+v ", a.OperationName, a.Code, a.Response)
}
