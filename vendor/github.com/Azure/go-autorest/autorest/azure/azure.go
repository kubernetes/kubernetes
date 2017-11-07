/*
Package azure provides Azure-specific implementations used with AutoRest.

See the included examples for more detail.
*/
package azure

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"

	"github.com/Azure/go-autorest/autorest"
)

const (
	// HeaderClientID is the Azure extension header to set a user-specified request ID.
	HeaderClientID = "x-ms-client-request-id"

	// HeaderReturnClientID is the Azure extension header to set if the user-specified request ID
	// should be included in the response.
	HeaderReturnClientID = "x-ms-return-client-request-id"

	// HeaderRequestID is the Azure extension header of the service generated request ID returned
	// in the response.
	HeaderRequestID = "x-ms-request-id"
)

// ServiceError encapsulates the error response from an Azure service.
type ServiceError struct {
	Code    string         `json:"code"`
	Message string         `json:"message"`
	Details *[]interface{} `json:"details"`
}

func (se ServiceError) Error() string {
	if se.Details != nil {
		d, err := json.Marshal(*(se.Details))
		if err != nil {
			return fmt.Sprintf("Code=%q Message=%q Details=%v", se.Code, se.Message, *se.Details)
		}
		return fmt.Sprintf("Code=%q Message=%q Details=%v", se.Code, se.Message, string(d))
	}
	return fmt.Sprintf("Code=%q Message=%q", se.Code, se.Message)
}

// RequestError describes an error response returned by Azure service.
type RequestError struct {
	autorest.DetailedError

	// The error returned by the Azure service.
	ServiceError *ServiceError `json:"error"`

	// The request id (from the x-ms-request-id-header) of the request.
	RequestID string
}

// Error returns a human-friendly error message from service error.
func (e RequestError) Error() string {
	return fmt.Sprintf("autorest/azure: Service returned an error. Status=%v %v",
		e.StatusCode, e.ServiceError)
}

// IsAzureError returns true if the passed error is an Azure Service error; false otherwise.
func IsAzureError(e error) bool {
	_, ok := e.(*RequestError)
	return ok
}

// NewErrorWithError creates a new Error conforming object from the
// passed packageType, method, statusCode of the given resp (UndefinedStatusCode
// if resp is nil), message, and original error. message is treated as a format
// string to which the optional args apply.
func NewErrorWithError(original error, packageType string, method string, resp *http.Response, message string, args ...interface{}) RequestError {
	if v, ok := original.(*RequestError); ok {
		return *v
	}

	statusCode := autorest.UndefinedStatusCode
	if resp != nil {
		statusCode = resp.StatusCode
	}
	return RequestError{
		DetailedError: autorest.DetailedError{
			Original:    original,
			PackageType: packageType,
			Method:      method,
			StatusCode:  statusCode,
			Message:     fmt.Sprintf(message, args...),
		},
	}
}

// WithReturningClientID returns a PrepareDecorator that adds an HTTP extension header of
// x-ms-client-request-id whose value is the passed, undecorated UUID (e.g.,
// "0F39878C-5F76-4DB8-A25D-61D2C193C3CA"). It also sets the x-ms-return-client-request-id
// header to true such that UUID accompanies the http.Response.
func WithReturningClientID(uuid string) autorest.PrepareDecorator {
	preparer := autorest.CreatePreparer(
		WithClientID(uuid),
		WithReturnClientID(true))

	return func(p autorest.Preparer) autorest.Preparer {
		return autorest.PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err != nil {
				return r, err
			}
			return preparer.Prepare(r)
		})
	}
}

// WithClientID returns a PrepareDecorator that adds an HTTP extension header of
// x-ms-client-request-id whose value is passed, undecorated UUID (e.g.,
// "0F39878C-5F76-4DB8-A25D-61D2C193C3CA").
func WithClientID(uuid string) autorest.PrepareDecorator {
	return autorest.WithHeader(HeaderClientID, uuid)
}

// WithReturnClientID returns a PrepareDecorator that adds an HTTP extension header of
// x-ms-return-client-request-id whose boolean value indicates if the value of the
// x-ms-client-request-id header should be included in the http.Response.
func WithReturnClientID(b bool) autorest.PrepareDecorator {
	return autorest.WithHeader(HeaderReturnClientID, strconv.FormatBool(b))
}

// ExtractClientID extracts the client identifier from the x-ms-client-request-id header set on the
// http.Request sent to the service (and returned in the http.Response)
func ExtractClientID(resp *http.Response) string {
	return autorest.ExtractHeaderValue(HeaderClientID, resp)
}

// ExtractRequestID extracts the Azure server generated request identifier from the
// x-ms-request-id header.
func ExtractRequestID(resp *http.Response) string {
	return autorest.ExtractHeaderValue(HeaderRequestID, resp)
}

// WithErrorUnlessStatusCode returns a RespondDecorator that emits an
// azure.RequestError by reading the response body unless the response HTTP status code
// is among the set passed.
//
// If there is a chance service may return responses other than the Azure error
// format and the response cannot be parsed into an error, a decoding error will
// be returned containing the response body. In any case, the Responder will
// return an error if the status code is not satisfied.
//
// If this Responder returns an error, the response body will be replaced with
// an in-memory reader, which needs no further closing.
func WithErrorUnlessStatusCode(codes ...int) autorest.RespondDecorator {
	return func(r autorest.Responder) autorest.Responder {
		return autorest.ResponderFunc(func(resp *http.Response) error {
			err := r.Respond(resp)
			if err == nil && !autorest.ResponseHasStatusCode(resp, codes...) {
				var e RequestError
				defer resp.Body.Close()

				// Copy and replace the Body in case it does not contain an error object.
				// This will leave the Body available to the caller.
				b, decodeErr := autorest.CopyAndDecode(autorest.EncodedAsJSON, resp.Body, &e)
				resp.Body = ioutil.NopCloser(&b)
				if decodeErr != nil {
					return fmt.Errorf("autorest/azure: error response cannot be parsed: %q error: %v", b.String(), decodeErr)
				} else if e.ServiceError == nil {
					// Check if error is unwrapped ServiceError
					if err := json.Unmarshal(b.Bytes(), &e.ServiceError); err != nil || e.ServiceError.Message == "" {
						e.ServiceError = &ServiceError{
							Code:    "Unknown",
							Message: "Unknown service error",
						}
					}
				}

				e.RequestID = ExtractRequestID(resp)
				if e.StatusCode == nil {
					e.StatusCode = resp.StatusCode
				}
				err = &e
			}
			return err
		})
	}
}
