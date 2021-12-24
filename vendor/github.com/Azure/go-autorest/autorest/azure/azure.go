// Package azure provides Azure-specific implementations used with AutoRest.
// See the included examples for more detail.
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
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"github.com/Azure/go-autorest/autorest"
)

const (
	// HeaderClientID is the Azure extension header to set a user-specified request ID.
	HeaderClientID = "x-ms-client-request-id"

	// HeaderReturnClientID is the Azure extension header to set if the user-specified request ID
	// should be included in the response.
	HeaderReturnClientID = "x-ms-return-client-request-id"

	// HeaderContentType is the type of the content in the HTTP response.
	HeaderContentType = "Content-Type"

	// HeaderRequestID is the Azure extension header of the service generated request ID returned
	// in the response.
	HeaderRequestID = "x-ms-request-id"
)

// ServiceError encapsulates the error response from an Azure service.
// It adhears to the OData v4 specification for error responses.
type ServiceError struct {
	Code           string                   `json:"code"`
	Message        string                   `json:"message"`
	Target         *string                  `json:"target"`
	Details        []map[string]interface{} `json:"details"`
	InnerError     map[string]interface{}   `json:"innererror"`
	AdditionalInfo []map[string]interface{} `json:"additionalInfo"`
}

func (se ServiceError) Error() string {
	result := fmt.Sprintf("Code=%q Message=%q", se.Code, se.Message)

	if se.Target != nil {
		result += fmt.Sprintf(" Target=%q", *se.Target)
	}

	if se.Details != nil {
		d, err := json.Marshal(se.Details)
		if err != nil {
			result += fmt.Sprintf(" Details=%v", se.Details)
		}
		result += fmt.Sprintf(" Details=%v", string(d))
	}

	if se.InnerError != nil {
		d, err := json.Marshal(se.InnerError)
		if err != nil {
			result += fmt.Sprintf(" InnerError=%v", se.InnerError)
		}
		result += fmt.Sprintf(" InnerError=%v", string(d))
	}

	if se.AdditionalInfo != nil {
		d, err := json.Marshal(se.AdditionalInfo)
		if err != nil {
			result += fmt.Sprintf(" AdditionalInfo=%v", se.AdditionalInfo)
		}
		result += fmt.Sprintf(" AdditionalInfo=%v", string(d))
	}

	return result
}

// UnmarshalJSON implements the json.Unmarshaler interface for the ServiceError type.
func (se *ServiceError) UnmarshalJSON(b []byte) error {
	// http://docs.oasis-open.org/odata/odata-json-format/v4.0/os/odata-json-format-v4.0-os.html#_Toc372793091

	type serviceErrorInternal struct {
		Code           string                   `json:"code"`
		Message        string                   `json:"message"`
		Target         *string                  `json:"target,omitempty"`
		AdditionalInfo []map[string]interface{} `json:"additionalInfo,omitempty"`
		// not all services conform to the OData v4 spec.
		// the following fields are where we've seen discrepancies

		// spec calls for []map[string]interface{} but have seen map[string]interface{}
		Details interface{} `json:"details,omitempty"`

		// spec calls for map[string]interface{} but have seen []map[string]interface{} and string
		InnerError interface{} `json:"innererror,omitempty"`
	}

	sei := serviceErrorInternal{}
	if err := json.Unmarshal(b, &sei); err != nil {
		return err
	}

	// copy the fields we know to be correct
	se.AdditionalInfo = sei.AdditionalInfo
	se.Code = sei.Code
	se.Message = sei.Message
	se.Target = sei.Target

	// converts an []interface{} to []map[string]interface{}
	arrayOfObjs := func(v interface{}) ([]map[string]interface{}, bool) {
		arrayOf, ok := v.([]interface{})
		if !ok {
			return nil, false
		}
		final := []map[string]interface{}{}
		for _, item := range arrayOf {
			as, ok := item.(map[string]interface{})
			if !ok {
				return nil, false
			}
			final = append(final, as)
		}
		return final, true
	}

	// convert the remaining fields, falling back to raw JSON if necessary

	if c, ok := arrayOfObjs(sei.Details); ok {
		se.Details = c
	} else if c, ok := sei.Details.(map[string]interface{}); ok {
		se.Details = []map[string]interface{}{c}
	} else if sei.Details != nil {
		// stuff into Details
		se.Details = []map[string]interface{}{
			{"raw": sei.Details},
		}
	}

	if c, ok := sei.InnerError.(map[string]interface{}); ok {
		se.InnerError = c
	} else if c, ok := arrayOfObjs(sei.InnerError); ok {
		// if there's only one error extract it
		if len(c) == 1 {
			se.InnerError = c[0]
		} else {
			// multiple errors, stuff them into the value
			se.InnerError = map[string]interface{}{
				"multi": c,
			}
		}
	} else if c, ok := sei.InnerError.(string); ok {
		se.InnerError = map[string]interface{}{"error": c}
	} else if sei.InnerError != nil {
		// stuff into InnerError
		se.InnerError = map[string]interface{}{
			"raw": sei.InnerError,
		}
	}
	return nil
}

// RequestError describes an error response returned by Azure service.
type RequestError struct {
	autorest.DetailedError

	// The error returned by the Azure service.
	ServiceError *ServiceError `json:"error" xml:"Error"`

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

// Resource contains details about an Azure resource.
type Resource struct {
	SubscriptionID string
	ResourceGroup  string
	Provider       string
	ResourceType   string
	ResourceName   string
}

// String function returns a string in form of azureResourceID
func (r Resource) String() string {
	return fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/%s/%s/%s", r.SubscriptionID, r.ResourceGroup, r.Provider, r.ResourceType, r.ResourceName)
}

// ParseResourceID parses a resource ID into a ResourceDetails struct.
// See https://docs.microsoft.com/en-us/azure/azure-resource-manager/templates/template-functions-resource?tabs=json#resourceid.
func ParseResourceID(resourceID string) (Resource, error) {

	const resourceIDPatternText = `(?i)subscriptions/(.+)/resourceGroups/(.+)/providers/(.+?)/(.+?)/(.+)`
	resourceIDPattern := regexp.MustCompile(resourceIDPatternText)
	match := resourceIDPattern.FindStringSubmatch(resourceID)

	if len(match) == 0 {
		return Resource{}, fmt.Errorf("parsing failed for %s. Invalid resource Id format", resourceID)
	}

	v := strings.Split(match[5], "/")
	resourceName := v[len(v)-1]

	result := Resource{
		SubscriptionID: match[1],
		ResourceGroup:  match[2],
		Provider:       match[3],
		ResourceType:   match[4],
		ResourceName:   resourceName,
	}

	return result, nil
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

				encodedAs := autorest.EncodedAsJSON
				if strings.Contains(resp.Header.Get("Content-Type"), "xml") {
					encodedAs = autorest.EncodedAsXML
				}

				// Copy and replace the Body in case it does not contain an error object.
				// This will leave the Body available to the caller.
				b, decodeErr := autorest.CopyAndDecode(encodedAs, resp.Body, &e)
				resp.Body = ioutil.NopCloser(&b)
				if decodeErr != nil {
					return fmt.Errorf("autorest/azure: error response cannot be parsed: %q error: %v", b.String(), decodeErr)
				}
				if e.ServiceError == nil {
					// Check if error is unwrapped ServiceError
					decoder := autorest.NewDecoder(encodedAs, bytes.NewReader(b.Bytes()))
					if err := decoder.Decode(&e.ServiceError); err != nil {
						return fmt.Errorf("autorest/azure: error response cannot be parsed: %q error: %v", b.String(), err)
					}

					// for example, should the API return the literal value `null` as the response
					if e.ServiceError == nil {
						e.ServiceError = &ServiceError{
							Code:    "Unknown",
							Message: "Unknown service error",
							Details: []map[string]interface{}{
								{
									"HttpResponse.Body": b.String(),
								},
							},
						}
					}
				}

				if e.ServiceError != nil && e.ServiceError.Message == "" {
					// if we're here it means the returned error wasn't OData v4 compliant.
					// try to unmarshal the body in hopes of getting something.
					rawBody := map[string]interface{}{}
					decoder := autorest.NewDecoder(encodedAs, bytes.NewReader(b.Bytes()))
					if err := decoder.Decode(&rawBody); err != nil {
						return fmt.Errorf("autorest/azure: error response cannot be parsed: %q error: %v", b.String(), err)
					}

					e.ServiceError = &ServiceError{
						Code:    "Unknown",
						Message: "Unknown service error",
					}
					if len(rawBody) > 0 {
						e.ServiceError.Details = []map[string]interface{}{rawBody}
					}
				}
				e.Response = resp
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
