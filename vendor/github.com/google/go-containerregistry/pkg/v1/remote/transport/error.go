// Copyright 2018 Google LLC All Rights Reserved.
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

package transport

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/google/go-containerregistry/internal/redact"
)

// Error implements error to support the following error specification:
// https://github.com/docker/distribution/blob/master/docs/spec/api.md#errors
type Error struct {
	Errors []Diagnostic `json:"errors,omitempty"`
	// The http status code returned.
	StatusCode int
	// The request that failed.
	Request *http.Request
	// The raw body if we couldn't understand it.
	rawBody string

	// Bit of a hack to make it easier to force a retry.
	temporary bool
}

// Check that Error implements error
var _ error = (*Error)(nil)

// Error implements error
func (e *Error) Error() string {
	prefix := ""
	if e.Request != nil {
		prefix = fmt.Sprintf("%s %s: ", e.Request.Method, redact.URL(e.Request.URL))
	}
	return prefix + e.responseErr()
}

func (e *Error) responseErr() string {
	switch len(e.Errors) {
	case 0:
		if len(e.rawBody) == 0 {
			if e.Request != nil && e.Request.Method == http.MethodHead {
				return fmt.Sprintf("unexpected status code %d %s (HEAD responses have no body, use GET for details)", e.StatusCode, http.StatusText(e.StatusCode))
			}
			return fmt.Sprintf("unexpected status code %d %s", e.StatusCode, http.StatusText(e.StatusCode))
		}
		return fmt.Sprintf("unexpected status code %d %s: %s", e.StatusCode, http.StatusText(e.StatusCode), e.rawBody)
	case 1:
		return e.Errors[0].String()
	default:
		var errors []string
		for _, d := range e.Errors {
			errors = append(errors, d.String())
		}
		return fmt.Sprintf("multiple errors returned: %s",
			strings.Join(errors, "; "))
	}
}

// Temporary returns whether the request that preceded the error is temporary.
func (e *Error) Temporary() bool {
	if e.temporary {
		return true
	}

	if len(e.Errors) == 0 {
		_, ok := temporaryStatusCodes[e.StatusCode]
		return ok
	}
	for _, d := range e.Errors {
		if _, ok := temporaryErrorCodes[d.Code]; !ok {
			return false
		}
	}
	return true
}

// Diagnostic represents a single error returned by a Docker registry interaction.
type Diagnostic struct {
	Code    ErrorCode `json:"code"`
	Message string    `json:"message,omitempty"`
	Detail  any       `json:"detail,omitempty"`
}

// String stringifies the Diagnostic in the form: $Code: $Message[; $Detail]
func (d Diagnostic) String() string {
	msg := fmt.Sprintf("%s: %s", d.Code, d.Message)
	if d.Detail != nil {
		msg = fmt.Sprintf("%s; %v", msg, d.Detail)
	}
	return msg
}

// ErrorCode is an enumeration of supported error codes.
type ErrorCode string

// The set of error conditions a registry may return:
// https://github.com/docker/distribution/blob/master/docs/spec/api.md#errors-2
const (
	BlobUnknownErrorCode         ErrorCode = "BLOB_UNKNOWN"
	BlobUploadInvalidErrorCode   ErrorCode = "BLOB_UPLOAD_INVALID"
	BlobUploadUnknownErrorCode   ErrorCode = "BLOB_UPLOAD_UNKNOWN"
	DigestInvalidErrorCode       ErrorCode = "DIGEST_INVALID"
	ManifestBlobUnknownErrorCode ErrorCode = "MANIFEST_BLOB_UNKNOWN"
	ManifestInvalidErrorCode     ErrorCode = "MANIFEST_INVALID"
	ManifestUnknownErrorCode     ErrorCode = "MANIFEST_UNKNOWN"
	ManifestUnverifiedErrorCode  ErrorCode = "MANIFEST_UNVERIFIED"
	NameInvalidErrorCode         ErrorCode = "NAME_INVALID"
	NameUnknownErrorCode         ErrorCode = "NAME_UNKNOWN"
	SizeInvalidErrorCode         ErrorCode = "SIZE_INVALID"
	TagInvalidErrorCode          ErrorCode = "TAG_INVALID"
	UnauthorizedErrorCode        ErrorCode = "UNAUTHORIZED"
	DeniedErrorCode              ErrorCode = "DENIED"
	UnsupportedErrorCode         ErrorCode = "UNSUPPORTED"
	TooManyRequestsErrorCode     ErrorCode = "TOOMANYREQUESTS"
	UnknownErrorCode             ErrorCode = "UNKNOWN"

	// This isn't defined by either docker or OCI spec, but is defined by docker/distribution:
	// https://github.com/distribution/distribution/blob/6a977a5a754baa213041443f841705888107362a/registry/api/errcode/register.go#L60
	UnavailableErrorCode ErrorCode = "UNAVAILABLE"
)

// TODO: Include other error types.
var temporaryErrorCodes = map[ErrorCode]struct{}{
	BlobUploadInvalidErrorCode: {},
	TooManyRequestsErrorCode:   {},
	UnknownErrorCode:           {},
	UnavailableErrorCode:       {},
}

var temporaryStatusCodes = map[int]struct{}{
	http.StatusRequestTimeout:      {},
	http.StatusInternalServerError: {},
	http.StatusBadGateway:          {},
	http.StatusServiceUnavailable:  {},
	http.StatusGatewayTimeout:      {},
}

// CheckError returns a structured error if the response status is not in codes.
func CheckError(resp *http.Response, codes ...int) error {
	for _, code := range codes {
		if resp.StatusCode == code {
			// This is one of the supported status codes.
			return nil
		}
	}

	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	return makeError(resp, b)
}

func makeError(resp *http.Response, body []byte) *Error {
	// https://github.com/docker/distribution/blob/master/docs/spec/api.md#errors
	structuredError := &Error{}

	// This can fail if e.g. the response body is not valid JSON. That's fine,
	// we'll construct an appropriate error string from the body and status code.
	_ = json.Unmarshal(body, structuredError)

	structuredError.rawBody = string(body)
	structuredError.StatusCode = resp.StatusCode
	structuredError.Request = resp.Request

	return structuredError
}

func retryError(resp *http.Response) error {
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	rerr := makeError(resp, b)
	rerr.temporary = true
	return rerr
}
