// Copyright 2021, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Package apierror implements a wrapper error for parsing error details from
// API calls. Currently, only errors representing a gRPC status are supported.
package apierror

import (
	"fmt"
	"strings"

	jsonerror "github.com/googleapis/gax-go/v2/apierror/internal/proto"
	"google.golang.org/api/googleapi"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protojson"
)

// ErrDetails holds the google/rpc/error_details.proto messages.
type ErrDetails struct {
	ErrorInfo           *errdetails.ErrorInfo
	BadRequest          *errdetails.BadRequest
	PreconditionFailure *errdetails.PreconditionFailure
	QuotaFailure        *errdetails.QuotaFailure
	RetryInfo           *errdetails.RetryInfo
	ResourceInfo        *errdetails.ResourceInfo
	RequestInfo         *errdetails.RequestInfo
	DebugInfo           *errdetails.DebugInfo
	Help                *errdetails.Help
	LocalizedMessage    *errdetails.LocalizedMessage

	// Unknown stores unidentifiable error details.
	Unknown []interface{}
}

func (e ErrDetails) String() string {
	var d strings.Builder
	if e.ErrorInfo != nil {
		d.WriteString(fmt.Sprintf("error details: name = ErrorInfo reason = %s domain = %s metadata = %s\n",
			e.ErrorInfo.GetReason(), e.ErrorInfo.GetDomain(), e.ErrorInfo.GetMetadata()))
	}

	if e.BadRequest != nil {
		v := e.BadRequest.GetFieldViolations()
		var f []string
		var desc []string
		for _, x := range v {
			f = append(f, x.GetField())
			desc = append(desc, x.GetDescription())
		}
		d.WriteString(fmt.Sprintf("error details: name = BadRequest field = %s desc = %s\n",
			strings.Join(f, " "), strings.Join(desc, " ")))
	}

	if e.PreconditionFailure != nil {
		v := e.PreconditionFailure.GetViolations()
		var t []string
		var s []string
		var desc []string
		for _, x := range v {
			t = append(t, x.GetType())
			s = append(s, x.GetSubject())
			desc = append(desc, x.GetDescription())
		}
		d.WriteString(fmt.Sprintf("error details: name = PreconditionFailure type = %s subj = %s desc = %s\n", strings.Join(t, " "),
			strings.Join(s, " "), strings.Join(desc, " ")))
	}

	if e.QuotaFailure != nil {
		v := e.QuotaFailure.GetViolations()
		var s []string
		var desc []string
		for _, x := range v {
			s = append(s, x.GetSubject())
			desc = append(desc, x.GetDescription())
		}
		d.WriteString(fmt.Sprintf("error details: name = QuotaFailure subj = %s desc = %s\n",
			strings.Join(s, " "), strings.Join(desc, " ")))
	}

	if e.RequestInfo != nil {
		d.WriteString(fmt.Sprintf("error details: name = RequestInfo id = %s data = %s\n",
			e.RequestInfo.GetRequestId(), e.RequestInfo.GetServingData()))
	}

	if e.ResourceInfo != nil {
		d.WriteString(fmt.Sprintf("error details: name = ResourceInfo type = %s resourcename = %s owner = %s desc = %s\n",
			e.ResourceInfo.GetResourceType(), e.ResourceInfo.GetResourceName(),
			e.ResourceInfo.GetOwner(), e.ResourceInfo.GetDescription()))

	}
	if e.RetryInfo != nil {
		d.WriteString(fmt.Sprintf("error details: retry in %s\n", e.RetryInfo.GetRetryDelay().AsDuration()))

	}
	if e.Unknown != nil {
		var s []string
		for _, x := range e.Unknown {
			s = append(s, fmt.Sprintf("%v", x))
		}
		d.WriteString(fmt.Sprintf("error details: name = Unknown  desc = %s\n", strings.Join(s, " ")))
	}

	if e.DebugInfo != nil {
		d.WriteString(fmt.Sprintf("error details: name = DebugInfo detail = %s stack = %s\n", e.DebugInfo.GetDetail(),
			strings.Join(e.DebugInfo.GetStackEntries(), " ")))
	}
	if e.Help != nil {
		var desc []string
		var url []string
		for _, x := range e.Help.Links {
			desc = append(desc, x.GetDescription())
			url = append(url, x.GetUrl())
		}
		d.WriteString(fmt.Sprintf("error details: name = Help desc = %s url = %s\n",
			strings.Join(desc, " "), strings.Join(url, " ")))
	}
	if e.LocalizedMessage != nil {
		d.WriteString(fmt.Sprintf("error details: name = LocalizedMessage locale = %s msg = %s\n",
			e.LocalizedMessage.GetLocale(), e.LocalizedMessage.GetMessage()))
	}

	return d.String()
}

// APIError wraps either a gRPC Status error or a HTTP googleapi.Error. It
// implements error and Status interfaces.
type APIError struct {
	err     error
	status  *status.Status
	httpErr *googleapi.Error
	details ErrDetails
}

// Details presents the error details of the APIError.
func (a *APIError) Details() ErrDetails {
	return a.details
}

// Unwrap extracts the original error.
func (a *APIError) Unwrap() error {
	return a.err
}

// Error returns a readable representation of the APIError.
func (a *APIError) Error() string {
	var msg string
	if a.status != nil {
		msg = a.err.Error()
	} else if a.httpErr != nil {
		// Truncate the googleapi.Error message because it dumps the Details in
		// an ugly way.
		msg = fmt.Sprintf("googleapi: Error %d: %s", a.httpErr.Code, a.httpErr.Message)
	}
	return strings.TrimSpace(fmt.Sprintf("%s\n%s", msg, a.details))
}

// GRPCStatus extracts the underlying gRPC Status error.
// This method is necessary to fulfill the interface
// described in https://pkg.go.dev/google.golang.org/grpc/status#FromError.
func (a *APIError) GRPCStatus() *status.Status {
	return a.status
}

// Reason returns the reason in an ErrorInfo.
// If ErrorInfo is nil, it returns an empty string.
func (a *APIError) Reason() string {
	return a.details.ErrorInfo.GetReason()
}

// Domain returns the domain in an ErrorInfo.
// If ErrorInfo is nil, it returns an empty string.
func (a *APIError) Domain() string {
	return a.details.ErrorInfo.GetDomain()
}

// Metadata returns the metadata in an ErrorInfo.
// If ErrorInfo is nil, it returns nil.
func (a *APIError) Metadata() map[string]string {
	return a.details.ErrorInfo.GetMetadata()

}

// FromError parses a Status error or a googleapi.Error and builds an APIError.
func FromError(err error) (*APIError, bool) {
	if err == nil {
		return nil, false
	}

	ae := APIError{err: err}
	st, isStatus := status.FromError(err)
	herr, isHTTPErr := err.(*googleapi.Error)

	switch {
	case isStatus:
		ae.status = st
		ae.details = parseDetails(st.Details())
	case isHTTPErr:
		ae.httpErr = herr
		ae.details = parseHTTPDetails(herr)
	default:
		return nil, false
	}

	return &ae, true

}

// parseDetails accepts a slice of interface{} that should be backed by some
// sort of proto.Message that can be cast to the google/rpc/error_details.proto
// types.
//
// This is for internal use only.
func parseDetails(details []interface{}) ErrDetails {
	var ed ErrDetails
	for _, d := range details {
		switch d := d.(type) {
		case *errdetails.ErrorInfo:
			ed.ErrorInfo = d
		case *errdetails.BadRequest:
			ed.BadRequest = d
		case *errdetails.PreconditionFailure:
			ed.PreconditionFailure = d
		case *errdetails.QuotaFailure:
			ed.QuotaFailure = d
		case *errdetails.RetryInfo:
			ed.RetryInfo = d
		case *errdetails.ResourceInfo:
			ed.ResourceInfo = d
		case *errdetails.RequestInfo:
			ed.RequestInfo = d
		case *errdetails.DebugInfo:
			ed.DebugInfo = d
		case *errdetails.Help:
			ed.Help = d
		case *errdetails.LocalizedMessage:
			ed.LocalizedMessage = d
		default:
			ed.Unknown = append(ed.Unknown, d)
		}
	}

	return ed
}

// parseHTTPDetails will convert the given googleapi.Error into the protobuf
// representation then parse the Any values that contain the error details.
//
// This is for internal use only.
func parseHTTPDetails(gae *googleapi.Error) ErrDetails {
	e := &jsonerror.Error{}
	if err := protojson.Unmarshal([]byte(gae.Body), e); err != nil {
		// If the error body does not conform to the error schema, ignore it
		// altogther. See https://cloud.google.com/apis/design/errors#http_mapping.
		return ErrDetails{}
	}

	// Coerce the Any messages into proto.Message then parse the details.
	details := []interface{}{}
	for _, any := range e.GetError().GetDetails() {
		m, err := any.UnmarshalNew()
		if err != nil {
			// Ignore malformed Any values.
			continue
		}
		details = append(details, m)
	}

	return parseDetails(details)
}
