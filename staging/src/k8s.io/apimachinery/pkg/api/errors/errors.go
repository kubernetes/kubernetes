/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package errors

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// StatusError is an error intended for consumption by a REST API server; it can also be
// reconstructed by clients from a REST response. Public to allow easy type switches.
type StatusError struct {
	ErrStatus metav1.Status
}

// APIStatus is exposed by errors that can be converted to an api.Status object
// for finer grained details.
type APIStatus interface {
	Status() metav1.Status
}

var _ error = &StatusError{}

// Error implements the Error interface.
func (e *StatusError) Error() string {
	return e.ErrStatus.Message
}

// Status allows access to e's status without having to know the detailed workings
// of StatusError.
func (e *StatusError) Status() metav1.Status {
	return e.ErrStatus
}

// DebugError reports extended info about the error to debug output.
func (e *StatusError) DebugError() (string, []interface{}) {
	if out, err := json.MarshalIndent(e.ErrStatus, "", "  "); err == nil {
		return "server response object: %s", []interface{}{string(out)}
	}
	return "server response object: %#v", []interface{}{e.ErrStatus}
}

// UnexpectedObjectError can be returned by FromObject if it's passed a non-status object.
type UnexpectedObjectError struct {
	Object runtime.Object
}

// Error returns an error message describing 'u'.
func (u *UnexpectedObjectError) Error() string {
	return fmt.Sprintf("unexpected object: %v", u.Object)
}

// FromObject generates an StatusError from an metav1.Status, if that is the type of obj; otherwise,
// returns an UnexpecteObjectError.
func FromObject(obj runtime.Object) error {
	switch t := obj.(type) {
	case *metav1.Status:
		return &StatusError{*t}
	}
	return &UnexpectedObjectError{obj}
}

// NewNotFound returns a new error which indicates that the resource of the kind and the name was not found.
func NewNotFound(qualifiedResource schema.GroupResource, name string) *StatusError {
	return &StatusError{metav1.Status{
		Status: metav1.StatusFailure,
		Code:   http.StatusNotFound,
		Reason: metav1.StatusReasonNotFound,
		Details: &metav1.StatusDetails{
			Group: qualifiedResource.Group,
			Kind:  qualifiedResource.Resource,
			Name:  name,
		},
		Message: fmt.Sprintf("%s %q not found", qualifiedResource.String(), name),
	}}
}

// NewAlreadyExists returns an error indicating the item requested exists by that identifier.
func NewAlreadyExists(qualifiedResource schema.GroupResource, name string) *StatusError {
	return &StatusError{metav1.Status{
		Status: metav1.StatusFailure,
		Code:   http.StatusConflict,
		Reason: metav1.StatusReasonAlreadyExists,
		Details: &metav1.StatusDetails{
			Group: qualifiedResource.Group,
			Kind:  qualifiedResource.Resource,
			Name:  name,
		},
		Message: fmt.Sprintf("%s %q already exists", qualifiedResource.String(), name),
	}}
}

// NewUnauthorized returns an error indicating the client is not authorized to perform the requested
// action.
func NewUnauthorized(reason string) *StatusError {
	message := reason
	if len(message) == 0 {
		message = "not authorized"
	}
	return &StatusError{metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    http.StatusUnauthorized,
		Reason:  metav1.StatusReasonUnauthorized,
		Message: message,
	}}
}

// NewForbidden returns an error indicating the requested action was forbidden
func NewForbidden(qualifiedResource schema.GroupResource, name string, err error) *StatusError {
	return &StatusError{metav1.Status{
		Status: metav1.StatusFailure,
		Code:   http.StatusForbidden,
		Reason: metav1.StatusReasonForbidden,
		Details: &metav1.StatusDetails{
			Group: qualifiedResource.Group,
			Kind:  qualifiedResource.Resource,
			Name:  name,
		},
		Message: fmt.Sprintf("%s %q is forbidden: %v", qualifiedResource.String(), name, err),
	}}
}

// NewConflict returns an error indicating the item can't be updated as provided.
func NewConflict(qualifiedResource schema.GroupResource, name string, err error) *StatusError {
	return &StatusError{metav1.Status{
		Status: metav1.StatusFailure,
		Code:   http.StatusConflict,
		Reason: metav1.StatusReasonConflict,
		Details: &metav1.StatusDetails{
			Group: qualifiedResource.Group,
			Kind:  qualifiedResource.Resource,
			Name:  name,
		},
		Message: fmt.Sprintf("Operation cannot be fulfilled on %s %q: %v", qualifiedResource.String(), name, err),
	}}
}

// NewGone returns an error indicating the item no longer available at the server and no forwarding address is known.
func NewGone(message string) *StatusError {
	return &StatusError{metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    http.StatusGone,
		Reason:  metav1.StatusReasonGone,
		Message: message,
	}}
}

// NewInvalid returns an error indicating the item is invalid and cannot be processed.
func NewInvalid(qualifiedKind schema.GroupKind, name string, errs field.ErrorList) *StatusError {
	causes := make([]metav1.StatusCause, 0, len(errs))
	for i := range errs {
		err := errs[i]
		causes = append(causes, metav1.StatusCause{
			Type:    metav1.CauseType(err.Type),
			Message: err.ErrorBody(),
			Field:   err.Field,
		})
	}
	return &StatusError{metav1.Status{
		Status: metav1.StatusFailure,
		Code:   http.StatusUnprocessableEntity,
		Reason: metav1.StatusReasonInvalid,
		Details: &metav1.StatusDetails{
			Group:  qualifiedKind.Group,
			Kind:   qualifiedKind.Kind,
			Name:   name,
			Causes: causes,
		},
		Message: fmt.Sprintf("%s %q is invalid: %v", qualifiedKind.String(), name, errs.ToAggregate()),
	}}
}

// NewBadRequest creates an error that indicates that the request is invalid and can not be processed.
func NewBadRequest(reason string) *StatusError {
	return &StatusError{metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    http.StatusBadRequest,
		Reason:  metav1.StatusReasonBadRequest,
		Message: reason,
	}}
}

// NewTooManyRequests creates an error that indicates that the client must try again later because
// the specified endpoint is not accepting requests. More specific details should be provided
// if client should know why the failure was limited4.
func NewTooManyRequests(message string, retryAfterSeconds int) *StatusError {
	return &StatusError{metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    http.StatusTooManyRequests,
		Reason:  metav1.StatusReasonTooManyRequests,
		Message: message,
		Details: &metav1.StatusDetails{
			RetryAfterSeconds: int32(retryAfterSeconds),
		},
	}}
}

// NewServiceUnavailable creates an error that indicates that the requested service is unavailable.
func NewServiceUnavailable(reason string) *StatusError {
	return &StatusError{metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    http.StatusServiceUnavailable,
		Reason:  metav1.StatusReasonServiceUnavailable,
		Message: reason,
	}}
}

// NewMethodNotSupported returns an error indicating the requested action is not supported on this kind.
func NewMethodNotSupported(qualifiedResource schema.GroupResource, action string) *StatusError {
	return &StatusError{metav1.Status{
		Status: metav1.StatusFailure,
		Code:   http.StatusMethodNotAllowed,
		Reason: metav1.StatusReasonMethodNotAllowed,
		Details: &metav1.StatusDetails{
			Group: qualifiedResource.Group,
			Kind:  qualifiedResource.Resource,
		},
		Message: fmt.Sprintf("%s is not supported on resources of kind %q", action, qualifiedResource.String()),
	}}
}

// NewServerTimeout returns an error indicating the requested action could not be completed due to a
// transient error, and the client should try again.
func NewServerTimeout(qualifiedResource schema.GroupResource, operation string, retryAfterSeconds int) *StatusError {
	return &StatusError{metav1.Status{
		Status: metav1.StatusFailure,
		Code:   http.StatusInternalServerError,
		Reason: metav1.StatusReasonServerTimeout,
		Details: &metav1.StatusDetails{
			Group:             qualifiedResource.Group,
			Kind:              qualifiedResource.Resource,
			Name:              operation,
			RetryAfterSeconds: int32(retryAfterSeconds),
		},
		Message: fmt.Sprintf("The %s operation against %s could not be completed at this time, please try again.", operation, qualifiedResource.String()),
	}}
}

// NewServerTimeoutForKind should not exist.  Server timeouts happen when accessing resources, the Kind is just what we
// happened to be looking at when the request failed.  This delegates to keep code sane, but we should work towards removing this.
func NewServerTimeoutForKind(qualifiedKind schema.GroupKind, operation string, retryAfterSeconds int) *StatusError {
	return NewServerTimeout(schema.GroupResource{Group: qualifiedKind.Group, Resource: qualifiedKind.Kind}, operation, retryAfterSeconds)
}

// NewInternalError returns an error indicating the item is invalid and cannot be processed.
func NewInternalError(err error) *StatusError {
	return &StatusError{metav1.Status{
		Status: metav1.StatusFailure,
		Code:   http.StatusInternalServerError,
		Reason: metav1.StatusReasonInternalError,
		Details: &metav1.StatusDetails{
			Causes: []metav1.StatusCause{{Message: err.Error()}},
		},
		Message: fmt.Sprintf("Internal error occurred: %v", err),
	}}
}

// NewTimeoutError returns an error indicating that a timeout occurred before the request
// could be completed.  Clients may retry, but the operation may still complete.
func NewTimeoutError(message string, retryAfterSeconds int) *StatusError {
	return &StatusError{metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    http.StatusGatewayTimeout,
		Reason:  metav1.StatusReasonTimeout,
		Message: fmt.Sprintf("Timeout: %s", message),
		Details: &metav1.StatusDetails{
			RetryAfterSeconds: int32(retryAfterSeconds),
		},
	}}
}

// NewGenericServerResponse returns a new error for server responses that are not in a recognizable form.
func NewGenericServerResponse(code int, verb string, qualifiedResource schema.GroupResource, name, serverMessage string, retryAfterSeconds int, isUnexpectedResponse bool) *StatusError {
	reason := metav1.StatusReasonUnknown
	message := fmt.Sprintf("the server responded with the status code %d but did not return more information", code)
	switch code {
	case http.StatusConflict:
		if verb == "POST" {
			reason = metav1.StatusReasonAlreadyExists
		} else {
			reason = metav1.StatusReasonConflict
		}
		message = "the server reported a conflict"
	case http.StatusNotFound:
		reason = metav1.StatusReasonNotFound
		message = "the server could not find the requested resource"
	case http.StatusBadRequest:
		reason = metav1.StatusReasonBadRequest
		message = "the server rejected our request for an unknown reason"
	case http.StatusUnauthorized:
		reason = metav1.StatusReasonUnauthorized
		message = "the server has asked for the client to provide credentials"
	case http.StatusForbidden:
		reason = metav1.StatusReasonForbidden
		// the server message has details about who is trying to perform what action.  Keep its message.
		message = serverMessage
	case http.StatusMethodNotAllowed:
		reason = metav1.StatusReasonMethodNotAllowed
		message = "the server does not allow this method on the requested resource"
	case http.StatusUnprocessableEntity:
		reason = metav1.StatusReasonInvalid
		message = "the server rejected our request due to an error in our request"
	case http.StatusGatewayTimeout:
		reason = metav1.StatusReasonTimeout
		message = "the server was unable to return a response in the time allotted, but may still be processing the request"
	case http.StatusTooManyRequests:
		reason = metav1.StatusReasonTooManyRequests
		message = "the server has received too many requests and has asked us to try again later"
	default:
		if code >= 500 {
			reason = metav1.StatusReasonInternalError
			message = fmt.Sprintf("an error on the server (%q) has prevented the request from succeeding", serverMessage)
		}
	}
	switch {
	case !qualifiedResource.Empty() && len(name) > 0:
		message = fmt.Sprintf("%s (%s %s %s)", message, strings.ToLower(verb), qualifiedResource.String(), name)
	case !qualifiedResource.Empty():
		message = fmt.Sprintf("%s (%s %s)", message, strings.ToLower(verb), qualifiedResource.String())
	}
	var causes []metav1.StatusCause
	if isUnexpectedResponse {
		causes = []metav1.StatusCause{
			{
				Type:    metav1.CauseTypeUnexpectedServerResponse,
				Message: serverMessage,
			},
		}
	} else {
		causes = nil
	}
	return &StatusError{metav1.Status{
		Status: metav1.StatusFailure,
		Code:   int32(code),
		Reason: reason,
		Details: &metav1.StatusDetails{
			Group: qualifiedResource.Group,
			Kind:  qualifiedResource.Resource,
			Name:  name,

			Causes:            causes,
			RetryAfterSeconds: int32(retryAfterSeconds),
		},
		Message: message,
	}}
}

// IsNotFound returns true if the specified error was created by NewNotFound.
func IsNotFound(err error) bool {
	return reasonForError(err) == metav1.StatusReasonNotFound
}

// IsAlreadyExists determines if the err is an error which indicates that a specified resource already exists.
func IsAlreadyExists(err error) bool {
	return reasonForError(err) == metav1.StatusReasonAlreadyExists
}

// IsConflict determines if the err is an error which indicates the provided update conflicts.
func IsConflict(err error) bool {
	return reasonForError(err) == metav1.StatusReasonConflict
}

// IsInvalid determines if the err is an error which indicates the provided resource is not valid.
func IsInvalid(err error) bool {
	return reasonForError(err) == metav1.StatusReasonInvalid
}

// IsMethodNotSupported determines if the err is an error which indicates the provided action could not
// be performed because it is not supported by the server.
func IsMethodNotSupported(err error) bool {
	return reasonForError(err) == metav1.StatusReasonMethodNotAllowed
}

// IsBadRequest determines if err is an error which indicates that the request is invalid.
func IsBadRequest(err error) bool {
	return reasonForError(err) == metav1.StatusReasonBadRequest
}

// IsUnauthorized determines if err is an error which indicates that the request is unauthorized and
// requires authentication by the user.
func IsUnauthorized(err error) bool {
	return reasonForError(err) == metav1.StatusReasonUnauthorized
}

// IsForbidden determines if err is an error which indicates that the request is forbidden and cannot
// be completed as requested.
func IsForbidden(err error) bool {
	return reasonForError(err) == metav1.StatusReasonForbidden
}

// IsTimeout determines if err is an error which indicates that request times out due to long
// processing.
func IsTimeout(err error) bool {
	return reasonForError(err) == metav1.StatusReasonTimeout
}

// IsServerTimeout determines if err is an error which indicates that the request needs to be retried
// by the client.
func IsServerTimeout(err error) bool {
	return reasonForError(err) == metav1.StatusReasonServerTimeout
}

// IsInternalError determines if err is an error which indicates an internal server error.
func IsInternalError(err error) bool {
	return reasonForError(err) == metav1.StatusReasonInternalError
}

// IsTooManyRequests determines if err is an error which indicates that there are too many requests
// that the server cannot handle.
func IsTooManyRequests(err error) bool {
	if reasonForError(err) == metav1.StatusReasonTooManyRequests {
		return true
	}
	switch t := err.(type) {
	case APIStatus:
		return t.Status().Code == http.StatusTooManyRequests
	}
	return false
}

// IsUnexpectedServerError returns true if the server response was not in the expected API format,
// and may be the result of another HTTP actor.
func IsUnexpectedServerError(err error) bool {
	switch t := err.(type) {
	case APIStatus:
		if d := t.Status().Details; d != nil {
			for _, cause := range d.Causes {
				if cause.Type == metav1.CauseTypeUnexpectedServerResponse {
					return true
				}
			}
		}
	}
	return false
}

// IsUnexpectedObjectError determines if err is due to an unexpected object from the master.
func IsUnexpectedObjectError(err error) bool {
	_, ok := err.(*UnexpectedObjectError)
	return err != nil && ok
}

// SuggestsClientDelay returns true if this error suggests a client delay as well as the
// suggested seconds to wait, or false if the error does not imply a wait. It does not
// address whether the error *should* be retried, since some errors (like a 3xx) may
// request delay without retry.
func SuggestsClientDelay(err error) (int, bool) {
	switch t := err.(type) {
	case APIStatus:
		if t.Status().Details != nil {
			switch t.Status().Reason {
			// this StatusReason explicitly requests the caller to delay the action
			case metav1.StatusReasonServerTimeout:
				return int(t.Status().Details.RetryAfterSeconds), true
			}
			// If the client requests that we retry after a certain number of seconds
			if t.Status().Details.RetryAfterSeconds > 0 {
				return int(t.Status().Details.RetryAfterSeconds), true
			}
		}
	}
	return 0, false
}

func reasonForError(err error) metav1.StatusReason {
	switch t := err.(type) {
	case APIStatus:
		return t.Status().Reason
	}
	return metav1.StatusReasonUnknown
}
