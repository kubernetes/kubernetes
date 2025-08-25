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
	"errors"
	"fmt"
	"net/http"
	"reflect"
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

var knownReasons = map[metav1.StatusReason]struct{}{
	// metav1.StatusReasonUnknown : {}
	metav1.StatusReasonUnauthorized:          {},
	metav1.StatusReasonForbidden:             {},
	metav1.StatusReasonNotFound:              {},
	metav1.StatusReasonAlreadyExists:         {},
	metav1.StatusReasonConflict:              {},
	metav1.StatusReasonGone:                  {},
	metav1.StatusReasonInvalid:               {},
	metav1.StatusReasonServerTimeout:         {},
	metav1.StatusReasonStoreReadError:        {},
	metav1.StatusReasonTimeout:               {},
	metav1.StatusReasonTooManyRequests:       {},
	metav1.StatusReasonBadRequest:            {},
	metav1.StatusReasonMethodNotAllowed:      {},
	metav1.StatusReasonNotAcceptable:         {},
	metav1.StatusReasonRequestEntityTooLarge: {},
	metav1.StatusReasonUnsupportedMediaType:  {},
	metav1.StatusReasonInternalError:         {},
	metav1.StatusReasonExpired:               {},
	metav1.StatusReasonServiceUnavailable:    {},
}

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

// HasStatusCause returns true if the provided error has a details cause
// with the provided type name.
// It supports wrapped errors and returns false when the error is nil.
func HasStatusCause(err error, name metav1.CauseType) bool {
	_, ok := StatusCause(err, name)
	return ok
}

// StatusCause returns the named cause from the provided error if it exists and
// the error unwraps to the type APIStatus. Otherwise it returns false.
func StatusCause(err error, name metav1.CauseType) (metav1.StatusCause, bool) {
	status, ok := err.(APIStatus)
	if (ok || errors.As(err, &status)) && status.Status().Details != nil {
		for _, cause := range status.Status().Details.Causes {
			if cause.Type == name {
				return cause, true
			}
		}
	}
	return metav1.StatusCause{}, false
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
		return &StatusError{ErrStatus: *t}
	case runtime.Unstructured:
		var status metav1.Status
		obj := t.UnstructuredContent()
		if !reflect.DeepEqual(obj["kind"], "Status") {
			break
		}
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(t.UnstructuredContent(), &status); err != nil {
			return err
		}
		if status.APIVersion != "v1" && status.APIVersion != "meta.k8s.io/v1" {
			break
		}
		return &StatusError{ErrStatus: status}
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

// NewGenerateNameConflict returns an error indicating the server
// was not able to generate a valid name for a resource.
func NewGenerateNameConflict(qualifiedResource schema.GroupResource, name string, retryAfterSeconds int) *StatusError {
	return &StatusError{metav1.Status{
		Status: metav1.StatusFailure,
		Code:   http.StatusConflict,
		Reason: metav1.StatusReasonAlreadyExists,
		Details: &metav1.StatusDetails{
			Group:             qualifiedResource.Group,
			Kind:              qualifiedResource.Resource,
			Name:              name,
			RetryAfterSeconds: int32(retryAfterSeconds),
		},
		Message: fmt.Sprintf(
			"%s %q already exists, the server was not able to generate a unique name for the object",
			qualifiedResource.String(), name),
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
	var message string
	if qualifiedResource.Empty() {
		message = fmt.Sprintf("forbidden: %v", err)
	} else if name == "" {
		message = fmt.Sprintf("%s is forbidden: %v", qualifiedResource.String(), err)
	} else {
		message = fmt.Sprintf("%s %q is forbidden: %v", qualifiedResource.String(), name, err)
	}
	return &StatusError{metav1.Status{
		Status: metav1.StatusFailure,
		Code:   http.StatusForbidden,
		Reason: metav1.StatusReasonForbidden,
		Details: &metav1.StatusDetails{
			Group: qualifiedResource.Group,
			Kind:  qualifiedResource.Resource,
			Name:  name,
		},
		Message: message,
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

// NewApplyConflict returns an error including details on the requests apply conflicts
func NewApplyConflict(causes []metav1.StatusCause, message string) *StatusError {
	return &StatusError{ErrStatus: metav1.Status{
		Status: metav1.StatusFailure,
		Code:   http.StatusConflict,
		Reason: metav1.StatusReasonConflict,
		Details: &metav1.StatusDetails{
			// TODO: Get obj details here?
			Causes: causes,
		},
		Message: message,
	}}
}

// NewGone returns an error indicating the item no longer available at the server and no forwarding address is known.
// DEPRECATED: Please use NewResourceExpired instead.
func NewGone(message string) *StatusError {
	return &StatusError{metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    http.StatusGone,
		Reason:  metav1.StatusReasonGone,
		Message: message,
	}}
}

// NewResourceExpired creates an error that indicates that the requested resource content has expired from
// the server (usually due to a resourceVersion that is too old).
func NewResourceExpired(message string) *StatusError {
	return &StatusError{metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    http.StatusGone,
		Reason:  metav1.StatusReasonExpired,
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
	err := &StatusError{metav1.Status{
		Status: metav1.StatusFailure,
		Code:   http.StatusUnprocessableEntity,
		Reason: metav1.StatusReasonInvalid,
		Details: &metav1.StatusDetails{
			Group:  qualifiedKind.Group,
			Kind:   qualifiedKind.Kind,
			Name:   name,
			Causes: causes,
		},
	}}
	aggregatedErrs := errs.ToAggregate()
	if aggregatedErrs == nil {
		err.ErrStatus.Message = fmt.Sprintf("%s %q is invalid", qualifiedKind.String(), name)
	} else {
		err.ErrStatus.Message = fmt.Sprintf("%s %q is invalid: %v", qualifiedKind.String(), name, aggregatedErrs)
	}
	return err
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
// if client should know why the failure was limited.
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

// NewTooManyRequestsError returns an error indicating that the request was rejected because
// the server has received too many requests. Client should wait and retry. But if the request
// is perishable, then the client should not retry the request.
func NewTooManyRequestsError(message string) *StatusError {
	return &StatusError{metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    http.StatusTooManyRequests,
		Reason:  metav1.StatusReasonTooManyRequests,
		Message: fmt.Sprintf("Too many requests: %s", message),
	}}
}

// NewRequestEntityTooLargeError returns an error indicating that the request
// entity was too large.
func NewRequestEntityTooLargeError(message string) *StatusError {
	return &StatusError{metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    http.StatusRequestEntityTooLarge,
		Reason:  metav1.StatusReasonRequestEntityTooLarge,
		Message: fmt.Sprintf("Request entity too large: %s", message),
	}}
}

// NewGenericServerResponse returns a new error for server responses that are not in a recognizable form.
func NewGenericServerResponse(code int, verb string, qualifiedResource schema.GroupResource, name, serverMessage string, retryAfterSeconds int, isUnexpectedResponse bool) *StatusError {
	reason := metav1.StatusReasonUnknown
	message := fmt.Sprintf("the server responded with the status code %d but did not return more information", code)
	switch code {
	case http.StatusConflict:
		if verb == http.MethodPost {
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
	case http.StatusNotAcceptable:
		reason = metav1.StatusReasonNotAcceptable
		// the server message has details about what types are acceptable
		if len(serverMessage) == 0 || serverMessage == "unknown" {
			message = "the server was unable to respond with a content type that the client supports"
		} else {
			message = serverMessage
		}
	case http.StatusUnsupportedMediaType:
		reason = metav1.StatusReasonUnsupportedMediaType
		// the server message has details about what types are acceptable
		message = serverMessage
	case http.StatusMethodNotAllowed:
		reason = metav1.StatusReasonMethodNotAllowed
		message = "the server does not allow this method on the requested resource"
	case http.StatusUnprocessableEntity:
		reason = metav1.StatusReasonInvalid
		message = "the server rejected our request due to an error in our request"
	case http.StatusServiceUnavailable:
		reason = metav1.StatusReasonServiceUnavailable
		message = "the server is currently unable to handle the request"
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
// It supports wrapped errors and returns false when the error is nil.
func IsNotFound(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonNotFound {
		return true
	}
	if _, ok := knownReasons[reason]; !ok && code == http.StatusNotFound {
		return true
	}
	return false
}

// IsAlreadyExists determines if the err is an error which indicates that a specified resource already exists.
// It supports wrapped errors and returns false when the error is nil.
func IsAlreadyExists(err error) bool {
	return ReasonForError(err) == metav1.StatusReasonAlreadyExists
}

// IsConflict determines if the err is an error which indicates the provided update conflicts.
// It supports wrapped errors and returns false when the error is nil.
func IsConflict(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonConflict {
		return true
	}
	if _, ok := knownReasons[reason]; !ok && code == http.StatusConflict {
		return true
	}
	return false
}

// IsInvalid determines if the err is an error which indicates the provided resource is not valid.
// It supports wrapped errors and returns false when the error is nil.
func IsInvalid(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonInvalid {
		return true
	}
	if _, ok := knownReasons[reason]; !ok && code == http.StatusUnprocessableEntity {
		return true
	}
	return false
}

// IsGone is true if the error indicates the requested resource is no longer available.
// It supports wrapped errors and returns false when the error is nil.
func IsGone(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonGone {
		return true
	}
	if _, ok := knownReasons[reason]; !ok && code == http.StatusGone {
		return true
	}
	return false
}

// IsResourceExpired is true if the error indicates the resource has expired and the current action is
// no longer possible.
// It supports wrapped errors and returns false when the error is nil.
func IsResourceExpired(err error) bool {
	return ReasonForError(err) == metav1.StatusReasonExpired
}

// IsNotAcceptable determines if err is an error which indicates that the request failed due to an invalid Accept header
// It supports wrapped errors and returns false when the error is nil.
func IsNotAcceptable(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonNotAcceptable {
		return true
	}
	if _, ok := knownReasons[reason]; !ok && code == http.StatusNotAcceptable {
		return true
	}
	return false
}

// IsUnsupportedMediaType determines if err is an error which indicates that the request failed due to an invalid Content-Type header
// It supports wrapped errors and returns false when the error is nil.
func IsUnsupportedMediaType(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonUnsupportedMediaType {
		return true
	}
	if _, ok := knownReasons[reason]; !ok && code == http.StatusUnsupportedMediaType {
		return true
	}
	return false
}

// IsMethodNotSupported determines if the err is an error which indicates the provided action could not
// be performed because it is not supported by the server.
// It supports wrapped errors and returns false when the error is nil.
func IsMethodNotSupported(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonMethodNotAllowed {
		return true
	}
	if _, ok := knownReasons[reason]; !ok && code == http.StatusMethodNotAllowed {
		return true
	}
	return false
}

// IsServiceUnavailable is true if the error indicates the underlying service is no longer available.
// It supports wrapped errors and returns false when the error is nil.
func IsServiceUnavailable(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonServiceUnavailable {
		return true
	}
	if _, ok := knownReasons[reason]; !ok && code == http.StatusServiceUnavailable {
		return true
	}
	return false
}

// IsBadRequest determines if err is an error which indicates that the request is invalid.
// It supports wrapped errors and returns false when the error is nil.
func IsBadRequest(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonBadRequest {
		return true
	}
	if _, ok := knownReasons[reason]; !ok && code == http.StatusBadRequest {
		return true
	}
	return false
}

// IsUnauthorized determines if err is an error which indicates that the request is unauthorized and
// requires authentication by the user.
// It supports wrapped errors and returns false when the error is nil.
func IsUnauthorized(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonUnauthorized {
		return true
	}
	if _, ok := knownReasons[reason]; !ok && code == http.StatusUnauthorized {
		return true
	}
	return false
}

// IsForbidden determines if err is an error which indicates that the request is forbidden and cannot
// be completed as requested.
// It supports wrapped errors and returns false when the error is nil.
func IsForbidden(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonForbidden {
		return true
	}
	if _, ok := knownReasons[reason]; !ok && code == http.StatusForbidden {
		return true
	}
	return false
}

// IsTimeout determines if err is an error which indicates that request times out due to long
// processing.
// It supports wrapped errors and returns false when the error is nil.
func IsTimeout(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonTimeout {
		return true
	}
	if _, ok := knownReasons[reason]; !ok && code == http.StatusGatewayTimeout {
		return true
	}
	return false
}

// IsServerTimeout determines if err is an error which indicates that the request needs to be retried
// by the client.
// It supports wrapped errors and returns false when the error is nil.
func IsServerTimeout(err error) bool {
	// do not check the status code, because no https status code exists that can
	// be scoped to retryable timeouts.
	return ReasonForError(err) == metav1.StatusReasonServerTimeout
}

// IsInternalError determines if err is an error which indicates an internal server error.
// It supports wrapped errors and returns false when the error is nil.
func IsInternalError(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonInternalError {
		return true
	}
	if _, ok := knownReasons[reason]; !ok && code == http.StatusInternalServerError {
		return true
	}
	return false
}

// IsTooManyRequests determines if err is an error which indicates that there are too many requests
// that the server cannot handle.
// It supports wrapped errors and returns false when the error is nil.
func IsTooManyRequests(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonTooManyRequests {
		return true
	}

	// IsTooManyRequests' checking of code predates the checking of the code in
	// the other Is* functions. In order to maintain backward compatibility, this
	// does not check that the reason is unknown.
	if code == http.StatusTooManyRequests {
		return true
	}
	return false
}

// IsRequestEntityTooLargeError determines if err is an error which indicates
// the request entity is too large.
// It supports wrapped errors and returns false when the error is nil.
func IsRequestEntityTooLargeError(err error) bool {
	reason, code := reasonAndCodeForError(err)
	if reason == metav1.StatusReasonRequestEntityTooLarge {
		return true
	}

	// IsRequestEntityTooLargeError's checking of code predates the checking of
	// the code in the other Is* functions. In order to maintain backward
	// compatibility, this does not check that the reason is unknown.
	if code == http.StatusRequestEntityTooLarge {
		return true
	}
	return false
}

// IsUnexpectedServerError returns true if the server response was not in the expected API format,
// and may be the result of another HTTP actor.
// It supports wrapped errors and returns false when the error is nil.
func IsUnexpectedServerError(err error) bool {
	status, ok := err.(APIStatus)
	if (ok || errors.As(err, &status)) && status.Status().Details != nil {
		for _, cause := range status.Status().Details.Causes {
			if cause.Type == metav1.CauseTypeUnexpectedServerResponse {
				return true
			}
		}
	}
	return false
}

// IsUnexpectedObjectError determines if err is due to an unexpected object from the master.
// It supports wrapped errors and returns false when the error is nil.
func IsUnexpectedObjectError(err error) bool {
	uoe, ok := err.(*UnexpectedObjectError)
	return err != nil && (ok || errors.As(err, &uoe))
}

// IsStoreReadError determines if err is due to either failure to transform the
// data from the storage, or failure to decode the object appropriately.
func IsStoreReadError(err error) bool {
	return ReasonForError(err) == metav1.StatusReasonStoreReadError
}

// SuggestsClientDelay returns true if this error suggests a client delay as well as the
// suggested seconds to wait, or false if the error does not imply a wait. It does not
// address whether the error *should* be retried, since some errors (like a 3xx) may
// request delay without retry.
// It supports wrapped errors and returns false when the error is nil.
func SuggestsClientDelay(err error) (int, bool) {
	t, ok := err.(APIStatus)
	if (ok || errors.As(err, &t)) && t.Status().Details != nil {
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
	return 0, false
}

// ReasonForError returns the HTTP status for a particular error.
// It supports wrapped errors and returns StatusReasonUnknown when
// the error is nil or doesn't have a status.
func ReasonForError(err error) metav1.StatusReason {
	if status, ok := err.(APIStatus); ok || errors.As(err, &status) {
		return status.Status().Reason
	}
	return metav1.StatusReasonUnknown
}

func reasonAndCodeForError(err error) (metav1.StatusReason, int32) {
	if status, ok := err.(APIStatus); ok || errors.As(err, &status) {
		return status.Status().Reason, status.Status().Code
	}
	return metav1.StatusReasonUnknown, 0
}

// ErrorReporter converts generic errors into runtime.Object errors without
// requiring the caller to take a dependency on meta/v1 (where Status lives).
// This prevents circular dependencies in core watch code.
type ErrorReporter struct {
	code   int
	verb   string
	reason string
}

// NewClientErrorReporter will respond with valid v1.Status objects that report
// unexpected server responses. Primarily used by watch to report errors when
// we attempt to decode a response from the server and it is not in the form
// we expect. Because watch is a dependency of the core api, we can't return
// meta/v1.Status in that package and so much inject this interface to convert a
// generic error as appropriate. The reason is passed as a unique status cause
// on the returned status, otherwise the generic "ClientError" is returned.
func NewClientErrorReporter(code int, verb string, reason string) *ErrorReporter {
	return &ErrorReporter{
		code:   code,
		verb:   verb,
		reason: reason,
	}
}

// AsObject returns a valid error runtime.Object (a v1.Status) for the given
// error, using the code and verb of the reporter type. The error is set to
// indicate that this was an unexpected server response.
func (r *ErrorReporter) AsObject(err error) runtime.Object {
	status := NewGenericServerResponse(r.code, r.verb, schema.GroupResource{}, "", err.Error(), 0, true)
	if status.ErrStatus.Details == nil {
		status.ErrStatus.Details = &metav1.StatusDetails{}
	}
	reason := r.reason
	if len(reason) == 0 {
		reason = "ClientError"
	}
	status.ErrStatus.Details.Causes = append(status.ErrStatus.Details.Causes, metav1.StatusCause{
		Type:    metav1.CauseType(reason),
		Message: err.Error(),
	})
	return &status.ErrStatus
}
