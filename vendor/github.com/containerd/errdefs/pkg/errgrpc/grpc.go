/*
   Copyright The containerd Authors.

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

// Package errgrpc provides utility functions for translating errors to
// and from a gRPC context.
//
// The functions ToGRPC and ToNative can be used to map server-side and
// client-side errors to the correct types.
package errgrpc

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	spb "google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/protoadapt"
	"google.golang.org/protobuf/types/known/anypb"

	"github.com/containerd/typeurl/v2"

	"github.com/containerd/errdefs"
	"github.com/containerd/errdefs/pkg/internal/cause"
	"github.com/containerd/errdefs/pkg/internal/types"
)

// ToGRPC will attempt to map the error into a grpc error, from the error types
// defined in the the errdefs package and attempign to preserve the original
// description. Any type which does not resolve to a defined error type will
// be assigned the unknown error code.
//
// Further information may be extracted from certain errors depending on their
// type. The grpc error details will be used to attempt to preserve as much of
// the error structures and types as possible.
//
// Errors which can be marshaled using protobuf or typeurl will be considered
// for including as GRPC error details.
// Additionally, use the following interfaces in errors to preserve custom types:
//
//	WrapError(error) error     - Used to wrap the previous error
//	JoinErrors(...error) error - Used to join all previous errors
//	CollapseError()            - Used for errors which carry information but
//	                             should not have their error message shown.
func ToGRPC(err error) error {
	if err == nil {
		return nil
	}

	if _, ok := status.FromError(err); ok {
		// error has already been mapped to grpc
		return err
	}
	st := statusFromError(err)
	if st != nil {
		if details := errorDetails(err, false); len(details) > 0 {
			if ds, _ := st.WithDetails(details...); ds != nil {
				st = ds
			}
		}
		err = st.Err()
	}
	return err
}

func statusFromError(err error) *status.Status {
	switch errdefs.Resolve(err) {
	case errdefs.ErrInvalidArgument:
		return status.New(codes.InvalidArgument, err.Error())
	case errdefs.ErrNotFound:
		return status.New(codes.NotFound, err.Error())
	case errdefs.ErrAlreadyExists:
		return status.New(codes.AlreadyExists, err.Error())
	case errdefs.ErrPermissionDenied:
		return status.New(codes.PermissionDenied, err.Error())
	case errdefs.ErrResourceExhausted:
		return status.New(codes.ResourceExhausted, err.Error())
	case errdefs.ErrFailedPrecondition, errdefs.ErrConflict, errdefs.ErrNotModified:
		return status.New(codes.FailedPrecondition, err.Error())
	case errdefs.ErrAborted:
		return status.New(codes.Aborted, err.Error())
	case errdefs.ErrOutOfRange:
		return status.New(codes.OutOfRange, err.Error())
	case errdefs.ErrNotImplemented:
		return status.New(codes.Unimplemented, err.Error())
	case errdefs.ErrInternal:
		return status.New(codes.Internal, err.Error())
	case errdefs.ErrUnavailable:
		return status.New(codes.Unavailable, err.Error())
	case errdefs.ErrDataLoss:
		return status.New(codes.DataLoss, err.Error())
	case errdefs.ErrUnauthenticated:
		return status.New(codes.Unauthenticated, err.Error())
	case context.DeadlineExceeded:
		return status.New(codes.DeadlineExceeded, err.Error())
	case context.Canceled:
		return status.New(codes.Canceled, err.Error())
	case errdefs.ErrUnknown:
		return status.New(codes.Unknown, err.Error())
	}
	return nil
}

// errorDetails returns an array of errors which make up the provided error.
// If firstIncluded is true, then all encodable errors will be used, otherwise
// the first error in an error list will be not be used, to account for the
// the base status error which details are added to via wrap or join.
//
// The errors are ordered in way that they can be applied in order by either
// wrapping or joining the errors to recreate an error with the same structure
// when `WrapError` and `JoinErrors` interfaces are used.
//
// The intent is that when re-applying the errors to create a single error, the
// results of calls to `Error()`, `errors.Is`, `errors.As`, and "%+v" formatting
// is the same as the original error.
func errorDetails(err error, firstIncluded bool) []protoadapt.MessageV1 {
	switch uerr := err.(type) {
	case interface{ Unwrap() error }:
		details := errorDetails(uerr.Unwrap(), firstIncluded)

		// If the type is able to wrap, then include if proto
		if _, ok := err.(interface{ WrapError(error) error }); ok {
			// Get proto message
			if protoErr := toProtoMessage(err); protoErr != nil {
				details = append(details, protoErr)
			}
		}

		return details
	case interface{ Unwrap() []error }:
		var details []protoadapt.MessageV1
		for i, e := range uerr.Unwrap() {
			details = append(details, errorDetails(e, firstIncluded || i > 0)...)
		}

		if _, ok := err.(interface{ JoinErrors(...error) error }); ok {
			// Get proto message
			if protoErr := toProtoMessage(err); protoErr != nil {
				details = append(details, protoErr)
			}
		}
		return details
	}

	if firstIncluded {
		if protoErr := toProtoMessage(err); protoErr != nil {
			return []protoadapt.MessageV1{protoErr}
		}
		if gs, ok := status.FromError(ToGRPC(err)); ok {
			return []protoadapt.MessageV1{gs.Proto()}
		}
		// TODO: Else include unknown extra error type?
	}

	return nil
}

func toProtoMessage(err error) protoadapt.MessageV1 {
	// Do not double encode proto messages, otherwise use Any
	if pm, ok := err.(protoadapt.MessageV1); ok {
		return pm
	}
	if pm, ok := err.(proto.Message); ok {
		return protoadapt.MessageV1Of(pm)
	}

	if reflect.TypeOf(err).Kind() == reflect.Ptr {
		a, aerr := typeurl.MarshalAny(err)
		if aerr == nil {
			return &anypb.Any{
				TypeUrl: a.GetTypeUrl(),
				Value:   a.GetValue(),
			}
		}
	}
	return nil
}

// ToGRPCf maps the error to grpc error codes, assembling the formatting string
// and combining it with the target error string.
//
// This is equivalent to grpc.ToGRPC(fmt.Errorf("%s: %w", fmt.Sprintf(format, args...), err))
func ToGRPCf(err error, format string, args ...interface{}) error {
	return ToGRPC(fmt.Errorf("%s: %w", fmt.Sprintf(format, args...), err))
}

// ToNative returns the underlying error from a grpc service based on the grpc
// error code. The grpc details are used to add wrap the error in more context
// or support multiple errors.
func ToNative(err error) error {
	if err == nil {
		return nil
	}

	s, isGRPC := status.FromError(err)

	var (
		desc string
		code codes.Code
	)

	if isGRPC {
		desc = s.Message()
		code = s.Code()
	} else {
		desc = err.Error()
		code = codes.Unknown
	}

	var cls error // divide these into error classes, becomes the cause

	switch code {
	case codes.InvalidArgument:
		cls = errdefs.ErrInvalidArgument
	case codes.AlreadyExists:
		cls = errdefs.ErrAlreadyExists
	case codes.NotFound:
		cls = errdefs.ErrNotFound
	case codes.Unavailable:
		cls = errdefs.ErrUnavailable
	case codes.FailedPrecondition:
		// TODO: Has suffix is not sufficient for conflict and not modified
		// Message should start with ": " or be at beginning of a line
		// Message should end with ": " or be at the end of a line
		// Compile a regex
		if desc == errdefs.ErrConflict.Error() || strings.HasSuffix(desc, ": "+errdefs.ErrConflict.Error()) {
			cls = errdefs.ErrConflict
		} else if desc == errdefs.ErrNotModified.Error() || strings.HasSuffix(desc, ": "+errdefs.ErrNotModified.Error()) {
			cls = errdefs.ErrNotModified
		} else {
			cls = errdefs.ErrFailedPrecondition
		}
	case codes.Unimplemented:
		cls = errdefs.ErrNotImplemented
	case codes.Canceled:
		cls = context.Canceled
	case codes.DeadlineExceeded:
		cls = context.DeadlineExceeded
	case codes.Aborted:
		cls = errdefs.ErrAborted
	case codes.Unauthenticated:
		cls = errdefs.ErrUnauthenticated
	case codes.PermissionDenied:
		cls = errdefs.ErrPermissionDenied
	case codes.Internal:
		cls = errdefs.ErrInternal
	case codes.DataLoss:
		cls = errdefs.ErrDataLoss
	case codes.OutOfRange:
		cls = errdefs.ErrOutOfRange
	case codes.ResourceExhausted:
		cls = errdefs.ErrResourceExhausted
	default:
		if idx := strings.LastIndex(desc, cause.UnexpectedStatusPrefix); idx > 0 {
			if status, uerr := strconv.Atoi(desc[idx+len(cause.UnexpectedStatusPrefix):]); uerr == nil && status >= 200 && status < 600 {
				cls = cause.ErrUnexpectedStatus{Status: status}
			}
		}
		if cls == nil {
			cls = errdefs.ErrUnknown
		}
	}

	msg := rebaseMessage(cls, desc)
	if msg == "" {
		err = cls
	} else if msg != desc {
		err = fmt.Errorf("%s: %w", msg, cls)
	} else if wm, ok := cls.(interface{ WithMessage(string) error }); ok {
		err = wm.WithMessage(msg)
	} else {
		err = fmt.Errorf("%s: %w", msg, cls)
	}

	if isGRPC {
		errs := []error{err}
		for _, a := range s.Details() {
			var derr error

			// First decode error if needed
			if s, ok := a.(*spb.Status); ok {
				derr = ToNative(status.ErrorProto(s))
			} else if e, ok := a.(error); ok {
				derr = e
			} else if dany, ok := a.(typeurl.Any); ok {
				i, uerr := typeurl.UnmarshalAny(dany)
				if uerr == nil {
					if e, ok = i.(error); ok {
						derr = e
					} else {
						derr = fmt.Errorf("non-error unmarshalled detail: %v", i)
					}
				} else {
					derr = fmt.Errorf("error of type %q with failure to unmarshal: %v", dany.GetTypeUrl(), uerr)
				}
			} else {
				derr = fmt.Errorf("non-error detail: %v", a)
			}

			switch werr := derr.(type) {
			case interface{ WrapError(error) error }:
				errs[len(errs)-1] = werr.WrapError(errs[len(errs)-1])
			case interface{ JoinErrors(...error) error }:
				// TODO: Consider whether this should support joining a subset
				errs[0] = werr.JoinErrors(errs...)
			case interface{ CollapseError() }:
				errs[len(errs)-1] = types.CollapsedError(errs[len(errs)-1], derr)
			default:
				errs = append(errs, derr)
			}

		}
		if len(errs) > 1 {
			err = errors.Join(errs...)
		} else {
			err = errs[0]
		}
	}

	return err
}

// rebaseMessage removes the repeats for an error at the end of an error
// string. This will happen when taking an error over grpc then remapping it.
//
// Effectively, we just remove the string of cls from the end of err if it
// appears there.
func rebaseMessage(cls error, desc string) string {
	clss := cls.Error()
	if desc == clss {
		return ""
	}

	return strings.TrimSuffix(desc, ": "+clss)
}
