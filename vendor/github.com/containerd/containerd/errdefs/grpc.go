package errdefs

import (
	"strings"

	"github.com/pkg/errors"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ToGRPC will attempt to map the backend containerd error into a grpc error,
// using the original error message as a description.
//
// Further information may be extracted from certain errors depending on their
// type.
//
// If the error is unmapped, the original error will be returned to be handled
// by the regular grpc error handling stack.
func ToGRPC(err error) error {
	if err == nil {
		return nil
	}

	if isGRPCError(err) {
		// error has already been mapped to grpc
		return err
	}

	switch {
	case IsInvalidArgument(err):
		return status.Errorf(codes.InvalidArgument, err.Error())
	case IsNotFound(err):
		return status.Errorf(codes.NotFound, err.Error())
	case IsAlreadyExists(err):
		return status.Errorf(codes.AlreadyExists, err.Error())
	case IsFailedPrecondition(err):
		return status.Errorf(codes.FailedPrecondition, err.Error())
	case IsUnavailable(err):
		return status.Errorf(codes.Unavailable, err.Error())
	case IsNotImplemented(err):
		return status.Errorf(codes.Unimplemented, err.Error())
	}

	return err
}

// ToGRPCf maps the error to grpc error codes, assembling the formatting string
// and combining it with the target error string.
//
// This is equivalent to errors.ToGRPC(errors.Wrapf(err, format, args...))
func ToGRPCf(err error, format string, args ...interface{}) error {
	return ToGRPC(errors.Wrapf(err, format, args...))
}

// FromGRPC returns the underlying error from a grpc service based on the grpc error code
func FromGRPC(err error) error {
	if err == nil {
		return nil
	}

	var cls error // divide these into error classes, becomes the cause

	switch grpc.Code(err) {
	case codes.InvalidArgument:
		cls = ErrInvalidArgument
	case codes.AlreadyExists:
		cls = ErrAlreadyExists
	case codes.NotFound:
		cls = ErrNotFound
	case codes.Unavailable:
		cls = ErrUnavailable
	case codes.FailedPrecondition:
		cls = ErrFailedPrecondition
	case codes.Unimplemented:
		cls = ErrNotImplemented
	default:
		cls = ErrUnknown
	}

	msg := rebaseMessage(cls, err)
	if msg != "" {
		err = errors.Wrapf(cls, msg)
	} else {
		err = errors.WithStack(cls)
	}

	return err
}

// rebaseMessage removes the repeats for an error at the end of an error
// string. This will happen when taking an error over grpc then remapping it.
//
// Effectively, we just remove the string of cls from the end of err if it
// appears there.
func rebaseMessage(cls error, err error) string {
	desc := grpc.ErrorDesc(err)
	clss := cls.Error()
	if desc == clss {
		return ""
	}

	return strings.TrimSuffix(desc, ": "+clss)
}

func isGRPCError(err error) bool {
	_, ok := status.FromError(err)
	return ok
}
