package oc

import (
	"errors"
	"io"
	"net"
	"os"

	"github.com/containerd/errdefs"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// todo: break import cycle with "internal/hcs/errors.go" and reference errors defined there
// todo: add errors defined in "internal/guest/gcserror" (Hresult does not implement error)

func toStatusCode(err error) codes.Code {
	// checks if err implements GRPCStatus() *"google.golang.org/grpc/status".Status,
	// wraps an error defined in "github.com/containerd/errdefs", or is a
	// context timeout or cancelled error
	if s, ok := status.FromError(errdefs.ToGRPC(err)); ok {
		return s.Code()
	}

	switch {
	// case isAny(err):
	// 	return codes.Cancelled
	case isAny(err, os.ErrInvalid):
		return codes.InvalidArgument
	case isAny(err, os.ErrDeadlineExceeded):
		return codes.DeadlineExceeded
	case isAny(err, os.ErrNotExist):
		return codes.NotFound
	case isAny(err, os.ErrExist):
		return codes.AlreadyExists
	case isAny(err, os.ErrPermission):
		return codes.PermissionDenied
	// case isAny(err):
	// 	return codes.ResourceExhausted
	case isAny(err, os.ErrClosed, net.ErrClosed, io.ErrClosedPipe, io.ErrShortBuffer):
		return codes.FailedPrecondition
	// case isAny(err):
	// 	return codes.Aborted
	// case isAny(err):
	// 	return codes.OutOfRange
	// case isAny(err):
	// 	return codes.Unimplemented
	case isAny(err, io.ErrNoProgress):
		return codes.Internal
	// case isAny(err):
	// 	return codes.Unavailable
	case isAny(err, io.ErrShortWrite, io.ErrUnexpectedEOF):
		return codes.DataLoss
	// case isAny(err):
	// 	return codes.Unauthenticated
	default:
		return codes.Unknown
	}
}

// isAny returns true if errors.Is is true for any of the provided errors, errs.
func isAny(err error, errs ...error) bool {
	for _, e := range errs {
		if errors.Is(err, e) {
			return true
		}
	}
	return false
}
