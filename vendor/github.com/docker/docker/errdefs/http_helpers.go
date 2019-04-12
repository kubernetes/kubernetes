package errdefs // import "github.com/docker/docker/errdefs"

import (
	"fmt"
	"net/http"

	"github.com/docker/distribution/registry/api/errcode"
	"github.com/sirupsen/logrus"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// GetHTTPErrorStatusCode retrieves status code from error message.
func GetHTTPErrorStatusCode(err error) int {
	if err == nil {
		logrus.WithFields(logrus.Fields{"error": err}).Error("unexpected HTTP error handling")
		return http.StatusInternalServerError
	}

	var statusCode int

	// Stop right there
	// Are you sure you should be adding a new error class here? Do one of the existing ones work?

	// Note that the below functions are already checking the error causal chain for matches.
	switch {
	case IsNotFound(err):
		statusCode = http.StatusNotFound
	case IsInvalidParameter(err):
		statusCode = http.StatusBadRequest
	case IsConflict(err):
		statusCode = http.StatusConflict
	case IsUnauthorized(err):
		statusCode = http.StatusUnauthorized
	case IsUnavailable(err):
		statusCode = http.StatusServiceUnavailable
	case IsForbidden(err):
		statusCode = http.StatusForbidden
	case IsNotModified(err):
		statusCode = http.StatusNotModified
	case IsNotImplemented(err):
		statusCode = http.StatusNotImplemented
	case IsSystem(err) || IsUnknown(err) || IsDataLoss(err) || IsDeadline(err) || IsCancelled(err):
		statusCode = http.StatusInternalServerError
	default:
		statusCode = statusCodeFromGRPCError(err)
		if statusCode != http.StatusInternalServerError {
			return statusCode
		}
		statusCode = statusCodeFromDistributionError(err)
		if statusCode != http.StatusInternalServerError {
			return statusCode
		}
		if e, ok := err.(causer); ok {
			return GetHTTPErrorStatusCode(e.Cause())
		}

		logrus.WithFields(logrus.Fields{
			"module":     "api",
			"error_type": fmt.Sprintf("%T", err),
		}).Debugf("FIXME: Got an API for which error does not match any expected type!!!: %+v", err)
	}

	if statusCode == 0 {
		statusCode = http.StatusInternalServerError
	}

	return statusCode
}

// FromStatusCode creates an errdef error, based on the provided HTTP status-code
func FromStatusCode(err error, statusCode int) error {
	if err == nil {
		return err
	}
	switch statusCode {
	case http.StatusNotFound:
		err = NotFound(err)
	case http.StatusBadRequest:
		err = InvalidParameter(err)
	case http.StatusConflict:
		err = Conflict(err)
	case http.StatusUnauthorized:
		err = Unauthorized(err)
	case http.StatusServiceUnavailable:
		err = Unavailable(err)
	case http.StatusForbidden:
		err = Forbidden(err)
	case http.StatusNotModified:
		err = NotModified(err)
	case http.StatusNotImplemented:
		err = NotImplemented(err)
	case http.StatusInternalServerError:
		if !IsSystem(err) && !IsUnknown(err) && !IsDataLoss(err) && !IsDeadline(err) && !IsCancelled(err) {
			err = System(err)
		}
	default:
		logrus.WithFields(logrus.Fields{
			"module":      "api",
			"status_code": fmt.Sprintf("%d", statusCode),
		}).Debugf("FIXME: Got an status-code for which error does not match any expected type!!!: %d", statusCode)

		switch {
		case statusCode >= 200 && statusCode < 400:
			// it's a client error
		case statusCode >= 400 && statusCode < 500:
			err = InvalidParameter(err)
		case statusCode >= 500 && statusCode < 600:
			err = System(err)
		default:
			err = Unknown(err)
		}
	}
	return err
}

// statusCodeFromGRPCError returns status code according to gRPC error
func statusCodeFromGRPCError(err error) int {
	switch status.Code(err) {
	case codes.InvalidArgument: // code 3
		return http.StatusBadRequest
	case codes.NotFound: // code 5
		return http.StatusNotFound
	case codes.AlreadyExists: // code 6
		return http.StatusConflict
	case codes.PermissionDenied: // code 7
		return http.StatusForbidden
	case codes.FailedPrecondition: // code 9
		return http.StatusBadRequest
	case codes.Unauthenticated: // code 16
		return http.StatusUnauthorized
	case codes.OutOfRange: // code 11
		return http.StatusBadRequest
	case codes.Unimplemented: // code 12
		return http.StatusNotImplemented
	case codes.Unavailable: // code 14
		return http.StatusServiceUnavailable
	default:
		if e, ok := err.(causer); ok {
			return statusCodeFromGRPCError(e.Cause())
		}
		// codes.Canceled(1)
		// codes.Unknown(2)
		// codes.DeadlineExceeded(4)
		// codes.ResourceExhausted(8)
		// codes.Aborted(10)
		// codes.Internal(13)
		// codes.DataLoss(15)
		return http.StatusInternalServerError
	}
}

// statusCodeFromDistributionError returns status code according to registry errcode
// code is loosely based on errcode.ServeJSON() in docker/distribution
func statusCodeFromDistributionError(err error) int {
	switch errs := err.(type) {
	case errcode.Errors:
		if len(errs) < 1 {
			return http.StatusInternalServerError
		}
		if _, ok := errs[0].(errcode.ErrorCoder); ok {
			return statusCodeFromDistributionError(errs[0])
		}
	case errcode.ErrorCoder:
		return errs.ErrorCode().Descriptor().HTTPStatusCode
	default:
		if e, ok := err.(causer); ok {
			return statusCodeFromDistributionError(e.Cause())
		}
	}
	return http.StatusInternalServerError
}
