package errdefs // import "github.com/docker/docker/errdefs"

import (
	"net/http"

	"github.com/sirupsen/logrus"
)

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
		logrus.WithError(err).WithFields(logrus.Fields{
			"module":      "api",
			"status_code": statusCode,
		}).Debug("FIXME: Got an status-code for which error does not match any expected type!!!")

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
