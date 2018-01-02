package distribution

import (
	"net/url"
	"strings"
	"syscall"

	"github.com/docker/distribution"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/api/errcode"
	"github.com/docker/distribution/registry/api/v2"
	"github.com/docker/distribution/registry/client"
	"github.com/docker/distribution/registry/client/auth"
	"github.com/docker/docker/distribution/xfer"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// ErrNoSupport is an error type used for errors indicating that an operation
// is not supported. It encapsulates a more specific error.
type ErrNoSupport struct{ Err error }

func (e ErrNoSupport) Error() string {
	if e.Err == nil {
		return "not supported"
	}
	return e.Err.Error()
}

// fallbackError wraps an error that can possibly allow fallback to a different
// endpoint.
type fallbackError struct {
	// err is the error being wrapped.
	err error
	// confirmedV2 is set to true if it was confirmed that the registry
	// supports the v2 protocol. This is used to limit fallbacks to the v1
	// protocol.
	confirmedV2 bool
	// transportOK is set to true if we managed to speak HTTP with the
	// registry. This confirms that we're using appropriate TLS settings
	// (or lack of TLS).
	transportOK bool
}

// Error renders the FallbackError as a string.
func (f fallbackError) Error() string {
	return f.Cause().Error()
}

func (f fallbackError) Cause() error {
	return f.err
}

// shouldV2Fallback returns true if this error is a reason to fall back to v1.
func shouldV2Fallback(err errcode.Error) bool {
	switch err.Code {
	case errcode.ErrorCodeUnauthorized, v2.ErrorCodeManifestUnknown, v2.ErrorCodeNameUnknown:
		return true
	}
	return false
}

// TranslatePullError is used to convert an error from a registry pull
// operation to an error representing the entire pull operation. Any error
// information which is not used by the returned error gets output to
// log at info level.
func TranslatePullError(err error, ref reference.Named) error {
	switch v := err.(type) {
	case errcode.Errors:
		if len(v) != 0 {
			for _, extra := range v[1:] {
				logrus.Infof("Ignoring extra error returned from registry: %v", extra)
			}
			return TranslatePullError(v[0], ref)
		}
	case errcode.Error:
		var newErr error
		switch v.Code {
		case errcode.ErrorCodeDenied:
			// ErrorCodeDenied is used when access to the repository was denied
			newErr = errors.Errorf("pull access denied for %s, repository does not exist or may require 'docker login'", reference.FamiliarName(ref))
		case v2.ErrorCodeManifestUnknown:
			newErr = errors.Errorf("manifest for %s not found", reference.FamiliarString(ref))
		case v2.ErrorCodeNameUnknown:
			newErr = errors.Errorf("repository %s not found", reference.FamiliarName(ref))
		}
		if newErr != nil {
			logrus.Infof("Translating %q to %q", err, newErr)
			return newErr
		}
	case xfer.DoNotRetry:
		return TranslatePullError(v.Err, ref)
	}

	return err
}

// continueOnError returns true if we should fallback to the next endpoint
// as a result of this error.
func continueOnError(err error) bool {
	switch v := err.(type) {
	case errcode.Errors:
		if len(v) == 0 {
			return true
		}
		return continueOnError(v[0])
	case ErrNoSupport:
		return continueOnError(v.Err)
	case errcode.Error:
		return shouldV2Fallback(v)
	case *client.UnexpectedHTTPResponseError:
		return true
	case ImageConfigPullError:
		return false
	case error:
		return !strings.Contains(err.Error(), strings.ToLower(syscall.ESRCH.Error()))
	}
	// let's be nice and fallback if the error is a completely
	// unexpected one.
	// If new errors have to be handled in some way, please
	// add them to the switch above.
	return true
}

// retryOnError wraps the error in xfer.DoNotRetry if we should not retry the
// operation after this error.
func retryOnError(err error) error {
	switch v := err.(type) {
	case errcode.Errors:
		if len(v) != 0 {
			return retryOnError(v[0])
		}
	case errcode.Error:
		switch v.Code {
		case errcode.ErrorCodeUnauthorized, errcode.ErrorCodeUnsupported, errcode.ErrorCodeDenied, errcode.ErrorCodeTooManyRequests, v2.ErrorCodeNameUnknown:
			return xfer.DoNotRetry{Err: err}
		}
	case *url.Error:
		switch v.Err {
		case auth.ErrNoBasicAuthCredentials, auth.ErrNoToken:
			return xfer.DoNotRetry{Err: v.Err}
		}
		return retryOnError(v.Err)
	case *client.UnexpectedHTTPResponseError:
		return xfer.DoNotRetry{Err: err}
	case error:
		if err == distribution.ErrBlobUnknown {
			return xfer.DoNotRetry{Err: err}
		}
		if strings.Contains(err.Error(), strings.ToLower(syscall.ENOSPC.Error())) {
			return xfer.DoNotRetry{Err: err}
		}
	}
	// let's be nice and fallback if the error is a completely
	// unexpected one.
	// If new errors have to be handled in some way, please
	// add them to the switch above.
	return err
}
