package errdefs

import "context"

type errNotFound struct{ error }

func (errNotFound) NotFound() {}

func (e errNotFound) Cause() error {
	return e.error
}

// NotFound is a helper to create an error of the class with the same name from any error type
func NotFound(err error) error {
	if err == nil {
		return nil
	}
	return errNotFound{err}
}

type errInvalidParameter struct{ error }

func (errInvalidParameter) InvalidParameter() {}

func (e errInvalidParameter) Cause() error {
	return e.error
}

// InvalidParameter is a helper to create an error of the class with the same name from any error type
func InvalidParameter(err error) error {
	if err == nil {
		return nil
	}
	return errInvalidParameter{err}
}

type errConflict struct{ error }

func (errConflict) Conflict() {}

func (e errConflict) Cause() error {
	return e.error
}

// Conflict is a helper to create an error of the class with the same name from any error type
func Conflict(err error) error {
	if err == nil {
		return nil
	}
	return errConflict{err}
}

type errUnauthorized struct{ error }

func (errUnauthorized) Unauthorized() {}

func (e errUnauthorized) Cause() error {
	return e.error
}

// Unauthorized is a helper to create an error of the class with the same name from any error type
func Unauthorized(err error) error {
	if err == nil {
		return nil
	}
	return errUnauthorized{err}
}

type errUnavailable struct{ error }

func (errUnavailable) Unavailable() {}

func (e errUnavailable) Cause() error {
	return e.error
}

// Unavailable is a helper to create an error of the class with the same name from any error type
func Unavailable(err error) error {
	return errUnavailable{err}
}

type errForbidden struct{ error }

func (errForbidden) Forbidden() {}

func (e errForbidden) Cause() error {
	return e.error
}

// Forbidden is a helper to create an error of the class with the same name from any error type
func Forbidden(err error) error {
	if err == nil {
		return nil
	}
	return errForbidden{err}
}

type errSystem struct{ error }

func (errSystem) System() {}

func (e errSystem) Cause() error {
	return e.error
}

// System is a helper to create an error of the class with the same name from any error type
func System(err error) error {
	if err == nil {
		return nil
	}
	return errSystem{err}
}

type errNotModified struct{ error }

func (errNotModified) NotModified() {}

func (e errNotModified) Cause() error {
	return e.error
}

// NotModified is a helper to create an error of the class with the same name from any error type
func NotModified(err error) error {
	if err == nil {
		return nil
	}
	return errNotModified{err}
}

type errAlreadyExists struct{ error }

func (errAlreadyExists) AlreadyExists() {}

func (e errAlreadyExists) Cause() error {
	return e.error
}

// AlreadyExists is a helper to create an error of the class with the same name from any error type
func AlreadyExists(err error) error {
	if err == nil {
		return nil
	}
	return errAlreadyExists{err}
}

type errNotImplemented struct{ error }

func (errNotImplemented) NotImplemented() {}

func (e errNotImplemented) Cause() error {
	return e.error
}

// NotImplemented is a helper to create an error of the class with the same name from any error type
func NotImplemented(err error) error {
	if err == nil {
		return nil
	}
	return errNotImplemented{err}
}

type errUnknown struct{ error }

func (errUnknown) Unknown() {}

func (e errUnknown) Cause() error {
	return e.error
}

// Unknown is a helper to create an error of the class with the same name from any error type
func Unknown(err error) error {
	if err == nil {
		return nil
	}
	return errUnknown{err}
}

type errCancelled struct{ error }

func (errCancelled) Cancelled() {}

func (e errCancelled) Cause() error {
	return e.error
}

// Cancelled is a helper to create an error of the class with the same name from any error type
func Cancelled(err error) error {
	if err == nil {
		return nil
	}
	return errCancelled{err}
}

type errDeadline struct{ error }

func (errDeadline) DeadlineExceeded() {}

func (e errDeadline) Cause() error {
	return e.error
}

// Deadline is a helper to create an error of the class with the same name from any error type
func Deadline(err error) error {
	if err == nil {
		return nil
	}
	return errDeadline{err}
}

type errDataLoss struct{ error }

func (errDataLoss) DataLoss() {}

func (e errDataLoss) Cause() error {
	return e.error
}

// DataLoss is a helper to create an error of the class with the same name from any error type
func DataLoss(err error) error {
	if err == nil {
		return nil
	}
	return errDataLoss{err}
}

// FromContext returns the error class from the passed in context
func FromContext(ctx context.Context) error {
	e := ctx.Err()
	if e == nil {
		return nil
	}

	if e == context.Canceled {
		return Cancelled(e)
	}
	if e == context.DeadlineExceeded {
		return Deadline(e)
	}
	return Unknown(e)
}
