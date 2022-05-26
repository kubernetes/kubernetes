package errchkjson

type unsupported interface {
	unsupported()
}

var _ unsupported = unsupportedError{}

type unsupportedError struct {
	err error
}

func newUnsupportedError(err error) error {
	return unsupportedError{
		err: err,
	}
}

func (u unsupportedError) unsupported() {}

func (u unsupportedError) Error() string {
	return u.err.Error()
}
