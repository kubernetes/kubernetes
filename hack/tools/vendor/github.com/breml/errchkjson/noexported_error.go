package errchkjson

type noexported interface {
	noexported()
}

var _ noexported = noexportedError{}

type noexportedError struct {
	err error
}

func newNoexportedError(err error) error {
	return noexportedError{
		err: err,
	}
}

func (u noexportedError) noexported() {}

func (u noexportedError) Error() string {
	return u.err.Error()
}
