package testing

type WantStatusError struct {
	Err error
}

func (e WantStatusError) Error() string {
	return e.Err.Error()
}

func (e WantStatusError) Is(target error) bool {
	return e.Error() == target.Error()
}
