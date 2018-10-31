package eventstreamapi

import "fmt"

type messageError struct {
	code string
	msg  string
}

func (e messageError) Code() string {
	return e.code
}

func (e messageError) Message() string {
	return e.msg
}

func (e messageError) Error() string {
	return fmt.Sprintf("%s: %s", e.code, e.msg)
}

func (e messageError) OrigErr() error {
	return nil
}
