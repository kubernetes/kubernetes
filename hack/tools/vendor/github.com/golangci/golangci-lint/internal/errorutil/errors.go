package errorutil

import (
	"fmt"
)

// PanicError can be used to not print stacktrace twice
type PanicError struct {
	recovered interface{}
	stack     []byte
}

func NewPanicError(recovered interface{}, stack []byte) *PanicError {
	return &PanicError{recovered: recovered, stack: stack}
}

func (e PanicError) Error() string {
	return fmt.Sprint(e.recovered)
}

func (e PanicError) Stack() []byte {
	return e.stack
}
