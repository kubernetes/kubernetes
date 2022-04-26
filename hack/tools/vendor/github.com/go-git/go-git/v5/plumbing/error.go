package plumbing

import "fmt"

type PermanentError struct {
	Err error
}

func NewPermanentError(err error) *PermanentError {
	if err == nil {
		return nil
	}

	return &PermanentError{Err: err}
}

func (e *PermanentError) Error() string {
	return fmt.Sprintf("permanent client error: %s", e.Err.Error())
}

type UnexpectedError struct {
	Err error
}

func NewUnexpectedError(err error) *UnexpectedError {
	if err == nil {
		return nil
	}

	return &UnexpectedError{Err: err}
}

func (e *UnexpectedError) Error() string {
	return fmt.Sprintf("unexpected client error: %s", e.Err.Error())
}
