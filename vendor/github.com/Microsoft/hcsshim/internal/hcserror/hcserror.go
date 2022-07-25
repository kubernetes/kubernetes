package hcserror

import (
	"fmt"
	"syscall"
)

const ERROR_GEN_FAILURE = syscall.Errno(31)

type HcsError struct {
	title string
	rest  string
	Err   error
}

func (e *HcsError) Error() string {
	s := e.title
	if len(s) > 0 && s[len(s)-1] != ' ' {
		s += " "
	}
	s += fmt.Sprintf("failed in Win32: %s (0x%x)", e.Err, Win32FromError(e.Err))
	if e.rest != "" {
		if e.rest[0] != ' ' {
			s += " "
		}
		s += e.rest
	}
	return s
}

func New(err error, title, rest string) error {
	// Pass through DLL errors directly since they do not originate from HCS.
	if _, ok := err.(*syscall.DLLError); ok {
		return err
	}
	return &HcsError{title, rest, err}
}

func Win32FromError(err error) uint32 {
	if herr, ok := err.(*HcsError); ok {
		return Win32FromError(herr.Err)
	}
	if code, ok := err.(syscall.Errno); ok {
		return uint32(code)
	}
	return uint32(ERROR_GEN_FAILURE)
}
