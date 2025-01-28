//go:build windows

package hns

import (
	"errors"
	"fmt"
	"syscall"

	"golang.org/x/sys/windows"
)

var (
	// ErrElementNotFound is an error encountered when the object being referenced does not exist
	ErrElementNotFound = syscall.Errno(0x490)

	// ErrInvalidData is an error encountered when the request being sent to hcs is invalid/unsupported
	// decimal -2147024883 / hex 0x8007000d
	ErrInvalidData = syscall.Errno(0xd)
)

type HnsError struct {
	title string
	rest  string
	Err   error
}

func (e *HnsError) Error() string {
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

func NewHnsError(err error, title, rest string) error {
	// Pass through DLL errors directly since they do not originate from HCS.
	var e *windows.DLLError
	if errors.As(err, &e) {
		return err
	}
	return &HnsError{title, rest, err}
}

func Win32FromError(err error) uint32 {
	var herr *HnsError
	if errors.As(err, &herr) {
		return Win32FromError(herr.Err)
	}
	var code windows.Errno
	if errors.As(err, &code) {
		return uint32(code)
	}
	return uint32(windows.ERROR_GEN_FAILURE)
}
