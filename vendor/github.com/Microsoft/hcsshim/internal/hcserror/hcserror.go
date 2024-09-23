//go:build windows

package hcserror

import (
	"errors"
	"fmt"

	"golang.org/x/sys/windows"
)

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
	var e *windows.DLLError
	if errors.As(err, &e) {
		return err
	}
	return &HcsError{title, rest, err}
}

func Win32FromError(err error) uint32 {
	var herr *HcsError
	if errors.As(err, &herr) {
		return Win32FromError(herr.Err)
	}
	var code windows.Errno
	if errors.As(err, &code) {
		return uint32(code)
	}
	return uint32(windows.ERROR_GEN_FAILURE)
}
