package internal

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/cilium/ebpf/internal/unix"
	"github.com/pkg/errors"
)

// ErrorWithLog returns an error that includes logs from the
// kernel verifier.
//
// logErr should be the error returned by the syscall that generated
// the log. It is used to check for truncation of the output.
func ErrorWithLog(err error, log []byte, logErr error) error {
	logStr := strings.Trim(CString(log), "\t\r\n ")
	if errors.Cause(logErr) == unix.ENOSPC {
		logStr += " (truncated...)"
	}

	return &loadError{err, logStr}
}

type loadError struct {
	cause error
	log   string
}

func (le *loadError) Error() string {
	if le.log == "" {
		return le.cause.Error()
	}

	return fmt.Sprintf("%s: %s", le.cause, le.log)
}

func (le *loadError) Cause() error {
	return le.cause
}

// CString turns a NUL / zero terminated byte buffer into a string.
func CString(in []byte) string {
	inLen := bytes.IndexByte(in, 0)
	if inLen == -1 {
		return ""
	}
	return string(in[:inLen])
}
