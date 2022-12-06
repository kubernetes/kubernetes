package libcontainer

import (
	"strconv"

	"golang.org/x/sys/unix"
)

// mountError holds an error from a failed mount or unmount operation.
type mountError struct {
	op     string
	source string
	target string
	procfd string
	flags  uintptr
	data   string
	err    error
}

// Error provides a string error representation.
func (e *mountError) Error() string {
	out := e.op + " "

	if e.source != "" {
		out += e.source + ":" + e.target
	} else {
		out += e.target
	}
	if e.procfd != "" {
		out += " (via " + e.procfd + ")"
	}

	if e.flags != uintptr(0) {
		out += ", flags: 0x" + strconv.FormatUint(uint64(e.flags), 16)
	}
	if e.data != "" {
		out += ", data: " + e.data
	}

	out += ": " + e.err.Error()
	return out
}

// Unwrap returns the underlying error.
// This is a convention used by Go 1.13+ standard library.
func (e *mountError) Unwrap() error {
	return e.err
}

// mount is a simple unix.Mount wrapper. If procfd is not empty, it is used
// instead of target (and the target is only used to add context to an error).
func mount(source, target, procfd, fstype string, flags uintptr, data string) error {
	dst := target
	if procfd != "" {
		dst = procfd
	}
	if err := unix.Mount(source, dst, fstype, flags, data); err != nil {
		return &mountError{
			op:     "mount",
			source: source,
			target: target,
			procfd: procfd,
			flags:  flags,
			data:   data,
			err:    err,
		}
	}
	return nil
}

// unmount is a simple unix.Unmount wrapper.
func unmount(target string, flags int) error {
	err := unix.Unmount(target, flags)
	if err != nil {
		return &mountError{
			op:     "unmount",
			target: target,
			flags:  uintptr(flags),
			err:    err,
		}
	}
	return nil
}
