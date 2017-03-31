// +build linux

// Package osutils provide access to the Get Child and Set Child prctl
// flags.
// See http://man7.org/linux/man-pages/man2/prctl.2.html
package sys

import (
	"syscall"
	"unsafe"
)

// PR_SET_CHILD_SUBREAPER allows setting the child subreaper.
// If arg2 is nonzero, set the "child subreaper" attribute of the
// calling process; if arg2 is zero, unset the attribute.  When a
// process is marked as a child subreaper, all of the children
// that it creates, and their descendants, will be marked as
// having a subreaper.  In effect, a subreaper fulfills the role
// of init(1) for its descendant processes.  Upon termination of
// a process that is orphaned (i.e., its immediate parent has
// already terminated) and marked as having a subreaper, the
// nearest still living ancestor subreaper will receive a SIGCHLD
// signal and be able to wait(2) on the process to discover its
// termination status.
const prSetChildSubreaper = 36

// PR_GET_CHILD_SUBREAPER allows retrieving the current child
// subreaper.
// Returns the "child subreaper" setting of the caller, in the
// location pointed to by (int *) arg2.
const prGetChildSubreaper = 37

// GetSubreaper returns the subreaper setting for the calling process
func GetSubreaper() (int, error) {
	var i uintptr
	if _, _, err := syscall.RawSyscall(syscall.SYS_PRCTL, prGetChildSubreaper, uintptr(unsafe.Pointer(&i)), 0); err != 0 {
		return -1, err
	}
	return int(i), nil
}

// SetSubreaper sets the value i as the subreaper setting for the calling process
func SetSubreaper(i int) error {
	if _, _, err := syscall.RawSyscall(syscall.SYS_PRCTL, prSetChildSubreaper, uintptr(i), 0); err != 0 {
		return err
	}
	return nil
}
