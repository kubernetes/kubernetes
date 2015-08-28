// Package netns allows ultra-simple network namespace handling. NsHandles
// can be retrieved and set. Note that the current namespace is thread
// local so actions that set and reset namespaces should use LockOSThread
// to make sure the namespace doesn't change due to a goroutine switch.
// It is best to close NsHandles when you are done with them. This can be
// accomplished via a `defer ns.Close()` on the handle. Changing namespaces
// requires elevated privileges, so in most cases this code needs to be run
// as root.
package netns

import (
	"fmt"
	"syscall"
)

// NsHandle is a handle to a network namespace. It can be cast directly
// to an int and used as a file descriptor.
type NsHandle int

// Equal determines if two network handles refer to the same network
// namespace. This is done by comparing the device and inode that the
// file descripors point to.
func (ns NsHandle) Equal(other NsHandle) bool {
	if ns == other {
		return true
	}
	var s1, s2 syscall.Stat_t
	if err := syscall.Fstat(int(ns), &s1); err != nil {
		return false
	}
	if err := syscall.Fstat(int(other), &s2); err != nil {
		return false
	}
	return (s1.Dev == s2.Dev) && (s1.Ino == s2.Ino)
}

// String shows the file descriptor number and its dev and inode.
func (ns NsHandle) String() string {
	var s syscall.Stat_t
	if ns == -1 {
		return "NS(None)"
	}
	if err := syscall.Fstat(int(ns), &s); err != nil {
		return fmt.Sprintf("NS(%d: unknown)", ns)
	}
	return fmt.Sprintf("NS(%d: %d, %d)", ns, s.Dev, s.Ino)
}

// IsOpen returns true if Close() has not been called.
func (ns NsHandle) IsOpen() bool {
	return ns != -1
}

// Close closes the NsHandle and resets its file descriptor to -1.
// It is not safe to use an NsHandle after Close() is called.
func (ns *NsHandle) Close() error {
	if err := syscall.Close(int(*ns)); err != nil {
		return err
	}
	(*ns) = -1
	return nil
}

// Get an empty (closed) NsHandle
func None() NsHandle {
	return NsHandle(-1)
}
