package netns

import (
	"fmt"

	"golang.org/x/sys/unix"
)

// NsHandle is a handle to a network namespace. It can be cast directly
// to an int and used as a file descriptor.
type NsHandle int

// Equal determines if two network handles refer to the same network
// namespace. This is done by comparing the device and inode that the
// file descriptors point to.
func (ns NsHandle) Equal(other NsHandle) bool {
	if ns == other {
		return true
	}
	var s1, s2 unix.Stat_t
	if err := unix.Fstat(int(ns), &s1); err != nil {
		return false
	}
	if err := unix.Fstat(int(other), &s2); err != nil {
		return false
	}
	return (s1.Dev == s2.Dev) && (s1.Ino == s2.Ino)
}

// String shows the file descriptor number and its dev and inode.
func (ns NsHandle) String() string {
	if ns == -1 {
		return "NS(None)"
	}
	var s unix.Stat_t
	if err := unix.Fstat(int(ns), &s); err != nil {
		return fmt.Sprintf("NS(%d: unknown)", ns)
	}
	return fmt.Sprintf("NS(%d: %d, %d)", ns, s.Dev, s.Ino)
}

// UniqueId returns a string which uniquely identifies the namespace
// associated with the network handle.
func (ns NsHandle) UniqueId() string {
	if ns == -1 {
		return "NS(none)"
	}
	var s unix.Stat_t
	if err := unix.Fstat(int(ns), &s); err != nil {
		return "NS(unknown)"
	}
	return fmt.Sprintf("NS(%d:%d)", s.Dev, s.Ino)
}

// IsOpen returns true if Close() has not been called.
func (ns NsHandle) IsOpen() bool {
	return ns != -1
}

// Close closes the NsHandle and resets its file descriptor to -1.
// It is not safe to use an NsHandle after Close() is called.
func (ns *NsHandle) Close() error {
	if err := unix.Close(int(*ns)); err != nil {
		return err
	}
	*ns = -1
	return nil
}

// None gets an empty (closed) NsHandle.
func None() NsHandle {
	return NsHandle(-1)
}
