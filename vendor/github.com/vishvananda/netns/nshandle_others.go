//go:build !linux
// +build !linux

package netns

// NsHandle is a handle to a network namespace. It can only be used on Linux,
// but provides stub methods on other platforms.
type NsHandle int

// Equal determines if two network handles refer to the same network
// namespace. It is only implemented on Linux.
func (ns NsHandle) Equal(_ NsHandle) bool {
	return false
}

// String shows the file descriptor number and its dev and inode.
// It is only implemented on Linux, and returns "NS(none)" on other
// platforms.
func (ns NsHandle) String() string {
	return "NS(None)"
}

// UniqueId returns a string which uniquely identifies the namespace
// associated with the network handle. It is only implemented on Linux,
// and returns "NS(none)" on other platforms.
func (ns NsHandle) UniqueId() string {
	return "NS(none)"
}

// IsOpen returns true if Close() has not been called. It is only implemented
// on Linux and always returns false on other platforms.
func (ns NsHandle) IsOpen() bool {
	return false
}

// Close closes the NsHandle and resets its file descriptor to -1.
// It is only implemented on Linux.
func (ns *NsHandle) Close() error {
	return nil
}

// None gets an empty (closed) NsHandle.
func None() NsHandle {
	return NsHandle(-1)
}
