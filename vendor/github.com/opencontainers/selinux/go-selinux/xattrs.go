// +build selinux,linux

package selinux

import (
	"golang.org/x/sys/unix"
)

// Returns a []byte slice if the xattr is set and nil otherwise
// Requires path and its attribute as arguments
func lgetxattr(path string, attr string) ([]byte, error) {
	// Start with a 128 length byte array
	dest := make([]byte, 128)
	sz, errno := unix.Lgetxattr(path, attr, dest)
	for errno == unix.ERANGE {
		// Buffer too small, use zero-sized buffer to get the actual size
		sz, errno = unix.Lgetxattr(path, attr, []byte{})
		if errno != nil {
			return nil, errno
		}

		dest = make([]byte, sz)
		sz, errno = unix.Lgetxattr(path, attr, dest)
	}
	if errno != nil {
		return nil, errno
	}

	return dest[:sz], nil
}
