package selinux

import (
	"golang.org/x/sys/unix"
)

// lgetxattr returns a []byte slice containing the value of
// an extended attribute attr set for path.
func lgetxattr(path, attr string) ([]byte, error) {
	// Start with a 128 length byte array
	dest := make([]byte, 128)
	sz, errno := doLgetxattr(path, attr, dest)
	for errno == unix.ERANGE {
		// Buffer too small, use zero-sized buffer to get the actual size
		sz, errno = doLgetxattr(path, attr, []byte{})
		if errno != nil {
			return nil, errno
		}

		dest = make([]byte, sz)
		sz, errno = doLgetxattr(path, attr, dest)
	}
	if errno != nil {
		return nil, errno
	}

	return dest[:sz], nil
}

// doLgetxattr is a wrapper that retries on EINTR
func doLgetxattr(path, attr string, dest []byte) (int, error) {
	for {
		sz, err := unix.Lgetxattr(path, attr, dest)
		if err != unix.EINTR {
			return sz, err
		}
	}
}
