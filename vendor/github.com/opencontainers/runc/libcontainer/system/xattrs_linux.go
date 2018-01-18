package system

import "golang.org/x/sys/unix"

// Returns a []byte slice if the xattr is set and nil otherwise
// Requires path and its attribute as arguments
func Lgetxattr(path string, attr string) ([]byte, error) {
	var sz int
	// Start with a 128 length byte array
	dest := make([]byte, 128)
	sz, errno := unix.Lgetxattr(path, attr, dest)

	switch {
	case errno == unix.ENODATA:
		return nil, errno
	case errno == unix.ENOTSUP:
		return nil, errno
	case errno == unix.ERANGE:
		// 128 byte array might just not be good enough,
		// A dummy buffer is used to get the real size
		// of the xattrs on disk
		sz, errno = unix.Lgetxattr(path, attr, []byte{})
		if errno != nil {
			return nil, errno
		}
		dest = make([]byte, sz)
		sz, errno = unix.Lgetxattr(path, attr, dest)
		if errno != nil {
			return nil, errno
		}
	case errno != nil:
		return nil, errno
	}
	return dest[:sz], nil
}
