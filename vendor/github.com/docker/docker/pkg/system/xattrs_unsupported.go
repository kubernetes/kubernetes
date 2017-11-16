// +build !linux

package system

// Lgetxattr is not supported on platforms other than linux.
func Lgetxattr(path string, attr string) ([]byte, error) {
	return nil, ErrNotSupportedPlatform
}

// Lsetxattr is not supported on platforms other than linux.
func Lsetxattr(path string, attr string, data []byte, flags int) error {
	return ErrNotSupportedPlatform
}
