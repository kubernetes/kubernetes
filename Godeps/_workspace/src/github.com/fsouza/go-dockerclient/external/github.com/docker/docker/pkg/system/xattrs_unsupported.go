// +build !linux

package system

func Lgetxattr(path string, attr string) ([]byte, error) {
	return nil, ErrNotSupportedPlatform
}

func Lsetxattr(path string, attr string, data []byte, flags int) error {
	return ErrNotSupportedPlatform
}
