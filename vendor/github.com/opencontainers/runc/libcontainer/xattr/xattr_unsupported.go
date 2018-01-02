// +build !linux

package xattr

func Listxattr(path string) ([]string, error) {
	return nil, ErrNotSupportedPlatform
}

func Getxattr(path, attr string) (string, error) {
	return "", ErrNotSupportedPlatform
}

func Setxattr(path, xattr, value string) error {
	return ErrNotSupportedPlatform
}
