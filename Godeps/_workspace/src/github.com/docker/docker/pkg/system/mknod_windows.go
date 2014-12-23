// +build windows

package system

func Mknod(path string, mode uint32, dev int) error {
	// should not be called on cli code path
	return ErrNotSupportedPlatform
}

func Mkdev(major int64, minor int64) uint32 {
	panic("Mkdev not implemented on windows, should not be called on cli code")
}
