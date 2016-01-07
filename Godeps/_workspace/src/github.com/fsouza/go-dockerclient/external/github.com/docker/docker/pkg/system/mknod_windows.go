// +build windows

package system

func Mknod(path string, mode uint32, dev int) error {
	return ErrNotSupportedPlatform
}

func Mkdev(major int64, minor int64) uint32 {
	panic("Mkdev not implemented on Windows.")
}
