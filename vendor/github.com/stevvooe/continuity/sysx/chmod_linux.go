package sysx

import "syscall"

const (
	// AtSymlinkNoFollow defined from AT_SYMLINK_NOFOLLOW in /usr/include/linux/fcntl.h
	AtSymlinkNofollow = 0x100
)

func Fchmodat(dirfd int, path string, mode uint32, flags int) error {
	return syscall.Fchmodat(dirfd, path, mode, flags)
}
