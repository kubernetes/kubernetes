// +build !windows

package archive

import (
	"errors"
	"syscall"

	"github.com/docker/docker/vendor/src/code.google.com/p/go/src/pkg/archive/tar"
)

func setHeaderForSpecialDevice(hdr *tar.Header, ta *tarAppender, name string, stat interface{}) (nlink uint32, inode uint64, err error) {
	s, ok := stat.(*syscall.Stat_t)

	if !ok {
		err = errors.New("cannot convert stat value to syscall.Stat_t")
		return
	}

	nlink = uint32(s.Nlink)
	inode = uint64(s.Ino)

	// Currently go does not fil in the major/minors
	if s.Mode&syscall.S_IFBLK == syscall.S_IFBLK ||
		s.Mode&syscall.S_IFCHR == syscall.S_IFCHR {
		hdr.Devmajor = int64(major(uint64(s.Rdev)))
		hdr.Devminor = int64(minor(uint64(s.Rdev)))
	}

	return
}

func major(device uint64) uint64 {
	return (device >> 8) & 0xfff
}

func minor(device uint64) uint64 {
	return (device & 0xff) | ((device >> 12) & 0xfff00)
}
