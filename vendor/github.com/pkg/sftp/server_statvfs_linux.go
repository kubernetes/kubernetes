// +build !gccgo,linux

package sftp

import (
	"syscall"
)

func statvfsFromStatfst(stat *syscall.Statfs_t) (*StatVFS, error) {
	return &StatVFS{
		Bsize:   uint64(stat.Bsize),
		Frsize:  uint64(stat.Frsize),
		Blocks:  stat.Blocks,
		Bfree:   stat.Bfree,
		Bavail:  stat.Bavail,
		Files:   stat.Files,
		Ffree:   stat.Ffree,
		Favail:  stat.Ffree,                                                            // not sure how to calculate Favail
		Fsid:    uint64(uint64(stat.Fsid.X__val[1])<<32 | uint64(stat.Fsid.X__val[0])), // endianness?
		Flag:    uint64(stat.Flags),                                                    // assuming POSIX?
		Namemax: uint64(stat.Namelen),
	}, nil
}
