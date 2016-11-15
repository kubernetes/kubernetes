package sftp

import (
	"syscall"
)

func statvfsFromStatfst(stat *syscall.Statfs_t) (*StatVFS, error) {
	return &StatVFS{
		Bsize:   uint64(stat.Bsize),
		Frsize:  uint64(stat.Bsize), // fragment size is a linux thing; use block size here
		Blocks:  stat.Blocks,
		Bfree:   stat.Bfree,
		Bavail:  stat.Bavail,
		Files:   stat.Files,
		Ffree:   stat.Ffree,
		Favail:  stat.Ffree,                                                      // not sure how to calculate Favail
		Fsid:    uint64(uint64(stat.Fsid.Val[1])<<32 | uint64(stat.Fsid.Val[0])), // endianness?
		Flag:    uint64(stat.Flags),                                              // assuming POSIX?
		Namemax: 1024,                                                            // man 2 statfs shows: #define MAXPATHLEN      1024
	}, nil
}
