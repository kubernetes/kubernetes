//go:build freebsd || darwin
// +build freebsd darwin

package mountinfo

import "golang.org/x/sys/unix"

func getMountinfo(entry *unix.Statfs_t) *Info {
	return &Info{
		Mountpoint: unix.ByteSliceToString(entry.Mntonname[:]),
		FSType:     unix.ByteSliceToString(entry.Fstypename[:]),
		Source:     unix.ByteSliceToString(entry.Mntfromname[:]),
	}
}
