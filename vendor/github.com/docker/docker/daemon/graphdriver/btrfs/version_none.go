// +build linux,btrfs_noversion

package btrfs

// TODO(vbatts) remove this work-around once supported linux distros are on
// btrfs utililties of >= 3.16.1

func BtrfsBuildVersion() string {
	return "-"
}

func BtrfsLibVersion() int {
	return -1
}
