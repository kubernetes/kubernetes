// +build linux,btrfs_noversion

package btrfs

// TODO(vbatts) remove this work-around once supported linux distros are on
// btrfs utilities of >= 3.16.1

func btrfsBuildVersion() string {
	return "-"
}

func btrfsLibVersion() int {
	return -1
}
