// +build linux

package blkio

import (
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fs"
)

func init() {
	blkioSubsystem = &fs.BlkioGroup{}
	// blkioSubsystem = &BlkioGroupSubsystem{}
	// FindCgroupMountpointDir easy for test
	FindCgroupMountpointDir = cgroups.FindCgroupMountpointDir
}
