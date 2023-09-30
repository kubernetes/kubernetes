package fscommon

import (
	libcontainerfscommon "github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
)

var (
	GetCgroupParamUint   = libcontainerfscommon.GetCgroupParamUint
	GetCgroupParamString = libcontainerfscommon.GetCgroupParamString
)
