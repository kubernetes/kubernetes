package daemon

import (
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/pkg/sysinfo"
)

// FillPlatformInfo fills the platform related info.
func (daemon *Daemon) FillPlatformInfo(v *types.Info, sysInfo *sysinfo.SysInfo) {
}
