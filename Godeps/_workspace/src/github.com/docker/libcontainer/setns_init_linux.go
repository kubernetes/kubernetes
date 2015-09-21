// +build linux

package libcontainer

import (
	"os"

	"github.com/docker/libcontainer/apparmor"
	"github.com/docker/libcontainer/label"
	"github.com/docker/libcontainer/system"
)

// linuxSetnsInit performs the container's initialization for running a new process
// inside an existing container.
type linuxSetnsInit struct {
	config *initConfig
}

func (l *linuxSetnsInit) Init() error {
	if err := setupRlimits(l.config.Config); err != nil {
		return err
	}
	if err := finalizeNamespace(l.config); err != nil {
		return err
	}
	if err := apparmor.ApplyProfile(l.config.Config.AppArmorProfile); err != nil {
		return err
	}
	if l.config.Config.ProcessLabel != "" {
		if err := label.SetProcessLabel(l.config.Config.ProcessLabel); err != nil {
			return err
		}
	}
	return system.Execv(l.config.Args[0], l.config.Args[0:], os.Environ())
}
