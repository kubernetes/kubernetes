//+build !windows

package daemon

import (
	"github.com/docker/docker/container"
)

func (daemon *Daemon) saveApparmorConfig(container *container.Container) error {
	container.AppArmorProfile = "" //we don't care about the previous value.

	if !daemon.apparmorEnabled {
		return nil // if apparmor is disabled there is nothing to do here.
	}

	if err := parseSecurityOpt(container, container.HostConfig); err != nil {
		return validationError{err}
	}

	if !container.HostConfig.Privileged {
		if container.AppArmorProfile == "" {
			container.AppArmorProfile = defaultApparmorProfile
		}

	} else {
		container.AppArmorProfile = "unconfined"
	}
	return nil
}
