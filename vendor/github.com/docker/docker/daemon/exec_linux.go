package daemon

import (
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/caps"
	"github.com/docker/docker/daemon/exec"
	"github.com/docker/docker/libcontainerd"
	"github.com/opencontainers/runc/libcontainer/apparmor"
	"github.com/opencontainers/runtime-spec/specs-go"
)

func execSetPlatformOpt(c *container.Container, ec *exec.Config, p *libcontainerd.Process) error {
	if len(ec.User) > 0 {
		uid, gid, additionalGids, err := getUser(c, ec.User)
		if err != nil {
			return err
		}
		p.User = &specs.User{
			UID:            uid,
			GID:            gid,
			AdditionalGids: additionalGids,
		}
	}
	if ec.Privileged {
		p.Capabilities = caps.GetAllCapabilities()
	}
	if apparmor.IsEnabled() {
		var appArmorProfile string
		if c.AppArmorProfile != "" {
			appArmorProfile = c.AppArmorProfile
		} else if c.HostConfig.Privileged {
			appArmorProfile = "unconfined"
		} else {
			appArmorProfile = "docker-default"
		}

		if appArmorProfile == "docker-default" {
			// Unattended upgrades and other fun services can unload AppArmor
			// profiles inadvertently. Since we cannot store our profile in
			// /etc/apparmor.d, nor can we practically add other ways of
			// telling the system to keep our profile loaded, in order to make
			// sure that we keep the default profile enabled we dynamically
			// reload it if necessary.
			if err := ensureDefaultAppArmorProfile(); err != nil {
				return err
			}
		}
	}
	return nil
}
