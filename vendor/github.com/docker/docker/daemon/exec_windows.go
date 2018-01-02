package daemon

import (
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/exec"
	"github.com/docker/docker/libcontainerd"
)

func execSetPlatformOpt(c *container.Container, ec *exec.Config, p *libcontainerd.Process) error {
	// Process arguments need to be escaped before sending to OCI.
	if c.Platform == "windows" {
		p.Args = escapeArgs(p.Args)
		p.User.Username = ec.User
	}
	return nil
}
