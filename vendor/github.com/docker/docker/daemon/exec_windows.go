package daemon

import (
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/exec"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

func (daemon *Daemon) execSetPlatformOpt(c *container.Container, ec *exec.Config, p *specs.Process) error {
	// Process arguments need to be escaped before sending to OCI.
	if c.OS == "windows" {
		p.Args = escapeArgs(p.Args)
		p.User.Username = ec.User
	}
	return nil
}
