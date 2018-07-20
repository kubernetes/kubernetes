// +build linux,!seccomp

package daemon

import (
	"fmt"

	"github.com/docker/docker/container"
	"github.com/opencontainers/runtime-spec/specs-go"
)

var supportsSeccomp = false

func setSeccomp(daemon *Daemon, rs *specs.Spec, c *container.Container) error {
	if c.SeccompProfile != "" && c.SeccompProfile != "unconfined" {
		return fmt.Errorf("seccomp profiles are not supported on this daemon, you cannot specify a custom seccomp profile")
	}
	return nil
}
