// +build !windows

package main

import (
	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestInfoSecurityOptions(c *check.C) {
	testRequires(c, SameHostDaemon, seccompEnabled, Apparmor, DaemonIsLinux)

	out, _ := dockerCmd(c, "info")
	c.Assert(out, checker.Contains, "Security Options:\n apparmor\n seccomp\n  Profile: default\n")
}
