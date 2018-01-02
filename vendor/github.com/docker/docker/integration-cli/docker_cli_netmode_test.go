package main

import (
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/runconfig"
	"github.com/go-check/check"
)

// GH14530. Validates combinations of --net= with other options

// stringCheckPS is how the output of PS starts in order to validate that
// the command executed in a container did really run PS correctly.
const stringCheckPS = "PID   USER"

// DockerCmdWithFail executes a docker command that is supposed to fail and returns
// the output, the exit code. If the command returns a Nil error, it will fail and
// stop the tests.
func dockerCmdWithFail(c *check.C, args ...string) (string, int) {
	out, status, err := dockerCmdWithError(args...)
	c.Assert(err, check.NotNil, check.Commentf("%v", out))
	return out, status
}

func (s *DockerSuite) TestNetHostnameWithNetHost(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace)

	out, _ := dockerCmd(c, "run", "--net=host", "busybox", "ps")
	c.Assert(out, checker.Contains, stringCheckPS)
}

func (s *DockerSuite) TestNetHostname(c *check.C) {
	testRequires(c, DaemonIsLinux)

	out, _ := dockerCmd(c, "run", "-h=name", "busybox", "ps")
	c.Assert(out, checker.Contains, stringCheckPS)

	out, _ = dockerCmd(c, "run", "-h=name", "--net=bridge", "busybox", "ps")
	c.Assert(out, checker.Contains, stringCheckPS)

	out, _ = dockerCmd(c, "run", "-h=name", "--net=none", "busybox", "ps")
	c.Assert(out, checker.Contains, stringCheckPS)

	out, _ = dockerCmdWithFail(c, "run", "-h=name", "--net=container:other", "busybox", "ps")
	c.Assert(out, checker.Contains, runconfig.ErrConflictNetworkHostname.Error())

	out, _ = dockerCmdWithFail(c, "run", "--net=container", "busybox", "ps")
	c.Assert(out, checker.Contains, "Invalid network mode: invalid container format container:<name|id>")

	out, _ = dockerCmdWithFail(c, "run", "--net=weird", "busybox", "ps")
	c.Assert(out, checker.Contains, "network weird not found")
}

func (s *DockerSuite) TestConflictContainerNetworkAndLinks(c *check.C) {
	testRequires(c, DaemonIsLinux)

	out, _ := dockerCmdWithFail(c, "run", "--net=container:other", "--link=zip:zap", "busybox", "ps")
	c.Assert(out, checker.Contains, runconfig.ErrConflictContainerNetworkAndLinks.Error())
}

func (s *DockerSuite) TestConflictContainerNetworkHostAndLinks(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace)

	out, _ := dockerCmdWithFail(c, "run", "--net=host", "--link=zip:zap", "busybox", "ps")
	c.Assert(out, checker.Contains, runconfig.ErrConflictHostNetworkAndLinks.Error())
}

func (s *DockerSuite) TestConflictNetworkModeNetHostAndOptions(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace)

	out, _ := dockerCmdWithFail(c, "run", "--net=host", "--mac-address=92:d0:c6:0a:29:33", "busybox", "ps")
	c.Assert(out, checker.Contains, runconfig.ErrConflictContainerNetworkAndMac.Error())
}

func (s *DockerSuite) TestConflictNetworkModeAndOptions(c *check.C) {
	testRequires(c, DaemonIsLinux)

	out, _ := dockerCmdWithFail(c, "run", "--net=container:other", "--dns=8.8.8.8", "busybox", "ps")
	c.Assert(out, checker.Contains, runconfig.ErrConflictNetworkAndDNS.Error())

	out, _ = dockerCmdWithFail(c, "run", "--net=container:other", "--add-host=name:8.8.8.8", "busybox", "ps")
	c.Assert(out, checker.Contains, runconfig.ErrConflictNetworkHosts.Error())

	out, _ = dockerCmdWithFail(c, "run", "--net=container:other", "--mac-address=92:d0:c6:0a:29:33", "busybox", "ps")
	c.Assert(out, checker.Contains, runconfig.ErrConflictContainerNetworkAndMac.Error())

	out, _ = dockerCmdWithFail(c, "run", "--net=container:other", "-P", "busybox", "ps")
	c.Assert(out, checker.Contains, runconfig.ErrConflictNetworkPublishPorts.Error())

	out, _ = dockerCmdWithFail(c, "run", "--net=container:other", "-p", "8080", "busybox", "ps")
	c.Assert(out, checker.Contains, runconfig.ErrConflictNetworkPublishPorts.Error())

	out, _ = dockerCmdWithFail(c, "run", "--net=container:other", "--expose", "8000-9000", "busybox", "ps")
	c.Assert(out, checker.Contains, runconfig.ErrConflictNetworkExposePorts.Error())
}
