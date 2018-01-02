// +build !windows

package main

import (
	"io/ioutil"
	"os"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestLinksEtcHostsContentMatch(c *check.C) {
	// In a _unix file as using Unix specific files, and must be on the
	// same host as the daemon.
	testRequires(c, SameHostDaemon, NotUserNamespace)

	out, _ := dockerCmd(c, "run", "--net=host", "busybox", "cat", "/etc/hosts")
	hosts, err := ioutil.ReadFile("/etc/hosts")
	if os.IsNotExist(err) {
		c.Skip("/etc/hosts does not exist, skip this test")
	}

	c.Assert(out, checker.Equals, string(hosts), check.Commentf("container: %s\n\nhost:%s", out, hosts))

}
