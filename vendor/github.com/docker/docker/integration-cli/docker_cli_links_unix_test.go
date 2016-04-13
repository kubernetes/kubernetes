// +build !windows

package main

import (
	"io/ioutil"
	"os"
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestLinksEtcHostsRegularFile(c *check.C) {
	out, _ := dockerCmd(c, "run", "--net=host", "busybox", "ls", "-la", "/etc/hosts")
	if !strings.HasPrefix(out, "-") {
		c.Errorf("/etc/hosts should be a regular file")
	}
}

func (s *DockerSuite) TestLinksEtcHostsContentMatch(c *check.C) {
	testRequires(c, SameHostDaemon)

	out, _ := dockerCmd(c, "run", "--net=host", "busybox", "cat", "/etc/hosts")
	hosts, err := ioutil.ReadFile("/etc/hosts")
	if os.IsNotExist(err) {
		c.Skip("/etc/hosts does not exist, skip this test")
	}

	if out != string(hosts) {
		c.Errorf("container: %s\n\nhost:%s", out, hosts)
	}

}

func (s *DockerSuite) TestLinksNetworkHostContainer(c *check.C) {
	dockerCmd(c, "run", "-d", "--net", "host", "--name", "host_container", "busybox", "top")
	out, _, err := dockerCmdWithError(c, "run", "--name", "should_fail", "--link", "host_container:tester", "busybox", "true")
	if err == nil || !strings.Contains(out, "--net=host can't be used with links. This would result in undefined behavior") {
		c.Fatalf("Running container linking to a container with --net host should have failed: %s", out)
	}

}
