// +build !windows

package main

import (
	"net"
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestPortHostBinding(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "-p", "9876:80", "busybox",
		"nc", "-l", "-p", "80")
	firstID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "port", firstID, "80")

	if !assertPortList(c, out, []string{"0.0.0.0:9876"}) {
		c.Error("Port list is not correct")
	}

	dockerCmd(c, "run", "--net=host", "busybox",
		"nc", "localhost", "9876")

	dockerCmd(c, "rm", "-f", firstID)

	if _, _, err := dockerCmdWithError(c, "run", "--net=host", "busybox",
		"nc", "localhost", "9876"); err == nil {
		c.Error("Port is still bound after the Container is removed")
	}
}

func (s *DockerSuite) TestPortExposeHostBinding(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "-P", "--expose", "80", "busybox",
		"nc", "-l", "-p", "80")
	firstID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "port", firstID, "80")

	_, exposedPort, err := net.SplitHostPort(out)

	if err != nil {
		c.Fatal(out, err)
	}

	dockerCmd(c, "run", "--net=host", "busybox",
		"nc", "localhost", strings.TrimSpace(exposedPort))

	dockerCmd(c, "rm", "-f", firstID)

	if _, _, err = dockerCmdWithError(c, "run", "--net=host", "busybox",
		"nc", "localhost", strings.TrimSpace(exposedPort)); err == nil {
		c.Error("Port is still bound after the Container is removed")
	}
}
