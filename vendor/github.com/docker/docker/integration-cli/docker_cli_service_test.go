// +build experimental

package main

import (
	"fmt"
	"strings"

	"github.com/go-check/check"
)

func assertSrvIsAvailable(c *check.C, sname, name string) {
	if !isSrvPresent(c, sname, name) {
		c.Fatalf("Service %s on network %s not found in service ls o/p", sname, name)
	}
}

func assertSrvNotAvailable(c *check.C, sname, name string) {
	if isSrvPresent(c, sname, name) {
		c.Fatalf("Found service %s on network %s in service ls o/p", sname, name)
	}
}

func isSrvPresent(c *check.C, sname, name string) bool {
	out, _, _ := dockerCmdWithStdoutStderr(c, "service", "ls")
	lines := strings.Split(out, "\n")
	for i := 1; i < len(lines)-1; i++ {
		if strings.Contains(lines[i], sname) && strings.Contains(lines[i], name) {
			return true
		}
	}
	return false
}

func isCntPresent(c *check.C, cname, sname, name string) bool {
	out, _, _ := dockerCmdWithStdoutStderr(c, "service", "ls", "--no-trunc")
	lines := strings.Split(out, "\n")
	for i := 1; i < len(lines)-1; i++ {
		fmt.Println(lines)
		if strings.Contains(lines[i], name) && strings.Contains(lines[i], sname) && strings.Contains(lines[i], cname) {
			return true
		}
	}
	return false
}

func (s *DockerSuite) TestDockerServiceCreateDelete(c *check.C) {
	dockerCmdWithStdoutStderr(c, "network", "create", "test")
	assertNwIsAvailable(c, "test")

	dockerCmdWithStdoutStderr(c, "service", "publish", "s1.test")
	assertSrvIsAvailable(c, "s1", "test")

	dockerCmdWithStdoutStderr(c, "service", "unpublish", "s1.test")
	assertSrvNotAvailable(c, "s1", "test")

	dockerCmdWithStdoutStderr(c, "network", "rm", "test")
	assertNwNotAvailable(c, "test")
}

func (s *DockerSuite) TestDockerPublishServiceFlag(c *check.C) {
	// Run saying the container is the backend for the specified service on the specified network
	out, _ := dockerCmd(c, "run", "-d", "--expose=23", "--publish-service", "telnet.production", "busybox", "top")
	cid := strings.TrimSpace(out)

	// Verify container is attached in service ps o/p
	assertSrvIsAvailable(c, "telnet", "production")
	dockerCmd(c, "rm", "-f", cid)
}
