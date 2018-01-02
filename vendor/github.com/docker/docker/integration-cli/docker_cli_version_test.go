package main

import (
	"strings"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

// ensure docker version works
func (s *DockerSuite) TestVersionEnsureSucceeds(c *check.C) {
	out, _ := dockerCmd(c, "version")
	stringsToCheck := map[string]int{
		"Client:":       1,
		"Server:":       1,
		" Version:":     2,
		" API version:": 2,
		" Go version:":  2,
		" Git commit:":  2,
		" OS/Arch:":     2,
		" Built:":       2,
	}

	for k, v := range stringsToCheck {
		c.Assert(strings.Count(out, k), checker.Equals, v, check.Commentf("The count of %v in %s does not match excepted", k, out))
	}
}

// ensure the Windows daemon return the correct platform string
func (s *DockerSuite) TestVersionPlatform_w(c *check.C) {
	testRequires(c, DaemonIsWindows)
	testVersionPlatform(c, "windows/amd64")
}

// ensure the Linux daemon return the correct platform string
func (s *DockerSuite) TestVersionPlatform_l(c *check.C) {
	testRequires(c, DaemonIsLinux)
	testVersionPlatform(c, "linux")
}

func testVersionPlatform(c *check.C, platform string) {
	out, _ := dockerCmd(c, "version")
	expected := "OS/Arch:      " + platform

	split := strings.Split(out, "\n")
	c.Assert(len(split) >= 14, checker.Equals, true, check.Commentf("got %d lines from version", len(split)))

	// Verify the second 'OS/Arch' matches the platform. Experimental has
	// more lines of output than 'regular'
	bFound := false
	for i := 14; i < len(split); i++ {
		if strings.Contains(split[i], expected) {
			bFound = true
			break
		}
	}
	c.Assert(bFound, checker.Equals, true, check.Commentf("Could not find server '%s' in '%s'", expected, out))
}
