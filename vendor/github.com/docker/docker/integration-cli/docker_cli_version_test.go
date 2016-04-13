package main

import (
	"strings"

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
		if strings.Count(out, k) != v {
			c.Errorf("%v expected %d instances found %d", k, v, strings.Count(out, k))
		}
	}
}
