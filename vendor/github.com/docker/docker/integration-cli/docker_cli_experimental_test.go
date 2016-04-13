// +build experimental

package main

import (
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestExperimentalVersion(c *check.C) {
	out, _ := dockerCmd(c, "version")
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "Experimental (client):") || strings.HasPrefix(line, "Experimental (server):") {
			c.Assert(line, check.Matches, "*true")
		}
	}

	out, _ = dockerCmd(c, "-v")
	if !strings.Contains(out, ", experimental") {
		c.Fatalf("docker version did not contain experimental: %s", out)
	}
}
