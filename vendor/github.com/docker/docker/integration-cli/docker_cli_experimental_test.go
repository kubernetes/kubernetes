package main

import (
	"strings"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestExperimentalVersionTrue(c *check.C) {
	testExperimentalInVersion(c, ExperimentalDaemon, "*true")
}

func (s *DockerSuite) TestExperimentalVersionFalse(c *check.C) {
	testExperimentalInVersion(c, NotExperimentalDaemon, "*false")
}

func testExperimentalInVersion(c *check.C, requirement func() bool, expectedValue string) {
	testRequires(c, requirement)
	out, _ := dockerCmd(c, "version")
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "Experimental:") {
			c.Assert(line, checker.Matches, expectedValue)
			return
		}
	}

	c.Fatal(`"Experimental" not found in version output`)
}
