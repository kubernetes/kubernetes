package main

import (
	"strings"

	"github.com/docker/docker/utils"
	"github.com/go-check/check"
)

// ensure docker info succeeds
func (s *DockerSuite) TestInfoEnsureSucceeds(c *check.C) {
	out, _ := dockerCmd(c, "info")

	// always shown fields
	stringsToCheck := []string{
		"ID:",
		"Containers:",
		"Images:",
		"Execution Driver:",
		"Logging Driver:",
		"Operating System:",
		"CPUs:",
		"Total Memory:",
		"Kernel Version:",
		"Storage Driver:",
	}

	if utils.ExperimentalBuild() {
		stringsToCheck = append(stringsToCheck, "Experimental: true")
	}

	for _, linePrefix := range stringsToCheck {
		if !strings.Contains(out, linePrefix) {
			c.Errorf("couldn't find string %v in output", linePrefix)
		}
	}
}
