package main

import (
	"net/http"
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestInfoApi(c *check.C) {
	endpoint := "/info"

	status, body, err := sockRequest("GET", endpoint, nil)
	c.Assert(status, check.Equals, http.StatusOK)
	c.Assert(err, check.IsNil)

	// always shown fields
	stringsToCheck := []string{
		"ID",
		"Containers",
		"Images",
		"ExecutionDriver",
		"LoggingDriver",
		"OperatingSystem",
		"NCPU",
		"MemTotal",
		"KernelVersion",
		"Driver"}

	out := string(body)
	for _, linePrefix := range stringsToCheck {
		if !strings.Contains(out, linePrefix) {
			c.Errorf("couldn't find string %v in output", linePrefix)
		}
	}
}
