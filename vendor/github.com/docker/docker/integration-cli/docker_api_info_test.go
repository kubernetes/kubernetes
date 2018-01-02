package main

import (
	"net/http"

	"encoding/json"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/request"
	"github.com/docker/docker/pkg/testutil"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestInfoAPI(c *check.C) {
	endpoint := "/info"

	status, body, err := request.SockRequest("GET", endpoint, nil, daemonHost())
	c.Assert(status, checker.Equals, http.StatusOK)
	c.Assert(err, checker.IsNil)

	// always shown fields
	stringsToCheck := []string{
		"ID",
		"Containers",
		"ContainersRunning",
		"ContainersPaused",
		"ContainersStopped",
		"Images",
		"LoggingDriver",
		"OperatingSystem",
		"NCPU",
		"OSType",
		"Architecture",
		"MemTotal",
		"KernelVersion",
		"Driver",
		"ServerVersion",
		"SecurityOptions"}

	out := string(body)
	for _, linePrefix := range stringsToCheck {
		c.Assert(out, checker.Contains, linePrefix)
	}
}

// TestInfoAPIRuncCommit tests that dockerd is able to obtain RunC version
// information, and that the version matches the expected version
func (s *DockerSuite) TestInfoAPIRuncCommit(c *check.C) {
	testRequires(c, DaemonIsLinux) // Windows does not have RunC version information

	res, body, err := request.Get("/v1.30/info")
	c.Assert(res.StatusCode, checker.Equals, http.StatusOK)
	c.Assert(err, checker.IsNil)

	b, err := testutil.ReadBody(body)
	c.Assert(err, checker.IsNil)

	var i types.Info

	c.Assert(json.Unmarshal(b, &i), checker.IsNil)
	c.Assert(i.RuncCommit.ID, checker.Not(checker.Equals), "N/A")
	c.Assert(i.RuncCommit.ID, checker.Equals, i.RuncCommit.Expected)
}

func (s *DockerSuite) TestInfoAPIVersioned(c *check.C) {
	testRequires(c, DaemonIsLinux) // Windows only supports 1.25 or later
	endpoint := "/v1.20/info"

	status, body, err := request.SockRequest("GET", endpoint, nil, daemonHost())
	c.Assert(status, checker.Equals, http.StatusOK)
	c.Assert(err, checker.IsNil)

	out := string(body)
	c.Assert(out, checker.Contains, "ExecutionDriver")
	c.Assert(out, checker.Contains, "not supported")
}
